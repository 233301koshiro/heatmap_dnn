import xml.etree.ElementTree as ET
import math
import numpy as np
import networkx as nx
import torch
from typing import Optional, Tuple, Dict, List
from torch_geometric.data import Data

# ========= 定数定義 =========
JTYPE_ORDER = ["revolute", "continuous", "prismatic", "fixed", "floating", "planar"]
JTYPE_TO_IDX = {name: i for i, name in enumerate(JTYPE_ORDER)}

FEATURE_NAMES = (
    ["deg", "depth", "mass"] +
    [f"jtype_is_{n}" for n in JTYPE_ORDER] +
    ["axis_x", "axis_y", "axis_z",
     "origin_x", "origin_y", "origin_z"]
)

SHORT_NAME_MAP = {
    "jtype_is_revolute":  "j_revo", "jtype_is_continuous":"j_cont",
    "jtype_is_prismatic": "j_pris", "jtype_is_fixed":     "j_fix",
    "jtype_is_floating":  "j_float","jtype_is_planar":    "j_plan",
}

# ========= ヘルパー関数 =========
def assert_finite_data(d, name_hint=""):
    """
    データに非有限値（NaN/Inf）が含まれていないことを検証します。
    あれば詳細を表示して例外をスローします。
    """
    x = d.x.detach().cpu()
    bad = ~torch.isfinite(x)
    if bad.any():
        idx = bad.nonzero(as_tuple=False)[:10]
        print(f"[NONFINITE in Data.x] {name_hint} count={bad.sum().item()}")
        for r, c in idx:
            print(f"  node={int(r)}, col={int(c)}, val={float(x[r,c])}")
        raise RuntimeError("non-finite features in Data.x")


def _parse_xyz(s: Optional[str]) -> Tuple[float, float, float]:
    """
    URDFの 'x y z' 文字列をパースしてfloatのタプルに変換します。
    
    Args:
        s (Optional[str]): 空白区切りの数値文字列（例: "1.0 0.0 -0.5"）。Noneまたは不正な形式の場合は (0.0, 0.0, 0.0) を返します。
        
    Returns:
        Tuple[float, float, float]: (x, y, z) 座標。
    """
    if not s: return (0.0, 0.0, 0.0)
    try:
        a = [float(x) for x in s.strip().split()]
        return (a[0], a[1], a[2]) if len(a) == 3 else (0.0, 0.0, 0.0)
    except Exception: return (0.0, 0.0, 0.0)

def jtype_to_onehot(name: str) -> Tuple[List[int], int]:
    """
    ジョイントタイプ名をone-hotベクトルとインデックスに変換します。
    
    Args:
        name (str): ジョイントタイプ名（例: "revolute"）。
        
    Returns:
        Tuple[List[int], int]: (one-hotリスト, インデックス)。未知のタイプの場合は全て0のリストと-1を返します。
    """
    v = [0] * len(JTYPE_ORDER)
    i = JTYPE_TO_IDX.get(name, None)
    if i is not None:
        v[i] = 1
        return v, i
    return v, -1

def shorten_feature_names(names: List[str]) -> List[str]:
    """
    特徴量名のリストを短縮形に変換します（表示用）。
    
    Args:
        names (List[str]): 元の特徴量名のリスト。
        
    Returns:
        List[str]: 短縮された特徴量名のリスト。
    """
    return [SHORT_NAME_MAP.get(n, n) for n in names]

# ========= コア関数 (URDF -> Graph) =========
def load_urdf_any(path: str) -> ET.Element:
    """
    指定されたパスのURDFファイルを読み込み、XMLのルート要素を返します。
    
    Args:
        path (str): URDFファイルのパス。
        
    Returns:
        ET.Element: XMLツリーのルート要素。
    """
    return ET.parse(path).getroot()

def urdf_to_graph(root: ET.Element) -> Tuple[nx.DiGraph, Dict[str, dict]]:
    """
    URDFのXMLルート要素からNetworkXの有向グラフを構築します。
    ノードはリンク、エッジはジョイントに対応します。
    
    Args:
        root (ET.Element): URDFのXMLルート要素。
        
    Returns:
        Tuple[nx.DiGraph, Dict[str, dict]]: (構築されたグラフ, 子リンク名をキーとするジョイント属性の辞書)。
    """
    G = nx.DiGraph()
    # 1. Linkノード作成
    for link in root.findall("link"):
        name = link.attrib.get("name")
        mass = 0.0
        inertial = link.find("inertial")
        if inertial is not None:
            m = inertial.find("mass")
            if m is not None:
                try:
                    mv = float(m.attrib.get("value", 0.0))
                    mass = mv if math.isfinite(mv) else 0.0
                except: pass
        G.add_node(name, mass=float(mass))

    # 2. Jointエッジ・属性作成
    child_joint_attr = {}
    fixed_idx = JTYPE_TO_IDX["fixed"]
    for joint in root.findall("joint"):
        jname = joint.attrib.get("name")
        jtype = joint.attrib.get("type", "fixed")
        parent = joint.find("parent").attrib.get("link")
        child = joint.find("child").attrib.get("link")
        
        axis = _parse_xyz(joint.find("axis").attrib.get("xyz")) if joint.find("axis") is not None else (1.0, 0.0, 0.0)
        orig = joint.find("origin")
        ox, oy, oz = _parse_xyz(orig.attrib.get("xyz")) if orig is not None else (0.0, 0.0, 0.0)

        child_joint_attr[child] = dict(
            joint_name=jname,
            joint_type=jtype,
            joint_type_idx=JTYPE_TO_IDX.get(jtype, fixed_idx),
            joint_axis=tuple(axis),
            joint_origin_xyz=(ox, oy, oz),
        )
        G.add_edge(parent, child)

    # 3. ルートノードなどへのデフォルト値適用
    for n in G.nodes():
         if n not in child_joint_attr:
             G.nodes[n].update(dict(
                 joint_name="", joint_type="fixed", joint_type_idx=fixed_idx,
                 joint_axis=(0.0, 0.0, 0.0), joint_origin_xyz=(0.0, 0.0, 0.0)
             ))
         else:
             G.nodes[n].update(child_joint_attr[n])
    return G, child_joint_attr

def get_graph_depths(S: nx.DiGraph) -> Dict[str, int]:
    """
    グラフの各ノードのルートからの深さを計算します。
    
    Args:
        S (nx.DiGraph): 入力有向グラフ。
        
    Returns:
        Dict[str, int]: ノード名をキー、深さを値とする辞書。サイクルがある場合は計算可能な範囲で返すか、全て0になります。
    """
    roots = [n for n in S.nodes() if S.in_degree(n) == 0]
    d = {n: 0 for n in S.nodes()}
    if roots:
        try:
            for n in nx.topological_sort(S):
                for _, v in S.out_edges(n):
                    d[v] = d[n] + 1
        except nx.NetworkXUnfeasible:
            pass # サイクル検出時、デフォルトの0を維持
    return d

def layered_layout(S: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    """
    グラフの階層構造に基づいたノード配置（レイアウト）を計算します。
    深さをY座標、同じ深さのノード群をX座標に展開します。
    
    Args:
        S (nx.DiGraph): 入力有向グラフ。
        
    Returns:
        Dict[str, Tuple[float, float]]: ノード名をキー、(x, y)座標を値とする辞書。
    """
    depth = get_graph_depths(S)
    levels: Dict[int, List[str]] = {}
    for n, d in depth.items():
        levels.setdefault(d, []).append(n)
    pos = {}
    for d, ns in levels.items():
        # 同じ深さのノードを横に並べる
        xs = np.linspace(-1.2, 1.2, len(ns)) if len(ns) > 1 else [0.0]
        for x, n in zip(xs, ns):
            pos[n] = (float(x), float(-d))
    return pos

def graph_features(G: nx.DiGraph, cfg=None) -> Tuple[nx.DiGraph, List[str], np.ndarray, np.ndarray, np.ndarray, float, List]:
    """
    NetworkXグラフからGNN用の数値特徴量を抽出します。
    
    Args:
        G (nx.DiGraph): 入力グラフ。
        cfg (Optional): 設定オブジェクト（現在は未使用、将来の拡張用）。
        
    Returns:
        Tuple containing:
            - S (nx.DiGraph): 処理済みグラフ（コピー）。
            - nodes (List[str]): ノード名のリスト。
            - X (np.ndarray): ノード特徴量行列 [num_nodes, num_features]。
            - edge_index (np.ndarray): エッジインデックス [2, num_edges]。
            - E (np.ndarray): エッジ特徴量（現在は空）。
            - scale (float): スケールファクタ（常に1.0）。
            - removed (List): 削除された要素（現在は空）。
    """
    S = G.copy()
    nodes = list(S.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    depth = get_graph_depths(S)
    X = []

    for n in nodes:
        nd = S.nodes[n]
        deg = float(S.degree(n))
        dep = float(depth.get(n, 0))
        mass = float(nd.get("mass", 0.0))
        j_name = str(nd.get("joint_type", "fixed"))
        j_one, _ = jtype_to_onehot(j_name)
        
        ax = np.array(nd.get("joint_axis", (0.0, 0.0, 0.0)), dtype=np.float64)
        ax_norm = np.linalg.norm(ax)
        if ax_norm > 1e-9:
            ax = ax / ax_norm
            
        org = np.array(nd.get("joint_origin_xyz", (0.0, 0.0, 0.0)), dtype=np.float64)

        feat = [deg, dep, mass] + j_one + ax.tolist() + org.tolist()
        X.append(feat)

    X_np = np.array(X, dtype=np.float32)
    X_np[~np.isfinite(X_np)] = 0.0 # NaN/Infを0で置換

    edges = list(S.edges())
    if edges:
        edge_index = np.vstack([[node_idx[u] for u, v in edges], [node_idx[v] for u, v in edges]]).astype(np.int64)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    
    E = np.zeros((edge_index.shape[1], 0), dtype=np.float32)
    return S, nodes, X_np, edge_index, E, 1.0, []

def urdf_to_feature_graph(path: str, cfg=None):
    """
    URDFファイルパスから直接特徴量付きグラフを生成する便利関数です。
    
    Args:
        path (str): URDFファイルのパス。
        cfg (Optional): 設定オブジェクト。
        
    Returns:
        See `graph_features` return values.
    """
    root = load_urdf_any(path)
    G, _ = urdf_to_graph(root)
    return graph_features(G, cfg)

def to_pyg(X: np.ndarray, edge_index: np.ndarray, E: np.ndarray, y: Optional[float] = None) -> Data:
    """
    NumPy配列形式のグラフデータをPyTorch GeometricのDataオブジェクトに変換します。
    
    Args:
        X (np.ndarray): ノード特徴量 [num_nodes, num_features]。
        edge_index (np.ndarray): エッジインデックス [2, num_edges]。
        E (np.ndarray): エッジ属性 [num_edges, num_edge_features]。
        y (Optional[float]): グラフ全体のラベル（任意）。
        
    Returns:
        Data: PyTorch Geometricのデータオブジェクト。
    """
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(E, dtype=torch.float32)
    )
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float32)
    return data