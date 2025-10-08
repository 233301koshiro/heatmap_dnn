# urdf_graph_utils.py
# pip install urdfpy networkx numpy matplotlib
from dataclasses import dataclass
from typing import Tuple, Dict, List
import numpy as np
import networkx as nx
import re
import subprocess
import math

# ヘッドレス環境でもPNG保存できるように
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from urdfpy import URDF

# ---------------------------
# 設定
# ---------------------------
@dataclass
class ExtractConfig:
    drop_endeffector_like: bool = True   # finger/gripper/tool0/flange を末端から剪定
    movable_only_graph: bool = True      # fixed 関節を除いた骨格グラフを作る
    normalize_by: str = "mean_edge_len"  # 距離の正規化尺度: "mean_edge_len" or "none"

FINGER_PAT = re.compile(r"(finger|gripper|hand|thumb|eef|tool0|flange)", re.I)
MOVABLE = {"revolute", "continuous", "prismatic"}
JTYPES = ["revolute", "continuous", "prismatic", "fixed", "planar", "floating"]
J2IDX = {j:i for i,j in enumerate(JTYPES)}

# ---------------------------
# ロード（xacroにも対応）
# ---------------------------
def load_urdf_any(path: str) -> URDF:
    if path.endswith(".xacro"):
        # xacro コマンドが必要（例: sudo apt install ros-noetic-xacro）
        xml = subprocess.check_output(["xacro", path]).decode("utf-8")
        return URDF.from_xml_string(xml)
    return URDF.load(path)

# urdfpy の Joint.origin から平行移動xyzを安全に取り出す
def _joint_origin_xyz(j) -> Tuple[float,float,float]:
    try:
        o = getattr(j, "origin", None)
        if o is None:
            return (0.0, 0.0, 0.0)
        # urdfpyの多くは4x4行列
        if isinstance(o, np.ndarray) and o.shape == (4, 4):
            t = o[:3, 3]
            return (float(t[0]), float(t[1]), float(t[2]))
        # Transformライクにtranslation属性がある場合
        if hasattr(o, "translation"):
            t = o.translation
            return (float(t[0]), float(t[1]), float(t[2]))
        # タプル/リスト保険
        if isinstance(o, (list, tuple)) and len(o) >= 3:
            return (float(o[0]), float(o[1]), float(o[2]))
    except Exception:
        pass
    return (0.0, 0.0, 0.0)

# ---------------------------
# 末端から“手先っぽい”ものを剪定（グラフ上で）
# ---------------------------
def prune_endeffector_branches(G: nx.DiGraph) -> nx.DiGraph:
    H = G.copy()
    # リンク名とジョイント名の両方を対象に
    targets = set()
    targets |= {n for n in H.nodes() if n and FINGER_PAT.search(n)}
    for u, v, ed in H.edges(data=True):
        jn = ed.get("name", "")
        if FINGER_PAT.search(jn):
            targets.add(v)
    # 末端から子孫を削除
    for t in list(targets):
        if t in H:
            descendants = nx.descendants(H, t)
            H.remove_nodes_from(descendants | {t})
    return H

# ---------------------------
# URDF -> 全ジョイントの有向グラフ
# ノード属性: mass, has_inertial, has_visual, has_collision
# エッジ属性: type, axis(3), origin_xyz(3), limit_width, movable(0/1),
#             is_mimic, has_damping, has_friction, name
# ---------------------------
def urdf_to_graph(urdf: URDF) -> Tuple[nx.DiGraph, Dict[Tuple[str,str], dict]]:
    G = nx.DiGraph()

    # ノード（リンク）
    for link in urdf.links:
        mass = 0.0
        has_inertial = 0
        if link.inertial is not None and link.inertial.mass is not None:
            mass = float(link.inertial.mass)
            has_inertial = 1
        has_visual = 1 if (link.visuals and len(link.visuals)>0) else 0
        has_collision = 1 if (link.collisions and len(link.collisions)>0) else 0
        G.add_node(link.name,
                   mass=mass,
                   has_inertial=has_inertial,
                   has_visual=has_visual,
                   has_collision=has_collision)

    # エッジ（ジョイント）
    joint_attr_map: Dict[Tuple[str,str], dict] = {}
    for j in urdf.joints:
        parent = j.parent
        child = j.child
        jt = j.joint_type or "fixed"
        axis = tuple(j.axis) if j.axis is not None else (1.0, 0.0, 0.0)
        xyz = _joint_origin_xyz(j)
        lower = j.limit.lower if (j.limit and j.limit.lower is not None) else None
        upper = j.limit.upper if (j.limit and j.limit.upper is not None) else None
        limit_width = 0.0
        if lower is not None and upper is not None:
            limit_width = float(upper - lower)
        if jt == "continuous":
            limit_width = 2*math.pi

        is_mimic = 1 if getattr(j, "mimic", None) is not None else 0
        has_damping = 1 if (j.dynamics and j.dynamics.damping is not None) else 0
        has_friction = 1 if (j.dynamics and j.dynamics.friction is not None) else 0

        attr = dict(
            name=j.name,
            type=jt,
            axis=axis,
            origin_xyz=xyz,
            limit_width=limit_width,
            movable=1 if jt in MOVABLE else 0,
            is_mimic=is_mimic,
            has_damping=has_damping,
            has_friction=has_friction
        )
        G.add_edge(parent, child, **attr)
        joint_attr_map[(parent, child)] = attr

    return G, joint_attr_map

# ---------------------------
# 可動骨格グラフを抽出（fixed を除外）
# ---------------------------
def make_movable_skeleton(G: nx.DiGraph) -> nx.DiGraph:
    H = nx.DiGraph()
    for n, nd in G.nodes(data=True):
        H.add_node(n, **nd)
    for u, v, ed in G.edges(data=True):
        if ed.get("movable", 0) == 1:
            H.add_edge(u, v, **ed)
    return H

def _compute_depths(S: nx.DiGraph):
    roots = [n for n in S.nodes() if S.in_degree(n)==0]
    root = roots[0] if roots else (list(S.nodes())[0] if S.nodes() else None)
    depth = {n: 0 for n in S.nodes()}
    if root is not None:
        for n in nx.topological_sort(S):
            for _, v in S.out_edges(n):
                depth[v] = depth[n] + 1
    return depth

# ---------------------------
# 特徴量の組み立て
#  - ノード行列 X: [deg, mass, has_inertial, has_visual, has_collision, depth, mean_out_len_norm]
#  - エッジ行列 E: onehot(type:6) + axis(3) + origin_xyz_norm(3) + [movable, limit_width, is_mimic, has_damping, has_friction]
#  - edge_index: (2, M)
# ---------------------------
def graph_features(G: nx.DiGraph, cfg: ExtractConfig):
    # 1) 指/ツール剪定
    H = prune_endeffector_branches(G) if cfg.drop_endeffector_like else G.copy()

    # 2) 可動骨格
    S = make_movable_skeleton(H) if cfg.movable_only_graph else H

    # 3) 正規化スケール
    dists = []
    for _,_,e in S.edges(data=True):
        dx,dy,dz = e.get("origin_xyz", (0.0,0.0,0.0))
        dists.append(float(np.linalg.norm([dx,dy,dz])))
    scale = np.mean(dists) if (cfg.normalize_by=="mean_edge_len" and len(dists)>0) else 1.0
    if not scale or scale <= 1e-9:
        scale = 1.0

    # 4) 深さ
    depth = _compute_depths(S)

    # 5) ノード特徴
    node_list = list(S.nodes())
    node_index = {n:i for i,n in enumerate(node_list)}
    X = []
    for n in node_list:
        nd = S.nodes[n]
        deg = S.degree(n)
        mass = float(nd.get("mass", 0.0))
        has_inertial = float(nd.get("has_inertial", 0))
        has_visual = float(nd.get("has_visual", 0))
        has_collision = float(nd.get("has_collision", 0))
        d = float(depth.get(n, 0))

        outs = list(S.out_edges(n, data=True))
        if outs:
            lens = []
            for _,_,ed in outs:
                dx,dy,dz = ed.get("origin_xyz", (0.0,0.0,0.0))
                lens.append(np.linalg.norm([dx,dy,dz]) / scale)
            mean_out_len = float(np.mean(lens))
        else:
            mean_out_len = 0.0

        X.append([deg, mass, has_inertial, has_visual, has_collision, d, mean_out_len])
    X = np.asarray(X, dtype=np.float32)

    # 6) エッジ特徴
    edges = list(S.edges(data=True))
    if not edges:
        edge_index = np.zeros((2,0), dtype=np.int64)
        E = np.zeros((0, 6 + 3 + 3 + 5), dtype=np.float32)
    else:
        idx_u = [node_index[u] for u,_,_ in edges]
        idx_v = [node_index[v] for _,v,_ in edges]
        edge_index = np.vstack([idx_u, idx_v]).astype(np.int64)

        e_rows = []
        for u, v, ed in edges:
            onehot = np.zeros(len(JTYPES), dtype=np.float32)
            onehot[J2IDX.get(ed.get("type","fixed"), 3)] = 1.0
            axis = np.array(ed.get("axis", (1.0,0.0,0.0)), dtype=np.float32)
            ox,oy,oz = ed.get("origin_xyz", (0.0,0.0,0.0))
            origin_xyz = np.array([ox/scale, oy/scale, oz/scale], dtype=np.float32)
            movable = float(ed.get("movable", 0))
            limit_width = float(ed.get("limit_width", 0.0))
            is_mimic = float(ed.get("is_mimic", 0))
            has_damping = float(ed.get("has_damping", 0))
            has_friction = float(ed.get("has_friction", 0))
            e_rows.append(np.hstack([onehot, axis, origin_xyz,
                                     [movable, limit_width, is_mimic, has_damping, has_friction]]))
        E = np.asarray(e_rows, dtype=np.float32)

    return S, node_list, X, edge_index, E, scale

# ---------------------------
# ワンストップ：URDFパス -> (Graph, 特徴)
# ---------------------------
def urdf_to_feature_graph(path: str, cfg: ExtractConfig = ExtractConfig()):
    urdf = load_urdf_any(path)
    G_all, _ = urdf_to_graph(urdf)
    return graph_features(G_all, cfg)

# ---------------------------
# (任意) PyTorch Geometric へ変換
# ---------------------------
def to_pyg(S, node_list, X, edge_index, E, y: float = None):
    import torch
    from torch_geometric.data import Data
    x = torch.tensor(X, dtype=torch.float32)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(E, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float32)
    return data

# ---------------------------
# レイアウト＆描画（比較用PNG）
# ---------------------------
def _compute_depths_for_layout(S: nx.DiGraph):
    return _compute_depths(S)

def layered_layout(S: nx.DiGraph):
    depth = _compute_depths_for_layout(S)
    levels = {}
    for n, d in depth.items():
        levels.setdefault(d, []).append(n)
    pos = {}
    for d, nodes in levels.items():
        xs = [0.0] if len(nodes) == 1 else np.linspace(-1.2, 1.2, len(nodes))
        for x, n in zip(xs, nodes):
            pos[n] = (float(x), float(-d))
    for n in S.nodes():
        if n not in pos:
            pos[n] = (0.0, 0.0)
    return pos

def draw_graph_png(S: nx.DiGraph, out_png: str, title: str):
    plt.figure(figsize=(8, 6))
    pos = layered_layout(S)
    nx.draw_networkx(S, pos=pos, with_labels=True, arrows=True, node_size=600)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
# ---------------------------

# 例: ABB IRB7600 の xacro を URDFに展開しなくてもOK（xacroも可）
S, node_list, X, edge_index, E, scale = urdf_to_feature_graph(
    "/home/irsl/heatmap_dnn/gnn_arm_dataset/ur5_gripper.urdf",  # or .xacro
    ExtractConfig(
        drop_endeffector_like=True,   # 指/グリッパ/ツールの末端剪定するかどうか選ぶ
        movable_only_graph=True,      # 可動骨格だけでグラフを作るかどうか選ぶ
        normalize_by="mean_edge_len", # 親→子オフセットの平均長で正規化（"none"も可）
    )
)

# (任意) network x→PyTorch Geometric に
# data = to_pyg(S, node_list, X, edge_index, E, y=None)
