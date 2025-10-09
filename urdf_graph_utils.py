# urdf_graph_utils.py  — urdfpy/xacro なし版（Python 3.11 + networkx 3.x対応）
from dataclasses import dataclass
from typing import Tuple, Dict
import xml.etree.ElementTree as ET
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # ヘッドレスでもPNG保存
import matplotlib.pyplot as plt
import re, math

#前処理のonoffや正規かほ法をまとめる
@dataclass
class ExtractConfig:
    drop_endeffector_like: bool = True   # finger/gripper/tool0/flange 等を末端から剪定
    movable_only_graph: bool = True      # fixed 関節を除いた“可動骨格”グラフ
    normalize_by: str = "mean_edge_len"  # "mean_edge_len" or "none"

#正規表現で刈り取るためのパターン
#FINGER_PAT = re.compile(r"(finger|gripper|hand|thumb|eef|tool0|flange)", re.I)
FINGER_PAT = re.compile(
    r"\b("
    r"finger|thumb|"                  # 指
    r"gripper|claw|pinch(?:er)?|"     # グリッパ一般
    r"hand|"                          # 手（※境界 \b で 'handle' を誤検出しにくい）
    r"eef|ee_link|ee_tip|end[_-]?effector|"   # end-effector系の略称
    r"tcp|tcp_frame|"                 # Tool Center Point
    r"tool0|tool(?:_link|_frame)?|"   # ツール基準（ROS-I慣習）
    r"flange"                         # 取付けフランジ（ISO 9409-1）
    r")\b",
    re.I
)
# ファイル先頭の定義群に追加
ANCHOR_PAT = re.compile(r"^(world|map|odom|base|base_footprint|base_frame)$", re.I)

MOVABLE = {"revolute", "continuous", "prismatic"}# 可動関節タイプ
JTYPES  = ["revolute","continuous","prismatic","fixed","planar","floating"]# 関節タイプ一覧
J2IDX   = {j:i for i,j in enumerate(JTYPES)}# 関節タイプ→インデックス

# ---------- XMLユーティリティ ----------
#_始まりは非公開関数
def _parse_xyz(s):#urdfのxyzの文字列を(x,y,z)のタプルに変換
    if not s: return (0.0,0.0,0.0)
    try:
        a = [float(x) for x in s.strip().split()]#文字列sを空白で分割してfloatに変換
        return (a[0],a[1],a[2]) if len(a)==3 else (0.0,0.0,0.0)
    except:
        return (0.0,0.0,0.0)

def _str2f(s, default=None):#文字列をfloatに変換、失敗したらdefault
    try: return float(s) if s is not None else default
    except: return default

#urdfのrootノードを返す．
def load_urdf_any(path: str):#データはET.Element型といいツリー走査用のapiを持つ
    return ET.parse(path).getroot()

# ---------- URDF(XML) → 関節グラフ ----------
def urdf_to_graph(root) -> Tuple[nx.DiGraph, Dict[Tuple[str,str], dict]]:
    G = nx.DiGraph()

    # link ノード
    for link in root.findall("link"):#urdfのlink要素について
        #各要素の取得
        name = link.attrib.get("name")
        mass = 0.0; has_inertial = 0
        inertial = link.find("inertial")#慣性フラグ

        if inertial is not None:
            m = inertial.find("mass")#質量

            if m is not None and m.attrib.get("value") is not None:#massのvalue属性があるなら
                mass = _str2f(m.attrib["value"], 0.0); has_inertial = 1

        has_visual    = 1 if link.find("visual")    is not None else 0#可視形状があるか
        has_collision = 1 if link.find("collision") is not None else 0# 当たり判定があるか
        #リンクをノードとして追加(名前,質量,慣性フラグ,可視形状フラグ,当たり判定フラグ)
        G.add_node(name, mass=float(mass), has_inertial=int(has_inertial),
                   has_visual=int(has_visual), has_collision=int(has_collision))

    # joint エッジ
    joint_attr_map = {}# (親リンク,子リンク)→関節属性辞書
    for joint in root.findall("joint"):#urdfのjoint要素について
        jname = joint.attrib.get("name")#関節名
        jtype = joint.attrib.get("type", "fixed")#関節タイプ revolute|prismatic|...
        parent = joint.find("parent").attrib.get("link")#親リンク名
        child  = joint.find("child").attrib.get("link")#子リンク名

        #xyzのどの軸に動くか(指定がなければx軸方向)
        axis  = _parse_xyz(joint.find("axis").attrib.get("xyz")) if joint.find("axis") is not None else (1.0,0.0,0.0)
        
        #親リンクから子リンクへのオフセット(xyz:0 0 0.1ならz向きに0.1加算した位置)
        orig  = joint.find("origin")
        ox,oy,oz = _parse_xyz(orig.attrib.get("xyz")) if orig is not None else (0.0,0.0,0.0)


        limit = joint.find("limit")#関節可動域
        lower = _str2f(limit.attrib.get("lower"), None) if limit is not None else None
        upper = _str2f(limit.attrib.get("upper"), None) if limit is not None else None
        width = (float(upper-lower) if lower is not None and upper is not None else 0.0)
        if jtype == "continuous": width = 2*math.pi#continuousは無限に回るが2πに設定

        dynamics = joint.find("dynamics")#減衰・摩擦係数

        #??? 減衰・摩擦係数が設定されているか
        has_damping  = 1 if (dynamics is not None and dynamics.attrib.get("damping")  is not None) else 0
        has_friction = 1 if (dynamics is not None and dynamics.attrib.get("friction") is not None) else 0
        is_mimic     = 1 if joint.find("mimic") is not None else 0

        # 関節をエッジとして追加(関節名,タイプ,軸,親リンク→子リンクオフセット,可動域幅,可動フラグ,ミミックフラグ,減衰フラグ,摩擦フラグ)
        attr = dict(
            name=jname, type=jtype, axis=tuple(axis),
            origin_xyz=(ox,oy,oz), limit_width=width,
            movable=1 if jtype in MOVABLE else 0,
            is_mimic=is_mimic, has_damping=has_damping, has_friction=has_friction
        )
        G.add_edge(parent, child, **attr)
        joint_attr_map[(parent, child)] = attr

    return G, joint_attr_map

# ---------- 前処理（剪定など） ----------
def prune_endeffector_branches(G: nx.DiGraph, leaf_only: bool = True):
    H = G.copy()#もとのグラフを壊さないようにコピー

    # 候補ノード(ノード名がFINGER_PATにマッチするノード)
    cand = {n for n in H.nodes() if n and FINGER_PAT.search(n)}

    # leaf_only=Trueなら末端ノードだけに削除対象を絞る
    if leaf_only:
        cand = {n for n in cand if H.out_degree(n) == 0}  # 末端だけ

    removed = set()
    for t in list(cand):
        if t in H:
            desc = nx.descendants(H, t)# tの子孫ノード
            removed |= desc | {t}# 削除対象に追加(|は集合の和)
            H.remove_nodes_from(desc | {t})
    return H, removed


# 可動骨格のみのグラフを作成
def make_movable_skeleton(G: nx.DiGraph) -> nx.DiGraph:#->は戻り値の型を示しているだけ
    H = nx.DiGraph()
    for n, nd in G.nodes(data=True): H.add_node(n, **nd)#すべてのノードを追加
    #すべてのエッジを走査して，movable属性が1のものだけを追加
    for u, v, ed in G.edges(data=True):
        if ed.get("movable", 0) == 1: H.add_edge(u, v, **ed)
    return H

# ノードの深さを計算（根ノードからの距離）
def _depths(S: nx.DiGraph):#S:の:は型ヒント
    roots = [n for n in S.nodes() if S.in_degree(n) == 0]#in_degreeは入次数.0なら根ノード
    root = roots[0] if roots else (next(iter(S.nodes()), None))#根ノードがなければ適当に1つ
    d = {n: 0 for n in S.nodes()}#深さ辞書
    if root is not None:#根ノードがあるなら
        for n in nx.topological_sort(S):#トポロジカルソート(有向グラフなど依存関係を考慮したソート)でノードを走査
            for _, v in S.out_edges(n): d[v] = d[n] + 1#子ノードの深さは親ノードの深さ+1
    
    return d

# ---------- 特徴抽出 ----------
#ここで今までの前処理を組み合わせて特徴抽出を行う
def graph_features(G: nx.DiGraph, cfg: ExtractConfig):
    removed_records = []  # ここに {"node": 名称, "reason": 理由} を貯める

    # 1) 手先っぽい末端の剪定
    if cfg.drop_endeffector_like:
        H, removed_eef = prune_endeffector_branches(G, leaf_only=True)
        removed_records += [{"node": n, "reason": "eef_like"} for n in sorted(removed_eef)]
    else:
        H = G.copy()

    # 2) 可動骨格のみ
    S = make_movable_skeleton(H) if cfg.movable_only_graph else H

    # 3) 孤立ノード削除
    isolates = list(nx.isolates(S))
    if isolates:
        S.remove_nodes_from(isolates)
        removed_records += [{"node": n, "reason": "isolate"} for n in sorted(isolates)]

    # 4) アンカーフレーム削除（world/base等）
    anchors = [n for n in list(S.nodes()) if ANCHOR_PAT.match(n)]
    if anchors:
        S.remove_nodes_from(anchors)
        removed_records += [{"node": n, "reason": "anchor_frame"} for n in sorted(anchors)]

    # 5) 最大連結成分のみ残す（弱連結）
    comps = list(nx.weakly_connected_components(S))
    if len(comps) > 1:
        keep = max(comps, key=len)
        drop = set().union(*[c for c in comps if c is not keep])
        if drop:
            S = S.subgraph(keep).copy()
            removed_records += [{"node": n, "reason": "non_max_component"} for n in sorted(drop)]

    # --- ここから先は既存の特徴抽出そのまま ---
    # 正規化スケールの計算
    #
    dists = [np.linalg.norm(e.get("origin_xyz", (0,0,0))) for _,_,e in S.edges(data=True)]#各ノードの距離
    scale = (np.mean(dists) if (cfg.normalize_by=="mean_edge_len" and dists) else 1.0) or 1.0#0除算防止

    depth = _depths(S)#ノードの深さ
    nodes = list(S.nodes()); idx = {n:i for i,n in enumerate(nodes)}#ノード→インデックス辞書
    X = []
    # ノード特徴量行列 X の作成
    for n in nodes:
        nd = S.nodes[n]; deg = S.degree(n)#次数
        mass = float(nd.get("mass", 0.0))#質量
        outs = list(S.out_edges(n, data=True))# nから出るエッジ
        
        #その関節の取り付け位置の平均長さ
        mean_len = float(np.mean([np.linalg.norm(ed.get("origin_xyz",(0,0,0)))/scale for _,_,ed in outs])) if outs else 0.0
        
        X.append([deg, mass,
                  float(nd.get("has_inertial",0)), float(nd.get("has_visual",0)), float(nd.get("has_collision",0)),
                  float(depth.get(n,0)), mean_len])
    X = np.asarray(X, np.float32)

    edges = list(S.edges(data=True))
    if not edges:
        edge_index = np.zeros((2,0), np.int64)
        E = np.zeros((0, 6+3+3+5), np.float32)
    else:
        ui = [idx[u] for u,_,_ in edges]; vi = [idx[v] for _,v,_ in edges]
        edge_index = np.vstack([ui,vi]).astype(np.int64)
        rows=[]
        for _,_,ed in edges:
            one = np.zeros(len(JTYPES), np.float32); one[J2IDX.get(ed.get("type","fixed"),3)] = 1.0
            ax  = np.array(ed.get("axis",(1,0,0)), np.float32)
            ox,oy,oz = ed.get("origin_xyz",(0,0,0)); org = np.array([ox/scale, oy/scale, oz/scale], np.float32)
            rows.append(np.hstack([one, ax, org,
                                   [float(ed.get("movable",0)), float(ed.get("limit_width",0.0)),
                                    float(ed.get("is_mimic",0)), float(ed.get("has_damping",0)), float(ed.get("has_friction",0))]]))
        E = np.asarray(rows, np.float32)

    # 追加：removed_records も返す
    return S, nodes, X, edge_index, E, scale, removed_records


def urdf_to_feature_graph(path: str, cfg: ExtractConfig = ExtractConfig()):
    root = load_urdf_any(path)
    G, _ = urdf_to_graph(root)
    return graph_features(G, cfg)  # 7要素に増える（… , removed_records）


# ---------- レイアウト＆描画 ----------
def _depths_layout(S: nx.DiGraph): return _depths(S)
def layered_layout(S: nx.DiGraph):
    depth = _depths_layout(S); levels={}
    for n,d in depth.items(): levels.setdefault(d, []).append(n)
    pos={}
    for d,ns in levels.items():
        xs = [0.0] if len(ns)==1 else np.linspace(-1.2, 1.2, len(ns))
        for x,n in zip(xs, ns): pos[n] = (float(x), float(-d))
    for n in S.nodes():
        if n not in pos: pos[n] = (0.0, 0.0)
    return pos

def draw_graph_png(S: nx.DiGraph, out_png: str, title: str):
    plt.figure(figsize=(8,6))
    nx.draw_networkx(S, pos=layered_layout(S), with_labels=True, arrows=True, node_size=600)
    plt.title(title); plt.axis('off'); plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

# ---------- (任意) PyTorch Geometric ----------
def to_pyg(S, node_list, X, edge_index, E, y: float | None = None):
    import torch
    from torch_geometric.data import Data
    data = Data(x=torch.tensor(X, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(E, dtype=torch.float32))
    if y is not None: data.y = torch.tensor([y], dtype=torch.float32)
    return data

# ---------------------------
'''
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
'''