# urdf_graph_utils.py  — regex pruningなし / edge属性なし / joint情報は子ノードへ
# Python 3.11 / networkx 3.x
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List
import xml.etree.ElementTree as ET
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # ヘッドレス環境でPNG保存可
import matplotlib.pyplot as plt
import math
from typing import List
from torch_geometric.data import Data
# ========== 設定 ==========
@dataclass
class ExtractConfig:
    """
    前処理のスイッチは最小限。正規表現での剪定は行わない。
    """
    normalize_by: str = "mean_edge_len"  # "mean_edge_len" or "none"

# URDF標準ジョイントタイプ
JTYPES  = ["revolute", "continuous", "prismatic", "fixed", "planar", "floating"]
J2IDX   = {j: i for i, j in enumerate(JTYPES)}
MOVABLE = {"revolute", "continuous", "prismatic"}

# ========== XMLユーティリティ ==========
def _parse_xyz(s: str | None) -> tuple[float, float, float]:
    if not s:
        return (0.0, 0.0, 0.0)
    try:
        a = [float(x) for x in s.strip().split()]
        return (a[0], a[1], a[2]) if len(a) == 3 else (0.0, 0.0, 0.0)
    except Exception:
        return (0.0, 0.0, 0.0)

def _str2f(s: str | None, default=None):
    try:
        return float(s) if s is not None else default
    except Exception:
        return default

def load_urdf_any(path: str) -> ET.Element:
    """URDF/XML をパースして root Element を返す。"""
    return ET.parse(path).getroot()

# ========== URDF → 有向関節グラフ ==========
def urdf_to_graph(root: ET.Element) -> Tuple[nx.DiGraph, Dict[tuple[str, str], dict]]:
    """
    - ノード: link（基本属性のみ）
    - エッジ: parent→child（構造のみ。属性は付けない）
    - ★ ジョイント情報は child ノードの属性に格納する（1ノード1ジョイント前提 / ルートはNone）
    """
    G = nx.DiGraph()

    # 1) まず全リンクをノードとして追加（最小属性: mass のみ）
    for link in root.findall("link"):
        name = link.attrib.get("name")
        mass = 0.0
        inertial = link.find("inertial")
        if inertial is not None:
            m = inertial.find("mass")
            if m is not None and m.attrib.get("value") is not None:
                mass = _str2f(m.attrib["value"], 0.0) or 0.0

        # ここで has_inertial / has_visual / has_collision は持たせない（要件）
        G.add_node(name, mass=float(mass))

    # childノードへ付与するjoint属性の一時バッファ
    child_joint_attr: Dict[str, dict] = {}

    # 2) ジョイントを走査して、構造（エッジ）と child側の属性を作る
    for joint in root.findall("joint"):
        jname  = joint.attrib.get("name")
        jtype  = joint.attrib.get("type", "fixed")
        parent = joint.find("parent").attrib.get("link")
        child  = joint.find("child").attrib.get("link")

        axis = _parse_xyz(joint.find("axis").attrib.get("xyz")) if joint.find("axis") is not None else (1.0, 0.0, 0.0)
        orig = joint.find("origin")
        ox, oy, oz = _parse_xyz(orig.attrib.get("xyz")) if orig is not None else (0.0, 0.0, 0.0)

        limit = joint.find("limit")
        lower = _str2f(limit.attrib.get("lower"), None) if limit is not None else None
        upper = _str2f(limit.attrib.get("upper"), None) if limit is not None else None
        width = (float(upper - lower) if (lower is not None and upper is not None) else 0.0)
        if jtype == "continuous":
            # 表現上は 2π を幅としておく（学習で相対比較に使える）
            import math
            width = 2 * math.pi

        # ★ childノード側に格納する joint 属性（指定どおり: mimic/damping/friction は入れない）
        cj = dict(
            joint_name=jname,
            joint_type=jtype,
            joint_type_idx=int(J2IDX.get(jtype, J2IDX["fixed"])),
            joint_axis=tuple(axis),
            joint_origin_xyz=(ox, oy, oz),
            joint_movable=1 if jtype in MOVABLE else 0,
            joint_limit_width=width,
            joint_limit_lower=(float(lower) if lower is not None else 0.0),
            joint_limit_upper=(float(upper) if upper is not None else 0.0),
        )
        child_joint_attr[child] = cj

        # 構造のみのエッジ
        G.add_edge(parent, child)

    # 3) childノードへ joint属性をアタッチ（ルートはデフォルト値）
    DEFAULT_CHILD_JOINT = dict(
        joint_name="",
        joint_type="fixed",
        joint_type_idx=int(J2IDX["fixed"]),
        joint_axis=(0.0, 0.0, 0.0),
        joint_origin_xyz=(0.0, 0.0, 0.0),
        joint_movable=0,
        joint_limit_width=0.0,
        joint_limit_lower=0.0,
        joint_limit_upper=0.0,
    )
    for n in G.nodes:
        cj = child_joint_attr.get(n, DEFAULT_CHILD_JOINT)
        G.nodes[n].update(cj)

    return G, child_joint_attr

# ========== 構造ユーティリティ ==========
def _depths(S: nx.DiGraph) -> Dict[str, int]:
    """トポロジカル順に親→子で深さ（root=0）を計算。root未特定時は最初のノードを0扱い。"""
    roots = [n for n in S.nodes() if S.in_degree(n) == 0]
    root = roots[0] if roots else (next(iter(S.nodes()), None))
    d = {n: 0 for n in S.nodes()}
    if root is not None:
        for n in nx.topological_sort(S):
            for _, v in S.out_edges(n):
                d[v] = d[n] + 1
    return d

# ========== 特徴抽出 ==========
def graph_features(G: nx.DiGraph, cfg: ExtractConfig):
    """
    何も削らず、構造と物理量をそのまま特徴へ。
    - ノード特徴 X: 構造 + link.mass + (child側の) joint属性
    - エッジ特徴 E: なし（shape [M, 0]）
    """
    S = G.copy()  # 変換時に壊さない

    # 正規化スケール（親→子origin距離の平均）
    edge_offsets = []
    for u, v in S.edges():
        org = S.nodes[v].get("joint_origin_xyz", (0.0, 0.0, 0.0))
        edge_offsets.append(np.linalg.norm(org))
    scale = (np.mean(edge_offsets) if (cfg.normalize_by == "mean_edge_len" and edge_offsets) else 1.0) or 1.0

    # ノード特徴
    nodes = list(S.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    depth = _depths(S)

    X: List[List[float]] = []
    for n in nodes:
        nd = S.nodes[n]

        # 構造系
        deg = float(S.degree(n))
        dep = float(depth.get(n, 0))

        # link属性
        mass = float(nd.get("mass", 0.0))

        # child側の joint 属性（要件どおり）
        j_idx = float(nd.get("joint_type_idx", J2IDX["fixed"]))
        j_one = np.zeros(len(JTYPES), np.float32)
        j_one[int(j_idx)] = 1.0

        ax = np.array(nd.get("joint_axis", (0.0, 0.0, 0.0)), np.float32)
        org = np.array(nd.get("joint_origin_xyz", (0.0, 0.0, 0.0)), np.float32) / scale
        movable = float(nd.get("joint_movable", 0))
        width   = float(nd.get("joint_limit_width", 0.0))
        lower   = float(nd.get("joint_limit_lower", 0.0))
        upper   = float(nd.get("joint_limit_upper", 0.0))

        # X の並び（コメントで次元固定化）
        # [deg, depth, mass] + [jtype_onehot(6)] + [axis(3)] + [origin_xyz_norm(3)] + [movable, width, lower, upper]
        feat = [deg, dep, mass] + list(j_one.tolist()) + list(ax.tolist()) + list(org.tolist()) + \
               [movable, width, lower, upper]
        X.append(feat)

    X = np.asarray(X, np.float32)

    # エッジインデックス（構造のみ）
    edges = list(S.edges())
    if edges:
        ui = [idx[u] for u, _ in edges]
        vi = [idx[v] for _, v in edges]
        edge_index = np.vstack([ui, vi]).astype(np.int64)
    else:
        edge_index = np.zeros((2, 0), np.int64)

    # ★ エッジ属性は「なし」：形状は [M, 0]
    E = np.zeros((edge_index.shape[1], 0), np.float32)

    removed_records: list[dict] = []  # 互換用（今回は削除しないので空）
    return S, nodes, X, edge_index, E, scale, removed_records

def urdf_to_feature_graph(path: str, cfg: ExtractConfig = ExtractConfig()):
    root = load_urdf_any(path)
    G, _ = urdf_to_graph(root)
    return graph_features(G, cfg)

# ========== レイアウト & 描画 ==========
def _depths_layout(S: nx.DiGraph): return _depths(S)

def layered_layout(S: nx.DiGraph):
    depth = _depths_layout(S)
    levels: Dict[int, List[str]] = {}
    for n, d in depth.items():
        levels.setdefault(d, []).append(n)
    pos = {}
    for d, ns in levels.items():
        xs = [0.0] if len(ns) == 1 else np.linspace(-1.2, 1.2, len(ns))
        for x, n in zip(xs, ns):
            pos[n] = (float(x), float(-d))
    for n in S.nodes():
        if n not in pos:
            pos[n] = (0.0, 0.0)
    return pos

def draw_graph_png(S: nx.DiGraph, out_png: str, title: str):
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(S, pos=layered_layout(S), with_labels=True, arrows=True, node_size=600)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# ========== (任意) PyTorch Geometric ==========
def to_pyg(S, node_list, X, edge_index, E, y: float | None = None):
    import torch
    from torch_geometric.data import Data
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(E, dtype=torch.float32),  # [M, 0]
    )
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float32)
    return data

# ====== Dataset デバッグユーティリティ ======================================

def _normalize_spec(spec: str) -> str:
    # ":.6g" → ".6g" に直す。空ならデフォルト ".6g"
    if not spec:
        return ".6g"
    return spec[1:] if spec.startswith(":") else spec

def minimal_graph_report(d: Data, max_nodes: int = 10, float_fmt=":.6g"):
    name = getattr(d, "name", "(no name)")
    print(f"=== GRAPH: {name} ===")
    print(f"num_nodes: {d.num_nodes}  |  num_edges: {d.num_edges}")

    if getattr(d, "x", None) is None:
        print("[WARN] d.x (ノード特徴) がありません。")
        return

    F = d.x.size(1)
    feat_names = getattr(d, "feature_names", None)
    if not feat_names or len(feat_names) != F:
        feat_names = [f"f{i}" for i in range(F)]

    node_names = getattr(d, "node_names", None)
    show_n = d.num_nodes if max_nodes is None else min(d.num_nodes, max_nodes)
    x = d.x.detach().cpu()

    spec = _normalize_spec(float_fmt)

    # 文字列化（先に全行作って幅決定）
    rows = []
    for i in range(show_n):
        row = [str(i)]
        if node_names is not None:
            row.append(str(node_names[i]))

        cells = []
        for v in x[i].tolist():
            fv = float(v)
            if math.isnan(fv):
                cells.append("NaN")
            elif math.isinf(fv):
                cells.append("Inf" if fv > 0 else "-Inf")
            else:
                cells.append(format(fv, spec))
        row += cells
        rows.append(row)

    headers = ["node_index"]
    if node_names is not None:
        headers.append("node_name")
    headers += feat_names

    # 列幅
    ncols = len(headers)
    col_widths = [len(h) for h in headers]
    for r in rows:
        for c, cell in enumerate(r):
            col_widths[c] = max(col_widths[c], len(cell))

    def fmt_cell(cidx, text):
        # node_index: 右寄せ, node_name: 左寄せ, それ以外: 右寄せ
        if cidx == 0:
            return f"{text:>{col_widths[cidx]}}"
        if node_names is not None and cidx == 1:
            return f"{text:<{col_widths[cidx]}}"
        return f"{text:>{col_widths[cidx]}}"

    def rule():
        print("-" * (sum(col_widths) + 3 * (ncols - 1)))

    rule()
    print(" | ".join(fmt_cell(i, h) for i, h in enumerate(headers)))
    rule()
    for r in rows:
        print(" | ".join(fmt_cell(i, cell) for i, cell in enumerate(r)))
    if show_n < d.num_nodes:
        print(f"... (showing {show_n}/{d.num_nodes} nodes)")
    rule()

def minimal_dataset_report(dataset: List[Data], max_graphs: int = 1, max_nodes: int = 20, float_fmt=":.6g"):
    total = len(dataset)
    print(f"[dataset] total graphs: {total}")
    for gi, d in enumerate(dataset[:max_graphs]):
        print(f"\n--- Graph {gi} / {total-1} ---")
        minimal_graph_report(d, max_nodes=max_nodes, float_fmt=float_fmt)


def debug_edge_index(data, k: int = 10, title: str = ""):
    """
    edge_index の中身を human-readable にダンプする。
    - 先頭 k 本、末尾 k 本の (src→dst) を表示（E < 2k の場合は全件）。
    - batch があれば各ノードの graph_id も表示。
    """
    debug_edge_index(data, k=20, title=f"{tag} step {step}", show_up_first=0)
    ei = data.edge_index
    assert ei.dim() == 2 and ei.size(0) == 2, f"edge_index shape must be [2, E], got {tuple(ei.shape)}"
    E = ei.size(1)
    print(f"\n[edge_index dump] {title}  shape={tuple(ei.shape)}  E={E}")

    # CPU に持ってきて扱いやすく
    src = ei[0].detach().cpu()
    dst = ei[1].detach().cpu()

    # どれだけ表示するか（被り防止）
    if E <= 2 * k:
        idxs = list(range(E))
        head_end_split = E  # 全件表示
    else:
        idxs = list(range(k)) + list(range(E - k, E))
        head_end_split = k  # 先頭k/末尾kで区切る

    # batch（各ノードの属する graph_id）があれば併記
    b = getattr(data, "batch", None)
    if b is not None:
        b = b.detach().cpu()

    # ヘッダ
    if b is None:
        print(" index | src -> dst")
        print("-------+----------------")
    else:
        print(" index | src(g) -> dst(g)")
        print("-------+-------------------")

    # 本体
    for i, j in enumerate(idxs):
        if i == head_end_split and E > 2 * k:
            print("  ...  |  (skipped middle)")

        s = int(src[j]); d = int(dst[j])
        if b is None:
            print(f"{j:6d} | {s} -> {d}")
        else:
            gs = int(b[s]); gd = int(b[d])
            print(f"{j:6d} | {s}({gs}) -> {d}({gd})")

    # 片方向/反転の確認用に up の最初の数本だけも表示（任意）
    rev = ei[[1, 0]]
    rsrc = rev[0].detach().cpu(); rdst = rev[1].detach().cpu()
    show_r = min(k, E)
    print(f"\n[edge_index^T (up)] show first {show_r}:")
    for j in range(show_r):
        s = int(rsrc[j]); d = int(rdst[j])
        if b is None:
            print(f"{j:6d} | {s} -> {d}")
        else:
            gs = int(b[s]); gd = int(b[d])
            print(f"{j:6d} | {s}({gs}) -> {d}({gd})")

def print_step_header(step: int, tag: str = "train"):
    print(f"\n===== {tag.upper()} STEP {step} =====")

def print_step_footer(step: int, tag: str = "train"):
    print(f"===== END {tag.upper()} STEP {step} =====\n")


# ------------------------------------------------------------------
# 使い方メモ:
# S, nodes, X, edge_index, E, scale, _ = urdf_to_feature_graph(
#     "/path/to/robot.urdf",
#     ExtractConfig(normalize_by="mean_edge_len"),
# )
# print(X.shape)  #  ノード次元: 3(構造) + 6(jtype onehot) + 3(axis) + 3(origin) + 4(movable/width/lower/upper) = 19
# print(E.shape)  #  エッジ次元: 0
