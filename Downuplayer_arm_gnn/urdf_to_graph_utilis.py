# urdf_graph_utils.py — 安全化版（regex pruningなし / edge属性なし / joint情報は子ノード）
# 目的別にセクションを整理。関数本体の中身は変更していません。

from __future__ import annotations

# =========================================================
# 0) Imports
#    - 標準 / サードパーティを整理（Aggはpyplot前）
# =========================================================
# ----- Stdlib -----
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Dict, List
import xml.etree.ElementTree as ET
from pathlib import Path
from glob import glob
import math
import json
import sys
import csv
import csv as _csv 
# ----- Third-party -----
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # ヘッドレス環境でPNG保存可（pyplotより前）
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data


# =========================================================
# 1) 定数 / 設定（ジョイント種別・特徴名・正規化列）
# =========================================================
@dataclass
class ExtractConfig:
    """
    前処理のスイッチは最小限。正規表現での剪定は行わない。
    normalize_by:
      - "mean_edge_len": 親→子origin距離の平均で origin_xyz を割る
      - "none": 距離正規化を行わない
    """
    normalize_by: str = "mean_edge_len"  # "mean_edge_len" or "none"

# URDF標準ジョイントタイプ
JTYPES  = ["revolute", "continuous", "prismatic", "fixed", "planar", "floating"]
J2IDX   = {j: i for i, j in enumerate(JTYPES)}
MOVABLE = {"revolute", "continuous", "prismatic"}

JTYPE_ORDER = ["revolute", "continuous", "prismatic", "fixed", "floating", "planar"]
JTYPE_TO_IDX = {name: i for i, name in enumerate(JTYPE_ORDER)}

# === 追加: 一番上の定数群の近くに置く =========================
SHORT_NAME_MAP = {
    "jtype_is_revolute":  "j_revo",
    "jtype_is_continuous":"j_cont",
    "jtype_is_prismatic": "j_pris",
    "jtype_is_fixed":     "j_fix",
    "jtype_is_floating":  "j_float",
    "jtype_is_planar":    "j_plan",
    # sin/cos も短くしたければ
    "lower_sin":"lo_sin", "lower_cos":"lo_cos",
    "upper_sin":"up_sin", "upper_cos":"up_cos",
}



# 特徴名（19次元の内訳）/ Z正規化対象列
# 埋め込み「前」の列名（19列）
FEATURE_NAMES = (
    ["deg", "depth", "mass"] +
    [f"jtype_is_{n}" for n in JTYPE_ORDER] +
    ["axis_x", "axis_y", "axis_z",
     "origin_x", "origin_y", "origin_z",
     "movable", "width",
     "lower", "upper"]   # ← ココを lower/upper に戻す
)

#upperとlowerは角度なのでsin/cosに変換前
#DEFAULT_Z_COLS = [0, 1, 2, 9, 10, 11, 12, 13, 14, 16, 17, 18]
# 角度をsin/cosに変換する場合
DEFAULT_Z_COLS = [0, 1, 2, 9, 10, 11, 12, 13, 14, 16]
# =========================================================
# 2) 正規化ユーティリティ（Z統計 / 適用 / 表示 / 逆変換 / サニタイズ）
# =========================================================
'''
# Z正規化統計をデータセット全体から計算
def compute_global_z_stats_from_dataset(
    dataset: List[Data],
    z_cols: List[int] = DEFAULT_Z_COLS,
    eps: float = 1e-8
) -> Dict[str, Any]:
    stacked = []
    for d in dataset:
        A = d.x.detach().cpu().numpy()[:, z_cols].astype(np.float64, copy=True)
        A[~np.isfinite(A)] = np.nan  # 非有限は無視対象に
        stacked.append(A)
    M = np.vstack(stacked)
    mu = np.nanmean(M, axis=0)
    sd = np.nanstd(M,  axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where((sd > 0) & np.isfinite(sd), sd, 1.0)
    return {"z_cols": list(z_cols), "mean": mu.tolist(), "std": sd.tolist(), "eps": float(eps)}

def apply_global_z_inplace_to_dataset(dataset: List[Data], stats: Dict[str, Any]) -> None:
    import torch
    cols = stats["z_cols"]
    mu = torch.tensor(stats["mean"], dtype=torch.float32)
    sd = torch.tensor(stats["std"],  dtype=torch.float32)
    for d in dataset:
        mu_d = mu.to(d.x.device)
        sd_d = sd.to(d.x.device)
        d.x[:, cols] = (d.x[:, cols] - mu_d) / sd_d

def print_z_stats(stats: Dict[str, Any], feature_names=FEATURE_NAMES) -> None:
    zcols = stats["z_cols"]
    mu = np.asarray(stats["mean"])
    sd = np.asarray(stats["std"])
    print("-" * 72)
    print(f"{'col':>3} | {'feat':<18} | {'mu':>12} | {'sigma':>12}")
    print("-" * 72)
    for c, m, s in zip(zcols, mu, sd):
        fname = (feature_names[c] if 0 <= c < len(feature_names) else f"f{c}")
        print(f"{c:>3} | {fname:<18} | {m:>12.6g} | {s:>12.6g}")
    print("-" * 72)

def dump_normalized_feature_table(
    X: np.ndarray,
    stats: Dict[str, Any],
    max_rows: int = 5,
    show_header: bool = True
) -> None:
    cols = stats["z_cols"]
    mu = np.asarray(stats["mean"], dtype=np.float32)
    sd = np.asarray(stats["std"],  dtype=np.float32)
    Xn = X.copy()
    Xn[:, cols] = (Xn[:, cols] - mu) / sd

    N, F = Xn.shape
    show_n = N if (max_rows is None or max_rows >= N) else max_rows
    headers = ["node_index"] + [f"f{i}" for i in range(F)]
    def fmt_row(i): return [str(i)] + [f"{float(v):.6g}" for v in Xn[i]]

    rows = [fmt_row(i) for i in range(show_n)]
    colw = [len(h) for h in headers]
    for r in rows:
        for j, cell in enumerate(r):
            colw[j] = max(colw[j], len(cell))

    def fmt_line(parts): return " | ".join(p.rjust(colw[i]) for i, p in enumerate(parts))
    rule = "-" * (sum(colw) + 3 * (len(colw) - 1))
    print(rule)
    if show_header:
        print(fmt_line(headers)); print(rule)
    for r in rows: print(fmt_line(r))
    if show_n < N: print(f"... (showing {show_n}/{N} nodes)")
    print(rule)

def sanitize_dataset_inplace(dataset: List[Data]) -> None:
    import torch
    for d in dataset:
        x = d.x
        bad = ~torch.isfinite(x)
        if bad.any():
            x[bad] = 0.0
        if x.size(1) > 16:  # width 列
            x[:, 16] = torch.clamp(x[:, 16], min=0.0)

def _denorm_batch(xn: torch.Tensor, stats: Dict[str, Any]) -> torch.Tensor:
    """
    xn: (N,F) 正規化後
    stats: {"z_cols": [...], "mean": [...], "std": [...]}
    return: (N,F) 元スケール
    """
    x = xn.clone()
    cols = stats["z_cols"]
    mu = torch.tensor(stats["mean"], dtype=x.dtype, device=x.device)
    sd = torch.tensor(stats["std"],  dtype=x.dtype, device=x.device)
    x[:, cols] = xn[:, cols] * sd + mu
    return x
'''
#特徴量のうち周期性のあるupperとlowerの角度をsin/cosに変換して埋め込む
# === 角度の sin/cos 埋め込み =========================
ANGLE_COLS = [17, 18]  # lower, upper (rad)
ANGLE_NEW_NAMES = {
    17: ("lower_sin", "lower_cos"),
    18: ("upper_sin", "upper_cos"),
}

def embed_angles_sincos(X: torch.Tensor, feature_names: list[str]) -> tuple[torch.Tensor, list[str]]:
    cols = sorted(ANGLE_COLS)
    new_feats = []
    new_names = []
    cset = set(cols)
    for j, name in enumerate(feature_names):
        if j in cset:
            theta = X[:, j]
            new_feats.append(torch.sin(theta).unsqueeze(1))
            new_feats.append(torch.cos(theta).unsqueeze(1))
            sn, cn = ANGLE_NEW_NAMES[j]
            new_names += [sn, cn]
        else:
            new_feats.append(X[:, j].unsqueeze(1))
            new_names.append(name)
    X2 = torch.cat(new_feats, dim=1)
    return X2, new_names

# ======== Min-Max 正規化ユーティリティ =========
def compute_global_minmax_stats_from_dataset(
    dataset: List[Data],
    z_cols: List[int] = DEFAULT_Z_COLS,
    eps: float = 1e-8
) -> Dict[str, Any]:
    """
    データセット全体で各列の min/max を計算（NaN/Inf は無視）。
    max==min の列は幅を1にフォールバックしてゼロ割を回避。
    """
    stacked = []
    for d in dataset:
        A = d.x.detach().cpu().numpy()[:, z_cols].astype(np.float64, copy=True)
        A[~np.isfinite(A)] = np.nan
        stacked.append(A)
    M = np.vstack(stacked)
    vmin = np.nanmin(M, axis=0)
    vmax = np.nanmax(M, axis=0)

    # 非有限はデフォ値へ、幅0は1へ
    vmin = np.where(np.isfinite(vmin), vmin, 0.0)
    vmax = np.where(np.isfinite(vmax), vmax, 1.0)
    width = vmax - vmin
    width = np.where((width > 0) & np.isfinite(width), width, 1.0)

    return {
        "z_cols": list(z_cols),
        "min": vmin.tolist(),
        "max": vmax.tolist(),
        "width": width.tolist(),  # = max - min（幅0は1に置換済み）
        "eps": float(eps),
        "method": "minmax",
    }

def apply_global_minmax_inplace_to_dataset(dataset: List[Data], stats: Dict[str, Any]) -> None:
    cols  = stats["z_cols"]
    vmin  = torch.tensor(stats["min"],   dtype=torch.float32)
    width = torch.tensor(stats["width"], dtype=torch.float32)
    for d in dataset:
        vmin_d  = vmin.to(d.x.device)
        width_d = width.to(d.x.device)
        d.x[:, cols] = (d.x[:, cols] - vmin_d) / width_d

def print_minmax_stats(stats: Dict[str, Any], feature_names=FEATURE_NAMES) -> None:
    cols  = stats["z_cols"]
    vmin  = np.asarray(stats["min"])
    vmax  = np.asarray(stats["max"])
    print("-" * 72)
    print(f"{'col':>3} | {'feat':<18} | {'min':>12} | {'max':>12}")
    print("-" * 72)
    for c, mn, mx in zip(cols, vmin, vmax):
        fname = (feature_names[c] if 0 <= c < len(feature_names) else f"f{c}")
        print(f"{c:>3} | {fname:<18} | {mn:>12.6g} | {mx:>12.6g}")
    print("-" * 72)

def _denorm_batch(xn: torch.Tensor, stats: Dict[str, Any]) -> torch.Tensor:
    """
    xn: (N,F) 正規化後
    stats: Z用({mean,std}) または Min-Max用({min,max,width}) を許容
    return: (N,F) 元スケール
    """
    x = xn.clone()
    cols = stats["z_cols"]
    if "std" in stats and "mean" in stats:  # Zスコア
        mu = torch.tensor(stats["mean"], dtype=x.dtype, device=x.device)
        sd = torch.tensor(stats["std"],  dtype=x.dtype, device=x.device)
        x[:, cols] = xn[:, cols] * sd + mu
    elif "min" in stats and ("width" in stats or "max" in stats):  # Min-Max
        vmin  = torch.tensor(stats["min"],   dtype=x.dtype, device=x.device)
        width = torch.tensor(stats.get("width", (torch.tensor(stats["max"], dtype=x.dtype, device=x.device) - vmin).tolist()),
                             dtype=x.dtype, device=x.device)
        x[:, cols] = xn[:, cols] * width + vmin
    else:
        # 未知の形式→そのまま返す（もしくは例外でもよい）
        pass
    return x

def dump_normalized_feature_table(
    Xn,
    stats,
    max_rows: int = 5,
    feature_names: list[str] | None = None,
    cols_per_block: int = 8,
    show_orig: bool = True,
):
    """
    - z_cols を複数ブロックに分けて横幅を抑えた表で表示
    - normalized と original を「別テーブル」で出すので折り返しで崩れにくい
    - 数値は 0 を "0"、-0 を "0" に揃え、末尾の 0 と '.' を削って短縮
    """
    import math
    if not isinstance(Xn, torch.Tensor):
        Xn_t = torch.tensor(Xn, dtype=torch.float32)
    else:
        Xn_t = Xn.detach().clone().float()

    # 逆正規化して original も用意
    Xorig_t = _denorm_batch(Xn_t, stats)
    Xn_np    = Xn_t.cpu().numpy()
    Xorig_np = Xorig_t.cpu().numpy()

    cols = list(stats["z_cols"])
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(Xn_np.shape[1])]

    def _fmt_num(v: float) -> str:
        # 3桁程度に丸めて末尾0と小数点を削る
        try:
            fv = float(v)
        except Exception:
            return str(v)
        if not math.isfinite(fv):
            return "NaN" if math.isnan(fv) else ("Inf" if fv > 0 else "-Inf")
        s = f"{fv:.3f}"
        s = s.rstrip("0").rstrip(".")
        if s in ("-0", "-0.0", "-0."):
            s = "0"
        if s == "":
            s = "0"
        return s

    def _print_block(title: str, mat, col_ids):
        # ブロックの列名とデータ幅に合わせて整形して出力
        headers = ["row"] + [feature_names[c] for c in col_ids]
        rows = []
        R = min(len(mat), max_rows)
        for i in range(R):
            rows.append([str(i)] + [_fmt_num(mat[i, c]) for c in col_ids])

        # 幅計算
        widths = [len(h) for h in headers]
        for r in rows:
            for j, cell in enumerate(r):
                widths[j] = max(widths[j], len(cell))

        def fmt_row(parts):
            out = []
            for j, cell in enumerate(parts):
                if j == 0:  # row 番号は右寄せ
                    out.append(cell.rjust(widths[j]))
                else:       # 値・列名は右寄せ（桁を揃える）
                    out.append(cell.rjust(widths[j]))
            return " | ".join(out)

        rule = "-" * (sum(widths) + 3 * (len(widths) - 1))
        print(rule)
        print(f"{title}")
        print(rule)
        print(fmt_row(headers))
        print(rule)
        for r in rows:
            print(fmt_row(r))
        print(rule)

    # ---- 表示本体：ブロックに分割して表示 ----
    print("---- preview (normalized) ----")
    for i in range(0, len(cols), cols_per_block):
        blk = cols[i:i+cols_per_block]
        _print_block(f"[block {i//cols_per_block+1}] normalized", Xn_np, blk)

    if show_orig:
        print("---- preview (original scale) ----")
        for i in range(0, len(cols), cols_per_block):
            blk = cols[i:i+cols_per_block]
            _print_block(f"[block {i//cols_per_block+1}] original", Xorig_np, blk)


# =========================================================
# 3) XML/URDFユーティリティ（数値パース / リミット整形 / ルート読込）
# =========================================================
def _parse_xyz(s: str | None) -> tuple[float, float, float]:
    if not s:
        return (0.0, 0.0, 0.0)
    try:
        a = [float(x) for x in s.strip().split()]
        return (a[0], a[1], a[2]) if len(a) == 3 else (0.0, 0.0, 0.0)
    except Exception:
        return (0.0, 0.0, 0.0)

def _str2f_safe(s: str | None):
    """文字列→float。非有限や巨大値は None を返す（呼び出し側で既定値へ）。"""
    try:
        if s is None:
            return None
        v = float(s)
        if not np.isfinite(v):
            return None
        if abs(v) > 1e6:   # 無意味な巨大値は捨てる（必要なら閾値調整）
            return None
        return v
    except Exception:
        return None

def sanitize_joint_limits(jtype: str, lower_raw, upper_raw):
    """
    URDFの joint limit を学習向けに有限化:
      - continuous: 幅=2π, lower=upper=0 に固定
      - その他: lower/upper が無効(None/非有限/巨大)なら 0、幅= max(upper-lower, 0)
    """
    if jtype == "continuous":
        return 0.0, 0.0, float(2 * math.pi)

    lo = _str2f_safe(lower_raw)
    hi = _str2f_safe(upper_raw)
    lo = 0.0 if lo is None else float(lo)
    hi = 0.0 if hi is None else float(hi)

    width = hi - lo
    if not np.isfinite(width) or width < 0:
        width = 0.0

    return float(lo), float(hi), float(width)

def load_urdf_any(path: str) -> ET.Element:
    """URDF/XML をパースして root Element を返す。"""
    return ET.parse(path).getroot()


# =========================================================
# 4) URDF → 有向グラフ / 特徴抽出（19次元）/ 便宜関数
# =========================================================
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
                try:
                    mv = float(m.attrib["value"])
                    mass = mv if math.isfinite(mv) else 0.0
                except Exception:
                    mass = 0.0

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

            # ★ limit の安全化（continuous含む）
            limit = joint.find("limit")
            lower_raw = limit.attrib.get("lower") if limit is not None else None
            upper_raw = limit.attrib.get("upper") if limit is not None else None
            lower_sanit, upper_sanit, width = sanitize_joint_limits(jtype, lower_raw, upper_raw)

            # childノード側に格納する joint 属性
            cj = dict(
                joint_name=jname,
                joint_type=jtype,
                joint_type_idx=int(J2IDX.get(jtype, J2IDX["fixed"])),
                joint_axis=tuple(axis),
                joint_origin_xyz=(ox, oy, oz),
                joint_movable=1 if jtype in MOVABLE else 0,
                joint_limit_width=float(width),
                joint_limit_lower=float(lower_sanit),
                joint_limit_upper=float(upper_sanit),
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

def graph_features(G: nx.DiGraph, cfg: ExtractConfig):
    """
    何も削らず、構造と物理量をそのまま特徴へ。
    - ノード特徴 X: 構造 + link.mass + (child側の) joint属性
      [deg, depth, mass] + [jtype_onehot(6)] + [axis(3)] + [origin_xyz_norm(3)] + [movable, width, lower, upper]
      → 次元=19
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
        j_name = str(nd.get("joint_type", "fixed"))
        j_one, _ = jtype_to_onehot(j_name)  # ← JTYPE_ORDERに沿ったワンホットを返す
        j_one = np.array(j_one, dtype=np.float32)


        ax = np.array(nd.get("joint_axis", (0.0, 0.0, 0.0)), np.float64)
        org = np.array(nd.get("joint_origin_xyz", (0.0, 0.0, 0.0)), np.float64)
        if cfg.normalize_by == "mean_edge_len" and scale != 0:
            org = org / float(scale)

        movable = float(nd.get("joint_movable", 0))
        width   = float(nd.get("joint_limit_width", 0.0))
        lower   = float(nd.get("joint_limit_lower", 0.0))
        upper   = float(nd.get("joint_limit_upper", 0.0))

        # X の並び
        feat = [deg, dep, mass] + list(j_one.tolist()) + list(ax.tolist()) + list(org.tolist()) + \
               [movable, width, lower, upper]
        X.append(feat)

    # まずfloat64で保持→非有限は0.0に→float32へキャスト（溢れ/NaN対策）
    X = np.array(X, dtype=np.float64)
    X[~np.isfinite(X)] = 0.0
    X = X.astype(np.float32, copy=False)

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


# =========================================================
# 5) レイアウト / 描画（構造把握の可視化）
# =========================================================
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


# =========================================================
# 6) (任意) PyTorch Geometric 変換
# =========================================================
def to_pyg(S, node_list, X, edge_index, E, y: float | None = None):
    import torch
    from torch_geometric.data import Data as _Data
    data = _Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(E, dtype=torch.float32),  # [M, 0]
    )
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float32)
    return data


# =========================================================
# 7) URDF収集 / 個別ダンプ / ディレクトリ一括処理（フラット配置）
#    - 画像, nodes_processed.csv, meta.json, summary CSV まで
# =========================================================
def collect_urdf_paths(dir_path: str, exts=(".urdf", ".xml")):
    g = glob(str(Path(dir_path) / "**" / "*"), recursive=True)
    return sorted([p for p in g if Path(p).is_file() and p.lower().endswith(tuple(exts))])

def dump_one(urdf_path: str, outdir_robot: str, normalize_by="mean_edge_len"):
    outdir_robot = Path(outdir_robot)
    outdir_robot.mkdir(parents=True, exist_ok=True)

    cfg = ExtractConfig(normalize_by=normalize_by)

    root = load_urdf_any(urdf_path)
    G_full, _ = urdf_to_graph(root)
    draw_graph_png(G_full, str(outdir_robot / "urdf_full.png"), "URDF Full Joint Graph")

    S, node_list, X, edge_index, E, scale, removed_records = graph_features(G_full, cfg)

    rows = []
    for n in S.nodes():
        nd = S.nodes[n]
        rows.append({
            "node": n,
            "deg": int(S.degree(n)),
            "mass": float(nd.get("mass", 0.0)),
            "joint_name": nd.get("joint_name", ""),
            "joint_type": nd.get("joint_type", "fixed"),
            "joint_type_idx": int(nd.get("joint_type_idx", 3)),
            "joint_axis": json.dumps(list(nd.get("joint_axis", (0.0, 0.0, 0.0)))),
            "joint_origin_xyz": json.dumps(list(nd.get("joint_origin_xyz", (0.0, 0.0, 0.0)))),
            "joint_movable": int(nd.get("joint_movable", 0)),
            "joint_limit_width": float(nd.get("joint_limit_width", 0.0)),
            "joint_limit_lower": float(nd.get("joint_limit_lower", 0.0)),
            "joint_limit_upper": float(nd.get("joint_limit_upper", 0.0)),
        })
    pd.DataFrame(rows).to_csv(str(outdir_robot / "nodes_processed.csv"), index=False)

    draw_graph_png(S, str(outdir_robot / "processed.png"), "Processed Graph (no pruning / no edge attrs)")

    meta = {
        "input": str(Path(urdf_path).resolve()),
        "scale_mean_edge_len": float(scale),
        "n_nodes": int(len(S.nodes())),
        "n_edges": int(len(S.edges())),
        "x_shape": [int(X.shape[0]), int(X.shape[1])],
        "edge_index_shape": [int(edge_index.shape[0]), int(edge_index.shape[1])],
        "edge_attr_shape": [int(E.shape[0]), int(E.shape[1])],
    }
    with open(str(outdir_robot / "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def make_robot_outdir_flat(file_path: str, out_root: str, used_names: dict) -> Path:
    base = Path(file_path).stem
    name = base
    if name in used_names:
        used_names[name] += 1
        name = f"{base}_{used_names[name]}"
    else:
        used_names[name] = 1
    return Path(out_root) / name


# =========================================================
# 8) CSVユーティリティ（サイズ出力 / name昇順ソート / pre-post比較）
#    - process_dir_flat からも利用
# =========================================================
def export_graph_sizes_csv_from_urdf(urdf_paths, csv_path, normalize_by="mean_edge_len"):
    """
    入力: urdf_paths = ["/path/a.urdf", "/path/b.urdf", ...]
    出力: CSV (name, num_nodes, num_edges)
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "num_nodes", "num_edges"])

        for p in urdf_paths:
            try:
                S, nodes, X, edge_index, E, scale, _ = urdf_to_feature_graph(
                    p, ExtractConfig(normalize_by=normalize_by)
                )
                name = Path(p).name
                n_nodes = int(X.shape[0])
                n_edges = int(edge_index.shape[1])
                w.writerow([name, n_nodes, n_edges])
            except Exception as e:
                # 壊れたURDFなどは行として残しておく（num_*=-1）
                w.writerow([Path(p).name, -1, -1])
                print(f"[WARN] failed: {p} ({e})")

    print(f"[saved] {csv_path}")

def _sort_csv_by_name(csv_path: str):
    """CSVに 'name' 列がある前提で昇順ソートして上書き保存"""
    try:
        df = pd.read_csv(csv_path)
        if "name" in df.columns:
            df.sort_values("name", inplace=True, kind="mergesort")  # 安定ソート
            df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"[WARN] sort skipped for {csv_path}: {e}", file=sys.stderr)

def make_compare_csv(pre_csv_path: str, post_csv_path: str, out_csv_path: str):
    """
    pre/post の graph_sizes_summary.csv から、
    robot_name, pre_node, post_node, pre_edge, post_edge を出力。
    - pre 側のキー: stem(name)
    - post 側のキー: stem(name) から先頭の 'merge_' を除去
    - どちらにも存在するもののみ（inner join）
    """
    pre = pd.read_csv(pre_csv_path)
    post = pd.read_csv(post_csv_path)

    for df, tag in [(pre, "pre"), (post, "post")]:
        if "name" not in df.columns:
            raise SystemExit(f"[ERROR] 'name' 列が見つかりません: {tag} ({pre_csv_path if tag=='pre' else post_csv_path})")

    pre = pre.rename(columns={"name": "pre_name", "num_nodes": "pre_node", "num_edges": "pre_edge"})
    pre["pre_key"] = pre["pre_name"].map(lambda s: Path(str(s)).stem)

    post = post.rename(columns={"name": "post_name", "num_nodes": "post_node", "num_edges": "post_edge"})
    post["post_key"] = post["post_name"].map(lambda s: Path(str(s)).stem)
    post["post_key"] = post["post_key"].str.replace(r"^merge_", "", regex=True)

    comp = pd.merge(
        pre, post,
        left_on="pre_key", right_on="post_key",
        how="inner", suffixes=("_pre", "_post")
    )

    out = comp[["pre_key", "pre_node", "post_node", "pre_edge", "post_edge"]].copy()
    out = out.rename(columns={"pre_key": "robot_name"})
    out.sort_values("robot_name", inplace=True, kind="mergesort")

    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv_path, index=False)
    print(f"[saved] {out_csv_path}  ({len(out)} rows)")


# =========================================================
# 9) ディレクトリ一括処理（フラット配置ドライバ）
#    - 上の収集/個別処理/CSVを束ねるオーケストレータ
# =========================================================
def process_dir_flat(dir_path: str, out_root: str, normalize_by="mean_edge_len",
                     exts=(".urdf", ".xml")) -> list:
    Path(out_root).mkdir(parents=True, exist_ok=True)

    urdfs = collect_urdf_paths(dir_path, exts=exts)
    print(f"[INFO] scan {dir_path} -> {len(urdfs)} files (exts={exts})")
    if not urdfs:
        pd.DataFrame(columns=["name", "num_nodes", "num_edges"]).to_csv(
            str(Path(out_root) / "graph_sizes_summary.csv"), index=False
        )
        return []

    used = {}
    for p in urdfs:
        try:
            robot_dir = make_robot_outdir_flat(p, out_root, used)
            dump_one(p, str(robot_dir), normalize_by=normalize_by)
        except Exception as e:
            print(f"[WARN] failed: {p} ({e})", file=sys.stderr)

    # サマリCSV（出力後に name 昇順で並べ替え）
    csv_out = str(Path(out_root) / "graph_sizes_summary.csv")
    try:
        export_graph_sizes_csv_from_urdf(urdfs, csv_out, normalize_by=normalize_by)
    except Exception as e:
        print(f"[WARN] summary export failed: {e}", file=sys.stderr)
        pd.DataFrame({"name": [Path(p).stem for p in urdfs]}).to_csv(csv_out, index=False)
    _sort_csv_by_name(csv_out)

    return [str(Path(p).resolve()) for p in urdfs]


# =========================================================
# 10) デバッグユーティリティ（表示 / スキャン）
# =========================================================
def shorten_feature_names(names: list[str]) -> list[str]:
    return [SHORT_NAME_MAP.get(n, n) for n in names]


def jtype_to_onehot(name: str):
    v = [0]*len(JTYPE_ORDER)
    i = JTYPE_TO_IDX.get(name, None)
    if i is not None:
        v[i] = 1
    return v, (i if i is not None else -1)  # onehot, index


def print_step_header(step: int, tag: str = "train"):
    print(f"\n===== {tag.upper()} STEP {step} =====")

def print_step_footer(step: int, tag: str = "train"):
    print(f"===== END {tag.upper()} STEP {step} =====\n")  # ← 注: 本文は変更不可指定なのでそのまま

def grep_inf_in_urdf(path: str):
    bad = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            if "inf" in line.lower():  # -inf, +inf まとめて拾う
                bad.append((i, line.strip()[:200]))
    if bad:
        print(f"[URDF-INF] {path}")
        for i, s in bad[:5]:
            print(f"  L{i}: {s}")
    return bad

def assert_finite_data(d, name_hint=""):
    import torch, math
    x = d.x.detach().cpu()
    bad = ~torch.isfinite(x)
    if bad.any():
        idx = bad.nonzero(as_tuple=False)[:10]
        print(f"[NONFINITE in Data.x] {name_hint} count={bad.sum().item()}")
        for r, c in idx:
            print(f"  node={int(r)}, col={int(c)}, val={float(x[r,c])}")
        raise RuntimeError("non-finite features in Data.x")

def check_z_stats(stats):
    import math
    mu = stats["mean"]; sd = stats["std"]
    def _ok(a): return all(math.isfinite(float(v)) for v in a)
    if not (_ok(mu) and _ok(sd)):
        raise RuntimeError(f"Z-stats not finite: mu[0]={mu[0]}, sd[0]={sd[0]}")

def _normalize_spec(spec: str) -> str:
    # ":.6g" → ".6g" に直す。空ならデフォルト ".6g"
    if not spec:
        return ".6g"
    return spec[1:] if spec.startswith(":") else spec

def minimal_graph_report(d: Data, max_nodes: int = 5, float_fmt=":.6g"):
    name = getattr(d, "name", "(no name)")
    print(f"=== GRAPH: {name} ===")
    print(f"num_nodes: {d.num_nodes}  |  num_edges: {d.num_edges}")

    if getattr(d, "x", None) is None:
        print("[WARN] d.x (ノード特徴) がありません。")
        return

    F = d.x.size(1)
    feat_names = getattr(d, "feature_names", None)
    feat_names_disp = getattr(d, "feature_names_disp", None)
    if feat_names_disp:
        feat_names = feat_names_disp
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


def minimal_dataset_report(dataset: List[Data], max_graphs: int = 1, max_nodes: int = 5, float_fmt=":.6g"):
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
    import torch

    if not hasattr(data, "edge_index"):
        print("[debug_edge_index] data has no edge_index")
        return

    ei = data.edge_index
    assert ei.dim() == 2 and ei.size(0) == 2, f"edge_index shape must be [2, E], got {tuple(ei.shape)}"
    E = int(ei.size(1))

    src = ei[0].detach().cpu()
    dst = ei[1].detach().cpu()

    if E <= 2 * k:
        idxs = list(range(E))
        head_end_split = E
    else:
        idxs = list(range(k)) + list(range(E - k, E))
        head_end_split = k

    b = getattr(data, "batch", None)
    if b is not None:
        b = b.detach().cpu()

    print(f"\n[edge_index dump] {title}  shape={tuple(ei.shape)}  E={E}")
    if b is None:
        print(" index | src -> dst")
        print("-------+----------------")
    else:
        print(" index | src(g) -> dst(g)")
        print("-------+-------------------")

    for i, j in enumerate(idxs):
        if i == head_end_split and E > 2 * k:
            print("  ...  |  (skipped middle)")
        s = int(src[j]); d = int(dst[j])
        if b is None:
            print(f"{j:6d} | {s} -> {d}")
        else:
            gs = int(b[s]); gd = int(b[d])
            print(f"{j:6d} | {s}({gs}) -> {d}({gd})")

    # 反転（上り方向）を先頭いくつか表示
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

def scan_nonfinite_features(dataset: List[Data],
                            feature_names=FEATURE_NAMES,
                            max_print=80,
                            save_csv="out_compare/debug_nonfinite.csv",
                            extreme_abs=1e6):
    """
    - 正規化前の Data.x をスキャンして NaN/Inf/極端値 を報告
    """
    from csv import writer
    rows = []
    printed = 0
    total_bad = 0
    total_ext = 0

    for d in dataset:
        robot = Path(getattr(d, "name", "(unknown)")).name
        X = d.x.detach().cpu().numpy()
        N, F = X.shape

        # 非有限
        bad = ~np.isfinite(X)
        if bad.any():
            idxs = np.argwhere(bad)
            total_bad += len(idxs)
            for (i, j) in idxs:
                if printed < max_print:
                    fname = feature_names[j] if j < len(feature_names) else f"f{j}"
                    print(f"[NONFINITE] robot={robot} node={i} col={j}({fname}) value={X[i,j]!r}")
                    printed += 1
                rows.append([robot, int(i), int(j),
                             feature_names[j] if j < len(feature_names) else f"f{j}",
                             "nonfinite", "nan/inf"])

        # 極端値
        extreme = np.abs(X) > float(extreme_abs)
        extreme &= np.isfinite(X)
        if extreme.any():
            idxs2 = np.argwhere(extreme)
            total_ext += len(idxs2)
            for (i, j) in idxs2[:max_print - printed]:
                fname = feature_names[j] if j < len(feature_names) else f"f{j}"
                print(f"[EXTREME]  robot={robot} node={i} col={j}({fname}) value={X[i,j]:.6g}")
                printed += 1
            for (i, j) in idxs2:
                rows.append([robot, int(i), int(j),
                             feature_names[j] if j < len(feature_names) else f"f{j}",
                             "extreme", float(X[i,j])])

    if rows and save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            w = writer(f)
            w.writerow(["robot","node_index","col_index","feature_name","kind","value"])
            w.writerows(rows)
        print(f"[debug] wrote report: {save_csv}")

    if total_bad == 0 and total_ext == 0:
        print("[debug] nonfinite / extreme values: none")
    else:
        print(f"[debug] nonfinite={total_bad}  extreme(>|{extreme_abs}|)={total_ext}")


# =========================================================
# 11) 再構成誤差（元スケール）集計（MAE/RMSE + 任意CSV）
# =========================================================
@torch.no_grad()
def compute_recon_metrics_origscale(
    model: torch.nn.Module,
    loader,
    device,
    z_stats: Dict[str, Any],
    feature_names: Optional[List[str]] = None,
    out_csv: Optional[str] = None,                # 全体集計 (従来どおり)
    out_csv_by_robot: Optional[str] = None,       # 新規: ロボット別
    use_mask_only: bool = False,
) -> Dict[str, Any]:
    """
    モデル出力と正解の Data.x を「元スケール」に戻して誤差を集計。
    - 全体（ALL）の per-feature MAE/RMSE を out_csv へ
    - ロボット別（graph別）の per-feature MAE/RMSE を out_csv_by_robot へ
    - 出力が tuple/list/dict の場合も先頭/既定キーをとってテンソル化
    """
    def _as_tensor(y, ref: torch.Tensor) -> torch.Tensor:
        # tuple / list -> 先頭
        if isinstance(y, (tuple, list)):
            y = y[0]
        # dict -> よくあるキー名を優先
        if isinstance(y, dict):
            for k in ("pred", "recon", "logits", "output"):
                if k in y:
                    y = y[k]; break
            else:
                y = next(iter(y.values()))
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=ref.dtype, device=ref.device)
        return y

    # 特徴量数・名前
    F = loader.dataset[0].num_node_features
    feat_names = feature_names or [f"f{i}" for i in range(F)]

    # ---- 全体集計用バッファ ----
    abs_sum_all = torch.zeros(F, dtype=torch.float64, device=device)
    sqr_sum_all = torch.zeros(F, dtype=torch.float64, device=device)
    cnt_all     = torch.zeros(F, dtype=torch.float64, device=device)

    # ---- ロボット別（graph別）集計用バッファ ----
    # name -> {abs_sum, sqr_sum, count, n_nodes_total}
    per_robot: Dict[str, Dict[str, torch.Tensor]] = {}

    model.eval()
    for batch in loader:
        batch = batch.to(device)

        # まとめて推論
        raw_out = model(batch)
        target  = batch.x  # (N,F)

        # まとめてテンソル化
        pred_all = _as_tensor(raw_out, ref=target)

        # バッチ内を graph 単位に切り出す準備
        # PyG Batch には nodeの累積ポインタ batch.ptr が入る（長さ G+1）
        # 無い場合は to_data_list から長さを復元
        if hasattr(batch, "ptr") and batch.ptr is not None:
            ptr = batch.ptr.tolist()
            sizes = [ptr[i+1] - ptr[i] for i in range(len(ptr)-1)]
            data_list = batch.to_data_list()  # 名前などの属性を参照するため
        else:
            data_list = batch.to_data_list()
            sizes = [d.num_nodes for d in data_list]

        # （オプション）マスクノード（バッチ基準のindex）
        mask_idx_batch: Optional[torch.Tensor] = None
        if use_mask_only and hasattr(batch, "mask_idx") and batch.mask_idx is not None:
            mask_idx_batch = batch.mask_idx.to(device)

        # グラフごとにスライスして集計
        start = 0
        for g, (d_g, n_g) in enumerate(zip(data_list, sizes)):
            end = start + n_g
            name_g = getattr(d_g, "name", f"(graph_{g})")

            # 範囲の抽出
            pred_g   = pred_all[start:end]
            target_g = target[start:end]

            # マスク適用（各グラフ範囲に入るindexのみ）
            if mask_idx_batch is not None:
                sel = mask_idx_batch[(mask_idx_batch >= start) & (mask_idx_batch < end)] - start
                if sel.numel() > 0:
                    pred_g   = pred_g.index_select(0, sel)
                    target_g = target_g.index_select(0, sel)
                else:
                    # マスク対象が無ければこのグラフはスキップ（分母0）
                    start = end
                    continue

            # 元スケールへ
            pred_o   = _denorm_batch(pred_g,   z_stats)
            target_o = _denorm_batch(target_g, z_stats)

            # 誤差
            err = pred_o - target_o
            abs_sum = err.abs().sum(dim=0).to(torch.float64)
            sqr_sum = (err * err).sum(dim=0).to(torch.float64)
            count   = torch.tensor(pred_o.size(0), dtype=torch.float64, device=device)

            # 全体に加算
            abs_sum_all += abs_sum
            sqr_sum_all += sqr_sum
            cnt_all     += count

            # ロボット別に加算
            if name_g not in per_robot:
                per_robot[name_g] = {
                    "abs_sum": torch.zeros(F, dtype=torch.float64, device=device),
                    "sqr_sum": torch.zeros(F, dtype=torch.float64, device=device),
                    "count":   torch.zeros(F, dtype=torch.float64, device=device),
                    "n_nodes": torch.zeros(1, dtype=torch.float64, device=device),
                }
            per_robot[name_g]["abs_sum"] += abs_sum
            per_robot[name_g]["sqr_sum"] += sqr_sum
            per_robot[name_g]["count"]   += count
            per_robot[name_g]["n_nodes"] += torch.tensor(float(n_g), dtype=torch.float64, device=device)

            start = end  # 次のグラフへ

    # ===== 全体（ALL） =====
    denom = cnt_all.clamp_min(1.0)
    mae_all  = (abs_sum_all / denom).cpu().numpy()
    rmse_all = (sqr_sum_all / denom).sqrt().cpu().numpy()

    # 表示（従来どおり）
    print("\n--- Reconstruction error on ORIGINAL scale (per feature) ---")
    print(f"{'idx':>3} | {'feature':<18} | {'MAE':>12} | {'RMSE':>12}")
    print("-" * 56)
    for i in range(F):
        name = feat_names[i] if i < len(feat_names) else f"f{i}"
        print(f"{i:>3} | {name:<18} | {mae_all[i]:>12.6g} | {rmse_all[i]:>12.6g}")
    print("-" * 56)

    # ===== CSV出力 =====
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["index", "feature", "mae", "rmse"])
            for i in range(F):
                name = feat_names[i] if i < len(feat_names) else f"f{i}"
                w.writerow([i, name, float(mae_all[i]), float(rmse_all[i])])
        print(f"[metrics] wrote {out_csv}")

    if out_csv_by_robot:
        Path(out_csv_by_robot).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv_by_robot, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["robot", "n_nodes", "feature_index", "feature", "mae", "rmse"])
            # 各ロボット
            for robot, buf in sorted(per_robot.items(), key=lambda kv: str(kv[0])):
                denom_r = buf["count"].clamp_min(1.0)
                mae_r   = (buf["abs_sum"] / denom_r).cpu().numpy()
                rmse_r  = (buf["sqr_sum"] / denom_r).sqrt().cpu().numpy()
                n_nodes = float(buf["n_nodes"].item())
                for i in range(F):
                    name = feat_names[i] if i < len(feat_names) else f"f{i}"
                    w.writerow([robot, int(n_nodes), i, name, float(mae_r[i]), float(rmse_r[i])])
        print(f"[metrics] wrote {out_csv_by_robot}")

    return {
        "mae": mae_all,
        "rmse": rmse_all,
        "feature_names": feat_names,
        "per_robot": {k: {kk: vv.detach().cpu().numpy().tolist() if hasattr(vv, "detach") else float(vv)
                          for kk, vv in v.items()}
                      for k, v in per_robot.items()}
    }
def compute_feature_mean_std_from_dataset(
    dataset: Sequence[torch.Tensor] | Sequence["Data"],
    cols: Optional[Sequence[int]] = None,
    drop_nonfinite: bool = True,
    population_std: bool = True,
) -> Dict[str, Any]:
    """
    正規化前の dataset から、列ごとの mean/std/count を集計する。
    - cols=None: 先頭グラフの全列を対象
    - drop_nonfinite=True: 非有限(NaN/Inf)は無視
    - population_std=True: 母標準偏差（ddof=0）。Falseで標本標準偏差（ddof=1）
    """
    mats = []
    for d in dataset:
        X = d.x.detach().cpu().numpy()
        mats.append(X)
    M = np.vstack(mats)  # (total_nodes, F)

    if cols is None:
        cols = list(range(M.shape[1]))
    else:
        cols = list(cols)

    A = M[:, cols].astype(np.float64, copy=True)

    if drop_nonfinite:
        mask = np.isfinite(A)
        # 列ごとの有限フラグ（行方向にAND）
        valid = np.all(mask, axis=1)
        A = A[valid]

    ddof = 0 if population_std else 1
    mean = np.nanmean(A, axis=0)
    std  = np.nanstd(A, axis=0, ddof=ddof)
    cnt  = np.sum(np.all(np.isfinite(A), axis=1))

    return {
        "cols": cols,
        "mean": mean.tolist(),
        "std":  std.tolist(),
        "count": int(cnt),
        "ddof": ddof,
        "method": "data_stats",
    }

def print_feature_mean_std(stats: Dict[str, Any], feature_names=None) -> None:
    cols = stats["cols"]
    mean = stats["mean"]
    std  = stats["std"]
    print("-" * 72)
    print(f"{'col':>3} | {'feat':<18} | {'mean':>12} | {'std':>12}")
    print("-" * 72)
    for c, mu, sd in zip(cols, mean, std):
        fname = (feature_names[c] if feature_names and 0 <= c < len(feature_names) else f"f{c}")
        print(f"{c:>3} | {fname:<18} | {mu:>12.6g} | {sd:>12.6g}")
    print("-" * 72)



# =========================================================
# 12) おまけ：単体実行時のサンプル（merge後URDF群のサイズCSV）
# =========================================================
if __name__ == "__main__":
    # マージ後ロボットを拾う例
    train_list = sorted([p for p in glob("./merge_joint_robots/**/*", recursive=True)
                         if p.lower().endswith((".urdf", ".xml"))])
    export_graph_sizes_csv_from_urdf(train_list, "out/merge_graph_sizes.csv")
