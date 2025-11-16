import numpy as np
import torch
import csv
import math
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Tuple, Sequence
from torch_geometric.data import Data
from urdf_core_utils import FEATURE_NAMES
from urdf_norm_utils import denorm_batch

# ========= データ検査・レポート関数 =========

def scan_nonfinite_features(dataset: List[Data], feature_names: List[str] = FEATURE_NAMES, save_csv: str = "debug_nonfinite.csv") -> None:
    """
    データセット内の特徴量をスキャンし、NaNやInfが含まれていないかチェックします。
    発見された非有限値はコンソールに表示され、CSVにも保存されます。
    """
    rows = []
    total_bad = 0
    for d in dataset:
        robot_name = Path(getattr(d, "name", "unknown")).name
        X = d.x.detach().cpu().numpy()
        bad_mask = ~np.isfinite(X)
        if bad_mask.any():
            for node_idx, col_idx in np.argwhere(bad_mask):
                total_bad += 1
                feat_name = feature_names[col_idx] if col_idx < len(feature_names) else f"f{col_idx}"
                val = X[node_idx, col_idx]
                print(f"[NONFINITE] robot={robot_name} node={node_idx} col={col_idx}({feat_name}) value={val}")
                rows.append([robot_name, node_idx, col_idx, feat_name, "nonfinite", val])

    if rows and save_csv:
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["robot", "node_idx", "col_idx", "feature_name", "kind", "value"])
            writer.writerows(rows)
        print(f"[debug] Wrote non-finite value report to: {save_csv}")
    
    if total_bad == 0:
        print("[debug] No non-finite values found.")
        print("infやNaNは見つかりませんでした。")

def grep_inf_in_urdf(path: str) -> List[Tuple[int, str]]:
    """
    URDFファイル内を単純なテキスト検索で走査し、'inf' (大文字小文字区別なし) が含まれる行を報告します。
    XMLパースエラー時などの低レベルなデバッグに有効です。
    """
    bad_lines = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, 1):
                if "inf" in line.lower():
                    bad_lines.append((i, line.strip()[:200]))
    except Exception as e:
        print(f"[WARN] grep_inf_in_urdf failed for {path}: {e}")
        return []

    if bad_lines:
        print(f"[URDF-INF] {path}")
        for i, content in bad_lines[:5]:
            print(f"  L{i}: {content}")
        if len(bad_lines) > 5:
             print(f"  ... (total {len(bad_lines)} lines)")
    return bad_lines

def minimal_graph_report(d: Data, max_nodes: int = 5, float_fmt: str = ".6g") -> None:
    """
    単一のグラフデータの簡易レポート（ノード数、エッジ数、特徴量の先頭数行）を表示します。
    """
    name = getattr(d, "name", "(no name)")
    print(f"=== GRAPH: {name} ===")
    print(f"num_nodes: {d.num_nodes}  |  num_edges: {d.num_edges}")

    if d.x is None:
        print("[WARN] d.x (node features) is missing.")
        return

    F = d.x.size(1)
    # 特徴量名がデータに付与されていればそれを使用、なければデフォルト
    feat_names = getattr(d, "feature_names_disp", getattr(d, "feature_names", [f"f{i}" for i in range(F)]))
    
    show_n = min(d.num_nodes, max_nodes) if max_nodes is not None else d.num_nodes
    x_cpu = d.x.detach().cpu().numpy()
    node_names = getattr(d, "node_names", None)

    # ヘッダ作成
    headers = ["node_idx"] + (["node_name"] if node_names is not None else []) + feat_names[:F]
    
    # データ行作成
    # 安全対策: float_fmt に余計なコロンが含まれていたら削除する
    fmt = float_fmt.lstrip(":")
    
    rows = []
    for i in range(show_n):
        row = [str(i)]
        if node_names is not None: row.append(str(node_names[i]))
        for j in range(F):
             val = x_cpu[i, j]
             if math.isnan(val): row.append("NaN")
             elif math.isinf(val): row.append("Inf" if val > 0 else "-Inf")
             else: row.append(f"{val:{fmt}}")
        rows.append(row)

    # 列幅計算と表示
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r): col_widths[i] = max(col_widths[i], len(cell))

    def _print_row(r_items):
        print(" | ".join(f"{item:>{w}}" for item, w in zip(r_items, col_widths)))

    separator = "-" * (sum(col_widths) + 3 * (len(headers) - 1))
    print(separator)
    _print_row(headers)
    print(separator)
    for r in rows: _print_row(r)
    if show_n < d.num_nodes: print(f"... (showing {show_n}/{d.num_nodes} nodes)")
    print(separator)

def minimal_dataset_report(dataset: Sequence[Data], max_graphs: int = 1, max_nodes: int = 5) -> None:
    """
    データセット内の最初の数件のグラフについて簡易レポートを表示します。
    """
    print("データセットのグラフの一例を表示します。")
    print(f"[dataset] total graphs: {len(dataset)}")
    for i, d in enumerate(dataset[:max_graphs]):
        print(f"\n--- Graph {i} ---")
        minimal_graph_report(d, max_nodes=max_nodes)

def debug_edge_index(data: Data, k: int = 10, title: str = "") -> None:
    """
    edge_index の中身（接続関係）を人間が読みやすい形式でダンプします。
    先頭と末尾の k 本を表示します。
    """
    if not hasattr(data, "edge_index"):
        print("[debug_edge_index] No edge_index found.")
        return

    ei = data.edge_index.detach().cpu()
    E = ei.size(1)
    print(f"\n[edge_index dump] {title} shape={tuple(ei.shape)} E={E}")

    # 表示するインデックスの決定
    indices = list(range(min(k, E)))
    if E > 2 * k:
        indices += list(range(E - k, E))
    elif E > k:
         indices = list(range(E))

    print(" index | src -> dst")
    print("-------+-----------")
    last_i = -1
    for i in indices:
        if i > last_i + 1: print("   ... | ...")
        print(f"{i:6d} | {ei[0, i]:4d} -> {ei[1, i]:4d}")
        last_i = i

# ========= 統計・正規化デバッグ関数 =========

def compute_feature_mean_std_from_dataset(
    dataset: Sequence[Data],
    cols: Optional[Sequence[int]] = None,
    drop_nonfinite: bool = True,
    population_std: bool = True,
) -> Dict[str, Any]:
    """
    正規化前の dataset から、列ごとの mean/std/count を集計する。
    """
    mats = []
    for d in dataset:
        X = d.x.detach().cpu().numpy()
        mats.append(X)
    M = np.vstack(mats)

    if cols is None:
        cols = list(range(M.shape[1]))
    else:
        cols = list(cols)

    A = M[:, cols].astype(np.float64, copy=True)

    if drop_nonfinite:
        mask = np.isfinite(A)
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

def print_feature_mean_std(stats: Dict[str, Any], feature_names: Optional[List[str]] = None) -> None:
    """
    計算された平均と標準偏差を表示します。
    """
    print("特徴量の平均と標準偏差を表示します。(標準偏差はデータのばらつきを表す)")
    print("\n=== Feature Mean/Std ===")
    cols = stats["cols"]
    mean = stats["mean"]
    std  = stats["std"]
    print(f"{'col':>3} | {'feat':<18} | {'mean':>12} | {'std':>12}")
    print("-" * 72)
    for c, mu, sd in zip(cols, mean, std):
        fname = (feature_names[c] if feature_names and 0 <= c < len(feature_names) else f"f{c}")
        print(f"{c:>3} | {fname:<18} | {mu:>12.6g} | {sd:>12.6g}")
    print("-" * 72)

def print_minmax_stats(stats: Dict[str, Any], feature_names: List[str] = FEATURE_NAMES) -> None:
    """
    計算されたMin-Max統計量（最小値、最大値）を見やすく表示します。
    正規化が正しく行われているかの確認に有用です。
    """
    print("minimumおよびmaximum値を表示します。(特徴量の範囲を把握するため)")
    print("minmaxしていない特徴量も含めてる(axis,origin)")
    print("\n=== Min-Max Statistics ===")
    cols = stats.get("norm_cols", [])
    vmin = stats.get("min", [])
    vmax = stats.get("max", [])
    
    print(f"{'col':>3} | {'feature':<20} | {'min':>12} | {'max':>12} | {'width':>12}")
    print("-" * 68)
    for i, (c, mn, mx) in enumerate(zip(cols, vmin, vmax)):
        fname = feature_names[c] if c < len(feature_names) else f"f{c}"
        wd = mx - mn
        print(f"{c:>3d} | {fname:<20} | {mn:>12.6g} | {mx:>12.6g} | {wd:>12.6g}")
    print("-" * 68)

def dump_normalized_feature_table(Xn: np.ndarray, stats: Dict[str, Any], feature_names: List[str] = FEATURE_NAMES, max_rows: int = 5) -> None:
    """
    正規化された特徴量テーブルの一部を表示し、さらにそれを逆正規化した値も併記して確認します。
    """
    print("正規化された特徴量とその逆正規化後の値を表示します。")
    print("選定したノードは最初の数行です。")
    print("\n=== Normalized Feature Preview ===")
    # PyTorchテンソルに変換して逆正規化
    Xn_t = torch.as_tensor(Xn, dtype=torch.float32)
    X_orig = denorm_batch(Xn_t, stats).numpy()
    
    cols = stats.get("norm_cols", [])
    print(f"Showing first {max_rows} rows for {len(cols)} normalized columns.")

    for i in range(min(Xn.shape[0], max_rows)):
        print(f"\n--- Node {i} ---")
        print(f"{'feature':<20} | {'Original':>12} | {'Normalized':>12}")
        print("-" * 50)
        for c in cols:
            fname = feature_names[c] if c < len(feature_names) else f"f{c}"
            print(f"{fname:<20} | {X_orig[i, c]:>12.6g} | {Xn[i, c]:>12.6g}")

# ========= モデル評価・学習デバッグ関数 =========

def print_step_header(step: int, tag: str = "train"):
    print(f"\n===== {tag.upper()} STEP {step} =====")

def print_step_footer(step: int, tag: str = "train"):
    print(f"===== END {tag.upper()} STEP {step} =====\n")

@torch.no_grad()
def compute_recon_metrics_origscale(
    model,
    loader,
    device,
    z_stats: Optional[Dict[str, Any]],
    feature_names: Optional[List[str]],
    out_csv: Optional[str] = None,
    out_csv_by_robot: Optional[str] = None,
    use_mask_only: bool = False,
    postprocess_fn: Optional[Callable] = None,
    mask_mode: str = "none",
    mask_k: int = 1,
    mask_seed: Optional[int] = None,
    reduction: str = "mean",
):
    """
    元スケール（逆正規化後）で per-feature の誤差(MAE/RMSE)を集計して表示・CSV保存する。
    """
    model.eval()
    _rng = np.random.RandomState(mask_seed) if mask_seed is not None else np.random

    err_rows: List[torch.Tensor] = []
    targ_rows: List[torch.Tensor] = []
    per_robot_err: Dict[str, List[torch.Tensor]] = {}
    per_robot_targ: Dict[str, List[torch.Tensor]] = {}
    individual_results: List[dict] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            if isinstance(pred, (tuple, list)): pred = pred[0]
            targ = batch.x

            if postprocess_fn is not None:
                out = postprocess_fn(pred, targ, batch)
                if isinstance(out, (tuple, list)): pred, targ = out[0], out[1]
                else: pred, targ = out

            sel = None
            if use_mask_only:
                if hasattr(batch, "mask_idx"):
                    sel = batch.mask_idx.to(pred.device)
                elif hasattr(batch, "ptr"):
                    ptr = batch.ptr.detach().cpu().numpy()
                    picks = []
                    for gi in range(len(ptr) - 1):
                        a, b = int(ptr[gi]), int(ptr[gi + 1])
                        if b <= a: continue
                        if mask_mode == "one":
                            picks.append(int(_rng.randint(a, b)))
                        elif mask_mode == "k":
                            k = min(max(1, int(mask_k)), b - a)
                            idxs = _rng.choice(np.arange(a, b), size=k, replace=False)
                            picks.extend([int(i) for i in idxs])
                        else:
                            picks.extend(range(a, b))
                    if picks: sel = torch.tensor(picks, dtype=torch.long, device=pred.device)
                
                if sel is None: sel = torch.arange(pred.shape[0], device=pred.device)

            err = pred - targ
            if sel is not None:
                targ_sel = targ.index_select(0, sel)
                err = err.index_select(0, sel)
            else:
                targ_sel = targ

            err_rows.append(err.detach().cpu())
            targ_rows.append(targ_sel.detach().cpu())

            if sel is not None and (individual_results is not None or out_csv_by_robot is not None):
                M = sel.shape[0]
                graph_ids = batch.batch[sel]
                local_indices = sel - batch.ptr[graph_ids]
                robot_names = getattr(batch, "name", ["unknown"] * batch.num_graphs)
                e_cpu, t_cpu = err.detach().cpu(), targ_sel.detach().cpu()
                for i in range(M):
                    rname = Path(robot_names[graph_ids[i].item()]).name
                    individual_results.append({"robot": rname, "local_node_idx": local_indices[i].item(), "err": e_cpu[i], "targ": t_cpu[i]})
                    if out_csv_by_robot:
                        per_robot_err.setdefault(rname, []).append(e_cpu[i].unsqueeze(0))
                        per_robot_targ.setdefault(rname, []).append(t_cpu[i].unsqueeze(0))

    if not err_rows:
        print("[WARN] No data to evaluate.")
        return

    E = torch.cat(err_rows, dim=0)
    T = torch.cat(targ_rows, dim=0)
    D = E.shape[1]

    theta_deg_mean = np.nan
    try:
        _ft = feature_names if (feature_names and len(feature_names) == D) else [f"f{i}" for i in range(D)]
        ax, ay, az = _ft.index("axis_x"), _ft.index("axis_y"), _ft.index("axis_z")
        P = T + E
        vp, vt = P[:, [ax, ay, az]], T[:, [ax, ay, az]]
        vp = vp / (vp.norm(dim=1, keepdim=True) + 1e-9)
        vt = vt / (vt.norm(dim=1, keepdim=True) + 1e-9)
        #theta_deg_mean = torch.rad2deg(torch.acos((vp * vt).sum(dim=1).clamp(-1.0, 1.0))).mean().item()
        dot = (vp * vt).sum(dim=1).clamp(-1.0, 1.0)  # ← .abs() を削除し、clampの上限を-1.0に
        theta_deg_mean = torch.rad2deg(torch.acos(dot)).mean().item()
    except ValueError: pass

    mae_all = E.abs().mean(dim=0).numpy() if reduction == "mean" else E.abs().sum(dim=0).numpy()
    rmse_all = torch.sqrt((E**2).mean(dim=0)).numpy() if reduction == "mean" else torch.sqrt((E**2).sum(dim=0)).numpy()
    overall_mae = E.abs().mean().item() if reduction == "mean" else E.abs().sum().item()

    print(f"[TEST] masked={use_mask_only} mode={mask_mode} k={mask_k} red={reduction} recon={overall_mae:.4f}\n")
    feat_names = feature_names if (feature_names and len(feature_names) == D) else [f"f{i}" for i in range(D)]
    print("--- Reconstruction Error by Feature ---")
    print("全データをまとめて計算した誤差を表示")
    print(f"{'idx':>3} | {'feature':<18} | {'MAE':>12} | {'RMSE':>12}")
    print("-" * 56)
    for i in range(D):
        print(f"{i:>3} | {feat_names[i]:<18} | {mae_all[i]:>12.6g} | {rmse_all[i]:>12.6g}")
    print("-" * 56)
    
    print("\naxisベクトルの平均角度誤差θ")
    if not np.isnan(theta_deg_mean): print(f"Mean Axis Angle Error: {theta_deg_mean:.4f} deg\n")

    # === エラー率の計算 (分母を targ_abs_mean に変更) ===
    targ_abs_mean = T.abs().mean(dim=0).clamp(min=1e-9).numpy()
    mae_rate = mae_all / targ_abs_mean
    rmse_rate = rmse_all / targ_abs_mean
    order = np.argsort(mae_rate)[::-1]

    print("\n--- Error rate by Mean Absolute Target (sorted by MAE rate, desc) ---")
    print("test各データからランダム1ノード選択し，再現したときの誤差率を表示")
    print(f"{'rank':>4} | {'idx':>3} | {'feature':<18} | {'MAE':>9} | {'|Targ|':>9} | {'(MAE/|Targ|)%':>13} | {'(RMSE/|Targ|)%':>15}")
    print("-" * 80)
    for r, i in enumerate(order, 1):
        print(f"{r:>4} | {i:>3} | {feat_names[i]:<18} | {mae_all[i]:>9.4g} | {targ_abs_mean[i]:>9.4g} | {100.0*mae_rate[i]:>13.2f} | {100.0*rmse_rate[i]:>15.2f}")
    print("-" * 80)

    if individual_results:
        print("\n--- Reconstruction error per Node (MAE) ---")
        print(f"{'Robot':<20} | {'Node':>4} | {'MAE_All':>9} | {'|Targ|_All':>11}")
        print("-" * 58)
        for res in individual_results[:20]:
             print(f"{res['robot']:<20} | {res['local_node_idx']:>4} | {res['err'].abs().mean().item():>9.4g} | {res['targ'].abs().mean().item():>11.4g}")
        if len(individual_results) > 20: print(f"... (total {len(individual_results)} nodes evaluated)")

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "feature", "mae", "rmse", "targ_abs_mean", "mae_rate", "rmse_rate", "masked", "mask_mode", "overall_mae"])
            for i in range(D):
                w.writerow([i, feat_names[i], mae_all[i], rmse_all[i], targ_abs_mean[i], mae_all[i]/targ_abs_mean[i], rmse_all[i]/targ_abs_mean[i], use_mask_only, mask_mode, overall_mae])

    if out_csv_by_robot:
        Path(out_csv_by_robot).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv_by_robot, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["robot", "idx", "feature", "mae", "rmse", "targ_abs_mean", "mae_rate"])
            for rname, errs in per_robot_err.items():
                Ei, Ti = torch.cat(errs, dim=0), torch.cat(per_robot_targ[rname], dim=0)
                mae_i = Ei.abs().mean(dim=0).numpy()
                rmse_i = torch.sqrt((Ei**2).mean(dim=0)).numpy()
                tm_i = Ti.abs().mean(dim=0).clamp(min=1e-9).numpy()
                for j in range(D):
                    w.writerow([rname, j, feat_names[j], mae_i[j], rmse_i[j], tm_i[j], mae_i[j]/tm_i[j]])
