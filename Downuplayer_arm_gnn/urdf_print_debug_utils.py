import numpy as np
import torch
import csv
import math
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable, Tuple, Sequence
from torch_geometric.data import Data
from urdf_core import FEATURE_NAMES
from urdf_norm import denorm_batch

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

def minimal_graph_report(d: Data, max_nodes: int = 5, float_fmt: str = ":.6g") -> None:
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
    rows = []
    for i in range(show_n):
        row = [str(i)]
        if node_names is not None: row.append(str(node_names[i]))
        for j in range(F):
             val = x_cpu[i, j]
             if math.isnan(val): row.append("NaN")
             elif math.isinf(val): row.append("Inf" if val > 0 else "-Inf")
             else: row.append(f"{val{float_fmt}}")
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

def print_minmax_stats(stats: Dict[str, Any], feature_names: List[str] = FEATURE_NAMES) -> None:
    """
    計算されたMin-Max統計量（最小値、最大値）を見やすく表示します。
    正規化が正しく行われているかの確認に有用です。
    """
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

def _compute_axis_angle_error(pred: torch.Tensor, targ: torch.Tensor, feature_names: List[str]) -> Optional[float]:
    """(内部) 軸ベクトルの角度誤差（度）の平均を計算"""
    try:
        ax, ay, az = feature_names.index("axis_x"), feature_names.index("axis_y"), feature_names.index("axis_z")
    except ValueError: return None
    vp, vt = pred[:, [ax, ay, az]], targ[:, [ax, ay, az]]
    vp = vp / (vp.norm(dim=1, keepdim=True) + 1e-9)
    vt = vt / (vt.norm(dim=1, keepdim=True) + 1e-9)
    return torch.rad2deg(torch.acos((vp * vt).sum(dim=1).clamp(-1.0, 1.0))).mean().item()

def print_step_header(step: int, tag: str = "train"):
    print(f"\n===== {tag.upper()} STEP {step} =====")

def print_step_footer(step: int, tag: str = "train"):
    print(f"===== END {tag.upper()} STEP {step} =====\n")  # ← 注: 本文は変更不可指定なのでそのまま


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
    # 追加パラメータ（debug.py 側はフラグを渡すだけ）
    mask_mode: str = "none",      # "none" | "one" | "k"
    mask_k: int = 1,              # mask_mode="k" のとき各グラフから選ぶノード数
    mask_seed: Optional[int] = None,
    reduction: str = "mean",      # "mean" | "sum"
):
    """
    元スケール（逆正規化後）で per-feature の誤差(MAE/RMSE)を集計して表示・CSV保存する。

    引数:
      - model: 予測モデル（forward: batch → pred）
      - loader: PyG DataLoader（Batch を返す想定）
      - device: torch.device
      - z_stats: min-max の統計（dict）。幅 'width' と列 'z_cols' を使って誤差率を計算する
      - feature_names: 特徴名のリスト（len = D）
      - out_csv: per-feature 集計（全体）の CSV 出力先
      - out_csv_by_robot: ロボット（ファイル）ごとの集計 CSV 出力先（任意）
      - use_mask_only: True のとき、マスクしたノードのみで集計
      - postprocess_fn: (pred, targ, batch) -> (pred_orig, targ_orig) を返す関数（逆正規化など）
      - mask_mode: "none" | "one" | "k"（各グラフで選ぶノード数のモード）
      - mask_k: mask_mode="k" のときのノード数
      - mask_seed: ノード抽出の乱数シード
      - reduction: "mean" or "sum"（誤差の縮約方法。監視用途に sum も選べる）
    """
    model.eval()

    # ===== 乱数（マスク用） =====
    _rng = np.random.RandomState(mask_seed) if mask_seed is not None else np.random

    # ===== 収集バッファ =====
    #   err_rows: すべての（マスク適用済み）ノード行の誤差を後でまとめて計算
    err_rows: List[torch.Tensor] = []
    targ_rows: List[torch.Tensor] = [] # <--- 追加
    #   per-robot 集計用
    per_robot_err: Dict[str, List[torch.Tensor]] = {}
    per_robot_targ: Dict[str, List[torch.Tensor]] = {} # ロボット別も変更する場合はここも
    individual_results: List[dict] = [] # <--- この行を追加

    # ===== 走査 =====
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # 予測 / 目標
            pred = model(batch)
            # モデルが (pred, aux, ...) を返す場合に備えて先頭を使用
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            targ = batch.x                     # [N, D]

            # 逆正規化などが必要ならここで
            if postprocess_fn is not None:
                out = postprocess_fn(pred, targ, batch)
                # postprocess_fn が (pred, targ) か (pred, targ, …) を返す場合に対応
                if isinstance(out, (tuple, list)):
                    pred, targ = out[0], out[1]
                else:
                    pred, targ = out  # もともと (pred, targ) を返す設計ならそのまま
            # ===== マスク選定 =====
            sel = None  # index tensor on device
            if use_mask_only:
                if hasattr(batch, "mask_idx"):
                    # 既にどこかで作られていればそれを使う（グローバル index 想定）
                    sel = batch.mask_idx
                    if sel.device != pred.device:
                        sel = sel.to(pred.device)
                else:
                    # この関数内でバッチから作る（各グラフのノード範囲を batch.ptr で取得）
                    if hasattr(batch, "ptr"):
                        ptr = batch.ptr.detach().cpu().numpy()  # shape [G+1]
                        picks: List[int] = []
                        for gi in range(len(ptr) - 1):
                            a, b = int(ptr[gi]), int(ptr[gi + 1])
                            if b <= a:
                                continue
                            if mask_mode == "one":
                                picks.append(int(_rng.randint(a, b)))
                            elif mask_mode == "k":
                                k = max(1, int(mask_k))
                                k = min(k, b - a)
                                if k == 1:
                                    picks.append(int(_rng.randint(a, b)))
                                else:
                                    idxs = _rng.choice(np.arange(a, b), size=k, replace=False)
                                    picks.extend([int(i) for i in idxs])
                            else:  # "none" or 未知指定 → 全ノード
                                picks.extend(range(a, b))
                        if len(picks) > 0:
                            sel = torch.as_tensor(picks, dtype=torch.long, device=pred.device)
                    # ptr がない場合は全ノード
                    if sel is None:
                        sel = torch.arange(pred.shape[0], device=pred.device)

            # ===== 誤差（元スケール） =====
            err = pred - targ                  # [N, D]
            if sel is not None:
                targ_sel = targ.index_select(0, sel) # <--- 追加
                err = err.index_select(0, sel) # マスク適用
            else: # <--- 追加 (sel=None の場合)
                targ_sel = targ # <--- 追加

            # まとめて後で集計
            err_rows.append(err.detach().cpu())
            targ_rows.append(targ_sel.detach().cpu()) 

            # ロボット（ファイル）単位の集計（任意）
            # ===== 個別ノード結果の収集 (兼 per_robot_csv 用) =====
            if sel is not None:
                M = sel.shape[0]
                graph_ids = batch.batch[sel]  # [M] tensor, graph ID for each masked node
                local_indices = sel - batch.ptr[graph_ids] # [M] tensor, local node index
                
                # 'name' は build_data で d.name = urdf_path として設定されている
                # batch.name は list[str] of graph names
                robot_names_list = getattr(batch, "name", ["unknown"] * batch.num_graphs)
                
                err_vectors_cpu = err.detach().cpu()
                targ_vectors_cpu = targ_sel.detach().cpu()
        
                for i in range(M):
                    gid = graph_ids[i].item()
                    robot_name = Path(robot_names_list[gid]).name # ファイル名
                    local_idx = local_indices[i].item()
                    err_i = err_vectors_cpu[i].unsqueeze(0)   # [1, D] tensor
                    targ_i = targ_vectors_cpu[i].unsqueeze(0) # [1, D] tensor
        
                    # 1. ターミナル表示用のリストに追加
                    individual_results.append({
                        "robot": robot_name,
                        "local_node_idx": local_idx,
                        "err": err_vectors_cpu[i],   # [D] tensor
                        "targ": targ_vectors_cpu[i], # [D] tensor
                    })
                    
                    # 2. ロボット別CSV用の辞書に追加
                    if out_csv_by_robot is not None:
                        # robot_name (ファイル名) をキーにする
                        per_robot_err.setdefault(robot_name, []).append(err_i)
                        per_robot_targ.setdefault(robot_name, []).append(targ_i)

    # ===== 全ノード（マスク適用後）の誤差を一括テンソルに =====
    if len(err_rows) == 0:
        print("[WARN] No data to evaluate in compute_recon_metrics_origscale")
        return

    E = torch.cat(err_rows, dim=0)  # [M, D] (誤差)
    T = torch.cat(targ_rows, dim=0)  # [M, D] (真値) <--- 追加
    D = E.shape[1]
    # === 追加開始: Theta Error 計算 ===
    theta_deg_mean = np.nan
    # 特徴名から axis_x, axis_y, axis_z のインデックスを探す
    try:
        # feat_names がまだ定義前なら仮で作成（通常は引数で渡ってくるか、後で定義される）
        _ft = feature_names if (feature_names is not None and len(feature_names) == D) \
              else [f"f{i}" for i in range(D)]
        ax_idx = _ft.index("axis_x")
        ay_idx = _ft.index("axis_y")
        az_idx = _ft.index("axis_z")

        # 予測値 P = T + E を復元
        P = T + E

        # ベクトルを取り出す [M, 3]
        v_pred = P[:, [ax_idx, ay_idx, az_idx]]
        v_targ = T[:, [ax_idx, ay_idx, az_idx]]

        # 正規化 (念のため。ゼロベクトル回避でイプシロンを入れる)
        v_pred = v_pred / (v_pred.norm(dim=1, keepdim=True) + 1e-9)
        v_targ = v_targ / (v_targ.norm(dim=1, keepdim=True) + 1e-9)

        # 内積 -> clamp -> acos -> degree
        dot = (v_pred * v_targ).sum(dim=1).clamp(-1.0, 1.0)
        theta_rad = torch.acos(dot)
        theta_deg = torch.rad2deg(theta_rad)
        theta_deg_mean = theta_deg.mean().item()

    except ValueError:
        # axisカラムが見つからない場合はスキップ
        pass
    # ===== 集計（mean/sum） =====
    if reduction == "sum":
        mae_all = E.abs().sum(dim=0).numpy()
        rmse_all = torch.sqrt((E ** 2).sum(dim=0)).numpy()
        overall_mae = float(E.abs().sum().item() / max(1, E.shape[0]))  # 表示用に平均も出しておく
    else:  # "mean"
        mae_all = E.abs().mean(dim=0).numpy()
        rmse_all = torch.sqrt((E ** 2).mean(dim=0)).numpy()
        overall_mae = float(E.abs().mean().item())

    # ===== 表示（ヘッダ） =====
    print(f"[TEST] recon_only_masked={use_mask_only} | mask_mode={mask_mode} | "
          f"mask_k={mask_k} | red={reduction} | recon={overall_mae:.4f}\n")

    # 特徴名
    feat_names = feature_names if (feature_names is not None and len(feature_names) == D) \
        else [f"f{i}" for i in range(D)]

    # ===== per-feature の表 =====
    print("--- Reconstruction error on ORIGINAL scale (per feature) ---")
    print(f"{'idx':>3} | {'feature':<18} | {'MAE':>12} | {'RMSE':>12}")
    print("-" * 56)
    for i in range(D):
        name_i = feat_names[i]
        print(f"{i:>3} | {name_i:<18} | {mae_all[i]:>12.6g} | {rmse_all[i]:>12.6g}")
    print("-" * 56)
    if not np.isnan(theta_deg_mean):
         print(f"Axis Angle Error (Mean): {theta_deg_mean:.4f} deg")
         print("-" * 56)
         
    # ===
    targ_abs_mean = (T.abs().mean(dim=0) + 1e-9).numpy()
    targ_abs_mean = np.where(np.isfinite(targ_abs_mean), targ_abs_mean, 1e-9)

    mae_rate = mae_all / targ_abs_mean
    rmse_rate = rmse_all / targ_abs_mean
    order = np.argsort(mae_rate)[::-1]  # 大きい順

    print("\n--- Error rate by Mean Absolute Target (sorted by MAE rate, desc) ---")
    print(f"{'rank':>4} | {'idx':>3} | {'feature':<18} | {'MAE':>9} | {'|Targ|':>9} | {'(MAE/|Targ|)%':>12} | {'(RMSE/|Targ|)%':>14}")
    print("-" * 78)
    for r, i in enumerate(order, 1):
        name_i = feat_names[i]
        print(f"{r:>4} | {i:>3} | {name_i:<18} | "
              f"{mae_all[i]:>9.4g} | {targ_abs_mean[i]:>9.4g} | " # <--- width_all を targ_abs_mean に
              f"{100.0 * mae_rate[i]:>12.2f} | {100.0 * rmse_rate[i]:>14.2f}")
    print("-" * 78)

    # ===== 個別ノードごとの誤差（サマリ） =====
    if individual_results:
        print("\n--- Reconstruction error per Node (MAE) ---")
        # ヘッダ: robot, node_idx, MAE_all, MAE_deg, MAE_mass, MAE_j_revo
        try:
            # 'deg', 'mass', 'j_revo' の列インデックスを探す
            idx_deg = feat_names.index("deg")
            idx_mass = feat_names.index("mass")
            idx_revo = feat_names.index("j_revo")
        except ValueError:
            # もし 'deg' などが見つからなかった場合 (特徴名が変わった場合など)
            print(f"{'Robot':<20} | {'Node':>4} | {'MAE_All':>9} | {'|Targ|_All':>9}")
            print("-" * 56)
            for res in individual_results:
                mae_all_node = res["err"].abs().mean().item() # <--- 変更
                targ_all = res["targ"].abs().mean().item()
                print(f"{res['robot']:<20} | {res['local_node_idx']:>4} | {mae_all_node:>9.4g} | {targ_all:>9.4g}") # <--- 変更
        else:
            # 'deg', 'mass', 'j_revo' が見つかった場合 (詳細版)
            print(f"{'Robot':<20} | {'Node':>4} | {'MAE_All':>9} | {'MAE_deg':>9} | {'MAE_mass':>9} | {'MAE_j_revo':>9}")
            print("-" * 75)
            for res in individual_results:
                mae_all_node = res["err"].abs().mean().item() # <--- 変数名を mae_all から mae_all_node に変更
                mae_deg = res["err"][idx_deg].abs().item()
                mae_mass = res["err"][idx_mass].abs().item()
                mae_revo = res["err"][idx_revo].abs().item()
                print(f"{res['robot']:<20} | {res['local_node_idx']:>4} | "
                      f"{mae_all_node:>9.4g} | {mae_deg:>9.4g} | {mae_mass:>9.4g} | {mae_revo:>9.4g}") # <--- ここも変更
            print("-" * 75)

    # ===== CSV（全体） =====
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "feature", "mae", "rmse", "width", "mae_rate", "rmse_rate",
                        "masked", "mask_mode", "mask_k", "reduction", "overall_mae"])
            for i in range(D):
                w.writerow([
                    i, feat_names[i],
                    float(mae_all[i]), float(rmse_all[i]),
                    float(targ_abs_mean[i]),
                    float(mae_rate[i]), float(rmse_rate[i]),
                    int(bool(use_mask_only)), str(mask_mode), int(mask_k), str(reduction),
                    float(overall_mae),
                ])

    # ===== CSV（ロボットごと、任意） =====
    if out_csv_by_robot:
        os.makedirs(os.path.dirname(out_csv_by_robot), exist_ok=True)
        with open(out_csv_by_robot, "w", newline="") as f:
            w = csv.writer(f)
            # ヘッダ (robot 列を追加)
            w.writerow(["robot", "idx", "feature", "mae", "rmse",
                        "targ_abs_mean", "mae_rate", "rmse_rate"])
            
            # per_robot_err と per_robot_targ のキーが一致している前提でループ
            for robot in per_robot_err.keys():
                errs = per_robot_err.get(robot, [])
                targs = per_robot_targ.get(robot, []) # <--- ロボットごとの真値を取得

                if not errs or not targs:
                    continue

                Ei = torch.cat(errs, dim=0)  # [Mi, D] (誤差)
                Ti = torch.cat(targs, dim=0) # [Mi, D] (真値) <--- 追加

                if Ei.numel() == 0:
                    continue
                
                # ロボットごとの MAE/RMSE
                if reduction == "sum":
                    mae_i = Ei.abs().sum(dim=0).numpy()
                    rmse_i = torch.sqrt((Ei ** 2).sum(dim=0)).numpy()
                else:
                    mae_i = Ei.abs().mean(dim=0).numpy()
                    rmse_i = torch.sqrt((Ei ** 2).mean(dim=0)).numpy()
                
                # [変更 1] ロボットごとの分母（|Targ|）を計算
                targ_abs_mean_i = (Ti.abs().mean(dim=0) + 1e-9).numpy()
                targ_abs_mean_i = np.where(np.isfinite(targ_abs_mean_i), targ_abs_mean_i, 1e-9)

                # [変更 2] ロボットごとの分母で誤差率を計算
                mae_rate_i = mae_i / targ_abs_mean_i
                rmse_rate_i = rmse_i / targ_abs_mean_i

                for j in range(D):
                    w.writerow([
                        robot, j, feat_names[j],
                        float(mae_i[j]), float(rmse_i[j]),
                        float(targ_abs_mean_i[j]), # [変更 3] ロボットごとの分母を書き込む
                        float(mae_rate_i[j]), float(rmse_rate_i[j]),
                    ])
