import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Callable
from torch_geometric.data import Data

# 正規化対象のデフォルト列 (deg, depth, mass, origin_x, origin_y, origin_z)
DEFAULT_NORM_COLS = [0, 1, 2, 10, 11, 12]

def fix_mass_to_one(dataset: List[Data]) -> None:
    """
    データセット内のすべてのデータに対して、質量（mass）列を1.0に固定します。
    ※注意: MassのLog学習を行う場合は、この関数を呼び出さないでください。
    """
    mass_col_idx = 2  # 質量列のインデックス
    for d in dataset:
        d.x[:, mass_col_idx] = 1.0

def compute_global_minmax_stats(dataset: List[Data], norm_cols: List[int] = DEFAULT_NORM_COLS, eps: float = 1e-8) -> Dict[str, Any]:
    """
    データセット全体から統計情報を計算します。
    【変更点】
    - mass(2): Log10変換後に StandardScaler (平均/標準偏差)
    - origin(10-12): そのまま StandardScaler
    - その他: Min-Max Scaling
    """
    # 列ごとの処理方針を定義
    log_standard_cols = {2}          # Massは対数正規化
    standard_cols = {10, 11, 12}     # Originは標準正規化
    
    stacked = []
    for d in dataset:
        A = d.x.detach().cpu().numpy()[:, norm_cols].astype(np.float64, copy=True)
        A[~np.isfinite(A)] = np.nan
        stacked.append(A)
    
    M = np.vstack(stacked)
    
    centers = np.zeros(M.shape[1])
    scales = np.zeros(M.shape[1])

    for i in range(M.shape[1]):
        col_idx = norm_cols[i]
        col_data = M[:, i]
        valid_data = col_data[np.isfinite(col_data)]

        if len(valid_data) == 0:
            centers[i] = 0.0
            scales[i] = 1.0
            continue

        if col_idx in log_standard_cols:
            # === Log10 + StandardScaler (Mass用) ===
            # 0以下は対数がとれないので除外 (1mg以下はクリップされている前提)
            valid_data = valid_data[valid_data > 0]
            if len(valid_data) == 0:
                print(f"[Stats] Col {col_idx}: No positive values for Log10!")
                centers[i], scales[i] = 0.0, 1.0
                continue
                
            log_data = np.log10(valid_data)
            mean_val = np.mean(log_data)
            std_val = np.std(log_data)
            
            centers[i] = mean_val
            scales[i] = std_val if std_val > eps else 1.0
            print(f"[Stats] Col {col_idx} (Log-Standard): Mean={mean_val:.4f} (Log), Std={std_val:.4f} (Log)")

        elif col_idx in standard_cols:
            # === StandardScaler (Origin用) ===
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            
            centers[i] = mean_val
            scales[i] = std_val if std_val > eps else 1.0
            print(f"[Stats] Col {col_idx} (Standard): Mean={mean_val:.4f}, Std={std_val:.4f}")
            
        else:
            # === Min-Max Scaling (その他) ===
            c_min = np.min(valid_data)
            c_max = np.max(valid_data)
            width = c_max - c_min
            
            centers[i] = c_min
            scales[i] = width if width > eps else 1.0

    return {
        "norm_cols": list(norm_cols),
        "min": centers.tolist(),  # 実質 center (mean)
        "max": (centers + scales).tolist(),
        "width": scales.tolist(), # 実質 scale (std)
        "method": "hybrid_log_standard_minmax",
    }

def apply_global_minmax_inplace(dataset: List[Data], stats: Dict[str, Any]) -> None:
    """
    統計情報を使用してデータセットを正規化します。
    Mass(2)に対しては Log10 を適用してから正規化を行います。
    """
    cols = stats["norm_cols"]
    vmin = torch.tensor(stats["min"], dtype=torch.float32)
    width = torch.tensor(stats["width"], dtype=torch.float32)
    
    # クリッピング範囲 (標準偏差の10倍あれば十分)
    clip_min = -10.0
    clip_max = 10.0
    
    for d in dataset:
        # 1. Mass(2) の対数変換 (正規化の前に行う)
        if 2 in cols:
            # 0以下を防ぐためにclampしてからlog10
            d.x[:, 2] = torch.log10(torch.clamp(d.x[:, 2], min=1e-6))
            
        # 2. 正規化 (Log化されたMass、およびOrigin等はここで (x - mean) / std される)
        vmin_d = vmin.to(d.x.device)
        width_d = width.to(d.x.device)
        d.x[:, cols] = (d.x[:, cols] - vmin_d) / width_d
        
        # 3. 外れ値クリップ
        d.x[:, cols] = torch.clamp(d.x[:, cols], clip_min, clip_max)

def denorm_batch(xn: torch.Tensor, stats: Dict[str, Any]) -> torch.Tensor:
    """
    データを元のスケールに逆変換します。
    Mass(2)に対しては 10^x を適用して線形スケールに戻します。
    """
    x = xn.clone()
    cols = stats["norm_cols"]
    vmin = torch.tensor(stats["min"], dtype=x.dtype, device=x.device)
    width = torch.tensor(stats["width"], dtype=x.dtype, device=x.device)
    
    # 1. 線形逆変換: x * std + mean
    x[:, cols] = xn[:, cols] * width + vmin
    
    # 2. Mass(2) の逆対数変換 (10のべき乗)
    # colsリスト内のインデックスではなく、実際の列インデックスが2の場所を探す
    if 2 in cols:
        x[:, 2] = torch.pow(10, x[:, 2])
        
    return x

def make_postprocess_fn(names: List[str], snap_onehot: bool, unit_axis: bool) -> Optional[Callable]:
    """
    make_postprocess_fn の説明
    指定されたオプションに基づいて、予測後の後処理関数を作成します。
    - snap_onehot: True の場合、ジョイントタイプの one-hot ベクトルを最も高い値にスナップします。
    - unit_axis: True の場合
        軸ベクトル (axis_x, axis_y, axis_z) を単位ベクトルに正規化します。
    戻り値:
    - 後処理関数 (pred: torch.Tensor, targ: torch.Tensor)
        または、どちらのオプションも False の場合は None を返します。
    """
    if not (snap_onehot or unit_axis):
        return None
        
    jtype_cols = [i for i, n in enumerate(names) if n.startswith("jtype_is_")]
    try:
        axis_cols = (names.index("axis_x"), names.index("axis_y"), names.index("axis_z"))
    except ValueError:
        axis_cols = None

    def _fn(pred: torch.Tensor, targ: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if snap_onehot and jtype_cols:
            for M in (pred, targ):
                J = M[:, jtype_cols]
                if J.numel() > 0:
                    idx = torch.argmax(J, dim=1)
                    J.zero_().scatter_(1, idx.unsqueeze(1), 1.0)
                    M[:, jtype_cols] = J

        if unit_axis and axis_cols is not None:
            for M in (pred, targ):
                cx, cy, cz = axis_cols
                vec = M[:, [cx, cy, cz]]
                norm = torch.linalg.norm(vec, dim=1, keepdim=True).clamp(min=1e-9)
                M[:, [cx, cy, cz]] = vec / norm

        return pred, targ

    return _fn

def create_composite_post_fn(stats: Dict[str, Any], names: List[str], snap_onehot: bool = True, unit_axis: bool = True) -> Callable:
    """
    create_composite_post_fn の説明
    正規化の逆変換と指定された後処理を組み合わせた関数を作成します。
    具体的には、以下の処理を行います:
    1. denorm_batch を使用して正規化を元に戻す。
    2. make_postprocess_fn で作成された後処理関数を適用する (必要に応じて)。
    これにより、予測値とターゲット値の両方に対して、一貫した後処理が適用されます。
    例:
        post_fn = create_composite_post_fn(stats, names, snap_onehot=True, unit_axis=True)
        pred_orig, targ_orig = post_fn(pred_norm, targ_norm)
    なお、snap_onehot と unit_axis の両方が False の場合、denorm_batch のみが適用されます。 
    
    :param stats: 説明
    :type stats: Dict[str, Any]
    :param names: 説明
    :type names: List[str]
    :param snap_onehot: 説明
    :type snap_onehot: bool
    :param unit_axis: 説明
    :type unit_axis: bool
    :return: 説明
    :rtype: Callable[..., Any]
    """
    _snap_fn = make_postprocess_fn(names, snap_onehot, unit_axis)
    
    def post_fn(pred_norm: torch.Tensor, targ_norm: torch.Tensor, batch=None) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_orig = denorm_batch(pred_norm, stats)
        targ_orig = denorm_batch(targ_norm, stats)
        
        if _snap_fn is not None:
            pred_orig, targ_orig = _snap_fn(pred_orig, targ_orig)
            
        return pred_orig, targ_orig
        
    return post_fn