import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Callable
from torch_geometric.data import Data

# 正規化対象のデフォルト列 (deg, depth, mass, origin_x, origin_y, origin_z)
DEFAULT_NORM_COLS = [0, 1, 2, 12, 13, 14]

#massの推定が不安定な場合なのでデータを読み込んだあとにmassの列を1にしてみる
def fix_mass_to_one(dataset: List[Data]) -> None:
    """
    データセット内のすべてのデータに対して、質量（mass）列を1.0に固定します。
    これにより、質量推定の不安定さを回避します。
    
    Args:
        dataset (List[Data]): PyGデータオブジェクトのリスト。
    """
    mass_col_idx = 2  # 質量列のインデックス
    for d in dataset:
        d.x[:, mass_col_idx] = 1.0

def compute_global_minmax_stats(dataset: List[Data], norm_cols: List[int] = DEFAULT_NORM_COLS, eps: float = 1e-8) -> Dict[str, Any]:
    """
    データセット全体から指定列のMin-Max統計（最小値、最大値、幅）を計算します。
    NaN/Infは無視され、値の幅が0になる列は幅を1.0としてゼロ除算を防ぎます。
    
    Args:
        dataset (List[Data]): PyGデータオブジェクトのリスト。
        norm_cols (List[int]): 正規化対象の列インデックスのリスト。
        eps (float): 幅がこの値以下の場合は0とみなす閾値。
        
    Returns:
        Dict[str, Any]: 統計情報を含む辞書（'min', 'max', 'width', 'norm_cols'など）。
    """
    stacked = []
    for d in dataset:
        A = d.x.detach().cpu().numpy()[:, norm_cols].astype(np.float64, copy=True)
        A[~np.isfinite(A)] = np.nan # 非有限値をNaNにして集計から除外
        stacked.append(A)
    
    M = np.vstack(stacked)
    vmin = np.nanmin(M, axis=0)
    vmax = np.nanmax(M, axis=0)

    # NaNが残った場合（全データが非有限だった列など）はデフォルト値で埋める
    vmin = np.where(np.isfinite(vmin), vmin, 0.0)
    vmax = np.where(np.isfinite(vmax), vmax, 1.0)
    
    width = vmax - vmin
    # 幅が 0 または非有限になってしまった場合は 1.0 にフォールバック
    width = np.where((width > eps) & np.isfinite(width), width, 1.0)

    return {
        "norm_cols": list(norm_cols),
        "min": vmin.tolist(),
        "max": vmax.tolist(),
        "width": width.tolist(),
        "method": "minmax",
    }

def apply_global_minmax_inplace(dataset: List[Data], stats: Dict[str, Any]) -> None:
    """
    計算済みの統計情報を使用して、データセットをインプレースでMin-Max正規化します。
    計算式: (x - min) / width
    
    Args:
        dataset (List[Data]): 正規化するデータセット。
        stats (Dict[str, Any]): `compute_global_minmax_stats` で計算された統計情報。
    """
    cols = stats["norm_cols"]
    vmin = torch.tensor(stats["min"], dtype=torch.float32)
    width = torch.tensor(stats["width"], dtype=torch.float32)
    
    for d in dataset:
        vmin_d = vmin.to(d.x.device)
        width_d = width.to(d.x.device)
        d.x[:, cols] = (d.x[:, cols] - vmin_d) / width_d

def denorm_batch(xn: torch.Tensor, stats: Dict[str, Any]) -> torch.Tensor:
    """
    正規化されたバッチデータを元のスケールに逆変換します。
    計算式: x_norm * width + min
    
    Args:
        xn (torch.Tensor): 正規化済みのデータテンソル [N, D]。
        stats (Dict[str, Any]): 正規化に使用した統計情報。
        
    Returns:
        torch.Tensor: 元のスケールに戻されたデータテンソル。
    """
    x = xn.clone()
    cols = stats["norm_cols"]
    vmin = torch.tensor(stats["min"], dtype=x.dtype, device=x.device)
    width = torch.tensor(stats["width"], dtype=x.dtype, device=x.device)
    x[:, cols] = xn[:, cols] * width + vmin
    return x

def make_postprocess_fn(names: List[str], snap_onehot: bool, unit_axis: bool) -> Optional[Callable]:
    """
    モデルの推論結果に対する後処理関数（one-hot化、単位ベクトル化）を作成します。
    
    Args:
        names (List[str]): 特徴量名のリスト。
        snap_onehot (bool): Trueの場合、one-hot特徴量を0/1にスナップする処理を追加。
        unit_axis (bool): Trueの場合、軸ベクトルを単位ベクトル化する処理を追加。
        
    Returns:
        Optional[Callable]: (pred, targ)を受け取り、後処理後の(pred, targ)を返す関数。処理が不要な場合はNone。
    """
    if not (snap_onehot or unit_axis):
        return None
        
    jtype_cols = [i for i, n in enumerate(names) if n.startswith("jtype_is_")]
    try:
        axis_cols = (names.index("axis_x"), names.index("axis_y"), names.index("axis_z"))
    except ValueError:
        axis_cols = None

    def _fn(pred: torch.Tensor, targ: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. one-hot スナップ (最大値を持つインデックスのみ1にする)
        if snap_onehot and jtype_cols:
            for M in (pred, targ):
                J = M[:, jtype_cols]
                if J.numel() > 0:
                    idx = torch.argmax(J, dim=1)
                    J.zero_().scatter_(1, idx.unsqueeze(1), 1.0)
                    M[:, jtype_cols] = J

        # 2. 軸ベクトルの単位化 (L2ノルムで割る)
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
    逆正規化と上記の後処理を組み合わせた複合後処理関数を作成します。
    評価ループなどで、モデルの生の出力を人間が解釈可能な形式に戻すために使用します。
    
    Args:
        stats (Dict[str, Any]): 逆正規化用の統計情報。
        names (List[str]): 特徴量名リスト。
        snap_onehot (bool): one-hotスナップを有効にするか。
        unit_axis (bool): 軸単位化を有効にするか。
        
    Returns:
        Callable: (pred_norm, targ_norm, batch) -> (pred_orig, targ_orig) を行う関数。
    """
    _snap_fn = make_postprocess_fn(names, snap_onehot, unit_axis)
    
    def post_fn(pred_norm: torch.Tensor, targ_norm: torch.Tensor, batch=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. 逆正規化
        pred_orig = denorm_batch(pred_norm, stats)
        targ_orig = denorm_batch(targ_norm, stats)
        
        # 2. 後処理 (スナップ/単位化)
        if _snap_fn is not None:
            # _snap_fn は (pred, targ) を受け取る設計
            pred_orig, targ_orig = _snap_fn(pred_orig, targ_orig)
            
        return pred_orig, targ_orig
        
    return post_fn