# -*- coding: utf-8 -*-
# test.py — 学習済みモデルで任意スプリット（train/val/test/all）を評価し、CSVを test/ に保存
from __future__ import annotations

import os, csv, random
from glob import glob
from typing import List, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# ---- Local utils ------------------------------------------------------------
from urdf_to_graph_utilis import (
    urdf_to_feature_graph, ExtractConfig, to_pyg,
    FEATURE_NAMES,
    embed_angles_sincos,
    shorten_feature_names,
    compute_feature_mean_std_from_dataset, print_feature_mean_std,
    compute_global_minmax_stats_from_dataset, apply_global_minmax_inplace_to_dataset,
    print_minmax_stats, dump_normalized_feature_table,
    minimal_dataset_report, assert_finite_data, scan_nonfinite_features,
    compute_recon_metrics_origscale, make_postprocess_fn,
)

from gnn_network_min import (
    MaskedTreeAutoencoder,
    eval_loss,
)

# -----------------------------------------------------------------------------
# データ: URDF → PyG（学習で使う列だけに落とす & 角度は sin/cos へ）
# -----------------------------------------------------------------------------
def build_data(urdf_path: str, normalize_by: str = "none",
               drop_feats: List[str] | None = None) -> Data:
    S, nodes, X, edge_index, E, scale, _ = urdf_to_feature_graph(
        urdf_path, ExtractConfig(normalize_by=normalize_by)
    )
    d = to_pyg(S, nodes, X, edge_index, E)
    d.name = urdf_path

    # 角度を sin/cos に展開
    X2, names2 = embed_angles_sincos(d.x, FEATURE_NAMES)

    # 除外列を適用（movable,width,lower,upper など）
    drop_feats = drop_feats or []
    expand_map = {  # lower/upper を sin/cos の2列に展開して指定できるように
        "lower": {"lower_sin", "lower_cos"},
        "upper": {"upper_sin", "upper_cos"},
    }
    drop_set = set()
    for k in drop_feats:
        drop_set |= expand_map.get(k, {k})

    keep_idx = [i for i, n in enumerate(names2) if n not in drop_set]
    d.x = X2[:, keep_idx]

    kept_names = [names2[i] for i in keep_idx]
    d.feature_names = kept_names                      # 長い正式名
    d.feature_names_disp = shorten_feature_names(kept_names)  # 短縮名（表示用）
    return d


def split_dataset(ds: List[Data], seed: int | None,
                  p_tr=0.7, p_va=0.2) -> Tuple[List[Data], List[Data], List[Data]]:
    idx = list(range(len(ds)))
    if seed is not None:
        random.seed(seed)
    random.shuffle(idx)
    N = len(ds)
    n_tr = max(1, int(N * p_tr))
    n_va = max(1, int(N * p_va))
    if n_tr + n_va > N - 1:
        n_va = max(1, N - 1 - n_tr)
        if n_va < 1:
            n_tr = max(1, N - 2); n_va = 1
    n_te = N - n_tr - n_va
    tr = [ds[i] for i in idx[:n_tr]]
    va = [ds[i] for i in idx[n_tr:n_tr+n_va]]
    te = [ds[i] for i in idx[n_tr+n_va:]]
    return tr, va, te


def load_weights_flex(model: torch.nn.Module, wpath: str):
    ckpt = torch.load(wpath, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        return ckpt.get("cfg", None)
    # state_dict 単体として読む
    model.load_state_dict(ckpt)
    return None


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate trained AE and dump CSVs to test/")
    ap.add_argument("--merge-dir", default="./merge_joint_robots",
                    help="URDF群ディレクトリ（*.urdf / *.xml を再帰探索）")
    ap.add_argument("--normalize-by", choices=["none", "mean_edge_len"], default="none",
                    help="origin距離の平均で正規化（特徴抽出時）。学習には影響しない")
    ap.add_argument("--drop-feats", default="movable,width,lower,upper",
                    help="学習に使わない特徴名（カンマ区切り）。lower/upper は sin/cos の2列を両方除外")
    ap.add_argument("--weights", required=True, help="学習済みモデル（best.pt など）")
    ap.add_argument("--eval-split", choices=["train", "val", "test", "all"], default="train",
                    help="どのスプリットで評価するか（seed が同じなら debug.py と同じ分割になる）")
    ap.add_argument("--seed", type=int, default=42, help="分割の乱数シード")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default="test", help="CSV を保存するディレクトリ（自動作成）")
    ap.add_argument("--preview", action="store_true", help="最初のグラフを表でプレビュー（任意）")
    return ap.parse_args()


def main():
    args = parse_args()

    # 1) Dataset 構築 ---------------------------------------------------------
    paths = sorted([p for p in glob(f"{args.merge_dir}/**/*", recursive=True)
                    if p.lower().endswith((".urdf", ".xml"))])
    if not paths:
        raise RuntimeError(f"URDFが見つかりません: {args.merge_dir}")

    drop_list = [s.strip() for s in (args.drop_feats or "").split(",") if s.strip()]
    dataset: List[Data] = []
    for p in paths:
        d = build_data(p, normalize_by=args.normalize_by, drop_feats=drop_list)
        if d.num_nodes == 0:
            continue
        dataset.append(d)

    if not dataset:
        raise RuntimeError("有効なグラフが0件でした。")

    # 表示用短縮名 / 正式名
    names_long  = getattr(dataset[0], "feature_names", FEATURE_NAMES)
    names_short = getattr(dataset[0], "feature_names_disp", shorten_feature_names(names_long))
    USE_SHORT = True
    NAMES_DISP = names_short if USE_SHORT else names_long

    # 2) 統計（min-max）を全体から計算 → 適用 ---------------------------------
    stats_mm = compute_global_minmax_stats_from_dataset(dataset, z_cols=list(range(len(names_long))))
    apply_global_minmax_inplace_to_dataset(dataset, stats_mm)
    for d in dataset:
        assert_finite_data(d, getattr(d, "name", "(no name)"))
    scan_nonfinite_features(dataset, extreme_abs=1e6)

    # オプション表示
    minimal_dataset_report(dataset)
    print_minmax_stats(stats_mm, feature_names=NAMES_DISP)
    if args.preview:
        x0 = dataset[0].x.detach().cpu().numpy()
        dump_normalized_feature_table(
            x0, stats_mm, max_rows=5, feature_names=NAMES_DISP,
            cols_per_block=15, show_orig=True
        )

    # 3) Split & Loader -------------------------------------------------------
    tr, va, te = split_dataset(dataset, seed=args.seed)
    split_map = {
        "train": tr,
        "val": va,
        "test": te,
        "all": dataset,
    }
    target_ds = split_map[args.eval_split]
    bs = min(args.batch_size, max(1, len(target_ds)))
    loader = DataLoader(target_ds, batch_size=bs, shuffle=False, drop_last=False)

    # 4) Model 構築 & 読み込み ------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = target_ds[0].num_node_features
    model = MaskedTreeAutoencoder(
        in_dim=in_dim,
        hidden=128,
        bottleneck_dim=128,
        enc_rounds=2,
        dec_rounds=2,
        dropout=0.1,
        mask_strategy="none",
    ).to(device)
    load_weights_flex(model, args.weights)
    model.eval()

    # 5) 正規化空間でのざっくり再構成誤差（ALLノード平均） ----------------------
    recon_norm = eval_loss(model, loader, device, recon_only_masked=False, verbose=False)
    print(f"[{args.eval_split}] mean recon (normalized space): {recon_norm:.6f}")

    # 6) 逆正規化して per-feature MAE/RMSE を CSV に保存 ------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, f"metrics_{args.eval_split}_origscale.csv")
    post_fn = make_postprocess_fn(
        names=names_long,    # one-hot/axis の検出は正式名でOK（短縮名でも動く実装）
        snap_onehot=True,    # 表示専用: joint_type を最大要素にスナップ
        unit_axis=True       # 表示専用: axis を L2 正規化
    )
    compute_recon_metrics_origscale(
        model=model,
        loader=loader,
        device=device,
        z_stats=stats_mm,                   # ← min-max 統計で逆正規化
        feature_names=NAMES_DISP,           # 表示は短縮名
        out_csv=base,                       # test/metrics_<split>_origscale.csv
        out_csv_by_robot=base.replace(".csv", "_by_robot.csv"),
        use_mask_only=False,
        postprocess_fn=post_fn,             # 表示用後処理（学習には不影響）
    )

    # 7) サマリ & 使用列名を保存 -----------------------------------------------
    with open(os.path.join(args.out_dir, f"summary_{args.eval_split}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "recon_normalized_mean", "near_zero(thr=1e-3)"])
        w.writerow([args.eval_split, f"{recon_norm:.8f}", recon_norm < 1e-3])

    with open(os.path.join(args.out_dir, "feature_names.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["index", "name_display", "name_full"])
        for i, (sd, sl) in enumerate(zip(NAMES_DISP, names_long)):
            w.writerow([i, sd, sl])

    print(f"[done] CSVs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()

'''
mkdir -p test
python test.py \
  --weights checkpoints_debug/best.pt \
  --merge-dir ./merge_joint_robots \
  --normalize-by none \
  --drop-feats movable,width,lower,upper \
  --seed 42 \
  --eval-split train \
  --out-dir test \
  --preview

'''
