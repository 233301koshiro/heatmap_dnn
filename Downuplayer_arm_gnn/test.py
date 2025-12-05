# -*- coding: utf-8 -*-
# test.py
from __future__ import annotations
import argparse
import time
import random
import os
import csv
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from glob import glob
from typing import List

# ===== Local Imports =====
from urdf_core_utils import (
    urdf_to_feature_graph,
    to_pyg,
    FEATURE_NAMES,
    shorten_feature_names
)
from urdf_norm_utils import (
    compute_global_minmax_stats,
    apply_global_minmax_inplace,
    create_composite_post_fn,
    fix_mass_to_one,
    denorm_batch  # ★追加: 直接逆正規化するために必要
)
from urdf_print_debug_utils import (
    compute_recon_metrics_origscale,
    compute_feature_mean_std_from_dataset
)
from gnn_network_min import (
    MaskedTreeAutoencoder,
    TrainCfg,
    eval_loss,
)
from urdf_io_utils import (
    export_predictions_to_csv
)

# =========================================================
# Data Loading Helper
# =========================================================
def build_data(urdf_path: str, drop_feats: list[str] | None = None) -> Data:
    S, nodes, X_np, edge_index, E_np, scale, _ = urdf_to_feature_graph(urdf_path)
    d = to_pyg(X_np, edge_index, E_np)
    d.name = urdf_path
    d.node_names = nodes
    
    current_names = list(FEATURE_NAMES)
    if drop_feats:
        drop_set = set(drop_feats)
        keep_idx = [i for i, n in enumerate(current_names) if n not in drop_set]
        d.x = d.x[:, keep_idx]
        d.feature_names = [current_names[i] for i in keep_idx]
    else:
        d.feature_names = current_names

    d.feature_names_disp = shorten_feature_names(d.feature_names)
    return d

def load_dataset_from_dir(target_dir, drop_list, keywords=None, exclude=False):
    paths = sorted([p for p in glob(f"{target_dir}/**/*", recursive=True)
                    if p.lower().endswith((".urdf", ".xml"))])
    
    dataset = []
    print(f"[INFO] Scanning {len(paths)} files in {target_dir}...")
    
    for p in paths:
        filename = os.path.basename(p)
        
        if keywords:
            hit = any(k in filename for k in keywords)
            if exclude and hit:
                continue 
            if not exclude and not hit:
                continue 

        try:
            d = build_data(p, drop_feats=drop_list)
            if d.num_nodes > 0:
                dataset.append(d)
        except Exception as e:
            pass
            
    return dataset

# =========================================================
# Diff Export Helper (修正版: tuple対応)
# =========================================================
def export_node_diffs(model, loader, device, stats, feature_names, out_path):
    """
    各ノードごとに 正解(GT), 予測(Pred), 差分(Diff) を計算してCSVに出力する
    """
    model.eval()
    
    header = ["urdf_name", "node_idx", "node_name"]
    for feat in feature_names:
        header.extend([f"{feat}_GT", f"{feat}_Pred", f"{feat}_Diff"])
        
    with torch.no_grad():
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for batch in loader:
                batch = batch.to(device)
                
                # 推論
                out = model(batch)
                
                # ★修正: out がタプルの場合のハンドリング
                if isinstance(out, tuple):
                    out = out[0]
                
                # 逆正規化 (statsを使って直接戻す)
                x_gt_orig = denorm_batch(batch.x, stats).cpu().numpy()
                x_pred_orig = denorm_batch(out, stats).cpu().numpy()
                
                start_idx = 0
                for i in range(len(batch.ptr) - 1):
                    n_nodes = batch.ptr[i+1] - batch.ptr[i]
                    end_idx = start_idx + n_nodes
                    
                    urdf_name = os.path.basename(batch.name[i])
                    node_names = batch.node_names[i]
                    
                    for local_idx in range(n_nodes):
                        global_idx = start_idx + local_idx
                        row = [urdf_name, local_idx, node_names[local_idx]]
                        
                        for f_idx, _ in enumerate(feature_names):
                            val_gt = x_gt_orig[global_idx, f_idx]
                            val_pred = x_pred_orig[global_idx, f_idx]
                            val_diff = val_pred - val_gt
                            
                            row.extend([f"{val_gt:.6f}", f"{val_pred:.6f}", f"{val_diff:.6f}"])
                        
                        writer.writerow(row)
                    
                    start_idx = end_idx

    print(f"[INFO] Node-wise diffs exported to: {out_path}")

# =========================================================
# Arguments
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Test runner for hold-out robots")
    ap.add_argument("--checkpoint", required=True, help="学習済みモデルのパス (.pt)")
    ap.add_argument("--train-dir", default="./augmented_dataset", 
                    help="正規化の基準を作るために使う学習データのディレクトリ")
    ap.add_argument("--test-dir", default="./merge_joint_robots", 
                    help="テスト対象(kinova/a1)が入っているディレクトリ")
    ap.add_argument("--out-dir", default="./test_results", help="結果出力フォルダ")
    ap.add_argument("--target-robots", default="merge_kinova,merge_a1", 
                    help="テスト対象にするロボット名のキーワード(カンマ区切り)")
    ap.add_argument("--drop-feats", default="movable,width,jtype_is_planar,jtype_is_floating",
                    help="除外する特徴量")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--enc-rounds", type=int, default=1)
    
    return ap.parse_args()

# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drop_list = [s.strip() for s in args.drop_feats.split(",") if s.strip()]
    target_keywords = [s.strip() for s in args.target_robots.split(",") if s.strip()]

    print("==========================================")
    print(" 1. 正規化パラメータの計算 (Training Data)")
    print("==========================================")
    train_dataset = load_dataset_from_dir(args.train_dir, drop_list)
    if not train_dataset:
        raise RuntimeError("学習データが見つかりません。")
    
    fix_mass_to_one(train_dataset)
    
    feat_names = train_dataset[0].feature_names
    
    exclude_cols = {"axis_x", "axis_y", "axis_z"}
    jtype_names = {n for n in feat_names if n.startswith("jtype_is_")}
    exclude_cols.update(jtype_names)
    z_cols = [i for i, n in enumerate(feat_names) if n not in exclude_cols]

    stats_mm = compute_global_minmax_stats(train_dataset, norm_cols=z_cols)
    print(f"[INFO] Computed stats from {len(train_dataset)} training samples.")
    del train_dataset

    print("\n==========================================")
    print(f" 2. テストデータの読み込み ({args.target_robots})")
    print("==========================================")
    test_dataset = load_dataset_from_dir(args.test_dir, drop_list, keywords=target_keywords, exclude=False)
    
    if not test_dataset:
        raise RuntimeError(f"テスト対象のロボット ({args.target_robots}) が見つかりませんでした。")

    print(f"[INFO] Found {len(test_dataset)} test robots:")
    for d in test_dataset:
        print(f"  - {os.path.basename(d.name)}")

    fix_mass_to_one(test_dataset)
    apply_global_minmax_inplace(test_dataset, stats_mm)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("\n==========================================")
    print(" 3. モデルのロードと推論")
    print("==========================================")
    model = MaskedTreeAutoencoder(
        in_dim=test_dataset[0].num_node_features,
        hidden=args.hidden,
        bottleneck_dim=args.bottleneck,
        enc_rounds=args.enc_rounds,
        dec_rounds=0,
        dropout=0.0,
        mask_strategy="none"
    ).to(device)

    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    full_loss = eval_loss(model, test_loader, device, recon_only_masked=False)
    print(f"[RESULT] Test Recon Loss (Normalized): {full_loss:.6f}")

    metrics_csv = os.path.join(args.out_dir, "test_metrics.csv")
    preds_csv = os.path.join(args.out_dir, "test_predictions.csv")
    diffs_csv = os.path.join(args.out_dir, "test_node_diffs.csv")

    post_fn = create_composite_post_fn(
        stats=stats_mm,
        names=feat_names,
        snap_onehot=True,
        unit_axis=False
    )

    compute_recon_metrics_origscale(
        model=model,
        loader=test_loader,
        device=device,
        z_stats=stats_mm,
        feature_names=test_dataset[0].feature_names_disp,
        out_csv=metrics_csv,
        postprocess_fn=post_fn
    )

    export_predictions_to_csv(
        model=model,
        loader=test_loader,
        device=device,
        stats=stats_mm,
        feature_names=feat_names,
        output_csv_path=preds_csv
    )

    print("\n[INFO] Exporting node-wise diffs...")
    export_node_diffs(
        model=model,
        loader=test_loader,
        device=device,
        stats=stats_mm,
        feature_names=feat_names,
        out_path=diffs_csv
    )
    
    print(f"\n[DONE] Results saved to:\n  - {metrics_csv}\n  - {preds_csv}\n  - {diffs_csv}")

if __name__ == "__main__":
    main()