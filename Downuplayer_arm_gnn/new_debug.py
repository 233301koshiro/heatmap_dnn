# -*- coding: utf-8 -*-
# debug.py
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
from typing import List, Optional

# ===== Local Imports (分割されたモジュールから) =====
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

    #massの推定が不安定なので一旦1に固定する関数をインポート
    fix_mass_to_one
)
from urdf_print_debug_utils import (
    minimal_dataset_report,
    scan_nonfinite_features,
    print_minmax_stats,
    dump_normalized_feature_table,
    compute_recon_metrics_origscale,
    compute_feature_mean_std_from_dataset,
    print_feature_mean_std
)

from gnn_network_min import (
    MaskedTreeAutoencoder,
    TrainCfg,
    train_one_epoch,
    eval_loss,
)


# =========================================================
# Data Construction
# =========================================================
def build_data(urdf_path: str, drop_feats: list[str] | None = None) -> Data:
    # 1. URDFから特徴量グラフを抽出
    S, nodes, X_np, edge_index, E_np, scale, _ = urdf_to_feature_graph(urdf_path)

    # 2. PyGデータ形式に変換
    d = to_pyg(X_np, edge_index, E_np)
    d.name = urdf_path
    
    # 現在の特徴量名リスト (urdf_core.FEATURE_NAMES を初期値とする)
    current_names = list(FEATURE_NAMES)

    # 3. 不要な特徴量の除外
    if drop_feats:
        drop_set = set(drop_feats)
        keep_idx = [i for i, n in enumerate(current_names) if n not in drop_set]
        d.x = d.x[:, keep_idx]
        d.feature_names = [current_names[i] for i in keep_idx]
    else:
        d.feature_names = current_names

    d.feature_names_disp = shorten_feature_names(d.feature_names)
    return d


# =========================================================
# Arguments
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Debug runner with split modules")
    ap.add_argument("--merge-dir", default="./merge_joint_robots",
                    help="URDF群ディレクトリ（再帰探索）")
    ap.add_argument("--epochs", type=int, default=200, help="学習エポック数")
    ap.add_argument("--batch-size", type=int, default=8, help="バッチサイズ")
    ap.add_argument("--loss-weight", type=str, default=None,
                help="特徴量ごとの損失重み (カンマ区切り, 例: '1.0,1.0,0.5,...')")
    
    # マスク設定
    ap.add_argument("--mask-mode", default="none", choices=["none", "one", "k"],
                    help="評価時のノードマスク戦略")
    ap.add_argument("--mask-k", type=int, default=1,
                    help="--mask-mode=k のときのマスク数")
    ap.add_argument("--seed", type=int, default=None, help="乱数シード")
    ap.add_argument("--mask-seed", type=int, default=0, help="評価マスク用の乱数シード")

    # 保存・ログ
    ap.add_argument("--save-dir", default="", help="チェックポイント保存先")
    ap.add_argument("--log-csv", default="", help="学習ログCSV出力先")
    ap.add_argument("--metrics-csv", default="", help="最終評価メトリクスCSV出力先")
    ap.add_argument("--quiet", action="store_true", help="詳細ログを抑制")
    ap.add_argument("--log-interval", type=int, default=0, help="ステップ毎のログ間隔")
    ap.add_argument("--log-every", type=int, default=20, help="エポック毎のログ間隔")

    # 早期終了
    ap.add_argument("--early-stop-patience", type=int, default=0, help="早期終了のpatience")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="早期終了の改善閾値")

    # 特徴量選択 (lower/upper の特殊処理に関する説明を削除)
    ap.add_argument("--drop-feats", default="movable,width,jtype_is_planar,jtype_is_floating",
                    help="学習から除外する特徴名（カンマ区切り）")

    return ap.parse_args()


# =========================================================
# Main Loop
# =========================================================
def main():
    args = parse_args()

    # 乱数固定
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # -----------------------------------------------------
    # 1. データ読み込み
    # -----------------------------------------------------
    paths = sorted([p for p in glob(f"{args.merge_dir}/**/*", recursive=True)
                    if p.lower().endswith((".urdf", ".xml"))])
    if not paths:
        raise RuntimeError(f"URDFが見つかりません: {args.merge_dir}")

    dataset: List[Data] = []
    drop_list = [s.strip() for s in args.drop_feats.split(",") if s.strip()]
    
    print(f"[INFO] Loading {len(paths)} URDFs...")
    for p in paths:
        try:
            d = build_data(p, drop_feats=drop_list)
            if d.num_nodes > 0:
                dataset.append(d)
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")

    if not dataset:
        raise RuntimeError("有効なグラフがロードできませんでした。")

    # -----------------------------------------------------
    # 2. 統計計算と正規化
    # -----------------------------------------------------
    # mass列を1.0に固定（不安定な質量推定を回避）
    #fix_mass_to_one(dataset)

    minimal_dataset_report(dataset)

    feat_names = dataset[0].feature_names
    print("[check] num_features:", dataset[0].num_node_features)
    print("[check] feature_names:", feat_names)

    # 生データの統計を表示
    data_stats = compute_feature_mean_std_from_dataset(dataset, population_std=True)
    print_feature_mean_std(data_stats, feature_names=dataset[0].feature_names_disp)

    # 正規化対象カラムの決定 (axisとorigin以外を正規化する例)
    #exclude_cols = {"origin_x","origin_y","origin_z","axis_x", "axis_y", "axis_z"}
    exclude_cols = {"axis_x", "axis_y", "axis_z"}
    z_cols = [i for i, n in enumerate(feat_names) if n not in exclude_cols]

    # Min-Max統計の計算
    stats_mm = compute_global_minmax_stats(dataset, norm_cols=z_cols)
    
    # 正規化の適用
    apply_global_minmax_inplace(dataset, stats_mm)
    
    # 正規化後のデータチェック
    scan_nonfinite_features(dataset)
    print_minmax_stats(stats_mm, feature_names=feat_names)

    # 正規化結果のプレビュー
    try:
        X_ex = dataset[0].x.detach().cpu().numpy()
        dump_normalized_feature_table(X_ex, stats_mm, feature_names=feat_names, max_rows=5)
    except Exception as e:
        print(f"[INFO] Preview skipped: {e}")

    # -----------------------------------------------------
    # 3. データセット分割 & Loader作成
    # -----------------------------------------------------
    N = len(dataset)
    indices = torch.randperm(N).tolist()
    n_tr = int(0.7 * N)
    n_va = int(0.2 * N)
    
    train_set = [dataset[i] for i in indices[:n_tr]]
    val_set   = [dataset[i] for i in indices[n_tr:n_tr+n_va]]
    test_set  = [dataset[i] for i in indices[n_tr+n_va:]]

    bs = min(args.batch_size, max(1, len(train_set)))
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=max(1, len(val_set)), shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=max(1, len(test_set)), shuffle=False)

    print("\nデータセット内訳")
    print(f"[dataset] total={N} | train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    # -----------------------------------------------------
    # 4. モデル構築
    # -----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = TrainCfg(
        lr=1e-3,
        weight_decay=1e-4,
        epochs=args.epochs,
        #コマンドライン引数の文字列をカンマで分割してfloatリストに変換
        loss_weight=[float(w) for w in args.loss_weight.split(",")] if args.loss_weight else None,
        mask_strategy=args.mask_mode,
        mask_k=args.mask_k,
        verbose=(not args.quiet),
        log_interval=args.log_interval
    )

    model = MaskedTreeAutoencoder(
        in_dim=dataset[0].num_node_features,
        hidden=128,
        bottleneck_dim=128,
        enc_rounds=8,
        dec_rounds=8,
        dropout=0.1,
        mask_strategy=cfg.mask_strategy
    ).to(device)
    model._cfg = cfg

    # -----------------------------------------------------
    # 5. 学習ループ
    # -----------------------------------------------------
    # 保存設定
    do_save = bool(args.save_dir)
    if do_save:
        os.makedirs(args.save_dir, exist_ok=True)
        best_pt = os.path.join(args.save_dir, "best.pt")
        latest_pth = os.path.join(args.save_dir, "latest.pth")

    if args.log_csv:
        with open(args.log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_recon", "val_recon", "sec"])

    # Optimizer (lazy init)
    model._opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, device, cfg, log_every_steps=args.log_interval)
        val_loss = eval_loss(model, val_loader, device, recon_only_masked=True, 
                             mask_strategy=args.mask_mode, verbose=(not args.quiet))
        sec = time.time() - t0

        # ログ表示
        if (epoch % args.log_every == 0) or (epoch == 1) or (epoch == args.epochs):
            print(f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | {sec:.1f}s")

        if args.log_csv:
            with open(args.log_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss, val_loss, sec])

        # チェックポイント保存 & 早期終了
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            patience_counter = 0
            if do_save:
                torch.save(model.state_dict(), best_pt)
        else:
            patience_counter += 1
            if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                print(f"[Early Stop] Epoch {epoch}: No improvement for {patience_counter} epochs.")
                break
        
        if do_save:
            torch.save(model.state_dict(), latest_pth)

    # -----------------------------------------------------
    # 6. 最終評価 (テスト)
    # -----------------------------------------------------
    print("\n===== FINAL TEST =====")
    if args.mask_mode != "none":
        masked_loss = eval_loss(model, test_loader, device, recon_only_masked=True, mask_strategy=args.mask_mode)
        #print(f"[TEST] Masked({args.mask_mode}) Recon Loss: {masked_loss:.4f}")

    full_loss = eval_loss(model, train_loader, device, recon_only_masked=False)
    print(f"[TEST] Full Recon Loss: {full_loss:.4f}")

    # 詳細メトリクス (元スケールでの評価)
    if args.metrics_csv:
        # 逆正規化 + 後処理用の関数を作成
        post_fn = create_composite_post_fn(
            stats=stats_mm,
            names=feat_names,
            snap_onehot=True,
            unit_axis=True
        )
        
        # 詳細評価実行
        compute_recon_metrics_origscale(
            model=model,
            loader=train_loader,
            device=device,
            z_stats=stats_mm,  # min/max統計を渡す
            feature_names=dataset[0].feature_names_disp,
            out_csv=args.metrics_csv,
            use_mask_only=(args.mask_mode != "none"),
            postprocess_fn=post_fn,
            mask_mode=args.mask_mode,
            mask_k=args.mask_k,
            mask_seed=args.mask_seed
        )

if __name__ == "__main__":
    main()

#実行コマンド例:
"""
python new_debug.py --merge-dir ./merge_joint_robots --epochs 100 --batch-size 16 --loss-weight '0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,1,1,1,1,1' --mask-mode none --mask-k 0 --seed 42 --save-dir ./checkpoints --log-csv ./checkpoints/training_log.csv --metrics-csv ./checkpoints/test_metrics.csv
"""