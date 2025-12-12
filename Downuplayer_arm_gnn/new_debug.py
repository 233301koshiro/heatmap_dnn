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
from urdf_io_utils import (
    export_predictions_to_csv
)

# =========================================================
# Data Construction
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


# =========================================================
# Arguments
# =========================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Debug runner with split modules")
    # 通常データセット用
    ap.add_argument("--merge-dir", default="./merge_joint_robots",
                    help="通常URDF群ディレクトリ（再帰探索）")
    
    # ★追加: 拡張データセット切り替え用
    ap.add_argument("--use-augmented", action="store_true",
                    help="拡張データセットを使用して学習を行う")
    ap.add_argument("--aug-dir", default="./augmented_dataset",
                    help="拡張データセットのディレクトリ (--use-augmented時有効)")

    ap.add_argument("--epochs", type=int, default=200, help="学習エポック数")
    ap.add_argument("--batch-size", type=int, default=8, help="バッチサイズ")
    ap.add_argument("--loss-weight", type=str, default=None,
                help="特徴量ごとの損失重み (カンマ区切り, 例: '1.0,1.0,0.5,...')")
    
    # マスク設定
    ap.add_argument("--mask-mode", default="none", choices=["none", "one", "k"],
                    help="評価時のノードマスク戦略")
    ap.add_argument("--mask-k", type=int, default=0,
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

    # 特徴量選択
    ap.add_argument("--drop-feats", default="movable,width,jtype_is_planar,jtype_is_floating",
                    help="学習から除外する特徴名（カンマ区切り）")

    # 1つのサンプルにわざと過学習させて致命傷を炙り出すモード
    ap.add_argument("--overfit-one", default="", help="単一サンプルに過学習する。index または URDFパス（部分一致可）")

    ap.add_argument("--preds-csv", default="vlisualize_choreonoid_feature.csv", help="予測値CSVの出力先")
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
    # ディレクトリと保存先の切り替えロジック
    # -----------------------------------------------------
    if args.use_augmented:
        target_dir = args.aug_dir
        print(f"\n========== [MODE] USING AUGMENTED DATASET ==========")
        print(f"Target Directory: {target_dir}")
        
        # 1. 保存先ディレクトリ名を自動変更
        if args.save_dir:
            args.save_dir = args.save_dir.rstrip("/\\") + "_augmented"
            print(f"Save Directory switched to: {args.save_dir}")
        
        # 2. 各種出力ファイルのパスも新しい保存先に変更する関数
        def redirect_path(original_path, new_dir):
            if not original_path: return original_path
            filename = os.path.basename(original_path)
            # ファイル名に _aug を付ける場合はここで行う
            root, ext = os.path.splitext(filename)
            new_filename = f"{root}_aug{ext}"
            return os.path.join(new_dir, new_filename)

        # args.save_dir が設定されている場合のみリダイレクト
        if args.save_dir:
            args.log_csv = redirect_path(args.log_csv, args.save_dir)
            args.metrics_csv = redirect_path(args.metrics_csv, args.save_dir)
            args.preds_csv = redirect_path(args.preds_csv, args.save_dir)
            
        print(f"Log CSV: {args.log_csv}")
        print(f"Metrics CSV: {args.metrics_csv}")
        print(f"Preds CSV: {args.preds_csv}")
        print("====================================================\n")
    else:
        target_dir = args.merge_dir

    # -----------------------------------------------------
    # 1. データ読み込み
    # -----------------------------------------------------
    paths = sorted([p for p in glob(f"{target_dir}/**/*", recursive=True)
                    if p.lower().endswith((".urdf", ".xml"))])
    if not paths:
        raise RuntimeError(f"URDFが見つかりません: {target_dir}")

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
    
    print("[DEBUG] loaded URDFs (first 5):")
    for d in dataset[:5]:
        print("   ", d.name)
    if len(dataset) > 5:
        print(f"    ... and {len(dataset)-5} more.")

    if not dataset:
        raise RuntimeError("有効なグラフがロードできませんでした。")

    # -----------------------------------------------------
    # (optional) 単一サンプルに過学習モード
    # -----------------------------------------------------
    if args.overfit_one:
        print("overfitモードを使用しました")
        key = args.overfit_one.strip()
        sel: Optional[Data] = None
        if key.isdigit():
            idx = int(key)
            if not (0 <= idx < len(dataset)):
                raise IndexError(f"--overfit-one index {idx} is out of range (0..{len(dataset)-1})")
            sel = dataset[idx]
        else:
            cand = [d for d in dataset if (key in getattr(d, "name", ""))]
            if len(cand) == 0:
                raise FileNotFoundError(f"--overfit-one '{key}' に一致する URDF がありません")
            if len(cand) > 1:
                print(f"[WARN] --overfit-one '{key}' 部分一致で {len(cand)} 件ヒット。先頭を採用: {cand[0].name}")
            sel = cand[0]

        print(f"[OVERFIT] target = {getattr(sel, 'name', '(unknown)')}")
        dataset = [sel]

        # ★ overfitモードの場合は専用フォルダへ
        overfit_dir = "./checkpoints_overfit"
        os.makedirs(overfit_dir, exist_ok=True)
        args.save_dir = overfit_dir
        args.preds_csv = os.path.join(overfit_dir, os.path.basename(args.preds_csv))
        args.log_csv = os.path.join(overfit_dir, "training_log.csv")
        args.metrics_csv = os.path.join(overfit_dir, "test_metrics.csv")

    # -----------------------------------------------------
    # 2. 統計計算と正規化
    # -----------------------------------------------------
    fix_mass_to_one(dataset)
    minimal_dataset_report(dataset)

    feat_names = dataset[0].feature_names
    print("[check] num_features:", dataset[0].num_node_features)
    print("[check] feature_names:", feat_names)

    data_stats = compute_feature_mean_std_from_dataset(dataset, population_std=True)
    print_feature_mean_std(data_stats, feature_names=dataset[0].feature_names_disp)

    exclude_cols = {"axis_x", "axis_y", "axis_z","rot6d_0","rot6d_1","rot6d_2","rot6d_3","rot6d_4","rot6d_5"}
    jtype_names = {n for n in feat_names if n.startswith("jtype_is_")}
    exclude_cols.update(jtype_names)
    z_cols = [i for i, n in enumerate(feat_names) if n not in exclude_cols]

    stats_mm = compute_global_minmax_stats(dataset, norm_cols=z_cols)
    apply_global_minmax_inplace(dataset, stats_mm)
    scan_nonfinite_features(dataset)
    print_minmax_stats(stats_mm, feature_names=feat_names)

    try:
        X_ex = dataset[0].x.detach().cpu().numpy()
        dump_normalized_feature_table(X_ex, stats_mm, feature_names=feat_names, max_rows=5)
    except Exception as e:
        print(f"[INFO] Preview skipped: {e}")

    # -----------------------------------------------------
    # 3. データセット分割 & Loader作成
    # -----------------------------------------------------
    N = len(dataset)
    if N == 1:
        train_set = [dataset[0]]
        val_set   = [dataset[0]]
        test_set  = [dataset[0]]
        bs = 1
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=False)
        val_loader   = DataLoader(val_set,   batch_size=1, shuffle=False)
        test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False)
        print("\nデータセット内訳")
        print(f"[dataset] total=1 | train=1 val=1 test=1 (OVERFIT mode)")
    else:
        indices = torch.randperm(N).tolist()
        n_tr = int(0.7 * N)
        n_va = int(0.2 * N)
        train_set = [dataset[i] for i in indices[:n_tr]]
        val_set   = [dataset[i] for i in indices[n_tr:n_tr+n_va]]
        test_set  = [dataset[i] for i in indices[n_tr+n_va:]]
        bs = min(args.batch_size, max(1, len(train_set)))
        
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
        val_loader   = DataLoader(val_set,   batch_size=max(1, len(val_set)//4), shuffle=False)
        test_loader  = DataLoader(test_set,  batch_size=max(1, len(test_set)//4), shuffle=False)
        print("\nデータセット内訳")
        print(f"[dataset] total={N} | train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    # -----------------------------------------------------
    # 4. モデル構築
    # -----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_overfit = bool(args.overfit_one)
    lr = 3e-3 if is_overfit else 1e-4
    weight_decay = 0.0 if is_overfit else 1e-4
    dropout = 0.0 if is_overfit else 0.1
    if is_overfit:
        print(f"[OVERFIT HP] lr={lr} weight_decay={weight_decay} dropout={dropout}")

    cfg = TrainCfg(
        lr=lr,
        weight_decay=weight_decay,
        epochs=args.epochs,
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
        enc_rounds=1,
        dec_rounds=1,
        dropout=dropout,
        mask_strategy=cfg.mask_strategy
    ).to(device)
    model._cfg = cfg

    # -----------------------------------------------------
    # 5. 学習ループ
    # -----------------------------------------------------
    do_save = bool(args.save_dir)
    if do_save:
        os.makedirs(args.save_dir, exist_ok=True)
        best_pt = os.path.join(args.save_dir, "best.pt")
        latest_pth = os.path.join(args.save_dir, "latest.pth")
        print(f"[INFO] Checkpoints will be saved to: {args.save_dir}")

    if args.log_csv:
        # ディレクトリが存在するか確認し、なければ作成
        log_dir = os.path.dirname(args.log_csv)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        with open(args.log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_recon", "val_recon", "sec"])

    model._opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, device, cfg, log_every_steps=args.log_interval)
        val_loss = eval_loss(model, val_loader, device, recon_only_masked=False, 
                             mask_strategy=args.mask_mode, verbose=(not args.quiet))
        sec = time.time() - t0

        if (epoch % args.log_every == 0) or (epoch == 1) or (epoch == args.epochs):
            print(f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} | {sec:.1f}s")

        if args.log_csv:
            with open(args.log_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss, val_loss, sec])

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
        masked_loss = eval_loss(model, test_loader, device, recon_only_masked=False, mask_strategy=args.mask_mode)

    full_loss = eval_loss(model, test_loader, device, recon_only_masked=False)
    print(f"[TEST] Full Recon Loss: {full_loss:.4f}")

    if args.metrics_csv:
        metrics_dir = os.path.dirname(args.metrics_csv)
        if metrics_dir and not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir, exist_ok=True)

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
            feature_names=dataset[0].feature_names_disp,
            out_csv=args.metrics_csv,
            use_mask_only=(args.mask_mode != "none"),
            postprocess_fn=post_fn,
            mask_mode=args.mask_mode,
            mask_k=args.mask_k,
            mask_seed=args.mask_seed
        )
        if args.preds_csv:
            preds_dir = os.path.dirname(args.preds_csv)
            if preds_dir and not os.path.exists(preds_dir):
                os.makedirs(preds_dir, exist_ok=True)

            export_predictions_to_csv(
                model=model,
                loader=test_loader,
                device=device,
                stats=stats_mm,
                feature_names=feat_names,
                output_csv_path=args.preds_csv,
                mask_mode=args.mask_mode,
                mask_k=args.mask_k
            )

if __name__ == "__main__":
    main()