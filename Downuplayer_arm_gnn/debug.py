# -*- coding: utf-8 -*-
# debug.py — デフォは1epoch・保存なし。必要に応じて学習ユーティリティを引数でON。
from __future__ import annotations
import numpy as np
# ===== Stdlib =====
import argparse, time, random, os, csv,random
from glob import glob
from typing import List

# ===== Third-party =====
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# ===== Local =====
from urdf_to_graph_utilis import (
    # データ作成
    urdf_to_feature_graph, ExtractConfig, to_pyg,
    # レポート/デバッグ
    minimal_dataset_report, assert_finite_data, scan_nonfinite_features,
    # z-score正規化
    #DEFAULT_Z_COLS, compute_global_z_stats_from_dataset,
    #apply_global_z_inplace_to_dataset, print_z_stats, dump_normalized_feature_table,
    #min-max正規化用
    DEFAULT_Z_COLS,FEATURE_NAMES, compute_global_minmax_stats_from_dataset,dump_normalized_feature_table,
    # Z統計チェック / 追加メトリクス
    #check_z_stats, compute_recon_metrics_origscale,

    # min-max 正規化用
    apply_global_minmax_inplace_to_dataset, print_minmax_stats, compute_recon_metrics_origscale,compute_feature_mean_std_from_dataset, print_feature_mean_std
    #周期性upper/lower → sin/cos 変換
    ,embed_angles_sincos,

    #表示用joint特徴名短縮
    shorten_feature_names,

    #metrics用前処理
    make_postprocess_fn,
)

from gnn_network_min import (
    MaskedTreeAutoencoder,
    TrainCfg,
    train_one_epoch,
    eval_loss,
)

# ---------------------------------------------------------
# Data: URDF → PyG
# ---------------------------------------------------------
def build_data(urdf_path: str, normalize_by: str = "none", drop_feats: list[str] | None = None) -> Data:
    S, nodes, X, edge_index, E, scale, _ = urdf_to_feature_graph(
        urdf_path, ExtractConfig(normalize_by=normalize_by)
    )
    d = to_pyg(S, nodes, X, edge_index, E)
    d.name = urdf_path
    # === ここで lower/upper -> sin/cos に差し替え ===
    X2, names2 = embed_angles_sincos(d.x, FEATURE_NAMES)
        # --- 学習で使わない列を落とす（CSV等の生成は別ルートなので影響なし） ---
    drop_feats = drop_feats or []
    # ユーザ指定が lower/upper の場合でも sin/cos 展開後の名前にマッピング
    expand_map = {
        "lower": {"lower_sin", "lower_cos"},
        "upper": {"upper_sin", "upper_cos"},
    }
    drop_set = set()
    for k in drop_feats:
        drop_set |= expand_map.get(k, {k})
    keep_idx = [i for i, n in enumerate(names2) if n not in drop_set]
    d.x = X2[:, keep_idx]
    kept_names = [names2[i] for i in keep_idx]
    d.feature_names = kept_names  # レポート系で使えるように保持
    d.feature_names_disp = shorten_feature_names(kept_names)
    return d


def parse_args():
    ap = argparse.ArgumentParser(description="Minimal AE debug runner with optional training utilities")
    ap.add_argument("--merge-dir", default="./merge_joint_robots",
                    help="URDF群ディレクトリ（*.urdf / *.xml を再帰探索）")
    ap.add_argument("--normalize-by", choices=["none", "mean_edge_len"], default="none",
                    help="origin距離の平均で正規化するか（特徴抽出時）")
    ap.add_argument("--epochs", type=int, default=1, help="エポック数（デフォルト1）")
    ap.add_argument("--batch-size", type=int, default=8, help="バッチサイズ")
    ap.add_argument("--mask-mode", choices=["none", "one"], default="none", help="評価時のマスク戦略")
    ap.add_argument("--seed", type=int, default=None, help="乱数シード（指定時に固定）")
    # 保存/ログ/早停
    ap.add_argument("--save-dir", default="", help="保存先ディレクトリ（空なら保存しない）")
    ap.add_argument("--log-csv", default="", help="学習ログCSVのパス（空なら保存しない）")
    ap.add_argument("--early-stop-patience", type=int, default=0, help="早期終了patience（0で無効）")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="val改善とみなす最小差")
    # メトリクス
    ap.add_argument("--metrics-csv", default="", help="学習後に元スケールでMAE/RMSEをCSV出力（空なら出力しない）")

    # ログ詳細度
    ap.add_argument("--quiet", action="store_true",
                    help="学習中の詳細ログを抑制（epoch行と最終TESTのみ表示）")
    ap.add_argument("--log-interval", type=int, default=0,
                    help="このステップ間隔でバッチ内ログを出す（0ならバッチ内は無出力）")

    ap.add_argument("--log-every", type=int, default=20, help="何エポックごとにサマリを出すか")

    # 除外特徴
    ap.add_argument("--drop-feats", default="movable,width,lower,upper,jtype_is_planar,jtype_is_floating",
                    help="学習から除外する特徴名をカンマ区切りで指定。"
                         "lower/upper は sin/cos 展開後の2列ずつをまとめて除外します。"
                         "（例）movable,width,lower,upper")
    return ap.parse_args()

def main():
    args = parse_args()

    # 乱数固定（任意）
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # -----------------------------------------------------
    # データ読み込み
    # -----------------------------------------------------
    paths: List[str] = sorted(
        [p for p in glob(f"{args.merge_dir}/**/*", recursive=True)
         if p.lower().endswith((".urdf", ".xml"))]
    )
    if not paths:
        raise RuntimeError(f"URDFが見つかりません: {args.merge_dir}")

    dataset: List[Data] = []
    drop_list = [s.strip() for s in (args.drop_feats or "").split(",") if s.strip()]
    for p in paths:
        d = build_data(p, normalize_by=args.normalize_by, drop_feats=drop_list)
        if d.num_nodes == 0:
            print(f"[skip] empty graph: {p}")
            continue
        dataset.append(d)

    feat_names_long  = getattr(dataset[0], "feature_names", FEATURE_NAMES)
    feat_names_short = shorten_feature_names(feat_names_long)

    # ← ここで “短縮表示を使う/使わない” をトグル
    USE_SHORT_NAMES = True

    NAMES = feat_names_short if USE_SHORT_NAMES else feat_names_long
    print("[check] num_features:", dataset[0].num_node_features)
    print("[check] feature_names:", getattr(dataset[0], "feature_names", FEATURE_NAMES))
    #assert any(n.endswith("_sin") for n in getattr(dataset[0], "feature_names", [])), \
    #"sin/cos 列が入っていません（embed_angles_sincos 未適用）"

    if not dataset:
        raise RuntimeError("有効なグラフが0件。URDFや抽出設定を見直してください。")
    # もう学習用行列 d.x は「使用する列だけ」になっているので、z_colsは全列でOK
    feat_names = getattr(dataset[0], "feature_names", None) or FEATURE_NAMES
    Z_COLS = list(range(len(feat_names)))
    feat_names_short = shorten_feature_names(feat_names)
    #datasetの統計表示
    minimal_dataset_report(dataset)
    data_stats = compute_feature_mean_std_from_dataset(dataset, cols=None, population_std=True)
    print_feature_mean_std(data_stats, feature_names=NAMES)

    # 統計計算 → 適用 → 非有限チェック
    #stats_z = compute_global_z_stats_from_dataset(dataset, z_cols=DEFAULT_Z_COLS)
    #check_z_stats(stats_z)
    #apply_global_z_inplace_to_dataset(dataset, stats_z)
    # 1回目（適用＋チェック）
    #stats_mm = compute_global_minmax_stats_from_dataset(dataset, z_cols=DEFAULT_Z_COLS)
    stats_mm = compute_global_minmax_stats_from_dataset(dataset, z_cols=Z_COLS)

    # （任意）簡易チェック：min/max が有限か見るだけなら下記で十分 
    assert all(np.isfinite(stats_mm["min"])) and all(np.isfinite(stats_mm["max"])), "min/max not finite"
    apply_global_minmax_inplace_to_dataset(dataset, stats_mm)
    for d in dataset:
        assert_finite_data(d, getattr(d, "name", "(no name)"))

    
    scan_nonfinite_features(dataset, extreme_abs=1e6)

    # 再掲：Z 正規化統計の表示とサンプル可視
    # stats_z = compute_global_z_stats_from_dataset(dataset, z_cols=DEFAULT_Z_COLS)
    # apply_global_z_inplace_to_dataset(dataset, stats_z)
    # print_z_stats(stats_z)
    # try:
    #     X_example = dataset[0].x.detach().cpu().numpy()
    #     dump_normalized_feature_table(X_example, stats_z, max_rows=5)
    # except Exception as e:
    #     print(f"[WARN] preview dump skipped: {e}")

    # min-max 正規化統計の表示とサンプル可視
    #print_minmax_stats(stats_mm, feature_names=FEATURE_NAMES)

    #min-maxの統計表示
    print_minmax_stats(stats_mm, feature_names=NAMES)
    try:
        X_example = dataset[0].x.detach().cpu().numpy()
        # z 用の可視化ヘルパを流用するなら、min-max を使う前提で自前の表示に差し替え可
        # dump_normalized_feature_table(X_example, {
        #     "z_cols": stats_mm["z_cols"],
        #     "mean":   [0]*len(stats_mm["z_cols"]),   # ダミー（使わない）
        #     "std":    [1]*len(stats_mm["z_cols"])    # ダミー（使わない）
        # }, max_rows=5)
        #upper/lower → sin/cos埋込み前
        #dump_normalized_feature_table(X_example, stats_mm, max_rows=5, feature_names=FEATURE_NAMES)
        #埋め込み後
        # 例：X_example=最初のグラフの x、stats_mm=Min-Max stats、NAMES=feature_names
        dump_normalized_feature_table(
            X_example,
            stats_mm,
            max_rows=5,
            feature_names=NAMES,
            cols_per_block=15,   # ← 横幅を抑える（お好みで）
            show_orig=True      # ← 元スケールも別表で表示
        )


    except Exception as e:
        print(f"[WARN] preview dump skipped: {e}")
    # -----------------------------------------------------
    # Split
    # -----------------------------------------------------
    N = len(dataset)
    idx = list(range(N))
    random.shuffle(idx)
    p_tr, p_va = 0.7, 0.2
    n_tr = max(1, int(N * p_tr))
    n_va = max(1, int(N * p_va))
    if n_tr + n_va > N - 1:
        n_va = max(1, N - 1 - n_tr)
        if n_va < 1:
            n_tr = max(1, N - 2); n_va = 1
    n_te = N - n_tr - n_va

    tr_idx = idx[:n_tr]
    va_idx = idx[n_tr:n_tr+n_va]
    te_idx = idx[n_tr+n_va:]

    train_set = [dataset[i] for i in tr_idx]
    val_set   = [dataset[i] for i in va_idx]
    test_set  = [dataset[i] for i in te_idx]

    bs = min(args.batch_size, max(1, len(train_set)))
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=max(1, len(val_set)), shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_set,  batch_size=max(1, len(test_set)), shuffle=False, drop_last=False)

    print(f"[dataset] total={N} | train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    print("train_loader batches:", len(train_loader))

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_node = train_set[0].num_node_features
    # --- Model/CFG 構築時に反映 ---
    cfg = TrainCfg(
        lr=1e-3, weight_decay=1e-4, epochs=args.epochs,
        recon_only_masked=True, mask_strategy=args.mask_mode,
        verbose=(not args.quiet),          # ← ここを追加
        log_interval=max(0, args.log_interval),  # ← ここを追加
    )


    model = MaskedTreeAutoencoder(
        in_dim=in_node,
        hidden=128,
        bottleneck_dim=128,
        enc_rounds=2,
        dec_rounds=2,
        dropout=0.1,
        mask_strategy=cfg.mask_strategy
    ).to(device)
    model._cfg = cfg

    # -----------------------------------------------------
    # ログ/保存（任意）
    # -----------------------------------------------------
    do_save = bool(args.save_dir)
    if do_save:
        os.makedirs(args.save_dir, exist_ok=True)
        latest_pth = os.path.join(args.save_dir, "latest.pth")
        best_pt    = os.path.join(args.save_dir, "best.pt")
    else:
        latest_pth = best_pt = ""

    if args.log_csv:
        with open(args.log_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_recon", "val_recon", "sec"])

    # -----------------------------------------------------
    # Train (早期終了は任意)
    # -----------------------------------------------------
    best_val = float("inf")
    bad = 0
    patience = max(0, int(args.early_stop_patience))
    min_delta = float(args.min_delta)
    # --- optimizer lazy init (if not provided) ---
    if not hasattr(model, "_opt") or model._opt is None:
        model._opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=getattr(cfg, "lr", 1e-3),
            weight_decay=getattr(cfg, "weight_decay", 1e-4),
        )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_recon = train_one_epoch(model, train_loader, device, cfg, log_every_steps=args.log_interval)
        val_recon   = eval_loss(model, val_loader, device, recon_only_masked=True, log_every_steps=args.log_interval, verbose=(not args.quiet))
        sec = time.time() - t0
        # エポック出力は log-every ごと（1エポ目と最終エポックは必ず出す）
        if (epoch % args.log_every == 0) or (epoch == 1) or (epoch == args.epochs):
            print(f"epoch {epoch:03d} | train {train_recon:.4f} | val {val_recon:.4f} | {sec:.1f}s")

        if args.log_csv:
            with open(args.log_csv, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{train_recon:.6f}", f"{val_recon:.6f}", f"{sec:.2f}"])

        # best/early-stop
        improved = (val_recon < best_val - min_delta)
        if improved:
            best_val = val_recon
            bad = 0
            if do_save:
                # optimizerは model 側に _opt が作られている前提（gnn_network_minに準拠）
                ckpt = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": getattr(model, "_opt").state_dict(),
                    "cfg": cfg.__dict__,
                    "val_recon": val_recon,
                    "timestamp": time.time(),
                }
                torch.save(ckpt, best_pt)
        else:
            bad += 1
            if patience and bad >= patience:
                print(f"[early stop] no improvement for {patience} epochs at epoch {epoch}")
                break

        if do_save:
            torch.save(model.state_dict(), latest_pth)

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------
    if args.mask_mode == "none":
        test_recon_all = eval_loss(model, test_loader, device, recon_only_masked=False)
        print(f"[TEST] recon={test_recon_all:.4f}")
    else:
        test_recon = eval_loss(model, test_loader, device, recon_only_masked=True, mask_strategy=args.mask_mode)
        print(f"[TEST] recon_only_masked=True | recon={test_recon:.4f}")
        test_recon_all = eval_loss(model, test_loader, device, recon_only_masked=False, mask_strategy=args.mask_mode)
        print(f"[TEST] recon_only_masked=False | recon={test_recon_all:.4f}")

    # -----------------------------------------------------
    # 追加メトリクス（元スケール）※任意
    # -----------------------------------------------------
    # if args.metrics_csv:
    #     base = args.metrics_csv
    #     per_robot_csv = base.replace(".csv", "_by_robot.csv")
    #     compute_recon_metrics_origscale(
    #         model=model, loader=test_loader, device=device,
    #         z_stats=compute_global_z_stats_from_dataset(dataset, z_cols=DEFAULT_Z_COLS),
    #         feature_names=None,
    #         out_csv=base,                      # 全体
    #         out_csv_by_robot=per_robot_csv,    # 追加: ロボット別
    #         use_mask_only=(args.mask_mode != "none"),
    #     )
    if args.metrics_csv:
        base = args.metrics_csv
        per_robot_csv = base.replace(".csv", "_by_robot.csv")
        # compute_recon_metrics_origscale(
        #     model=model, loader=test_loader, device=device,
        #     z_stats=compute_global_minmax_stats_from_dataset(dataset, z_cols=DEFAULT_Z_COLS),  # ←ここをmin-maxに
        #     feature_names=None,
        #     out_csv=base,
        #     out_csv_by_robot=per_robot_csv,
        #     use_mask_only=(args.mask_mode != "none"),
        # )
        feat_names = getattr(test_loader.dataset[0], "feature_names",
                     [f"f{i}" for i in range(test_loader.dataset[0].num_node_features)])
        names_long = getattr(test_loader.dataset[0], "feature_names",
                     [f"f{i}" for i in range(test_loader.dataset[0].num_node_features)])
        names_disp = getattr(test_loader.dataset[0], "feature_names_disp", names_long)
        post_fn = make_postprocess_fn(
            names=names_long,
            snap_onehot=True,   # 表示専用で one-hot を最大値にスナップ
            unit_axis=True      # 表示専用で axis を L2 正規化
        )
        compute_recon_metrics_origscale(
            model=model,
            loader=test_loader,
            device=device,
            z_stats=stats_mm,                 # min/max/width が入っている dict
            feature_names=names_disp,
            out_csv=args.metrics_csv,
            out_csv_by_robot=getattr(args, "metrics_csv_by_robot", None),
            use_mask_only=(args.mask_mode != "none"),
            postprocess_fn=post_fn,           # 逆正規化をここでやっているならそれを渡す
            mask_mode=args.mask_mode,         # "none" | "one" | "k"
            mask_k=getattr(args, "mask_k", 1),
            mask_seed=getattr(args, "seed", None),
            reduction=getattr(args, "mask_reduction", "mean"),
        )







if __name__ == "__main__":
    main()


'''
デバッグ用に実行したい場合
python debug.py
#####ちゃんと実行したい場合
python debug.py \
  --epochs 100 \
  --batch-size 8 \
  --mask-mode none \
  --seed 42 \
  --save-dir checkpoints_debug \
  --log-csv checkpoints_debug/train_log.csv \
  --early-stop-patience 10 --min-delta 1e-4 \
  --metrics-csv checkpoints_debug/test_metrics_origscale.csv \
  --normalize-by mean_edge_len
'''