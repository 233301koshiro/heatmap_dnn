# ====== Imports ===============================================================
import os, csv, time, random
from typing import List, Tuple, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from new_urdf_graph_utils import urdf_to_feature_graph, ExtractConfig, to_pyg

# ★ ここを ArmLikeness 用から AE 用に差し替え ★
from masked_tree_autoencoder import (
    MaskedTreeAutoencoder,
    TrainCfg,           # lr, weight_decay, epochs, recon_only_masked など
    train_one_epoch,    # すでに masked_tree_autoencoder.py に実装済み
    eval_loss           # 同上（検証も再構成誤差で測る）
)

# ====== URDF → PyG Data 変換 =================================================
def build_data(urdf_path: str) -> Data:
    """
    URDFから特徴グラフを作って PyG Data に変換する（ラベル不要：自己教師あり）。
    """
    S, nodes, X, edge_index, E, scale, _ = urdf_to_feature_graph(
        urdf_path,
        ExtractConfig(normalize_by="mean_edge_len"),
    )
    # ★ y は不要（自己教師あり）。to_pyg の引数に y を渡さない/None でOK
    d = to_pyg(S, nodes, X, edge_index, E)
    d.name = urdf_path
    return d

# ====== 学習用の URDF 一覧（例）==============================================
train_list: List[str] = [
    # アーム/非アームなど混ざっていてOK（自己教師ありなのでラベル不要）
    "/home/irsl/heatmap_dnn/gnn_arm_dataset/baxter.urdf",
    "/home/irsl/heatmap_dnn/gnn_arm_dataset/kinova.urdf",
    "/home/irsl/heatmap_dnn/gnn_arm_dataset/ur5_robot.urdf",
    "/home/irsl/heatmap_dnn/urdf_robot_gnn/anymal_c.urdf",
    "/home/irsl/heatmap_dnn/urdf_robot_gnn/go1.urdf",
    "/home/irsl/heatmap_dnn/urdf_robot_gnn/romeo.urdf",
    # …必要に応じて追加 …
]

# ====== Data 構築（空グラフは除外）==========================================
dataset: List[Data] = []
for p in train_list:
    d = build_data(p)
    if d.num_nodes == 0:
        print(f"[skip] empty graph: {p}"); continue
    dataset.append(d)

if len(dataset) == 0:
    raise RuntimeError("有効なデータが0件です。URDFや抽出設定を見直してください。")

# ====== train/val/test 分割 ===================================================
random.seed(0); torch.manual_seed(0)
N = len(dataset)
idx = list(range(N)); random.shuffle(idx)
n_tr = int(N * 0.8); n_va = int(N * 0.1)
tr_idx, va_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]

train_set = [dataset[i] for i in tr_idx]
val_set   = [dataset[i] for i in va_idx]
test_set  = [dataset[i] for i in te_idx]

train_loader = DataLoader(train_set, batch_size=8, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_set,   batch_size=8, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_set,  batch_size=8, shuffle=False, drop_last=False)

print(f"[dataset] total={N} | train={len(train_set)} val={len(val_set)} test={len(test_set)}")

# ====== モデル作成（★ここが Arm→AE の主差分）===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_node = train_set[0].num_node_features  # 例: 19
# anchor に全特徴を渡すなら anchor_idx=None のまま
# 特定列だけをアンカーにしたい時は slice(start, stop) を指定（例: slice(0,7)）
model = MaskedTreeAutoencoder(
    in_dim=in_node,
    hidden=128,
    enc_rounds=2,
    dec_rounds=2,
    dropout=0.1,
    anchor_idx=None,
).to(device)

# AE 用 Train 設定：recon_only_masked=True なら「マスク行だけ」で誤差を計算
cfg = TrainCfg(lr=1e-3, weight_decay=1e-4, epochs=50, recon_only_masked=True)

# ====== ログ/チェックポイント ===============================================
os.makedirs("checkpoints", exist_ok=True)
log_csv = "checkpoints/ae_train_log.csv"
best_val = float("inf")

if not os.path.exists(log_csv):
    import csv
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_recon", "val_recon", "sec"])

# ====== 学習ループ（再構成誤差で監視）======================================
for epoch in range(1, cfg.epochs + 1):
    t0 = time.time()
    train_recon = train_one_epoch(model, train_loader, device, cfg)
    # 評価はマスクのみで見るか（True）/全ノードで見るか（False）は好み
    val_recon   = eval_loss(model, val_loader, device, recon_only_masked=True)

    print(f"epoch {epoch:03d} | train {train_recon:.4f} | val {val_recon:.4f} | {time.time()-t0:.1f}s")

    with open(log_csv, "a", newline="") as f:
        csv.writer(f).writerow([epoch, f"{train_recon:.6f}", f"{val_recon:.6f}", f"{time.time()-t0:.2f}"])

    # ベスト更新でフル保存
    if val_recon < best_val:
        best_val = val_recon
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": getattr(model, "_opt").state_dict(),
            "cfg": cfg.__dict__,
            "val_recon": val_recon,
            "timestamp": time.time(),
        }
        torch.save(ckpt, "checkpoints/masked_tree_ae_best.pt")

    # いつも最新も保存（軽量）
    torch.save(model.state_dict(), "checkpoints/masked_tree_ae_latest.pth")

# ====== テスト再構成誤差 ====================================================
test_recon = eval_loss(model, test_loader, device, recon_only_masked=True)
print(f"[TEST] recon_only_masked=True | recon={test_recon:.4f}")

# 安定性を見たい場合は全ノードでも
test_recon_all = eval_loss(model, test_loader, device, recon_only_masked=False)
print(f"[TEST] recon_only_masked=False | recon={test_recon_all:.4f}")
