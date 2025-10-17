# ====== Imports ===============================================================
import os, csv, time, random
from typing import List, Tuple, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from urdf_to_graph_utilis import urdf_to_feature_graph, ExtractConfig, to_pyg


from gnn_network import (
    MaskedTreeAutoencoder,
    TrainCfg,           
    train_one_epoch,    
    eval_loss          
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
train_list: List[Tuple[str, float]] = [
    # --- アームロボット群（正例） ---
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/baxter.urdf"),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/finger_edu.urdf"),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/kinova.urdf"),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/kr150_2.urdf"),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/panda.urdf"),
    #("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur3.urdf"),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur3_gripper.urdf"),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur3_robot.urdf"),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur5_gripper.urdf"),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur5_robot.urdf"),

    # --- 非アーム（負例 or 中間） ---
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/a1.urdf"),          # 四脚*アーム（中間）
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/anymal_c.urdf"),    # 四脚
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/b1-z1.urdf"),       # 四脚*アーム（中間）
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/b1.urdf"),          # 四脚
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/bolt.urdf"),        # 二足/その他
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/borinot_flying_arm_2.urdf"),  # 飛行+アーム
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/go1.urdf"),         # 四脚
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/hextilt_flying_arm_5.urdf"),  # 飛行+アーム
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/hyq_no_sensors.urdf"),        # 四脚
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/romeo_small.urdf"), # ヒューマノイド
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/romeo.urdf"),       # ヒューマノイド

    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur10_robot.urdf"),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/z1.urdf"),
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/solo.urdf"),              # 二足/ヒューマノイド
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/anymal-kinova.urdf"),     # 四脚*アーム
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/anymal_b.urdf"),          # 四脚
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

# ====== train/val/test 分割 ======
random.seed(0); torch.manual_seed(0)
N = len(dataset)
idx = list(range(N)); random.shuffle(idx)

# 極小データでも val/test を 1 件以上確保
n_tr = int(N * 0.8)
n_va = int(N * 0.1)
# 下限を設定
n_va = max(1, n_va)
n_te = max(1, N - (n_tr + n_va))
# 足りない/はみ出しを調整
if n_tr + n_va + n_te > N:
    n_tr = max(1, N - (1 + 1))  # val=1, test=1 を最低確保
    n_va = 1
    n_te = 1

tr_idx, va_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:n_tr+n_va+n_te]

train_set = [dataset[i] for i in tr_idx]
val_set   = [dataset[i] for i in va_idx]
test_set  = [dataset[i] for i in te_idx]

# バッチはデータ数に合わせて
bs = min(8, len(train_set))  # 今回は4
train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_set,   batch_size=max(1, len(val_set)), shuffle=False, drop_last=False)
test_loader  = DataLoader(test_set,  batch_size=max(1, len(test_set)), shuffle=False, drop_last=False)

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
cfg = TrainCfg(lr=1e-3, weight_decay=1e-4, epochs=100, recon_only_masked=True)

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
print(f"[TEST] recon_only_masked=True | recon={test_recon:.4f}")#1nodeだけマスクして評価はそのノードだけで誤差計算

# 安定性を見たい場合は全ノードでも
test_recon_all = eval_loss(model, test_loader, device, recon_only_masked=False)
print(f"[TEST] recon_only_masked=False | recon={test_recon_all:.4f}")#1nodeだけマスクして評価は全ノードで誤差計算
