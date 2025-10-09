# train_arm_linkness.py  — clean & commented
# 依存: torch, torch_geometric, urdf_graph_utils（あなたのモジュール）, arm_likeness_gnn（あなたのGNN）
# 目的: URDF→グラフ→GNNで「アームらしさ」回帰（0〜1）、学習/検証/評価まで一式

from __future__ import annotations

# ====== Imports（すべて先頭に集約）===========================================
import os
import csv
import time
import random
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from urdf_graph_utils import urdf_to_feature_graph, ExtractConfig, to_pyg
from arm_likeness_gnn import ArmLikenessGNN, TrainCfg, train_one_epoch, eval_loss

# ============================================================================


# ====== ユーティリティ: URDF→PyG Data 変換 =================================
def build_data(urdf_path: str, y: float) -> Data:
    """
    URDFから特徴グラフを作って PyG Data に変換する。
    - drop_endeffector_like: 末端（グリッパなど）の影響を抑える
    - movable_only_graph: 可動ジョイントのみでグラフ化（空グラフになる場合あり）
    - normalize_by: 辺長の平均で正規化
    """
    S, nodes, X, edge_index, E, scale, _ = urdf_to_feature_graph(
        urdf_path,
        ExtractConfig(
            drop_endeffector_like=True,
            movable_only_graph=True,
            normalize_by="mean_edge_len",
        ),
    )
    # d.x:[N,7], d.edge_attr:[M,17] を想定
    d = to_pyg(S, nodes, X, edge_index, E, y=y)
    d.name = urdf_path                            # ← 追加：後で表示に使う
    return d


# ====== 学習用のデータリスト（パスとラベル）===============================
# value は「アームロボットっぽさ(0〜1)」
train_list: List[Tuple[str, float]] = [
    # --- アームロボット群（正例） ---
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/baxter.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/finger_edu.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/kinova.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/kr150_2.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/panda.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur3.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur3_gripper.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur3_robot.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur5_gripper.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur5_robot.urdf", 1.0),

    # --- 非アーム（負例 or 中間） ---
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/a1.urdf", 0.4),          # 四脚*アーム（中間）
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/anymal_c.urdf", 0.0),    # 四脚
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/b1-z1.urdf", 0.4),       # 四脚*アーム（中間）
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/b1.urdf", 0.0),          # 四脚
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/bolt.urdf", 0.0),        # 二足/その他
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/borinot_flying_arm_2.urdf", 1.0),  # 飛行+アーム
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/go1.urdf", 0.0),         # 四脚
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/hextilt_flying_arm_5.urdf", 1.0),  # 飛行+アーム
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/hyq_no_sensors.urdf", 0.0),        # 四脚
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/romeo_small.urdf", 0.0), # ヒューマノイド
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/romeo.urdf", 0.0),       # ヒューマノイド

    # --- テスト用の保留候補（必要になったら有効化）
     ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur10_robot.urdf", 1.0),
     ("/home/irsl/heatmap_dnn/gnn_arm_dataset/z1.urdf", 1.0),
     ("/home/irsl/heatmap_dnn/urdf_robot_gnn/solo.urdf", 0.0),              # 二足/ヒューマノイド
     ("/home/irsl/heatmap_dnn/urdf_robot_gnn/anymal-kinova.urdf", 0.4),     # 四脚*アーム
     ("/home/irsl/heatmap_dnn/urdf_robot_gnn/anymal_b.urdf", 0.0),          # 四脚
]


# ====== データ生成（空グラフは除外）======================================
train_data: List[Data] = []
for p, y in train_list:
    d = build_data(p, y)
    if d.num_nodes == 0:  # 可動ジョイント0 などで空グラフ化したものは除外
        print(f"[skip] empty graph (movable joints 0): {p}")
        continue
    train_data.append(d)

if len(train_data) == 0:
    raise RuntimeError("有効な学習データが0件です。ExtractConfigやデータセットを見直してください。")


# ====== データ分割: train/val/test（8:1:1）=================================
random.seed(0); torch.manual_seed(0)

N = len(train_data)
idx = list(range(N)); random.shuffle(idx)
n_tr = int(N * 0.8); n_va = int(N * 0.1)
tr_idx, va_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

train_set = [train_data[i] for i in tr_idx]
val_set   = [train_data[i] for i in va_idx]
test_set  = [train_data[i] for i in te_idx]

# drop_last=True は BatchNorm回避や勾配安定に効く（LayerNormなら必須ではないが安定度UP）
train_loader = DataLoader(train_set, batch_size=8, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_set,   batch_size=8, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_set,  batch_size=8, shuffle=False, drop_last=False)

print(f"[dataset] total={N} | train={len(train_set)} val={len(val_set)} test={len(test_set)}")


# ====== モデル作成＆学習設定 ==============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ノード/エッジ次元は最初のサンプルから取得
in_node   = train_data[0].num_node_features       # 例: 7
base_edge = train_data[0].edge_attr.size(1)       # 例: 17（素のエッジ特徴）
in_edge   = base_edge + 1                         # 双方向化で「方向フラグ(+1)」を付与するため

model = ArmLikenessGNN(in_node=in_node, in_edge=in_edge, hidden=128, n_layers=3).to(device)

# pos_weight は正例が少ない場合 >1 にすると良い（例: 3.0 など）
cfg = TrainCfg(lr=1e-3, weight_decay=1e-4, epochs=50, pos_weight=1.0)

# ====== ロギング/チェックポイント =========================================
os.makedirs("checkpoints", exist_ok=True)
log_csv = "checkpoints/train_log.csv"
best_val = float("inf")

# CSVヘッダー（初回のみ）
if not os.path.exists(log_csv):
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "timestamp"])


# ====== 学習ループ（各エポックで train/val を表示&保存）====================
for epoch in range(1, cfg.epochs + 1):
    t0 = time.time()

    # 学習1エポック
    train_loss = train_one_epoch(model, train_loader, device, cfg)

    # 検証ロス（過学習監視）
    val_loss = eval_loss(model, val_loader, device)

    # コンソール出力
    print(f"epoch {epoch:03d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | {time.time()-t0:.1f}s")

    # CSV 追記
    with open(log_csv, "a", newline="") as f:
        csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", int(time.time())])

    # ベストモデルを上書き保存（完全復元用 .pt）
    if val_loss < best_val:
        best_val = val_loss
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": model._opt.state_dict(),
            "cfg": cfg.__dict__,
            "val_loss": val_loss,
            "timestamp": time.time(),
        }
        torch.save(ckpt, "checkpoints/arm_likeness_best.pt")

    # つねに最新パラメータも保存（軽量 .pth）
    torch.save(model.state_dict(), "checkpoints/arm_likeness_latest.pth")


# ====== 指標計算: 閾値最適化（val）→ テスト評価 ============================
@torch.no_grad()
def eval_metrics(model: torch.nn.Module, loader: DataLoader, device: torch.device, thr: float = 0.5):
    """
    2値タスクの基本指標（Acc/F1/Prec/Rec）＋可能なら AUC / AP を返す。
    - K≠B（空グラフ）対策として、実在グラフID（present）で y を揃える。
    """
    model.eval()
    ys, ps = [], []
    for data in loader:
        data = data.to(device)
        logit = model(data)  # [K]
        if logit.numel() == 0:
            continue
        present = torch.unique(data.batch)
        y = data.y.to(device).float()[present]  # [K]
        p = torch.sigmoid(logit)                # [K]
        ys.append(y.cpu()); ps.append(p.cpu())

    if not ys:  # 全て空だった場合のガード
        return dict(acc=float("nan"), f1=float("nan"), prec=float("nan"), rec=float("nan"),
                    auc=float("nan"), ap=float("nan"), tp=0, fp=0, tn=0, fn=0)

    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()

    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())

    acc  = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec  = tp / max(1, (tp + fn))
    f1   = 2 * prec * rec / max(1e-9, (prec + rec))

    # sklearn がある場合のみ AUC / AP を計算
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = float(roc_auc_score(y, p))
        ap  = float(average_precision_score(y, p))
    except Exception:
        auc = float("nan")
        ap  = float("nan")

    return dict(acc=acc, f1=f1, prec=prec, rec=rec, auc=auc, ap=ap, tp=tp, fp=fp, tn=tn, fn=fn)


@torch.no_grad()
def find_best_threshold(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    検証セット上でしきい値（0.05〜0.95）を総当りし、Youden's J（TPR−FPR）が最大の閾値を返す。
    """
    model.eval()
    ys, ps = [], []
    for data in loader:
        data = data.to(device)
        logit = model(data)
        if logit.numel() == 0:
            continue
        present = torch.unique(data.batch)
        ys.append(data.y.to(device).float()[present].cpu())
        ps.append(torch.sigmoid(logit).cpu())

    if not ys:  # まれに全て空
        return 0.5

    y = torch.cat(ys).numpy()
    p = torch.cat(ps).numpy()

    best_thr, best_j = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 19):
        pred = (p >= thr).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        tpr = tp / max(1, (tp + fn))
        fpr = fp / max(1, (fp + tn))
        j = tpr - fpr
        if j > best_j:
            best_j, best_thr = j, float(thr)
    return best_thr

@torch.no_grad()
def score_by_sample_with_loader(model, dataset, device, out_csv_path=None):
    model.eval()
    rows = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    for i, data in enumerate(loader):
        data = data.to(device)
        logit = model(data)              # [1] or [K]
        if logit.numel() == 0:
            continue
        present = torch.unique(data.batch)
        y = data.y.float()[present].cpu().numpy()
        p = torch.sigmoid(logit).detach().cpu().numpy()

        # 名前は元Dataから運ぶ: 事前に build_data 内で d.name = urdf_path 済み前提
        # DataLoader経由でも custom attr は保持されることが多いですが、
        # 念のため fallback を用意
        name = getattr(data, "name", f"sample_{i}")
        for k in range(len(p)):
            rows.append({
                "name": name,
                "index_in_batch": int(k),
                "label": float(y[k]),
                "logit": float(logit.detach().cpu().numpy()[k]),
                "prob": float(p[k]),
            })

    if out_csv_path and len(rows) > 0:
        import csv, os
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        with open(out_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return rows




# ====== 最終評価 ============================================================
best_thr = find_best_threshold(model, val_loader, device)
test_metrics = eval_metrics(model, test_loader, device, thr=best_thr)
print(f"[TEST] thr={best_thr:.2f} | {test_metrics}")

# --- 追加：テストセットの各URDFごとのスコアを出力 ---
per_sample = score_by_sample_with_loader(
    model, test_set, device, out_csv_path="checkpoints/per_sample_test.csv"
)


# 見やすいように確率の降順で上位を表示（最大10件）
per_sample_sorted = sorted(per_sample, key=lambda r: r["prob"], reverse=True)
print("\n[TEST per-sample top 10 by prob]")
for r in per_sample_sorted[:10]:
    print(f"{r['prob']:.3f} | y={r['label']:.1f} | {r['name']}")

print("\n保存: checkpoints/per_sample_test.csv")
