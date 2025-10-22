# ====== Imports ===============================================================
import os, csv, time, random
from typing import List, Tuple, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from urdf_to_graph_utilis import urdf_to_feature_graph,minimal_dataset_report, ExtractConfig, to_pyg


from gnn_network_min import (
    MaskedTreeAutoencoder,
    TrainCfg,           
    train_one_epoch,    
    eval_loss          
)
#1ノードマスクしたいとき
MASK_MODE = 'one'
#オートエンコーダ用の何もマスクしない
MASK_MODE = 'none'

cfg = TrainCfg(lr=1e-3, weight_decay=1e-4, epochs=1, recon_only_masked=True, mask_strategy=MASK_MODE)
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

minimal_dataset_report(dataset)#この感じだとdatasetは問題なさそう

#ここで正規化を行うが現状，どれに対して正規化を行うか，one-hotに対して正規化を行うのはあまりよろしくないかなどは不明なので保留

# ====== train/val/test 分割 ======
#ここは17/5/3で確認されてるため検証不要
random.seed(0); torch.manual_seed(0)
N = len(dataset)
idx = list(range(N)); random.shuffle(idx)
# 目標比率
p_tr, p_va = 0.7, 0.2
# まず切る（下限1を考慮しつつ）
n_tr = max(1, int(N * p_tr))
n_va = max(1, int(N * p_va))
# はみ出しを抑えて、残りを test に回す
# n_te は必ず 0 以上にし、合計を N に合わせる
if n_tr + n_va > N - 1:
    n_va = max(1, N - 1 - n_tr)
    if n_va < 1:          # まだダメなら train を削って調整
        n_tr = max(1, N - 2)
        n_va = 1
n_te = N - n_tr - n_va    # ここで必ず合計が N になる（n_te>=0保証）
# スライス
tr_idx = idx[:n_tr]
va_idx = idx[n_tr:n_tr+n_va]
te_idx = idx[n_tr+n_va:]
train_set = [dataset[i] for i in tr_idx]
val_set   = [dataset[i] for i in va_idx]
test_set  = [dataset[i] for i in te_idx]
# バッチはデータ数に合わせて
bs = min(8, len(train_set))  # 今回は4
train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, drop_last=False)
val_loader   = DataLoader(val_set,   batch_size=max(1, len(val_set)), shuffle=False, drop_last=False)
test_loader  = DataLoader(test_set,  batch_size=max(1, len(test_set)), shuffle=False, drop_last=False)
print(f"[dataset] total={N} | train={len(train_set)} val={len(val_set)} test={len(test_set)}")

# ====== モデル作成===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_node = train_set[0].num_node_features  # 例: 19
#num_node_featuresというのはPyG Dataオブジェクトの属性で，各ノードの特徴量の次元数を表す

model = MaskedTreeAutoencoder(#ここでモデルの構造を定義
    in_dim=in_node,#入力特徴量次元数
    hidden=128,#隠れ層次元数
    bottleneck_dim = 128,#ボトルネック次元数
    enc_rounds=2,#エンコーダの反復回数
    dec_rounds=2,#デコーダの反復回数
    dropout=0.1,#ドロップアウト率(ドロップアウトは過学習防止のための手法)
    anchor_idx=None,#アンカー特徴量のインデックス（Noneなら全特徴量を使用する)特定列だけをアンカーにしたい時は slice(start, stop) を指定（例: slice(0,7)）
).to(device)
model._cfg = cfg  # 型情報保持用
#学習させたいわけじゃなくて学習途中のモデルのデータを追いたいのでlogはいらない

for epoch in range(1, cfg.epochs + 1):
    t0 = time.time()
    #early stopは学習ループの最後で判定
    #判定基準は検証誤差の改善有無
    # 1) 学習1エポック
    train_recon = train_one_epoch(model, train_loader, device, cfg)

    # 2) 検証（安定さ重視なら recon_only_masked=False でも可）
    val_recon = eval_loss(model, val_loader, device, recon_only_masked=True)

    # 3) ログ表示
    print(f"epoch {epoch:03d} | train {train_recon:.4f} | val {val_recon:.4f} | {time.time()-t0:.1f}s")


if MASK_MODE == 'none':
    test_recon_all = eval_loss(model, test_loader, device, recon_only_masked=False, mask_strategy=MASK_MODE)
    print(f"[TEST] recon={test_recon_all:.4f}")
else:
    test_recon = eval_loss(model, test_loader, device, recon_only_masked=True, mask_strategy=MASK_MODE)
    print(f"[TEST] recon_only_masked=True | recon={test_recon:.4f}")#1nodeだけマスクして評価はマスクノードのみで誤差計算

    test_recon_all = eval_loss(model, test_loader, device, recon_only_masked=False, mask_strategy=MASK_MODE)
    print(f"[TEST] recon_only_masked=False | recon={test_recon_all:.4f}")#1nodeだけマスクして評価は全ノードで誤差計算
