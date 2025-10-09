# 例（学習スクリプト側）
from urdf_graph_utils import urdf_to_feature_graph, ExtractConfig, to_pyg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
from arm_likeness_gnn import ArmLikenessGNN, TrainCfg, train_one_epoch, eval_loss

def build_data(urdf_path: str, y: float) -> Data:
    S, nodes, X, edge_index, E, scale, _ = urdf_to_feature_graph(
        urdf_path,
        ExtractConfig(drop_endeffector_like=True, movable_only_graph=True, normalize_by="mean_edge_len"),
    )
    d = to_pyg(S, nodes, X, edge_index, E, y=y)  # d.x:[N,7], d.edge_attr:[M,17]
    return d

# データを用意（例）
#valuwはアームロボットっぽいかどうか0~1
train_list = [
    #アームロボット群
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/boxter.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/finger_edu.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/kinova.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/kr150_2.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/panda.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur3.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur3_gripper.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur3_robot.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur5_gripper.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur5_robot.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/ur10_robot.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/z1.urdf", 1.0),
    # ...
    
    # 非アームロボット群
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/a_1.urdf", 0.4),#四脚アームロボット
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/animal-kinova.urdf", 0.0),
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/animal_b.urdf", 0.0),
    ("/home/irsl/heatmap_dnn/gnn_arm_dataset/animal_c.urdf", 0.0),
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/b1-z1.urdf", 0.4),
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/b1.urdf", 0.0),
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/bolt.urdf", 0.0),
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/borinot_flying_arm_2.urdf", 1.0),#おそらくドローンのやつ
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/go1.urdf", 0.0),
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/hextilt_flying_arm_5.urdf", 1.0),
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/hyq_no_sensors.urdf", 0.0),
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/romeo_small.urdf", 0.0),
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/romeo.urdf", 0.0),
    ("/home/irsl/heatmap_dnn/urdf_robot_gnn/solo.urdf", 0.0),
    
]
train_data = [build_data(p, y) for p, y in train_list]
loader = DataLoader(train_data, batch_size=8, shuffle=True)

# モデル
in_node = train_data[0].num_node_features        # 7
in_edge = train_data[0].edge_attr.size(1)        # 17（方向フラグは内部で追加）
model = ArmLikenessGNN(in_node=in_node, in_edge=in_edge, hidden=128, n_layers=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

cfg = TrainCfg(lr=1e-3, weight_decay=1e-4, epochs=50, pos_weight=1.0)
for epoch in range(cfg.epochs):
    tr = train_one_epoch(model, loader, device, cfg)
    print(f"epoch {epoch+1:03d} | train loss {tr:.4f}")
