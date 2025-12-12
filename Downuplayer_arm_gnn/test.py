import torch
import os
import csv
import glob
import numpy as np
from torch_geometric.loader import DataLoader

# 既存のモジュールをインポート
from urdf_core_utils import urdf_to_feature_graph, to_pyg, FEATURE_NAMES, shorten_feature_names
from urdf_norm_utils import compute_global_minmax_stats, apply_global_minmax_inplace, denorm_batch
from gnn_network_min import MaskedTreeAutoencoder

def load_dataset(target_dir):
    """ディレクトリから全URDFを読み込んでデータセットを作成"""
    # 再帰的に検索するように変更
    paths = sorted([p for p in glob.glob(f"{target_dir}/**/*", recursive=True)
                    if p.lower().endswith((".urdf", ".xml"))])
    dataset = []
    print(f"[INFO] Loading training dataset from {target_dir} to compute stats...")
    if not paths:
        print(f"[WARN] No URDF files found in {target_dir}")
        return []
        
    for p in paths:
        try:
            S, nodes, X_np, edge_index, E_np, scale, _ = urdf_to_feature_graph(p)
            d = to_pyg(X_np, edge_index, E_np)
            dataset.append(d)
        except Exception:
            pass
    return dataset

def predict_robot(model, urdf_path, stats, device):
    """単一のロボットに対する予測を実行"""
    try:
        S, nodes, X_np, edge_index, E_np, scale, _ = urdf_to_feature_graph(urdf_path)
        data = to_pyg(X_np, edge_index, E_np).to(device)
    except Exception as e:
        print(f"[Error] Failed to load {urdf_path}: {e}")
        return None

    # massを1.0に固定
    data.x[:, 2] = 1.0

    # 正規化 (学習データから得た stats を使用)
    dummy_list = [data]
    apply_global_minmax_inplace(dummy_list, stats)
    
    # 推論
    model.eval()
    with torch.no_grad():
        # 全ノード再構成
        pred_norm, _ = model(data, mask_idx=None, recon_only_masked=False)
    
    # 逆正規化
    pred_orig = denorm_batch(pred_norm, stats)
    
    return data, nodes, pred_orig.cpu().numpy()

def main():
    # === 設定 ===
    train_dir = "./augmented_dataset"       # 統計計算用 (学習データと同じ場所を指定)
    test_dir = "./merge_joint_robots"       # テスト用URDFがあるディレクトリ
    
    test_urdf_files = [
        "merge_kinova.urdf", 
        "merge_a1.urdf", 
        "merge_nao.urdf"
    ]
    
    model_path = "./checkpoints_augmented/best.pt"
    output_csv = "test_results_6d/test_predictions.csv"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 統計情報の再計算
    train_dataset = load_dataset(train_dir)
    if not train_dataset:
        print(f"Error: Training dataset not found in {train_dir}.")
        print("Please check if 'augmented_dataset' exists.")
        return

    stats = compute_global_minmax_stats(train_dataset)
    print("[INFO] Normalization stats computed.")

    # 2. モデルの準備
    in_dim = train_dataset[0].num_node_features
    model = MaskedTreeAutoencoder(
        in_dim=in_dim,
        hidden=128,
        bottleneck_dim=128,
        enc_rounds=1,
        dec_rounds=1,
        dropout=0.0
    ).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[INFO] Model loaded from {model_path}")

    # 3. テスト実行 & 詳細表示
    header = ["robot", "node_name"] + FEATURE_NAMES + [f"pred_{n}" for n in FEATURE_NAMES]
    
    print("\n=== Prediction Start ===")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for filename in test_urdf_files:
            urdf_path = os.path.join(test_dir, filename)
            
            if not os.path.exists(urdf_path):
                print(f"[Skip] File not found: {urdf_path}")
                continue
                
            print(f"\n>>> Predicting: {filename}")
            res = predict_robot(model, urdf_path, stats, device)
            if res is None: continue
            
            data, node_names, preds = res
            
            # 元の値（正解値）
            inputs_orig = denorm_batch(data.x, stats).cpu().numpy()

            # --- 詳細表示 ---
            print("-" * 85)
            print(f"{'Node Name':<20} | {'Feature':<10} | {'True':>12} | {'Pred':>12} | {'Diff':>12}")
            print("-" * 85)

            for i, name in enumerate(node_names):
                # CSV書き込み
                row = [filename, name] + inputs_orig[i].tolist() + preds[i].tolist()
                writer.writerow(row)
                
                # Origin (位置) の詳細比較
                # 10: origin_x, 11: origin_y, 12: origin_z
                
                # Origin X
                val_t = inputs_orig[i, 10]
                val_p = preds[i, 10]
                diff = val_t - val_p
                print(f"{name:<20} | {'Origin X':<10} | {val_t:12.4f} | {val_p:12.4f} | {diff:12.4f}")
                
                # Origin Y
                val_t = inputs_orig[i, 11]
                val_p = preds[i, 11]
                diff = val_t - val_p
                print(f"{'':<20} | {'Origin Y':<10} | {val_t:12.4f} | {val_p:12.4f} | {diff:12.4f}")

                # Origin Z
                val_t = inputs_orig[i, 12]
                val_p = preds[i, 12]
                diff = val_t - val_p
                print(f"{'':<20} | {'Origin Z':<10} | {val_t:12.4f} | {val_p:12.4f} | {diff:12.4f}")
                
                print("-" * 85)

    print(f"\n[Done] Results saved to {output_csv}")

if __name__ == "__main__":
    main()