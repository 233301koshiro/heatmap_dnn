import sys
import csv
import json
from pathlib import Path
from glob import glob
from typing import List, Dict
import pandas as pd
from urdf_core_utils import load_urdf_any, urdf_to_graph, graph_features, urdf_to_feature_graph
from urdf_viz_utils import draw_graph_png
from urdf_norm_utils import denorm_batch # 逆正規化のためにインポート
from gnn_network_min import _pick_mask_indices_from_batch # マスク処理を合わせるためにインポート
import torch
from torch_geometric.loader import DataLoader
from typing import Any, Dict, List, Optional # Dict と Any 以外にも使っていそうな型を念のため追加
def collect_urdf_paths(dir_path: str, exts: tuple = (".urdf", ".xml")) -> List[str]:
    """
    指定ディレクトリ以下からURDFファイルを再帰的に検索します。
    
    Args:
        dir_path (str): 検索ルートディレクトリ。
        exts (tuple): 対象とする拡張子のタプル。
        
    Returns:
        List[str]: 発見されたファイルパスのソート済みリスト。
    """
    g = glob(str(Path(dir_path) / "**" / "*"), recursive=True)
    return sorted([p for p in g if Path(p).is_file() and p.lower().endswith(exts)])

def dump_one(urdf_path: str, outdir_robot: str) -> None:
    """
    1つのURDFファイルを処理し、グラフ画像、ノード情報CSV、メタデータJSONを出力します。
    
    Args:
        urdf_path (str): 入力URDFファイルのパス。
        outdir_robot (str): 出力先ディレクトリパス。
    """
    out = Path(outdir_robot)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 生のグラフ構造を可視化
    root = load_urdf_any(urdf_path)
    G_full, _ = urdf_to_graph(root)
    draw_graph_png(G_full, str(out / "urdf_full.png"), "URDF Full Joint Graph")

    # 2. 特徴量抽出と処理済みグラフの可視化
    S, nodes, X, edge_index, E, scale, _ = graph_features(G_full)
    draw_graph_png(S, str(out / "processed.png"), "Processed Graph")

    # 3. ノード情報のCSV出力
    rows = []
    for n in S.nodes():
        nd = S.nodes[n]
        # 必要な属性を辞書から取得して行データを作成
        rows.append({
            "node": n,
            "deg": int(S.degree(n)),
            "mass": float(nd.get("mass", 0.0)),
            "joint_type": nd.get("joint_type", "fixed"),
            # ... 他の必要な属性もここに追加 ...
        })
    pd.DataFrame(rows).to_csv(str(out / "nodes_processed.csv"), index=False)

    # 4. メタデータのJSON出力
    meta = {
        "input": str(Path(urdf_path).resolve()),
        "n_nodes": int(X.shape[0]),
        "n_edges": int(edge_index.shape[1]),
        "x_shape": list(X.shape),
    }
    with open(str(out / "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def make_compare_csv(pre_csv_path: str, post_csv_path: str, out_csv_path: str):
    """
    pre/post の graph_sizes_summary.csv から、
    robot_name, pre_node, post_node, pre_edge, post_edge を出力。
    - pre 側のキー: stem(name)
    - post 側のキー: stem(name) から先頭の 'merge_' を除去
    - どちらにも存在するもののみ（inner join）
    """
    pre = pd.read_csv(pre_csv_path)
    post = pd.read_csv(post_csv_path)

    for df, tag in [(pre, "pre"), (post, "post")]:
        if "name" not in df.columns:
            raise SystemExit(f"[ERROR] 'name' 列が見つかりません: {tag} ({pre_csv_path if tag=='pre' else post_csv_path})")

    pre = pre.rename(columns={"name": "pre_name", "num_nodes": "pre_node", "num_edges": "pre_edge"})
    pre["pre_key"] = pre["pre_name"].map(lambda s: Path(str(s)).stem)

    post = post.rename(columns={"name": "post_name", "num_nodes": "post_node", "num_edges": "post_edge"})
    post["post_key"] = post["post_name"].map(lambda s: Path(str(s)).stem)
    post["post_key"] = post["post_key"].str.replace(r"^merge_", "", regex=True)

    comp = pd.merge(
        pre, post,
        left_on="pre_key", right_on="post_key",
        how="inner", suffixes=("_pre", "_post")
    )

    out = comp[["pre_key", "pre_node", "post_node", "pre_edge", "post_edge"]].copy()
    out = out.rename(columns={"pre_key": "robot_name"})
    out.sort_values("robot_name", inplace=True, kind="mergesort")

    Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv_path, index=False)
    print(f"[saved] {out_csv_path}  ({len(out)} rows)")
    
def export_graph_sizes_csv(urdf_paths: List[str], csv_path: str) -> None:
    """
    複数のURDFファイルのグラフサイズ（ノード数、エッジ数）を集計してCSVに出力します。
    
    Args:
        urdf_paths (List[str]): URDFファイルパスのリスト。
        csv_path (str): 出力先CSVファイルのパス。
    """
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "num_nodes", "num_edges"])
        for p in urdf_paths:
            try:
                # 特徴量抽出のみ行いサイズを取得
                _, _, X, edge_index, _, _, _ = urdf_to_feature_graph(p)
                w.writerow([Path(p).name, X.shape[0], edge_index.shape[1]])
            except Exception as e:
                print(f"[WARN] failed to get size for {p}: {e}", file=sys.stderr)
                w.writerow([Path(p).name, -1, -1])
    print(f"[saved] {csv_path}")

def process_dir_flat(dir_path: str, out_root: str, exts: tuple = (".urdf", ".xml")) -> List[str]:
    """
    ディレクトリ内の全URDFを一括処理し、結果をフラットな構造で出力します。
    同名ファイルは連番を付与して区別します。
    
    Args:
        dir_path (str): 入力ルートディレクトリ。
        out_root (str): 出力ルートディレクトリ。
        exts (tuple): 対象拡張子。
        
    Returns:
        List[str]: 処理されたファイルのフルパスリスト。
    """
    urdfs = collect_urdf_paths(dir_path, exts)
    print(f"[INFO] Found {len(urdfs)} files in {dir_path}")
    
    used_names = {}
    for p in urdfs:
        # ファイル名の衝突回避
        stem = Path(p).stem
        if stem in used_names:
            used_names[stem] += 1
            name = f"{stem}_{used_names[stem]}"
        else:
            used_names[stem] = 1
            name = stem
            
        try:
            dump_one(p, str(Path(out_root) / name))
        except Exception as e:
            print(f"[WARN] Failed to process {p}: {e}", file=sys.stderr)

    # 最後にサマリCSVを出力
    export_graph_sizes_csv(urdfs, str(Path(out_root) / "graph_sizes_summary.csv"))
    return [str(Path(p).resolve()) for p in urdfs]

#choreonoidでビジュアライズするために特徴量のcsvを出力
#モデルの予測値をcsvに転記する(入力:test_loader, 出力:preds)
@torch.no_grad()
def export_predictions_to_csv(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    stats: Dict[str, Any], # 逆正規化のための統計情報 (stats_mm)
    feature_names: List[str], # CSVの列名用
    output_csv_path: str,
    mask_mode: str = "none",
    mask_k: int = 0
):
    """
    モデルの予測値（逆正規化済み）をCSVファイルとしてエクスポートする。
    """
    model.eval()
    all_rows = [] # CSVの行データを格納するリスト
    
    # CSVのヘッダーを作成 (識別子 + 特徴量名)
    headers = ["urdf_path", "node_name"] + feature_names

    print(f"\n===== EXPORTING PREDICTIONS =====")
    print(f"Saving to {output_csv_path}...")

    for data in loader:
        data = data.to(device)

        if not hasattr(data, 'node_names'):
             print("[ERROR] 'data.node_names' not found.", file=sys.stderr)
             return

        # ステップ1の修正が適用されているか確認
        if not hasattr(data, 'node_names'):
             print("[ERROR] 'data.node_names' not found. Did you modify 'build_data'?", file=sys.stderr)
             return # 処理中断

        # (1) 評価時と同様のマスクを（必要なら）作成
        mask_idx = None
        if mask_mode in ("one", "k") and hasattr(data, "ptr"):
            k = 1 if mask_mode == "one" else max(1, int(mask_k))
            mask_idx = _pick_mask_indices_from_batch(data, k=k).to(device)

        # (2) モデルで予測 (正規化済み)
        # recon_only_masked=False で全ノードの予測を取得
        pred_norm, out = model(data, mask_idx=mask_idx, recon_only_masked=False) 

        # (3) 予測値を逆正規化 (元のスケールに戻す)
        pred_orig = denorm_batch(pred_norm, stats)
        
        # (4) ノード識別子と予測値をマージ
        
        # バッチ内のノード名リスト (ステップ1で追加)
        try:
            node_names_list = [name for graph_nodes in data.node_names for name in graph_nodes]
        except TypeError:
            # もし既にフラットだった場合（将来の修正に備えて）
            node_names_list = data.node_names
        
        # バッチ内のURDFパスリスト (data.name と data.batch から作成)
        urdf_paths_list = [data.name[i] for i in data.batch]

        pred_np = pred_orig.cpu().numpy()

        print("---------------------------------")
        print(f"[DEBUG] data.num_nodes: {data.num_nodes}")
        print(f"[DEBUG] len(node_names_list) (After Flatten): {len(node_names_list)}")
        print(f"[DEBUG] data.num_graphs: {data.num_graphs}")
        print("---------------------------------")

        # 各ノードの情報をCSVの1行として追加
        for i in range(data.num_nodes):
            row_data = [
                urdf_paths_list[i],   # どのURDFか
                node_names_list[i]    # どのノードか
            ] + pred_np[i].tolist()   # 予測値
            all_rows.append(row_data)

    # (5) CSVに書き出し
    try:
        df = pd.DataFrame(all_rows, columns=headers)
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Predictions saved to {output_path} ({len(df)} rows)")

    except Exception as e:
        print(f"[ERROR] Failed to write CSV: {e}", file=sys.stderr)