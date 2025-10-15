# dump_urdf_graph.py — nodes-only CSV版
import argparse, os, json
import pandas as pd
from new_urdf_graph_utils import (
    ExtractConfig, load_urdf_any, urdf_to_graph, graph_features, draw_graph_png
)

def main():
    ap = argparse.ArgumentParser(description="URDF→グラフ→ノードCSV/PNG（剪定なし・エッジ属性なし）")
    ap.add_argument("urdf", help="URDFのパス（xacroは事前展開）")
    ap.add_argument("--outdir", default="out", help="出力ディレクトリ")
    ap.add_argument("--no-normalize", action="store_true", help="origin距離の平均で正規化しない")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.urdf))[0]
    out_base = os.path.join(args.outdir, stem)

    cfg = ExtractConfig(normalize_by=("none" if args.no_normalize else "mean_edge_len"))

    # フル構造の可視化
    root = load_urdf_any(args.urdf)
    G_full, _ = urdf_to_graph(root)
    draw_graph_png(G_full, out_base + "_urdf_full.png", "URDF Full Joint Graph")

    # 特徴抽出（削除なし・ジョイント情報は子ノード側・エッジ属性なし）
    S, node_list, X, edge_index, E, scale, removed_records = graph_features(G_full, cfg)

    # ノードCSV：mass と “子ノード側のジョイント属性” を展開
        # ノードCSV：mass と “子ノード側のジョイント属性” をそのまま出力
    # joint_axis / joint_origin_xyz は (x,y,z) を 1セルに JSON 文字列で入れる
    rows = []
    for n in S.nodes():
        nd = S.nodes[n]
        rows.append({
            "node": n,
            "deg": int(S.degree(n)),
            "mass": float(nd.get("mass", 0.0)),
            "joint_name": nd.get("joint_name", ""),
            "joint_type": nd.get("joint_type", "fixed"),
            "joint_type_idx": int(nd.get("joint_type_idx", 3)),
            # ここを1列化（JSONで保存）
            "joint_axis": json.dumps(list(nd.get("joint_axis", (0.0, 0.0, 0.0)))),
            "joint_origin_xyz": json.dumps(list(nd.get("joint_origin_xyz", (0.0, 0.0, 0.0)))),
            "joint_movable": int(nd.get("joint_movable", 0)),
            "joint_limit_width": float(nd.get("joint_limit_width", 0.0)),
            "joint_limit_lower": float(nd.get("joint_limit_lower", 0.0)),
            "joint_limit_upper": float(nd.get("joint_limit_upper", 0.0)),
        })
    df_nodes = pd.DataFrame(rows)
    nodes_csv = f"{out_base}_nodes_processed.csv"
    df_nodes.to_csv(nodes_csv, index=False)


    # 処理後グラフの可視化
    draw_graph_png(S, out_base + "_processed.png", "Processed Graph (no pruning / no edge attrs)")

    # メタ情報（学習監査用）
    meta = {
        "input": args.urdf,
        "scale_mean_edge_len": float(scale),
        "n_nodes": int(len(S.nodes())),
        "n_edges": int(len(S.edges())),
        "x_shape": [int(X.shape[0]), int(X.shape[1])],             # [N, node_dim]
        "edge_index_shape": [int(edge_index.shape[0]), int(edge_index.shape[1])],  # [2, M]
        "edge_attr_shape": [int(E.shape[0]), int(E.shape[1])],     # [M, 0]想定
    }
    with open(out_base + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


    print("Saved:")
    print("  ", out_base + "_urdf_full.png")
    print("  ", out_base + "_processed.png")
    print("  ", nodes_csv)
    print("  ", out_base + "_meta.json")

if __name__ == "__main__":
    main()
