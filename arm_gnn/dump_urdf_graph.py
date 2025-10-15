# dump_urdf_graph.py
import argparse, os, json
import pandas as pd
from urdf_graph_utils import (
    ExtractConfig, load_urdf_any, urdf_to_graph, graph_features, draw_graph_png
)

def export_csv(df_nodes: pd.DataFrame, df_edges: pd.DataFrame, out_base: str, suffix: str):
    np_nodes = f"{out_base}_nodes_{suffix}.csv"
    np_edges = f"{out_base}_edges_{suffix}.csv"
    df_nodes.to_csv(np_nodes, index=False)
    df_edges.to_csv(np_edges, index=False)
    return np_nodes, np_edges

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf", help="URDF or XACRO path")
    ap.add_argument("--outdir", default="out", help="output directory")
    ap.add_argument("--keep-eef", action="store_true", help="末端(手/指/tool0等)を剪定しない")
    ap.add_argument("--include-fixed", action="store_true", help="fixed関節も含める")
    ap.add_argument("--no-normalize", action="store_true", help="距離正規化をしない")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out_base = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.urdf))[0])

    cfg = ExtractConfig(
        drop_endeffector_like=not args.keep_eef,
        movable_only_graph=not args.include_fixed,
        normalize_by=("none" if args.no_normalize else "mean_edge_len"),
    )
    
    # フル構造PNG（比較用）
    urdf = load_urdf_any(args.urdf)
    G_full, _ = urdf_to_graph(urdf)
    draw_graph_png(G_full, out_base + "_urdf_full.png", "URDF Full Joint Graph")

    # 処理後（剪定 + 可動骨格）→ 特徴＋PNG
    # graph_features の戻りが 7要素になったので受け取りを変更
    S, node_list, X, edge_index, E, scale, removed_records = graph_features(G_full, cfg)

    # 既存のノード/エッジCSV作成はそのまま…
    # 追加：removed_nodes.csv を保存
    import pandas as pd
    df_removed = pd.DataFrame(removed_records) if removed_records else pd.DataFrame(columns=["node","reason"])
    removed_csv = out_base + "_removed_nodes.csv"
    df_removed.to_csv(removed_csv, index=False)

    print("  ", removed_csv)

    # DataFrame化
    import numpy as np, pandas as pd, networkx as nx
    df_nodes = pd.DataFrame([
        {"node": n, "deg": S.degree(n),
         **S.nodes[n]}  # mass/flags他（必要なら整形）
        for n in S.nodes()
    ])
    # エッジ表
    rows = []
    for u, v, ed in S.edges(data=True):
        rows.append({"parent": u, "child": v, **ed})
    df_edges = pd.DataFrame(rows)

    draw_graph_png(S, out_base + "_processed.png", "Processed Graph (pruned + movable-only)")
    nodes_csv, edges_csv = export_csv(df_nodes, df_edges, out_base, "processed")

    with open(out_base + "_meta.json", "w") as f:
        json.dump({"input": args.urdf, "scale_mean_edge_len": float(scale)}, f, indent=2)

    print("Saved:")
    print("  ", out_base + "_urdf_full.png")
    print("  ", out_base + "_processed.png")
    print("  ", nodes_csv)
    print("  ", edges_csv)
    print("  ", out_base + "_meta.json")

if __name__ == "__main__":
    main()
