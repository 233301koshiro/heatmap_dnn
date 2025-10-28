# create_csv_and_png.py
import argparse
import sys
from pathlib import Path
import pandas as pd

# 共通ユーティリティから必要なものだけ
from urdf_to_graph_utilis import (
    process_dir_flat,     # pre/post の一括処理（画像, nodes_processed.csv, meta.json, summary CSV まで）
    make_compare_csv,     # pre/post の summary から比較 CSV 生成
)

def _normalize_exts(exts):
    # 例: ["urdf", ".XML"] -> (".urdf", ".xml")
    return tuple(e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts)

def _require_dir(path_str: str, tag: str):
    p = Path(path_str)
    if not p.exists() or not p.is_dir():
        print(f"[ERROR] {tag} not found or not a directory: {p}", file=sys.stderr)
        sys.exit(2)
    return p

def main():
    ap = argparse.ArgumentParser(
        description="pre/post を“フラット名”で出力（重複は連番回避）＋ 比較CSV生成"
    )
    ap.add_argument("--pre-dir", required=True, help="merge前URDF群（例: ./fixed_joint_robots）")
    ap.add_argument("--post-dir", required=True, help="merge後URDF群（例: ./merge_joint_robots）")
    ap.add_argument("--outdir", default="out_compare", help="出力ルート（pre/ と post/ を作成）")
    ap.add_argument("--no-normalize", action="store_true", help="origin距離平均でのスケール正規化を無効化")
    ap.add_argument("--exts", nargs="*", default=[".urdf", ".xml"], help="対象拡張子（例: --exts .urdf .xml .xacro）")
    ap.add_argument("--compare-out", default=None, help="比較CSVの出力先（省略時: outdir/compare_nodes_edges.csv）")
    args = ap.parse_args()

    # 入力/出力の前処理
    pre_dir  = _require_dir(args.pre_dir,  "pre-dir")
    post_dir = _require_dir(args.post_dir, "post-dir")
    out_root = Path(args.outdir); out_root.mkdir(parents=True, exist_ok=True)

    exts = _normalize_exts(args.exts)
    normalize_by = "none" if args.no_normalize else "mean_edge_len"

    print("=== CONFIG ===")
    print(f" pre-dir     : {pre_dir}")
    print(f" post-dir    : {post_dir}")
    print(f" outdir      : {out_root}")
    print(f" normalize_by: {normalize_by}")
    print(f" exts        : {exts}")
    print("==============")

    # 出力先（pre/post）
    pre_out  = out_root / "pre"
    post_out = out_root / "post"

    # 実行
    pre_list = process_dir_flat(str(pre_dir),  str(pre_out),  normalize_by=normalize_by, exts=exts)
    post_list= process_dir_flat(str(post_dir), str(post_out), normalize_by=normalize_by, exts=exts)

    # 走査インデックスの保存
    idx_rows = [{"set": "pre", "path": p} for p in pre_list] + \
               [{"set": "post", "path": p} for p in post_list]
    if idx_rows:
        df_idx = pd.DataFrame(idx_rows)
        df_idx.sort_values(["set", "path"], inplace=True, kind="mergesort")
        (out_root / "index_paths.csv").parent.mkdir(parents=True, exist_ok=True)
        df_idx.to_csv(out_root / "index_paths.csv", index=False)

    # 比較CSV
    pre_csv_path  = pre_out  / "graph_sizes_summary.csv"
    post_csv_path = post_out / "graph_sizes_summary.csv"
    compare_out   = Path(args.compare_out) if args.compare_out else (out_root / "compare_nodes_edges.csv")
    try:
        make_compare_csv(str(pre_csv_path), str(post_csv_path), str(compare_out))
    except Exception as e:
        print(f"[WARN] failed to make compare CSV: {e}", file=sys.stderr)

    print("\n[done] outputs under:", out_root)
    print("  pre/: ロボット名でフラット配置（例: pre/panda/, pre/anymal/ ...）")
    print("  post/: 同上（例: post/panda/, post/anymal/ ... ※重複は *_2）")
    print("  各robotフォルダに urdf_full.png, processed.png, nodes_processed.csv, meta.json")
    print("  pre/post に graph_sizes_summary.csv（name昇順）")
    print("  ルートに index_paths.csv（set→path昇順）と compare_nodes_edges.csv（robot_name昇順）")

if __name__ == "__main__":
    main()
