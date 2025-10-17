# batch_dump_urdf_graph.py
# train_list に載っている全 URDF を dump_urdf_graph.py に通して
# output/<URDFベース名>_hash/ に出力するラッパー（nodes-only CSV 仕様対応）

import os
import sys
import subprocess
import hashlib
import argparse
from datetime import datetime

# ========== 既定設定 ==========
OUTPUT_ROOT = "output"          # でかい出力フォルダ
PYTHON_EXE  = sys.executable    # 使うPython
DUMP_SCRIPT = "create_csv_and_png.py"
DUMP_FLAGS  = [
    # 例: "--no-normalize"
    # 空のまま = 正規化あり（mean_edge_len）
]
# ==============================

def safe_name(path: str) -> str:
    """URDFのベース名 + 短いハッシュでサブフォルダ名を作る"""
    base = os.path.splitext(os.path.basename(path))[0]
    h = hashlib.md5(path.encode("utf-8")).hexdigest()[:6]
    return f"{base}_{h}"

def main():
    ap = argparse.ArgumentParser(description="URDF一括ダンプ（dump_urdf_graph.py ラッパー）")
    ap.add_argument("--outdir", default=OUTPUT_ROOT, help="出力ルートディレクトリ（既定: output）")
    ap.add_argument("--force", action="store_true", help="既存結果があっても再実行する")
    ap.add_argument("--no-normalize", action="store_true", help="dump_urdf_graph.py に --no-normalize を付ける")
    args = ap.parse_args()

    # train_list を import（(path, y) のリスト）
    try:
        from train import train_list
    except Exception as e:
        print("ERROR: train_list の import に失敗しました。batch_dump_urdf_graph.py と同じ環境で実行してください。")
        print(e)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    ok, ng, skipped = 0, 0, 0

    for i, (urdf_path, _y) in enumerate(train_list, 1):
        subdir = os.path.join(args.outdir, safe_name(urdf_path))
        os.makedirs(subdir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(urdf_path))[0]
        nodes_csv = os.path.join(subdir, f"{stem}_nodes_processed.csv")
        log_path  = os.path.join(subdir, "batch.log")

        # スキップ判定（既存ノードCSVがあればスキップ）
        if (not args.force) and os.path.exists(nodes_csv):
            skipped += 1
            print(f"[{i}/{len(train_list)}] SKIP (exists): {nodes_csv}")
            continue

        # dumpコマンド組み立て
        cmd = [PYTHON_EXE, DUMP_SCRIPT, urdf_path, "--outdir", subdir]
        # --no-normalize を付けたいとき
        if args.no_normalize:
            cmd.append("--no-normalize")
        # 追加の固定フラグ（必要なら上の DUMP_FLAGS に入れる）
        cmd += DUMP_FLAGS

        print(f"[{i}/{len(train_list)}] run: {' '.join(cmd)}")

        try:
            cp = subprocess.run(cmd, check=False, capture_output=True, text=True)
            # ログ保存
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"=== {datetime.now().isoformat()} ===\n")
                f.write("CMD: " + " ".join(cmd) + "\n")
                f.write("--- STDOUT ---\n")
                f.write(cp.stdout or "")
                f.write("\n--- STDERR ---\n")
                f.write(cp.stderr or "")
                f.write("\n")

            if cp.returncode == 0:
                ok += 1
                print(f"  -> OK  out: {subdir}")
            else:
                ng += 1
                print(f"  -> FAIL (code={cp.returncode})  path: {urdf_path}")
                if cp.stdout:
                    print("  ---- STDOUT ----\n", cp.stdout.strip())
                if cp.stderr:
                    print("  ---- STDERR ----\n", cp.stderr.strip())

        except Exception as e:
            ng += 1
            print(f"  -> EXCEPTION  path: {urdf_path}\n     {e}")

    print(f"\nDONE  success={ok}  failed={ng}  skipped={skipped}  out_root={os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()
