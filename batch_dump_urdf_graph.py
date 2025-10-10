# batch_dump_urdf_graph.py
# train_list に載っている全 URDF を dump_urdf_graph.py に通して
# output/<URDFベース名>_hash/ に出力するラッパー。

import os
import sys
import subprocess
import hashlib

# ========== 設定 ==========
OUTPUT_ROOT = "output"          # でかい出力フォルダ
PYTHON_EXE  = sys.executable    # 使うPython
DUMP_SCRIPT = "dump_urdf_graph.py"
DUMP_FLAGS  = [
    # 例: "--keep-eef", "--include-fixed", "--no-normalize"
    # 何も付けない = 末端剪定あり / fixed除外 / 正規化あり（あなたのデフォルト）
]
# ==========================

def safe_name(path: str) -> str:
    """URDFのベース名 + 短いハッシュでサブフォルダ名を作る"""
    base = os.path.splitext(os.path.basename(path))[0]
    h = hashlib.md5(path.encode("utf-8")).hexdigest()[:6]
    return f"{base}_{h}"

def main():
    # train_list を import（(path, y) のリスト）
    try:
        from train_arm_linkness import train_list
    except Exception as e:
        print("ERROR: train_list の import に失敗しました。batch_dump_urdf_graph.py と同じ環境で実行してください。")
        print(e)
        sys.exit(1)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    ok, ng = 0, 0

    for i, (urdf_path, _y) in enumerate(train_list, 1):
        subdir = os.path.join(OUTPUT_ROOT, safe_name(urdf_path))
        os.makedirs(subdir, exist_ok=True)

        cmd = [PYTHON_EXE, DUMP_SCRIPT, urdf_path, "--outdir", subdir] + DUMP_FLAGS
        print(f"[{i}/{len(train_list)}] run:", " ".join(cmd))

        try:
            cp = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if cp.returncode == 0:
                ok += 1
                # 進捗の要約だけ表示（詳細は各サブフォルダ内のログ/生成物を確認）
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

    print(f"\nDONE  success={ok}  failed={ng}  out_root={os.path.abspath(OUTPUT_ROOT)}")

if __name__ == "__main__":
    main()
