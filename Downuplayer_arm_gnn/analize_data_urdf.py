import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. URDF を置いてあるディレクトリ
#   ここを自分の環境に合わせて変えてください
urdf_root = Path("./merge_joint_robots")  # 例: ./merge_joint_robots/urdf など

rows = []

# 2. 再帰的に *.urdf を全部読む
for urdf_path in urdf_root.rglob("*.urdf"):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # 各 link の <inertial> から mass, origin を取得
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue

        mass_elem = inertial.find("mass")
        origin_elem = inertial.find("origin")

        # mass
        mass = None
        if mass_elem is not None and mass_elem.get("value") is not None:
            mass = float(mass_elem.get("value"))

        # origin (xyz 属性が無い場合は 0,0,0 にしておく)
        origin_x = origin_y = origin_z = 0.0
        if origin_elem is not None:
            xyz = origin_elem.get("xyz", "0 0 0").split()
            if len(xyz) == 3:
                origin_x, origin_y, origin_z = map(float, xyz)

        rows.append(
            {
                "urdf": str(urdf_path),
                "link": link.get("name", ""),
                "mass": mass,
                "origin_x": origin_x,
                "origin_y": origin_y,
                "origin_z": origin_z,
            }
        )

# 3. DataFrame 化（今まで CSV から読んでいた df と互換な形に）
df = pd.DataFrame(rows)

print("読み込んだ行数:", len(df))
print(df.head())

# 4. ここから下はほぼ元のスクリプトそのまま ---------------------------------

print("=== 確認したい項目を選択してください ===")
print("1: Origin (origin_x, origin_y, origin_z)")
print("2: Mass (mass)")
choice = input("選択 (1 or 2): ").strip()

if choice == "1":
    targets = ["origin_x", "origin_y", "origin_z"]
    xlabel = "Value (meters)"
    print_target = "Origin X"
elif choice == "2":
    targets = ["mass"]
    xlabel = "Value (kg)"
    print_target = "Mass"
else:
    print("Invalid choice. Defaulting to Origin.")
    targets = ["origin_x", "origin_y", "origin_z"]
    xlabel = "Value (meters)"
    print_target = "Origin X"

plt.figure(figsize=(18, 5))

for i, col in enumerate(targets):
    plt.subplot(1, len(targets), i + 1)
    sns.histplot(df[col].dropna(), bins=100, kde=False)
    plt.title(f"Distribution of {col}")
    plt.xlabel(xlabel)
    plt.ylabel("Count")

    mean_val = df[col].mean()
    plt.axvline(mean_val, color="r", linestyle="--", label=f"Mean: {mean_val:.2f}")
    plt.legend()

plt.tight_layout()
plt.show()

print(f"\n=== {print_target} Stats ===")
if choice == "1":
    print(df["origin_x"].describe())
    print(f"95% percentile: {df['origin_x'].quantile(0.95):.4f}")
    print(f"99% percentile: {df['origin_x'].quantile(0.99):.4f}")
elif choice == "2":
    print(df["mass"].describe())
    print(f"95% percentile: {df['mass'].quantile(0.95):.4f}")
    print(f"99% percentile: {df['mass'].quantile(0.99):.4f}")
