import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 保存された予測結果CSV（正解データも含まれているはず）を読み込み
# ※ パスは実際の保存先に合わせて変更してください
csv_path = "./checkpoints_augmented/vlisualize_choreonoid_feature_aug.csv"
df = pd.read_csv(csv_path)

# 2. 確認したい項目を選択
print("=== 確認したい項目を選択してください ===")
print("1: Origin (origin_x, origin_y, origin_z)")
print("2: Mass (mass)")
choice = input("選択 (1 or 2): ").strip()

if choice == "1":
    targets = ['origin_x', 'origin_y', 'origin_z']
    xlabel = "Value (meters)"
    print_target = "Origin X"
elif choice == "2":
    targets = ['mass']
    xlabel = "Value (kg)"
    print_target = "Mass"
else:
    print("Invalid choice. Defaulting to Origin.")
    targets = ['origin_x', 'origin_y', 'origin_z']
    xlabel = "Value (meters)"
    print_target = "Origin X"

# 3. 分布を描画
plt.figure(figsize=(18, 5))

for i, col in enumerate(targets):
    plt.subplot(1, len(targets), i+1)
    # 外れ値の影響で見にくくなるのを防ぐため、対数グラフも検討できますが
    # まずはそのまま、または範囲を絞って表示します
    sns.histplot(df[col], bins=100, kde=False)
    plt.title(f"Distribution of {col}")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    
    # 統計量を表示
    mean_val = df[col].mean()
    max_val = df[col].max()
    plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.legend()

plt.tight_layout()
plt.show()

# 4. 統計情報を表示
print(f"\n=== {print_target} Stats ===")
if choice == "1":
    print(df['origin_x'].describe())
    print(f"95% percentile: {df['origin_x'].quantile(0.95):.4f}")
    print(f"99% percentile: {df['origin_x'].quantile(0.99):.4f}")
elif choice == "2":
    print(df['mass'].describe())
    print(f"95% percentile: {df['mass'].quantile(0.95):.4f}")
    print(f"99% percentile: {df['mass'].quantile(0.99):.4f}")