import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import FEATURES, TARGET, TRAIN_FILE, OUTPUT_DIR

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(TRAIN_FILE)

report = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "target_min": float(df[TARGET].min()),
    "target_max": float(df[TARGET].max()),
    "target_mean": float(df[TARGET].mean()),
    "target_median": float(df[TARGET].median()),
}

with open(OUTPUT_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

correlations = df[FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
correlations.sort_values(ascending=False).to_csv(
    OUTPUT_DIR / "target_correlations.csv", header=["correlation"]
)

plt.figure(figsize=(8, 5))
sns.histplot(df[TARGET], bins=30, kde=True)
plt.title("Productivity Score Distribution")
plt.xlabel("Productivity Score")
plt.ylabel("Students")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "productivity_distribution.png", dpi=150)
plt.close()

for feature in FEATURES:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=feature, y=TARGET, alpha=0.35)
    plt.title(f"{feature} vs Productivity Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{feature}_vs_productivity.png", dpi=150)
    plt.close()

plt.figure(figsize=(10, 7))
sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, fmt=".2f", cmap="Blues")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
plt.close()

print("EDA completed.")
print(json.dumps(report, indent=2))
