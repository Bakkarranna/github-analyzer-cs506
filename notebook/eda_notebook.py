# eda_notebook.py
# CS506 – Big Data Analytics | GitHub Repository Intelligence Analyzer
# Muhammad Abubakar Siddique | 2023-AG-10411
#
# Run this in Jupyter Notebook or Google Colab (Step 3 of project).
# Copy each cell block into a separate notebook cell.
#
# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install dependencies
# ─────────────────────────────────────────────────────────────────────────────
# %pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost joblib

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Imports
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["figure.figsize"] = (12, 5)

print("✅ Imports OK")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Load cleaned data
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/cleaned_data.csv")
print(f"Shape: {df.shape}")
display(df.head())
display(df.describe())

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Missing value heatmap
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))
sns.heatmap(df.isnull(), cbar=False, cmap="Reds", yticklabels=False)
plt.title("Missing Value Heatmap (post-cleaning)")
plt.tight_layout()
plt.savefig("notebook/missing_values.png")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Stars distribution
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram (capped at 99th percentile)
cap = df["stars"].quantile(0.99)
axes[0].hist(df[df["stars"] < cap]["stars"], bins=60, color="#388bfd", edgecolor="none", alpha=0.85)
axes[0].axvline(df["stars"].median(), color="#f0883e", linestyle="--", label=f"Median: {df['stars'].median():.0f}")
axes[0].set_title("Stars Distribution (capped at 99th pct)")
axes[0].set_xlabel("Stars")
axes[0].legend()

# Box plot per popularity tier
if "popularity_tier" in df.columns:
    order = ["obscure", "low", "moderate", "high", "viral"]
    order = [o for o in order if o in df["popularity_tier"].unique()]
    sns.boxplot(data=df[df["stars"] < cap], x="popularity_tier", y="stars",
                order=order, ax=axes[1], palette="Blues")
    axes[1].set_title("Stars by Popularity Tier")
    axes[1].set_xlabel("Tier")

plt.tight_layout()
plt.savefig("notebook/stars_distribution.png")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df[["stars", "forks", "issues", "pull_requests",
                    "contributors", "engagement_score",
                    "pr_to_issue_ratio", "is_popular"]].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("notebook/correlation_heatmap.png")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Top languages
# ─────────────────────────────────────────────────────────────────────────────
if "language" in df.columns:
    top_langs = df[df["language"] != "Unknown"]["language"].value_counts().head(15)

    plt.figure(figsize=(12, 5))
    sns.barplot(x=top_langs.values, y=top_langs.index, palette="Blues_d")
    plt.title("Top 15 Programming Languages by Repository Count")
    plt.xlabel("Number of Repositories")
    plt.tight_layout()
    plt.savefig("notebook/top_languages.png")
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Engagement score distribution by popularity
# ─────────────────────────────────────────────────────────────────────────────
if all(c in df.columns for c in ["engagement_score", "is_popular"]):
    cap_eng = df["engagement_score"].quantile(0.98)
    plot_df = df[df["engagement_score"] < cap_eng]

    plt.figure(figsize=(10, 5))
    sns.histplot(data=plot_df, x="engagement_score", hue="is_popular",
                 bins=50, kde=True, palette={0: "#6e7681", 1: "#56d364"})
    plt.title("Engagement Score: Popular vs Not Popular")
    plt.xlabel("Engagement Score")
    plt.legend(title="Popular", labels=["No", "Yes"])
    plt.tight_layout()
    plt.savefig("notebook/engagement_by_popularity.png")
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Forks vs Stars scatter (log scale)
# ─────────────────────────────────────────────────────────────────────────────
if all(c in df.columns for c in ["stars", "forks", "is_popular"]):
    sample = df.sample(min(5000, len(df)), random_state=42)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        np.log1p(sample["stars"]),
        np.log1p(sample["forks"]),
        c=sample["is_popular"],
        cmap="coolwarm",
        alpha=0.5,
        s=12,
    )
    plt.colorbar(scatter, label="Popular (1=Yes)")
    plt.xlabel("log(Stars + 1)")
    plt.ylabel("log(Forks + 1)")
    plt.title("Stars vs Forks (log scale) — colored by popularity")
    plt.tight_layout()
    plt.savefig("notebook/stars_vs_forks.png")
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — Summary findings
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  EDA Summary Findings")
print("=" * 55)
print(f"  Total repositories analyzed : {len(df):,}")
print(f"  Popular repos (>median stars): {df['is_popular'].sum():,} ({df['is_popular'].mean()*100:.1f}%)")
if "language" in df.columns:
    print(f"  Most common language        : {df['language'].mode()[0]}")
print(f"  Median stars                : {df['stars'].median():,.0f}")
print(f"  Avg engagement score        : {df['engagement_score'].mean():,.1f}")
print(f"  Avg contributors            : {df['contributors'].mean():.1f}")
print("\n  📁 All plots saved to: notebook/")
print("=" * 55)
