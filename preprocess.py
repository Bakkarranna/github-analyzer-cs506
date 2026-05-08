"""
preprocess.py
CS506 – Big Data Analytics Project
GitHub Repository Intelligence Analyzer
Muhammad Abubakar Siddique | 2023-AG-10411

Step 2: Clean and preprocess the downloaded Kaggle dataset.
Run this AFTER downloading the dataset from Kaggle.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "github_dataset.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "cleaned_data.csv")


def load_data(filepath):
    print(f"[1/6] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"      Rows: {len(df):,}  |  Columns: {df.columns.tolist()}")
    return df


def rename_and_select_columns(df):
    """
    The Kaggle 'github_dataset.csv' has these columns:
    repositories, stars_count, forks_count, issues_count,
    pull_requests, contributors, language, licence
    We rename for clarity and add engineered features.
    """
    print("[2/6] Renaming and selecting relevant columns...")

    rename_map = {
        "repositories": "repo_name",
        "stars_count": "stars",
        "forks_count": "forks",
        "issues_count": "issues",
        "pull_requests": "pull_requests",
        "contributors": "contributors",
        "language": "language",
        "licence": "license",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def clean_data(df):
    print("[3/6] Cleaning data (nulls, duplicates, types)...")
    before = len(df)

    # Drop full duplicates
    df = df.drop_duplicates()

    # Drop rows where core numeric columns are missing
    numeric_cols = ["stars", "forks", "issues", "pull_requests", "contributors"]
    existing_numeric = [c for c in numeric_cols if c in df.columns]
    df = df.dropna(subset=existing_numeric)

    # Fill missing language with 'Unknown'
    if "language" in df.columns:
        df["language"] = df["language"].fillna("Unknown").str.strip()

    if "license" in df.columns:
        df["license"] = df["license"].fillna("No License").str.strip()

    # Convert numeric columns to int (remove any floats from NaN cleaning)
    for col in existing_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Remove negative values (data errors)
    for col in existing_numeric:
        df = df[df[col] >= 0]

    after = len(df)
    print(f"      Rows removed: {before - after:,}  |  Remaining: {after:,}")
    return df


def engineer_features(df):
    print("[4/6] Engineering features...")

    # Engagement score: weighted sum of social signals
    if all(c in df.columns for c in ["stars", "forks", "contributors"]):
        df["engagement_score"] = (
            df["stars"] * 0.5
            + df["forks"] * 0.3
            + df["contributors"] * 0.2
        ).round(2)

    # Issue resolution ratio (proxy for project health)
    if all(c in df.columns for c in ["pull_requests", "issues"]):
        df["pr_to_issue_ratio"] = np.where(
            df["issues"] > 0,
            (df["pull_requests"] / df["issues"]).round(3),
            0.0,
        )

    # Activity tier based on stars
    if "stars" in df.columns:
        df["popularity_tier"] = pd.cut(
            df["stars"],
            bins=[-1, 10, 100, 1000, 10000, np.inf],
            labels=["obscure", "low", "moderate", "high", "viral"],
        ).astype(str)

    # Contributor category
    if "contributors" in df.columns:
        df["contributor_category"] = pd.cut(
            df["contributors"],
            bins=[-1, 1, 5, 20, 100, np.inf],
            labels=["solo", "small_team", "mid_team", "large_team", "open_source"],
        ).astype(str)

    # Target variable: is_popular (binary — for classification model)
    if "stars" in df.columns:
        median_stars = df["stars"].median()
        df["is_popular"] = (df["stars"] > median_stars).astype(int)
        print(f"      Target median stars threshold: {median_stars:,.0f}")

    print(f"      New features added: engagement_score, pr_to_issue_ratio,")
    print(f"      popularity_tier, contributor_category, is_popular")
    return df


def encode_categoricals(df):
    print("[5/6] Encoding categorical variables...")

    if "language" in df.columns:
        top_languages = df["language"].value_counts().head(15).index.tolist()
        df["language_encoded"] = df["language"].apply(
            lambda x: x if x in top_languages else "Other"
        )
        lang_dummies = pd.get_dummies(df["language_encoded"], prefix="lang").astype(int)
        df = pd.concat([df, lang_dummies], axis=1)
        df.drop(columns=["language_encoded"], inplace=True)

    return df


def save_data(df, filepath):
    print(f"[6/6] Saving cleaned data to: {filepath}")
    df.to_csv(filepath, index=False)
    print(f"      Final dataset: {len(df):,} rows × {len(df.columns)} columns")
    print(f"\n✅  Preprocessing complete! File saved: {filepath}")


def main():
    print("=" * 55)
    print("  GitHub Repository Intelligence Analyzer")
    print("  Data Preprocessing Pipeline")
    print("=" * 55)

    if not os.path.exists(INPUT_FILE):
        print(f"\n❌  ERROR: Dataset not found at '{INPUT_FILE}'")
        print("     Please download from Kaggle and place in data/ folder.")
        print("     Expected file: data/github_dataset.csv")
        return

    df = load_data(INPUT_FILE)
    df = rename_and_select_columns(df)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    save_data(df, OUTPUT_FILE)

    print("\n📊  Quick Stats:")
    print(df[["stars", "forks", "issues", "contributors", "engagement_score"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
