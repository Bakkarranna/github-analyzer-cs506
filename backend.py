"""
backend.py
CS506 – Big Data Analytics Project
GitHub Repository Intelligence Analyzer
Muhammad Abubakar Siddique | 2023-AG-10411

Step 5 (backend/tools): Helper functions and routes that
the Streamlit frontend calls. Handles model loading,
predictions, and analytics computations.
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = "models"
DATA_DIR = "data"

MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")
FEATURES_FILE = os.path.join(MODEL_DIR, "feature_columns.json")
DATA_FILE = os.path.join(DATA_DIR, "cleaned_data.csv")


# ─────────────────────────────────────────────
#  LOADERS
# ─────────────────────────────────────────────

def load_model():
    """Load the trained model from disk."""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_FILE}'. Run train_model.py first."
        )
    return joblib.load(MODEL_FILE)


def load_metrics():
    """Load saved evaluation metrics."""
    if not os.path.exists(METRICS_FILE):
        return {}
    with open(METRICS_FILE, "r") as f:
        return json.load(f)


def load_feature_columns():
    """Load the list of feature column names used during training."""
    if not os.path.exists(FEATURES_FILE):
        return []
    with open(FEATURES_FILE, "r") as f:
        return json.load(f)


def load_dataset():
    """Load the cleaned dataset for analytics."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Cleaned data not found at '{DATA_FILE}'. Run preprocess.py first."
        )
    return pd.read_csv(DATA_FILE)


# ─────────────────────────────────────────────
#  PREDICTION ROUTE
# ─────────────────────────────────────────────

def predict_repository_popularity(stars, forks, issues, pull_requests, contributors):
    """
    Route: Predict whether a repository will be popular.

    Parameters
    ----------
    stars : int
    forks : int
    issues : int
    pull_requests : int
    contributors : int

    Returns
    -------
    dict with keys: prediction, confidence, label, insight
    """
    model = load_model()
    feature_columns = load_feature_columns()

    # Build engineered features (must match preprocess.py logic)
    engagement_score = stars * 0.5 + forks * 0.3 + contributors * 0.2
    pr_to_issue_ratio = pull_requests / issues if issues > 0 else 0.0

    base_features = {
        "stars": stars,
        "forks": forks,
        "issues": issues,
        "pull_requests": pull_requests,
        "contributors": contributors,
        "engagement_score": round(engagement_score, 2),
        "pr_to_issue_ratio": round(pr_to_issue_ratio, 3),
    }

    # Build full feature row aligned to training columns
    row = {col: 0 for col in feature_columns}
    for k, v in base_features.items():
        if k in row:
            row[k] = v

    X = pd.DataFrame([row])[feature_columns].values

    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = round(float(proba[prediction]) * 100, 1)

    label = "🔥 Likely Popular" if prediction == 1 else "📉 Not Likely Popular"

    insight = _generate_insight(stars, forks, issues, pull_requests, contributors, prediction)

    return {
        "prediction": int(prediction),
        "confidence": confidence,
        "label": label,
        "insight": insight,
        "engagement_score": round(engagement_score, 2),
        "pr_to_issue_ratio": round(pr_to_issue_ratio, 3),
    }


def _generate_insight(stars, forks, issues, pull_requests, contributors, prediction):
    """Generate a human-readable insight string."""
    strengths = []
    weaknesses = []

    if stars > 500:
        strengths.append("high star count")
    elif stars < 10:
        weaknesses.append("very few stars")

    if forks > 100:
        strengths.append("strong fork activity")
    elif forks < 5:
        weaknesses.append("low fork count")

    if contributors > 10:
        strengths.append("active contributor base")
    elif contributors <= 1:
        weaknesses.append("solo project with no collaborators")

    if pull_requests > issues:
        strengths.append("more PRs than issues (healthy resolution)")
    elif issues > pull_requests * 3:
        weaknesses.append("many unresolved issues")

    parts = []
    if strengths:
        parts.append("Strengths: " + ", ".join(strengths))
    if weaknesses:
        parts.append("Areas to improve: " + ", ".join(weaknesses))

    return " | ".join(parts) if parts else "Repository has average engagement metrics."


# ─────────────────────────────────────────────
#  ANALYTICS ROUTES
# ─────────────────────────────────────────────

def get_top_languages(df, top_n=10):
    """Route: Top programming languages by repository count."""
    if "language" not in df.columns:
        return pd.DataFrame()
    counts = (
        df[df["language"] != "Unknown"]["language"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    counts.columns = ["language", "count"]
    return counts


def get_stars_distribution(df):
    """Route: Stars distribution stats for histogram."""
    if "stars" not in df.columns:
        return {}
    data = df[df["stars"] < df["stars"].quantile(0.99)]["stars"]
    return {
        "values": data.tolist(),
        "mean": round(data.mean(), 1),
        "median": round(data.median(), 1),
        "max": int(data.max()),
    }


def get_engagement_by_language(df, top_n=10):
    """Route: Average engagement score per language."""
    if not all(c in df.columns for c in ["language", "engagement_score"]):
        return pd.DataFrame()
    result = (
        df[df["language"] != "Unknown"]
        .groupby("language")["engagement_score"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    result.columns = ["language", "avg_engagement"]
    result["avg_engagement"] = result["avg_engagement"].round(2)
    return result


def get_contributor_tier_distribution(df):
    """Route: Count of repos per contributor category."""
    if "contributor_category" not in df.columns:
        return pd.DataFrame()
    counts = df["contributor_category"].value_counts().reset_index()
    counts.columns = ["tier", "count"]
    order = ["solo", "small_team", "mid_team", "large_team", "open_source"]
    counts["tier"] = pd.Categorical(counts["tier"], categories=order, ordered=True)
    return counts.sort_values("tier").reset_index(drop=True)


def get_popularity_by_language(df, top_n=10):
    """Route: Popularity rate per language."""
    if not all(c in df.columns for c in ["language", "is_popular"]):
        return pd.DataFrame()
    result = (
        df[df["language"] != "Unknown"]
        .groupby("language")
        .agg(total=("is_popular", "count"), popular=("is_popular", "sum"))
        .reset_index()
    )
    result = result[result["total"] >= 5]
    result["popularity_rate"] = (result["popular"] / result["total"] * 100).round(1)
    return result.sort_values("popularity_rate", ascending=False).head(top_n)


def get_summary_kpis(df):
    """Route: High-level KPI summary for dashboard header."""
    kpis = {
        "total_repos": len(df),
        "total_stars": int(df["stars"].sum()) if "stars" in df.columns else 0,
        "avg_engagement": round(df["engagement_score"].mean(), 1) if "engagement_score" in df.columns else 0,
        "top_language": df["language"].mode()[0] if "language" in df.columns else "N/A",
        "popular_repos_pct": round(df["is_popular"].mean() * 100, 1) if "is_popular" in df.columns else 0,
        "avg_contributors": round(df["contributors"].mean(), 1) if "contributors" in df.columns else 0,
    }
    return kpis
