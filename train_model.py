"""
train_model.py
CS506 – Big Data Analytics Project
GitHub Repository Intelligence Analyzer
Muhammad Abubakar Siddique | 2023-AG-10411

Step 3: Train ML model on preprocessed data.
Run this AFTER preprocess.py has completed.
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.pipeline import Pipeline

DATA_DIR = "data"
MODEL_DIR = "models"
INPUT_FILE = os.path.join(DATA_DIR, "cleaned_data.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")
FEATURES_FILE = os.path.join(MODEL_DIR, "feature_columns.json")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    print("[1/7] Loading cleaned data...")
    df = pd.read_csv(INPUT_FILE)
    print(f"      Loaded: {len(df):,} rows x {df.shape[1]} columns")
    return df


def prepare_features(df):
    print("[2/7] Preparing feature matrix...")

    # Drop non-feature columns
    drop_cols = [
        "repo_name", "language", "license",
        "popularity_tier", "contributor_category", "is_popular"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    feature_df = df.drop(columns=drop_cols)

    # Drop columns with object dtype that slipped through
    feature_df = feature_df.select_dtypes(include=[np.number])

    # Drop the target if it accidentally remained
    if "is_popular" in feature_df.columns:
        feature_df.drop(columns=["is_popular"], inplace=True)

    feature_columns = feature_df.columns.tolist()
    print(f"      Features selected ({len(feature_columns)}): {feature_columns[:8]}{'...' if len(feature_columns)>8 else ''}")

    return feature_df, feature_columns


def split_data(feature_df, df):
    print("[3/7] Splitting train/test (80/20)...")
    X = feature_df.values
    y = df["is_popular"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"      Class balance (train): {np.bincount(y_train)}")
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    print("[4/7] Training Random Forest classifier...")
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(f"      CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    return rf_pipeline, cv_scores


def train_gradient_boosting(X_train, y_train):
    print("[5/7] Training Gradient Boosting classifier...")
    gb_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ))
    ])
    gb_pipeline.fit(X_train, y_train)
    cv_scores = cross_val_score(gb_pipeline, X_train, y_train, cv=5, scoring="accuracy")
    print(f"      CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    return gb_pipeline, cv_scores


def evaluate_and_select(rf_model, gb_model, rf_cv, gb_cv, X_test, y_test):
    print("[6/7] Evaluating both models on test set...")

    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_pred)
    gb_acc = accuracy_score(y_test, gb_pred)

    rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    gb_auc = roc_auc_score(y_test, gb_model.predict_proba(X_test)[:, 1])

    print(f"\n      {'='*40}")
    print(f"      {'Model':<25} {'Accuracy':<10} {'AUC-ROC':<10}")
    print(f"      {'-'*40}")
    print(f"      {'Random Forest':<25} {rf_acc:<10.4f} {rf_auc:<10.4f}")
    print(f"      {'Gradient Boosting':<25} {gb_acc:<10.4f} {gb_auc:<10.4f}")
    print(f"      {'='*40}")

    # Select best by AUC
    if rf_auc >= gb_auc:
        best_model = rf_model
        best_name = "Random Forest"
        best_acc = rf_acc
        best_auc = rf_auc
        best_cv = rf_cv.mean()
        y_pred = rf_pred
    else:
        best_model = gb_model
        best_name = "Gradient Boosting"
        best_acc = gb_acc
        best_auc = gb_auc
        best_cv = gb_cv.mean()
        y_pred = gb_pred

    print(f"\n      >>  Best model: {best_name} (AUC = {best_auc:.4f})")
    print(f"\n      Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Not Popular", "Popular"]))

    metrics = {
        "best_model": best_name,
        "test_accuracy": round(best_acc, 4),
        "roc_auc": round(best_auc, 4),
        "cv_accuracy": round(best_cv, 4),
        "rf_accuracy": round(rf_acc, 4),
        "gb_accuracy": round(gb_acc, 4),
        "rf_auc": round(rf_auc, 4),
        "gb_auc": round(gb_auc, 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return best_model, metrics


def save_artifacts(model, metrics, feature_columns):
    print("[7/7] Saving model and artifacts...")
    joblib.dump(model, MODEL_FILE)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(FEATURES_FILE, "w") as f:
        json.dump(feature_columns, f, indent=2)

    print(f"      Model saved:    {MODEL_FILE}")
    print(f"      Metrics saved:  {METRICS_FILE}")
    print(f"      Features saved: {FEATURES_FILE}")
    print(f"\n**  Training complete!")


def main():
    print("=" * 55)
    print("  GitHub Repository Intelligence Analyzer")
    print("  Model Training Pipeline")
    print("=" * 55)

    if not os.path.exists(INPUT_FILE):
        print(f"\n!!  ERROR: '{INPUT_FILE}' not found.")
        print("     Run preprocess.py first.")
        return

    df = load_data()
    feature_df, feature_columns = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(feature_df, df)
    rf_model, rf_cv = train_random_forest(X_train, y_train)
    gb_model, gb_cv = train_gradient_boosting(X_train, y_train)
    best_model, metrics = evaluate_and_select(rf_model, gb_model, rf_cv, gb_cv, X_test, y_test)
    save_artifacts(best_model, metrics, feature_columns)


if __name__ == "__main__":
    main()