import sys, os, traceback, json
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, '.')

# Check files exist
print("=== File existence check ===")
for f in ['models/model.pkl', 'models/feature_columns.json', 'models/metrics.json']:
    print(f"  {f}: {'EXISTS' if os.path.exists(f) else 'MISSING'}")

print("\n=== Load model ===")
try:
    model = joblib.load('models/model.pkl')
    print(f"  Type: {type(model)}")
    if hasattr(model, 'named_steps'):
        print(f"  Steps: {list(model.named_steps.keys())}")
        scaler = model.named_steps.get('scaler')
        clf = model.named_steps.get('clf')
        if scaler:
            print(f"  Scaler: {type(scaler).__name__}, mean shape: {scaler.mean_.shape if hasattr(scaler, 'mean_') else 'N/A'}")
        if clf:
            print(f"  Classifier: {type(clf).__name__}")
            print(f"  n_features_in_: {getattr(clf, 'n_features_in_', 'N/A')}")
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

print("\n=== Load feature columns ===")
try:
    with open('models/feature_columns.json') as f:
        feature_columns = json.load(f)
    print(f"  {len(feature_columns)} columns: {feature_columns}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== Build input row ===")
try:
    stars, forks, issues, pull_requests, contributors = 120, 35, 18, 22, 8
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
    
    row = {col: 0 for col in feature_columns}
    for k, v in base_features.items():
        if k in row:
            row[k] = v
    
    print(f"  row has {len(row)} keys")
    print(f"  lang columns all zero: {all(row.get(f'lang_{l}', -1) == 0 for l in ['Python','JavaScript','Java'])}")
    
    X = pd.DataFrame([row])[feature_columns].values
    print(f"  X shape: {X.shape}")
    print(f"  X dtype: {X.dtype}")
    
    print("\n=== Predict ===")
    prediction = model.predict(X)
    proba = model.predict_proba(X)
    print(f"  prediction: {prediction}")
    print(f"  proba: {proba}")
    
except Exception as e:
    print(f"  ERROR: {e}")
    traceback.print_exc()

print("\n=== Done ===")