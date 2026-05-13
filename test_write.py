import sys, os, json
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, '.')

with open('diag_output.txt', 'w') as f:
    f.write("=== Diagnosis ===\n")
    
    # Load model
    try:
        model = joblib.load('models/model.pkl')
        f.write(f"Model type: {type(model)}\n")
        if hasattr(model, 'named_steps'):
            f.write(f"Steps: {list(model.named_steps.keys())}\n")
            clf = model.named_steps.get('clf')
            if clf:
                f.write(f"Classifier: {type(clf).__name__}\n")
    except Exception as e:
        f.write(f"Model load error: {e}\n")
    
    # Load features
    try:
        with open('models/feature_columns.json') as jf:
            feature_columns = json.load(jf)
        f.write(f"Features ({len(feature_columns)}): {feature_columns}\n")
    except Exception as e:
        f.write(f"Features load error: {e}\n")
        feature_columns = []
    
    # Build input
    try:
        stars, forks, issues, pull_requests, contributors = 120, 35, 18, 22, 8
        engagement_score = stars * 0.5 + forks * 0.3 + contributors * 0.2
        pr_to_issue_ratio = pull_requests / issues if issues > 0 else 0.0
        
        base_features = {
            "stars": stars, "forks": forks, "issues": issues,
            "pull_requests": pull_requests, "contributors": contributors,
            "engagement_score": round(engagement_score, 2),
            "pr_to_issue_ratio": round(pr_to_issue_ratio, 3),
        }
        
        row = {col: 0 for col in feature_columns}
        for k, v in base_features.items():
            if k in row:
                row[k] = v
        
        f.write(f"Input row keys: {len(row)}\n")
        X = pd.DataFrame([row])[feature_columns]
        f.write(f"X shape: {X.shape}, dtype: {X.dtypes.tolist()[:5]}\n")
        f.write(f"X values sample: {X.values[0][:7]}\n")
        
        # Predict
        y_pred = model.predict(X.values)
        y_proba = model.predict_proba(X.values)
        f.write(f"Prediction: {y_pred}, Proba: {y_proba}\n")
        f.write("SUCCESS: Model works correctly!\n")
    except Exception as e:
        import traceback
        f.write(f"Prediction error: {e}\n")
        f.write(traceback.format_exc())
    
    f.write("=== Done ===\n")

print("Written to diag_output.txt")