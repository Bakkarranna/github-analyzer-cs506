import sys
sys.path.insert(0, '.')
from backend import predict_repository_popularity, load_model, load_feature_columns

# Test 1: Check model loading
print("=== Test 1: Model Loading ===")
try:
    model = load_model()
    print(f"Model loaded successfully: {type(model)}")
except Exception as e:
    print(f"ERROR loading model: {e}")

# Test 2: Check feature columns
print("\n=== Test 2: Feature Columns ===")
try:
    cols = load_feature_columns()
    print(f"Feature columns ({len(cols)}): {cols}")
except Exception as e:
    print(f"ERROR loading features: {e}")

# Test 3: Check prediction
print("\n=== Test 3: Prediction ===")
try:
    result = predict_repository_popularity(120, 35, 18, 22, 8)
    print(f"SUCCESS: {result}")
except Exception as e:
    import traceback
    print(f"ERROR predicting: {e}")
    traceback.print_exc()

print("\n=== Done ===")