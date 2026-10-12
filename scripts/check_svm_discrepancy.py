import sys
sys.path.insert(0, "src")
import numpy as np
from pathlib import Path
from spectral_forensics.classification.interface import load_model, predict_features
from spectral_forensics.features.extract import extract_feature_vector

model = load_model("models/svm_v1")
img_path = Path("datasets/verified_parquet_images/test/REAL/real_000007.png")
features = extract_feature_vector(img_path).reshape(1, -1)

pred = model.estimator.predict(features)[0]
dec_func = model.estimator.decision_function(features)[0]
prob = model.estimator.predict_proba(features)[0]

print(f"Prediction: {pred}")
print(f"Decision Function: {dec_func}")
print(f"Predict Proba: {prob}")
