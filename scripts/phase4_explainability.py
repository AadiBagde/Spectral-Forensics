"""Stage 6: Explainability and Frequency-Band Analysis."""

import json
from pathlib import Path
import sys
import os

# Support running from the scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from spectral_forensics.paths import get_project_paths

def main():
    paths = get_project_paths()
    root = paths.root
    out_dir = root / "experiments" / "phase4" / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_df = pd.read_csv(paths.processed / "verified_train_features.csv")
    feature_cols = [f"freq_{i}" for i in range(128)]
    
    # 1. Compare radial-spectrum feature distributions between REAL and FAKE
    real_means = train_df[train_df['label'] == 0][feature_cols].mean().values
    fake_means = train_df[train_df['label'] == 1][feature_cols].mean().values
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(real_means, label='REAL (mean)', color='green')
    ax.plot(fake_means, label='FAKE (mean)', color='red')
    ax.set_title("Average Radial Frequency Profile (Training Data)")
    ax.set_xlabel("Frequency Bin (Low -> High)")
    ax.set_ylabel("Mean Magnitude/Power")
    ax.legend()
    fig.savefig(out_dir / "mean_radial_profile_comparison.png")
    plt.close(fig)
    
    # 2. Feature Importance (using Random Forest if available or a new one)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_df[feature_cols].values, train_df['label'].values)
    importances = rf.feature_importances_
    
    top_indices = np.argsort(importances)[::-1][:10]
    top_features = [feature_cols[i] for i in top_indices]
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(top_features, importances[top_indices], color='purple')
    ax2.set_title("Top 10 Important Frequency Bins (Random Forest)")
    fig2.savefig(out_dir / "feature_importance.png")
    plt.close(fig2)
    
    report = {
        "top_10_features": top_features,
        "limitation": "Feature importance from Random Forest does not imply causality. It indicates which bins provide the most discriminative information for this specific dataset."
    }
    
    with open(out_dir / "explainability.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("Explainability analysis complete.")

if __name__ == "__main__":
    main()
