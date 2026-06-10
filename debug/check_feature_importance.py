"""
check_feature_importance.py - Debug script to view feature importance for models
Use this to understand which features are most important for predictions
"""
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "model" / "models"

def check_feature_importance(prop, model_type="short"):
    """Check feature importance for a specific model"""
    filename = f"batter_prop_{prop}_{model_type}_feature_importance.parquet"
    filepath = MODEL_DIR / filename
    
    if not filepath.exists():
        print(f"❌ Feature importance file not found: {filename}")
        return
    
    df = pd.read_parquet(filepath)
    
    print(f"=== {prop.upper()} {model_type.upper()} FEATURE IMPORTANCE ===")
    print(f"Total features: {len(df)}")
    print()
    print("Top 20 Features:")
    print(df.head(20).to_string(index=False))
    print()

def check_all_importances():
    """Check feature importance for all models"""
    props = ["hit", "hr", "multi", "k"]
    
    for prop in props:
        # Check short-term (most commonly used)
        check_feature_importance(prop, "short")
        print()

if __name__ == "__main__":
    check_all_importances()
