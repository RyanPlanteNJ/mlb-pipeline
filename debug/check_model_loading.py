"""
check_model_loading.py - Debug script to test if models can be loaded correctly
Use this to verify model files exist and can be loaded without errors
"""
import pickle
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "model" / "models"

def check_model_file(model_path):
    """Check if a model file exists and can be loaded"""
    if not model_path.exists():
        return False, "File not found"
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return True, "Loaded successfully"
    except Exception as e:
        return False, f"Load failed: {e}"

def check_all_models():
    """Check all model files"""
    print("=== MODEL LOADING CHECK ===")
    print()
    
    # Batter prop models
    batter_models = [
        "batter_prop_hit_long_ensemble.pkl",
        "batter_prop_hit_short_ensemble.pkl",
        "batter_prop_hr_long_ensemble.pkl",
        "batter_prop_hr_short_ensemble.pkl",
        "batter_prop_multi_long_ensemble.pkl",
        "batter_prop_multi_short_ensemble.pkl",
        "batter_prop_k_long_ensemble.pkl",
        "batter_prop_k_short_ensemble.pkl",
    ]
    
    # Season-level models
    season_models = [
        "batter_prop_hit_season_short_ensemble.pkl",
        "batter_prop_hr_season_short_ensemble.pkl",
        "batter_prop_multi_season_short_ensemble.pkl",
        "batter_prop_k_season_short_ensemble.pkl",
    ]
    
    # Win probability models
    win_prob_models = [
        "win_prob_ensemble_long.pkl",
        "win_prob_ensemble_short.pkl",
    ]
    
    all_models = [("Batter Props (Historical)", batter_models),
                  ("Batter Props (Season)", season_models),
                  ("Win Probability", win_prob_models)]
    
    total_checked = 0
    total_ok = 0
    
    for category, models in all_models:
        print(f"--- {category} ---")
        for model_name in models:
            model_path = MODEL_DIR / model_name
            ok, status = check_model_file(model_path)
            total_checked += 1
            if ok:
                print(f"✅ {model_name}: {status}")
                total_ok += 1
            else:
                print(f"❌ {model_name}: {status}")
        print()
    
    print(f"Summary: {total_ok}/{total_checked} models loaded successfully")

if __name__ == "__main__":
    check_all_models()
