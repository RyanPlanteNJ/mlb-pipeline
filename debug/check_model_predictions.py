"""
check_model_predictions.py - Debug script to verify model predictions are reasonable
Use this to check if model predictions are within expected ranges
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent / "data" / "normalized"

def check_batter_props():
    """Check batter prop predictions for reasonableness"""
    props = pd.read_parquet(BASE / "fact_player_props.parquet")
    
    print("=== BATTER PROP PREDICTIONS CHECK ===")
    print(f"Total predictions: {len(props)}")
    print()
    
    for prop in ["hit", "hr", "multi", "k"]:
        col = f"{prop}_prob"
        if col in props.columns:
            print(f"{prop.upper()} Predictions:")
            print(f"  Mean: {props[col].mean():.4f}")
            print(f"  Min: {props[col].min():.4f}")
            print(f"  Max: {props[col].max():.4f}")
            print(f"  Std: {props[col].std():.4f}")
            
            # Check for unreasonable values
            if props[col].min() < 0:
                print(f"  ⚠️ WARNING: Negative values found")
            if props[col].max() > 1:
                print(f"  ⚠️ WARNING: Values > 1 found")
            if props[col].mean() < 0.01:
                print(f"  ⚠️ WARNING: Mean very low, possible issue")
            print()

def check_team_props():
    """Check team prop predictions for reasonableness"""
    try:
        props = pd.read_parquet(BASE / "fact_team_props.parquet")
        
        print("=== TEAM PROP PREDICTIONS CHECK ===")
        print(f"Total predictions: {len(props)}")
        print()
        
        print("Win Probabilities:")
        print(f"  Home Win Mean: {props['home_win_prob'].mean():.4f}")
        print(f"  Away Win Mean: {props['away_win_prob'].mean():.4f}")
        print()
        
        print("Expected Runs:")
        print(f"  Home Expected Runs Mean: {props['home_expected_runs'].mean():.2f}")
        print(f"  Away Expected Runs Mean: {props['away_expected_runs'].mean():.2f}")
        print()
        
    except FileNotFoundError:
        print("Team props file not found")

def check_win_probability():
    """Check win probability predictions for reasonableness"""
    try:
        wp = pd.read_parquet(BASE / "fact_win_probability.parquet")
        
        print("=== WIN PROBABILITY CHECK ===")
        print(f"Total predictions: {len(wp)}")
        print()
        
        print("Home Win Probabilities:")
        print(f"  Mean: {wp['home_win_prob'].mean():.4f}")
        print(f"  Min: {wp['home_win_prob'].min():.4f}")
        print(f"  Max: {wp['home_win_prob'].max():.4f}")
        print()
        
        # Check for unreasonable values
        if wp['home_win_prob'].min() < 0 or wp['home_win_prob'].max() > 1:
            print("⚠️ WARNING: Win probabilities outside [0, 1] range")
        
    except FileNotFoundError:
        print("Win probability file not found")

if __name__ == "__main__":
    check_batter_props()
    check_team_props()
    check_win_probability()
