"""
check_pipeline_status.py - Debug script to check overall pipeline status
Use this to see which pipeline stages have run and their completion status
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent / "data" / "normalized"
MODEL_DIR = Path(__file__).parent.parent / "model" / "models"

def check_file_exists_and_fresh(filepath, max_age_hours=24):
    """Check if file exists and is fresh"""
    if not filepath.exists():
        return False, "Missing"
    
    mod_time = datetime.fromtimestamp(filepath.stat().st_mtime)
    age = datetime.now() - mod_time
    age_hours = age.total_seconds() / 3600
    
    is_fresh = age_hours <= max_age_hours
    status = f"Fresh ({age_hours:.1f}h old)" if is_fresh else f"Stale ({age_hours:.1f}h old)"
    return True, status

def check_pipeline_status():
    """Check status of all pipeline stages"""
    print("=== PIPELINE STATUS CHECK ===")
    print(f"Check time: {datetime.now()}")
    print()
    
    # Pipeline stages and their outputs
    stages = [
        ("Ingestion", [
            (BASE / "fact_games.parquet", 24),
            (BASE / "fact_player_stats.parquet", 24),
            (BASE / "fact_team_stats.parquet", 24),
        ]),
        ("Normalization", [
            (BASE / "fact_batter_pitcher_matchups.parquet", 24),
            (BASE / "fact_batter_game_logs.parquet", 24),
            (BASE / "fact_pitcher_game_logs.parquet", 24),
        ]),
        ("Model Training", [
            (MODEL_DIR / "batter_prop_hit_short_ensemble.pkl", 168),  # 7 days
            (MODEL_DIR / "batter_prop_hr_short_ensemble.pkl", 168),
            (MODEL_DIR / "batter_prop_multi_short_ensemble.pkl", 168),
            (MODEL_DIR / "batter_prop_k_short_ensemble.pkl", 168),
            (MODEL_DIR / "win_prob_ensemble_short.pkl", 168),
        ]),
        ("Prediction Scoring", [
            (BASE / "fact_player_props.parquet", 6),
            (BASE / "fact_team_props.parquet", 6),
            (BASE / "fact_win_probability.parquet", 6),
        ]),
    ]
    
    for stage_name, files in stages:
        print(f"--- {stage_name} ---")
        all_ok = True
        for filepath, max_age in files:
            exists, status = check_file_exists_and_fresh(filepath, max_age)
            if exists:
                print(f"✅ {filepath.name}: {status}")
            else:
                print(f"❌ {filepath.name}: {status}")
                all_ok = False
        
        if all_ok:
            print(f"Status: ✅ COMPLETE")
        else:
            print(f"Status: ⚠️  INCOMPLETE")
        print()

if __name__ == "__main__":
    check_pipeline_status()
