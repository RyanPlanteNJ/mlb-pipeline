"""
check_data_freshness.py - Debug script to check if data files are up to date
Use this to verify data freshness before running the pipeline
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

BASE = Path(__file__).parent.parent / "data" / "normalized"

def check_file_freshness(file_path, max_age_hours=24):
    """Check if a file is fresh enough"""
    if not file_path.exists():
        return None, "File not found"
    
    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
    age = datetime.now() - mod_time
    age_hours = age.total_seconds() / 3600
    
    is_fresh = age_hours <= max_age_hours
    return is_fresh, f"Age: {age_hours:.1f} hours (max: {max_age_hours}h)"

def check_all_data():
    """Check freshness of all critical data files"""
    print("=== DATA FRESHNESS CHECK ===")
    print(f"Check time: {datetime.now()}")
    print()
    
    critical_files = [
        ("fact_games.parquet", 24),
        ("fact_player_stats.parquet", 24),
        ("fact_team_stats.parquet", 24),
        ("fact_batter_pitcher_matchups.parquet", 24),
        ("fact_player_props.parquet", 6),  # Updated more frequently
        ("fact_team_props.parquet", 6),
        ("fact_win_probability.parquet", 6),
    ]
    
    stale_files = []
    
    for filename, max_age in critical_files:
        file_path = BASE / filename
        is_fresh, status = check_file_freshness(file_path, max_age)
        
        if is_fresh is None:
            print(f"❌ {filename}: {status}")
        elif is_fresh:
            print(f"✅ {filename}: {status}")
        else:
            print(f"⚠️  {filename}: {status}")
            stale_files.append(filename)
    
    print()
    if stale_files:
        print(f"⚠️  {len(stale_files)} file(s) may need updating:")
        for f in stale_files:
            print(f"  - {f}")
    else:
        print("✅ All critical files are fresh")

if __name__ == "__main__":
    check_all_data()
