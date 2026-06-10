"""
check_data_quality.py - Debug script for quick data quality checks
Use this to check for nulls, duplicates, and other data quality issues
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent / "data" / "normalized"

def check_nulls(df, table_name):
    """Check for null values in dataframe"""
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    
    if len(null_cols) > 0:
        print(f"⚠️  {table_name} - Null values found:")
        for col, count in null_cols.items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")
    else:
        print(f"✅ {table_name} - No null values")

def check_duplicates(df, table_name, key_cols):
    """Check for duplicate rows"""
    if len(key_cols) > 0:
        dupes = df.duplicated(subset=key_cols).sum()
        if dupes > 0:
            print(f"⚠️  {table_name} - {dupes} duplicate rows on {key_cols}")
        else:
            print(f"✅ {table_name} - No duplicates on {key_cols}")

def check_all_tables():
    """Check data quality for all tables"""
    print("=== DATA QUALITY CHECK ===")
    print()
    
    # Define tables to check and their key columns
    tables = {
        "fact_games.parquet": ["game_pk"],
        "fact_player_stats.parquet": ["player_id", "season", "group"],
        "fact_team_stats.parquet": ["team_id", "season", "group"],
        "fact_batter_pitcher_matchups.parquet": ["game_pk", "batter_id", "pitcher_id"],
        "fact_player_props.parquet": ["game_pk", "player_id"],
        "fact_team_props.parquet": ["game_pk"],
        "fact_win_probability.parquet": ["game_pk"],
    }
    
    for filename, key_cols in tables.items():
        filepath = BASE / filename
        if not filepath.exists():
            print(f"❌ {filename} - File not found")
            print()
            continue
        
        try:
            df = pd.read_parquet(filepath)
            print(f"--- {filename} ---")
            print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
            check_nulls(df, filename)
            check_duplicates(df, filename, key_cols)
            print()
        except Exception as e:
            print(f"❌ {filename} - Error reading: {e}")
            print()

if __name__ == "__main__":
    check_all_tables()
