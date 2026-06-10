"""
check_player_stats.py - Debug script to verify player stats data
Use this to check player statistics for data quality issues
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent / "data" / "normalized"

def check_player_stats():
    """Check player stats for data quality"""
    try:
        df = pd.read_parquet(BASE / "fact_player_stats.parquet")
        
        print("=== PLAYER STATS CHECK ===")
        print(f"Total rows: {len(df)}")
        print(f"Total unique players: {df['player_id'].nunique()}")
        print(f"Seasons: {df['season'].unique()}")
        print()
        
        # Check for groups
        print("Groups:")
        print(df['group'].value_counts())
        print()
        
        # Check for positions
        print("Positions (hitting):")
        hitting = df[df['group'] == 'hitting']
        print(hitting['position'].value_counts())
        print()
        
        # Check for string values in numeric columns
        numeric_cols = ['avg', 'ops', 'slg', 'obp', 'era', 'whip', 'hits', 'homeRuns', 'strikeOuts', 'atBats']
        print("Checking for string values in numeric columns:")
        for col in numeric_cols:
            if col in df.columns:
                string_count = df[col].apply(lambda x: isinstance(x, str)).sum()
                if string_count > 0:
                    print(f"  ⚠️  {col}: {string_count} string values")
                else:
                    print(f"  ✅ {col}: All numeric")
        print()
        
        # Check for null values
        print("Null values:")
        null_counts = df.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                pct = (count / len(df)) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
        print()
        
        # Sample data
        print("Sample data (first 5 rows):")
        print(df.head())
        
    except FileNotFoundError:
        print("❌ Player stats file not found")

if __name__ == "__main__":
    check_player_stats()
