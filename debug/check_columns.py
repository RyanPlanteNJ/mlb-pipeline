import pandas as pd

# Check column names in relevant files
print("=== fact_games.parquet ===")
games = pd.read_parquet('data/normalized/fact_games.parquet')
print(games.columns.tolist())

print("\n=== fact_batter_pitcher_matchups.parquet ===")
matchups = pd.read_parquet('data/normalized/fact_batter_pitcher_matchups.parquet')
print(matchups.columns.tolist())

print("\n=== fact_player_stats.parquet ===")
stats = pd.read_parquet('data/normalized/fact_player_stats.parquet')
print(stats.columns.tolist())
