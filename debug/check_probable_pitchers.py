import pandas as pd
from datetime import date

# Check probable pitcher columns
games = pd.read_parquet('data/normalized/fact_games.parquet')
games['game_date'] = pd.to_datetime(games['game_date'], errors='coerce').dt.date
today = date.today()
games_today = games[games['game_date'] == today]

print(f"Games today: {len(games_today)}")
if len(games_today) > 0:
    print("\nSample game data:")
    print(games_today[['game_pk', 'home_probable_pitcher', 'away_probable_pitcher']].head())
    
    # Check if these are names or IDs
    sample_home = games_today['home_probable_pitcher'].iloc[0] if len(games_today) > 0 else None
    sample_away = games_today['away_probable_pitcher'].iloc[0] if len(games_today) > 0 else None
    print(f"\nSample home_probable_pitcher: {sample_home} (type: {type(sample_home)})")
    print(f"Sample away_probable_pitcher: {sample_away} (type: {type(sample_away)})")
