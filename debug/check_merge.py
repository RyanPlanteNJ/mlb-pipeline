import pandas as pd
from datetime import date

# Load data
matchups = pd.read_parquet('data/normalized/fact_batter_pitcher_matchups.parquet')
games = pd.read_parquet('data/normalized/fact_games.parquet')
player_stats = pd.read_parquet('data/normalized/fact_player_stats.parquet')

# Filter to today's games
games['game_date'] = pd.to_datetime(games['game_date'], errors='coerce').dt.date
today = date.today()
games_today = games[games['game_date'] == today]

print(f"Today's games: {len(games_today)}")
print(f"Game PKs: {games_today['game_pk'].tolist()[:5]}")

# Check if there are historical matchups for today
matchups_today = matchups[matchups['game_pk'].isin(games_today['game_pk'])]
print(f"Historical matchups for today: {len(matchups_today)}")

# Check season stats
latest_season = player_stats['season'].max()
print(f"Latest season in stats: {latest_season}")

batter_season_stats = player_stats[
    (player_stats['group'] == 'hitting') & 
    (player_stats['season'] == latest_season)
]
pitcher_season_stats = player_stats[
    (player_stats['group'] == 'pitching') & 
    (player_stats['season'] == latest_season)
]

print(f"Batter season stats: {len(batter_season_stats)}")
print(f"Pitcher season stats: {len(pitcher_season_stats)}")

# Check if scheduled matchups have matching season stats
# First, generate scheduled matchups from probable pitchers
scheduled = []
for _, game in games_today.iterrows():
    home_pitcher = game.get('probable_pitcher_home_id')
    away_pitcher = game.get('probable_pitcher_away_id')
    
    if pd.notna(home_pitcher):
        scheduled.append({'game_pk': game['game_pk'], 'pitcher_id': home_pitcher})
    if pd.notna(away_pitcher):
        scheduled.append({'game_pk': game['game_pk'], 'pitcher_id': away_pitcher})

scheduled_df = pd.DataFrame(scheduled)
print(f"Scheduled pitchers: {len(scheduled_df)}")

# Check if these pitchers have season stats
pitcher_ids_with_stats = set(pitcher_season_stats['player_id'].tolist())
scheduled_pitcher_ids = set(scheduled_df['pitcher_id'].tolist())

missing_pitchers = scheduled_pitcher_ids - pitcher_ids_with_stats
print(f"Pitchers without season stats: {len(missing_pitchers)}")
if missing_pitchers:
    print(f"Missing pitcher IDs: {list(missing_pitchers)[:5]}")
