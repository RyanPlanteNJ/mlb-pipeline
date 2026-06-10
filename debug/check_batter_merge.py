import pandas as pd
from datetime import date

# Load data
player_stats = pd.read_parquet('data/normalized/fact_player_stats.parquet')
games = pd.read_parquet('data/normalized/fact_games.parquet')
games['game_date'] = pd.to_datetime(games['game_date'], errors='coerce').dt.date
today = date.today()
games_today = games[games['game_date'] == today]

# Get season stats
latest_season = player_stats['season'].max()
batter_season_stats = player_stats[
    (player_stats['group'] == 'hitting') & 
    (player_stats['season'] == latest_season)
]

print(f"Latest season: {latest_season}")
print(f"Batter season stats: {len(batter_season_stats)}")
print(f"Batter IDs in season stats: {batter_season_stats['player_id'].nunique()}")

# Generate scheduled matchups (same logic as scoring script)
scheduled_matchups = []
pitcher_stats = player_stats[player_stats["group"] == "pitching"]
pitcher_name_to_id = dict(zip(pitcher_stats["player_name"].str.lower(), pitcher_stats["player_id"]))

for _, game in games_today.iterrows():
    game_pk = game["game_pk"]
    home_team_id = game["home_team_id"]
    away_team_id = game["away_team_id"]
    home_probable_pitcher = game.get("home_probable_pitcher")
    away_probable_pitcher = game.get("away_probable_pitcher")
    
    home_batters = player_stats[player_stats["team_id"] == home_team_id]
    away_batters = player_stats[player_stats["team_id"] == away_team_id]
    
    if home_probable_pitcher and pd.notna(home_probable_pitcher):
        pitcher_id = pitcher_name_to_id.get(home_probable_pitcher.lower())
        if pitcher_id:
            for _, batter in away_batters.iterrows():
                scheduled_matchups.append({
                    "game_pk": game_pk,
                    "batter_id": batter["player_id"],
                    "pitcher_id": pitcher_id
                })

matchups_df = pd.DataFrame(scheduled_matchups)
print(f"\nScheduled matchups: {len(matchups_df)}")
print(f"Unique batter IDs in scheduled matchups: {matchups_df['batter_id'].nunique()}")

# Check if these batters have season stats
batter_ids_with_stats = set(batter_season_stats['player_id'].tolist())
scheduled_batter_ids = set(matchups_df['batter_id'].tolist())

missing_batters = scheduled_batter_ids - batter_ids_with_stats
print(f"\nBatters without season stats: {len(missing_batters)}")
if missing_batters:
    print(f"Missing batter IDs: {list(missing_batters)[:10]}")
    
    # Check if these batters exist in player_stats at all
    all_batter_ids = set(player_stats[player_stats["group"] == "hitting"]["player_id"].tolist())
    missing_from_all = missing_batters - all_batter_ids
    print(f"Batters missing from all player_stats: {len(missing_from_all)}")
