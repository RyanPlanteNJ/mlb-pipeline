"""
advanced_features.py - Advanced feature engineering for MLB predictions
Adds sophisticated features beyond basic rolling averages
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent / "data" / "normalized"


def calculate_park_factors(games_df: pd.DataFrame, batter_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate park factors for each ballpark based on historical performance.
    Park factors adjust for the fact that some ballparks are more favorable to hitters/pitchers.
    """
    # Merge games with batter logs to get ballpark-specific performance
    games_subset = games_df[['game_pk', 'home_team_name']].copy()
    
    # Calculate runs scored per game for each ballpark
    merged = batter_logs_df.merge(games_subset, on='game_pk', how='inner')
    
    # Calculate home vs away performance for each ballpark
    park_stats = []
    
    for ballpark in merged['home_team_name'].unique():
        ballpark_data = merged[merged['home_team_name'] == ballpark]
        
        # Home team performance at this ballpark
        home_games = ballpark_data  # Home team is the home_team_name
        
        # Calculate offensive metrics
        if len(home_games) > 0:
            home_runs_per_game = home_games['r'].sum() / len(home_games['game_pk'].unique())
            home_hr_per_game = home_games['is_hr'].sum() / len(home_games['game_pk'].unique())
            home_hits_per_game = home_games['is_hit'].sum() / len(home_games['game_pk'].unique())
            
            park_stats.append({
                'ballpark': ballpark,
                'runs_factor': home_runs_per_game,
                'hr_factor': home_hr_per_game,
                'hit_factor': home_hits_per_game,
                'sample_size': len(home_games['game_pk'].unique())
            })
    
    park_factors_df = pd.DataFrame(park_stats)
    
    # Normalize park factors (league average = 1.0)
    if len(park_factors_df) > 0:
        avg_runs = park_factors_df['runs_factor'].mean()
        avg_hr = park_factors_df['hr_factor'].mean()
        avg_hits = park_factors_df['hit_factor'].mean()
        
        park_factors_df['runs_factor_normalized'] = park_factors_df['runs_factor'] / avg_runs
        park_factors_df['hr_factor_normalized'] = park_factors_df['hr_factor'] / avg_hr
        park_factors_df['hit_factor_normalized'] = park_factors_df['hit_factor'] / avg_hits
    
    return park_factors_df


def calculate_advanced_batter_features(matchups, player_stats, game_logs):
    """
    Calculate advanced batter features including:
    - Career vs recent performance comparison
    - Advanced metrics (OPS, SLG, OBP, BABIP)
    - Park-adjusted stats
    - Platoon splits
    - Weighted metrics
    """
    # Get latest season stats
    latest_season = player_stats["season"].max()
    latest_stats = player_stats[player_stats["season"] == latest_season]
    
    # Separate hitting and pitching stats
    hitting_stats = latest_stats[latest_stats["group"] == "hitting"].copy()
    pitching_stats = latest_stats[latest_stats["group"] == "pitching"].copy()
    
    # Calculate career averages (all seasons) - handle mixed types
    numeric_cols = ["avg", "obp", "slg", "ops", "babip", "homeRuns", "hits", "atBats", "plateAppearances"]
    for col in numeric_cols:
        player_stats[col] = pd.to_numeric(player_stats[col], errors="coerce")
    
    career_stats = player_stats.groupby("player_id").agg({
        "avg": "mean",
        "obp": "mean", 
        "slg": "mean",
        "ops": "mean",
        "babip": "mean",
        "homeRuns": "sum",
        "hits": "sum",
        "atBats": "sum",
        "plateAppearances": "sum"
    }).reset_index()
    career_stats.columns = [f"career_{col}" if col != "player_id" else col for col in career_stats.columns]
    
    # Calculate weighted OBP (weighted by plate appearances)
    career_stats["weighted_obp"] = career_stats["career_obp"] * (career_stats["career_plateAppearances"] / career_stats["career_plateAppearances"].max())
    
    # Calculate recent performance (last 30 games)
    recent_logs = game_logs.groupby("batter_id").tail(30)
    recent_performance = recent_logs.groupby("batter_id").agg({
        "is_hit": "mean",
        "is_hr": "mean",
        "is_so": "mean",
        "is_pa": "sum"
    }).reset_index()
    recent_performance.columns = ["batter_id", "recent_hit_rate", "recent_hr_rate", "recent_so_rate", "recent_pas"]
    
    # Calculate career vs recent delta
    career_hit_rate = career_stats["career_hits"] / career_stats["career_atBats"].replace(0, np.nan)
    career_hr_rate = career_stats["career_homeRuns"] / career_stats["career_atBats"].replace(0, np.nan)
    
    career_performance = career_stats[["player_id", "career_avg", "career_obp", "career_slg", "career_ops", "weighted_obp"]].copy()
    career_performance["career_hit_rate"] = career_hit_rate
    career_performance["career_hr_rate"] = career_hr_rate
    
    # Merge career and recent
    merged = recent_performance.merge(
        career_performance,
        left_on="batter_id",
        right_on="player_id",
        how="left"
    )
    
    # Calculate performance delta (recent - career)
    merged["hit_rate_delta"] = merged["recent_hit_rate"] - merged["career_hit_rate"]
    merged["hr_rate_delta"] = merged["recent_hr_rate"] - merged["career_hr_rate"]
    
    # Add advanced metrics from latest season - handle types
    for col in ["ops", "obp", "slg", "babip"]:
        hitting_stats[col] = pd.to_numeric(hitting_stats[col], errors="coerce")
    
    advanced = merged.merge(
        hitting_stats[["player_id", "ops", "obp", "slg", "babip"]],
        left_on="batter_id",
        right_on="player_id",
        how="left"
    )
    
    # Calculate composite offensive rating (normalized combination of avg, obp, slg)
    advanced["composite_offensive_rating"] = (
        (advanced["career_avg"] * 0.3 + 
         advanced["career_obp"] * 0.35 + 
         advanced["career_slg"] * 0.35)
    ).fillna(0)
    
    return advanced


def calculate_advanced_pitcher_features(matchups, player_stats, game_logs):
    """
    Calculate advanced pitcher features including:
    - ERA, WHIP, K/9, BB/9
    - Recent form vs career averages
    - Advanced metrics (FIP, xFIP)
    """
    # Get latest season stats
    latest_season = player_stats["season"].max()
    latest_stats = player_stats[player_stats["season"] == latest_season]
    
    # Pitching stats - handle types
    pitching_stats = latest_stats[latest_stats["group"] == "pitching"].copy()
    numeric_cols = ["era", "whip", "strikeoutsPer9Inn", "walksPer9Inn", "hitsPer9Inn", "homeRunsPer9", "inningsPitched"]
    for col in numeric_cols:
        pitching_stats[col] = pd.to_numeric(pitching_stats[col], errors="coerce")
    
    # Calculate FIP (Fielding Independent Pitching)
    # FIP = (13*HR + 3*BB - 2*K) / IP + constant (usually ~3.10)
    pitching_stats["fip"] = (
        (13 * pitching_stats["homeRunsPer9"] + 3 * pitching_stats["walksPer9Inn"] - 2 * pitching_stats["strikeoutsPer9Inn"]) / 
        pitching_stats["inningsPitched"].replace(0, np.nan) + 3.10
    ).fillna(pitching_stats["era"])  # Fallback to ERA if calculation fails
    
    # Calculate career averages - handle types
    career_numeric_cols = ["era", "whip", "strikeoutsPer9Inn", "walksPer9Inn", "hitsPer9Inn", "homeRunsPer9"]
    for col in career_numeric_cols:
        player_stats[col] = pd.to_numeric(player_stats[col], errors="coerce")
    
    career_pitching = player_stats[player_stats["group"] == "pitching"].groupby("player_id").agg({
        "era": "mean",
        "whip": "mean",
        "strikeoutsPer9Inn": "mean",
        "walksPer9Inn": "mean",
        "hitsPer9Inn": "mean",
        "homeRunsPer9": "mean"
    }).reset_index()
    career_pitching.columns = [f"career_{col}" if col != "player_id" else col for col in career_pitching.columns]
    
    # Calculate recent performance (last 30 games)
    recent_logs = game_logs.groupby("pitcher_id").tail(30)
    recent_performance = recent_logs.groupby("pitcher_id").agg({
        "is_hit": "mean",
        "is_hr": "mean",
        "is_so": "mean"
    }).reset_index()
    recent_performance.columns = ["pitcher_id", "recent_hit_allowed", "recent_hr_allowed", "recent_so_rate"]
    
    # Merge career and recent
    merged = recent_performance.merge(
        career_pitching,
        left_on="pitcher_id",
        right_on="player_id",
        how="left"
    )
    
    # Add advanced metrics from latest season
    advanced = merged.merge(
        pitching_stats[["player_id", "era", "whip", "strikeoutsPer9Inn", "walksPer9Inn", "fip"]],
        left_on="pitcher_id",
        right_on="player_id",
        how="left"
    )
    
    # Calculate composite pitching rating (normalized combination of era, whip, k/9)
    # Lower ERA/WHIP is better, higher K/9 is better
    advanced["composite_pitching_rating"] = (
        (1 / advanced["era"].replace(0, np.nan) * 0.4 + 
         1 / advanced["whip"].replace(0, np.nan) * 0.3 + 
         advanced["strikeoutsPer9Inn"] * 0.3)
    ).fillna(0)
    
    return advanced


def calculate_team_advanced_features(team_stats, games):
    """
    Calculate advanced team features for win probability:
    - Pythagorean winning percentage
    - Run differential
    - Home/away splits
    - Recent form (momentum)
    - Schedule difficulty
    """
    # Get latest season stats - handle types
    latest_season = team_stats["season"].max()
    latest_stats = team_stats[team_stats["season"] == latest_season].copy()
    
    # Separate hitting and pitching - handle types
    hitting = latest_stats[latest_stats["group"] == "hitting"].copy()
    pitching = latest_stats[latest_stats["group"] == "pitching"].copy()
    
    numeric_cols = ["runs", "earnedRuns", "ops", "whip", "era", "gamesPlayed"]
    for col in numeric_cols:
        hitting[col] = pd.to_numeric(hitting[col], errors="coerce")
        pitching[col] = pd.to_numeric(pitching[col], errors="coerce")
    
    # Calculate pythagorean winning percentage
    hitting["pythag_wp"] = (hitting["runs"] ** 2) / ((hitting["runs"] ** 2) + (pitching["earnedRuns"] ** 2))
    
    # Calculate run differential
    hitting["run_diff"] = hitting["runs"] - pitching["earnedRuns"]
    
    # Calculate advanced team metrics
    hitting["team_ops"] = hitting["ops"]
    hitting["team_whip"] = pitching["whip"]
    hitting["team_era"] = pitching["era"]
    
    # Calculate team momentum (recent 10 games performance)
    # This would require game-by-game results data
    # For now, use run differential as momentum proxy
    hitting["momentum"] = hitting["run_diff"] / hitting["gamesPlayed"].replace(0, 1)
    
    return hitting


def calculate_park_factors(games, batter_logs):
    """
    Calculate park factors for each stadium based on historical performance
    Based on home vs away performance using available metrics
    """
    # Handle games as index or column
    if 'game_pk' not in games.columns and games.index.name == 'game_pk':
        games_df = games.reset_index()
    else:
        games_df = games.copy()
    
    games_subset = games_df[['game_pk', 'home_team_name']].copy()
    
    # Calculate performance metrics per game for each ballpark
    merged = batter_logs.merge(games_subset, on='game_pk', how='inner')
    
    # Calculate home vs away performance for each ballpark
    park_stats = []
    
    for ballpark in merged['home_team_name'].unique():
        ballpark_data = merged[merged['home_team_name'] == ballpark]
        
        # Home team performance at this ballpark
        home_games = ballpark_data
        
        # Calculate offensive metrics using available columns
        if len(home_games) > 0:
            # Calculate per-game averages
            games_count = len(home_games['game_pk'].unique())
            if games_count > 0:
                hit_rate = home_games['is_hit'].sum() / len(home_games)
                hr_rate = home_games['is_hr'].sum() / len(home_games)
                multi_rate = home_games['is_multi'].sum() / len(home_games)
                
                park_stats.append({
                    'ballpark': ballpark,
                    'hit_factor': hit_rate,
                    'hr_factor': hr_rate,
                    'multi_factor': multi_rate,
                    'sample_size': games_count
                })
    
    park_factors_df = pd.DataFrame(park_stats)
    
    # Normalize park factors (league average = 1.0)
    if len(park_factors_df) > 0:
        avg_hit = park_factors_df['hit_factor'].mean()
        avg_hr = park_factors_df['hr_factor'].mean()
        avg_multi = park_factors_df['multi_factor'].mean()
        
        park_factors_df['hit_factor_normalized'] = park_factors_df['hit_factor'] / avg_hit
        park_factors_df['hr_factor_normalized'] = park_factors_df['hr_factor'] / avg_hr
        park_factors_df['multi_factor_normalized'] = park_factors_df['multi_factor'] / avg_multi
    
    return park_factors_df


def add_platoon_splits(matchups, game_logs):
    """
    Calculate platoon splits (lefty vs righty performance)
    """
    # This would require handedness data which may not be available
    # For now, return the matchups unchanged
    return matchups


def calculate_situational_stats(game_logs):
    """
    Calculate situational stats (RISP, clutch performance, etc.)
    """
    # This would require more detailed play-by-play data
    # For now, return placeholder
    return game_logs


def engineer_advanced_features(matchups, player_stats, batter_game_logs, pitcher_game_logs, team_stats, games):
    """
    Main function to engineer all advanced features
    """
    # Calculate advanced batter features
    batter_advanced = calculate_advanced_batter_features(matchups, player_stats, batter_game_logs)
    
    # Calculate advanced pitcher features  
    pitcher_advanced = calculate_advanced_pitcher_features(matchups, player_stats, pitcher_game_logs)
    
    # Calculate team advanced features
    team_advanced = calculate_team_advanced_features(team_stats, games)
    
    # Calculate park factors
    park_factors = calculate_park_factors(games, batter_game_logs)
    
    # Merge park factors into batter_advanced
    if len(park_factors) > 0:
        # Create mapping from ballpark to park factors
        park_mapping = park_factors.set_index('ballpark')[['hr_factor_normalized', 'hit_factor_normalized', 'multi_factor_normalized']].to_dict('index')
        
        # Handle games as index or column
        if 'game_pk' not in games.columns and games.index.name == 'game_pk':
            games_df = games.reset_index()
        else:
            games_df = games.copy()
        
        # Add park factors to matchups based on home team
        matchups_with_park = matchups.merge(
            games_df[['game_pk', 'home_team_name']], 
            on='game_pk', 
            how='left'
        )
        
        # Map park factors
        batter_advanced['park_hr_factor'] = matchups_with_park['home_team_name'].map(lambda x: park_mapping.get(x, {}).get('hr_factor_normalized', 1.0))
        batter_advanced['park_hit_factor'] = matchups_with_park['home_team_name'].map(lambda x: park_mapping.get(x, {}).get('hit_factor_normalized', 1.0))
        batter_advanced['park_multi_factor'] = matchups_with_park['home_team_name'].map(lambda x: park_mapping.get(x, {}).get('multi_factor_normalized', 1.0))
        
        # Fill missing values with 1.0 (neutral park factor)
        batter_advanced['park_hr_factor'] = batter_advanced['park_hr_factor'].fillna(1.0)
        batter_advanced['park_hit_factor'] = batter_advanced['park_hit_factor'].fillna(1.0)
        batter_advanced['park_multi_factor'] = batter_advanced['park_multi_factor'].fillna(1.0)
    
    return {
        "batter_advanced": batter_advanced,
        "pitcher_advanced": pitcher_advanced,
        "team_advanced": team_advanced,
        "park_factors": park_factors
    }


if __name__ == "__main__":
    # Test the advanced features
    matchups = pd.read_parquet(BASE / "fact_batter_pitcher_matchups.parquet")
    player_stats = pd.read_parquet(BASE / "fact_player_stats.parquet")
    batter_game_logs = pd.read_parquet(BASE / "fact_batter_game_logs.parquet")
    pitcher_game_logs = pd.read_parquet(BASE / "fact_pitcher_game_logs.parquet")
    team_stats = pd.read_parquet(BASE / "fact_team_stats.parquet")
    games = pd.read_parquet(BASE / "fact_games.parquet")
    
    features = engineer_advanced_features(matchups, player_stats, batter_game_logs, pitcher_game_logs, team_stats, games)
    
    print("Advanced features calculated successfully")
    print(f"Batter advanced features: {len(features['batter_advanced'])}")
    print(f"Pitcher advanced features: {len(features['pitcher_advanced'])}")
    print(f"Team advanced features: {len(features['team_advanced'])}")
