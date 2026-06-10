"""
feature_builder.py - Unified feature pipeline for batter/pitcher prop models.
Ensures consistent feature engineering for both historical and scheduled game contexts.
"""
import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)


def rolling_features(df, id_col, date_col, stat_cols, windows=[7, 14, 30]):
    """
    Build rolling window statistics.
    Uses shift(1) to prevent target leakage (hide today's stats from today's prediction).
    """
    df = df.sort_values([id_col, date_col])
    for w in windows:
        for col in stat_cols:
            df[f"{col}_roll{w}"] = (
                df.groupby(id_col)[col]
                  .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            )
    return df


def build_features_historical(matchups, player_stats, batter_logs, pitcher_logs, team_stats, games, advanced_features=None):
    """
    Build feature matrix for historical/completed games.
    Uses rolling features, advanced features, and full game context.
    
    Args:
        matchups: fact_batter_pitcher_matchups.parquet
        player_stats: fact_player_stats.parquet
        batter_logs: fact_batter_game_logs.parquet (with rolling features already computed)
        pitcher_logs: fact_pitcher_game_logs.parquet (with rolling features already computed)
        team_stats: fact_team_stats.parquet
        games: fact_games.parquet (indexed by game_pk)
        advanced_features: dict with 'batter_advanced' and 'pitcher_advanced' DataFrames
        
    Returns:
        DataFrame with all features ready for historical model training/scoring
    """
    df = matchups.copy()
    
    # Merge rolling features (batter perspective)
    df = df.merge(
        batter_logs.add_prefix("batter_"),
        left_on=["batter_id", "game_pk"],
        right_on=["batter_batter_id", "batter_game_pk"],
        how="left"
    )
    
    # Merge rolling features (pitcher perspective)
    df = df.merge(
        pitcher_logs.add_prefix("pitcher_"),
        left_on=["pitcher_id", "game_pk"],
        right_on=["pitcher_pitcher_id", "pitcher_game_pk"],
        how="left"
    )
    
    # Merge advanced features if provided
    if advanced_features:
        batter_adv = advanced_features.get("batter_advanced", pd.DataFrame())
        pitcher_adv = advanced_features.get("pitcher_advanced", pd.DataFrame())
        
        if not batter_adv.empty:
            batter_adv = batter_adv.rename(columns={"batter_id": "batter_id_adv"})
            df = df.merge(batter_adv, left_on="batter_id", right_on="batter_id_adv", how="left")
        
        if not pitcher_adv.empty:
            pitcher_adv = pitcher_adv.rename(columns={"pitcher_id": "pitcher_id_adv"})
            df = df.merge(pitcher_adv, left_on="pitcher_id", right_on="pitcher_id_adv", how="left")
    
    return df


def build_features_scheduled(matchups, player_stats, team_stats, games):
    """
    Build feature matrix for pre-game (scheduled) predictions.
    Uses only season-level stats and park factors (available before game starts).
    No rolling features or advanced features that depend on in-game data.
    
    Args:
        matchups: fact_batter_pitcher_matchups.parquet or generated scheduled matchups
        player_stats: fact_player_stats.parquet
        team_stats: fact_team_stats.parquet
        games: fact_games.parquet
        
    Returns:
        DataFrame with all season-level features ready for scheduled game scoring
    """
    df = matchups.copy()
    
    # Get latest season stats (most recent year available)
    latest_season = player_stats["season"].max()
    
    batter_season_stats = player_stats[
        (player_stats["group"] == "hitting") & 
        (player_stats["season"] == latest_season)
    ].copy()
    
    pitcher_season_stats = player_stats[
        (player_stats["group"] == "pitching") & 
        (player_stats["season"] == latest_season)
    ].copy()
    
    # Get career stats as fallback (all seasons combined)
    batter_stats_all = player_stats[(player_stats["group"] == "hitting")].copy()
    for col in ["avg", "ops", "slg", "obp", "hits", "homeRuns", "strikeOuts", "atBats"]:
        batter_stats_all[col] = pd.to_numeric(batter_stats_all[col], errors="coerce")
    
    batter_career_stats = batter_stats_all.groupby("player_id").agg({
        "avg": "mean",
        "ops": "mean",
        "slg": "mean",
        "obp": "mean",
        "hits": "sum",
        "homeRuns": "sum",
        "strikeOuts": "sum",
        "atBats": "sum"
    }).reset_index()
    
    pitcher_stats_all = player_stats[(player_stats["group"] == "pitching")].copy()
    for col in ["era", "whip", "avg", "hits", "homeRuns", "strikeOuts", "ops", "slg", "obp"]:
        pitcher_stats_all[col] = pd.to_numeric(pitcher_stats_all[col], errors="coerce")
    
    pitcher_career_stats = pitcher_stats_all.groupby("player_id").agg({
        "era": "mean",
        "whip": "mean",
        "avg": "mean",
        "hits": "sum",
        "homeRuns": "sum",
        "strikeOuts": "sum",
        "ops": "mean",
        "slg": "mean",
        "obp": "mean"
    }).reset_index()
    
    # Merge season stats with career stats (career as fallback)
    batter_season_stats = batter_season_stats.merge(
        batter_career_stats,
        on="player_id",
        how="outer",
        suffixes=("", "_career")
    )
    
    pitcher_season_stats = pitcher_season_stats.merge(
        pitcher_career_stats,
        on="player_id",
        how="outer",
        suffixes=("", "_career")
    )
    
    # Fill missing season stats with career stats
    for col in ["avg", "ops", "slg", "obp", "hits", "homeRuns", "strikeOuts", "atBats"]:
        if f"{col}_career" in batter_season_stats.columns:
            batter_season_stats[col] = batter_season_stats[col].fillna(batter_season_stats[f"{col}_career"])
    
    for col in ["era", "whip", "avg", "hits", "homeRuns", "strikeOuts", "ops", "slg", "obp"]:
        if f"{col}_career" in pitcher_season_stats.columns:
            pitcher_season_stats[col] = pitcher_season_stats[col].fillna(pitcher_season_stats[f"{col}_career"])
    
    # Prepare batter features
    batter_cols = batter_season_stats[["player_id", "avg", "ops", "slg", "obp", "hits", "homeRuns", "strikeOuts", "atBats"]].copy()
    # Convert string values to numeric
    for col in ["avg", "ops", "slg", "obp", "hits", "homeRuns", "strikeOuts", "atBats"]:
        batter_cols[col] = pd.to_numeric(batter_cols[col], errors="coerce")
    batter_cols.columns = [f"{col}_batter" if col != "player_id" else col for col in batter_cols.columns]
    
    # Prepare pitcher features
    pitcher_cols = pitcher_season_stats[["player_id", "era", "whip", "avg", "hits", "homeRuns", "strikeOuts", "ops", "slg", "obp"]].copy()
    # Convert string values to numeric
    for col in ["era", "whip", "avg", "hits", "homeRuns", "strikeOuts", "ops", "slg", "obp"]:
        pitcher_cols[col] = pd.to_numeric(pitcher_cols[col], errors="coerce")
    pitcher_cols.columns = [f"{col}_pitcher" if col != "player_id" else col for col in pitcher_cols.columns]
    
    # Merge into matchups
    df = df.merge(
        batter_cols,
        left_on="batter_id",
        right_on="player_id",
        how="left"
    ).merge(
        pitcher_cols,
        left_on="pitcher_id",
        right_on="player_id",
        how="left"
    )
    
    # Calculate derived stats from season data
    df["batter_avg"] = df["avg_batter"].fillna(0.25)
    df["batter_ops"] = df["ops_batter"].fillna(0.74)
    df["batter_slg"] = df["slg_batter"].fillna(0.42)
    df["batter_obp"] = df["obp_batter"].fillna(0.32)
    df["batter_hr_rate"] = (df["homeRuns_batter"] / df["atBats_batter"].replace(0, 1)).fillna(0.03)
    df["batter_so_rate"] = (df["strikeOuts_batter"] / df["atBats_batter"].replace(0, 1)).fillna(0.20)
    
    df["pitcher_era"] = df["era_pitcher"].fillna(4.50)
    df["pitcher_whip"] = df["whip_pitcher"].fillna(1.35)
    df["pitcher_avg"] = df["avg_pitcher"].fillna(0.25)
    df["pitcher_hr_rate"] = (df["homeRuns_pitcher"] / 100).fillna(0.03)
    df["pitcher_so_rate"] = (df["strikeOuts_pitcher"] / 100).fillna(0.20)
    
    # Fill rolling feature columns with season stats (maps to model training feature names)
    for window in [7, 14, 30]:
        df[f"batter_is_hit_roll{window}"] = df["avg_batter"].fillna(0.25)
        
        
        df[f"batter_is_hr_roll{window}"] = df["batter_hr_rate"].fillna(0.03)
        
        df[f"batter_is_so_roll{window}"] = df["batter_so_rate"].fillna(0.20)

        # Multi-hit rate (simple approximation)
        df[f"batter_is_multi_roll{window}"] = (df["avg_batter"].fillna(0.25) * 0.3).fillna(0.075)
        
        df[f"pitcher_is_hit_roll{window}"] = df["avg_pitcher"].fillna(0.25)
        df[f"pitcher_is_hr_roll{window}"] = df["pitcher_hr_rate"].fillna(0.03)
        df[f"pitcher_is_so_roll{window}"] = df["pitcher_so_rate"].fillna(0.20)
    
    # Add park factors (league average for now)
    df["park_hit_factor"] = 1.0
    df["park_hr_factor"] = 1.0
    df["park_multi_factor"] = 1.0
    
    # Add career/recent stats (same as season stats in pre-game context)
    df["career_avg"] = df["avg_batter"].fillna(0.25)
    df["career_hr_rate"] = df["batter_hr_rate"].fillna(0.03)
    df["career_hit_rate"] = df["avg_batter"].fillna(0.25)
    df["career_obp"] = df["obp_batter"].fillna(0.32)
    df["career_slg"] = df["slg_batter"].fillna(0.42)
    df["career_ops"] = df["ops_batter"].fillna(0.74)
    df["career_era"] = df["era_pitcher"].fillna(4.50)
    df["career_whip"] = df["whip_pitcher"].fillna(1.35)
    df["career_homeRunsPer9"] = (df["homeRuns_pitcher"] / 100 * 9).fillna(1.0)
    df["career_hitsPer9Inn"] = (df["hits_pitcher"] / 100 * 9).fillna(8.0)
    df["career_strikeoutsPer9Inn"] = (df["strikeOuts_pitcher"] / 100 * 9).fillna(8.0)
    df["career_walksPer9Inn"] = 3.0
    
    df["recent_hit_rate"] = df["avg_batter"].fillna(0.25)
    df["recent_hr_rate"] = df["batter_hr_rate"].fillna(0.03)
    df["recent_so_rate"] = df["batter_so_rate"].fillna(0.20)
    df["recent_so_rate_x"] = df["batter_so_rate"].fillna(0.20)
    df["recent_so_rate_y"] = df["batter_so_rate"].fillna(0.20)
    df["recent_pas"] = df["atBats_batter"].fillna(100)
    
    # Fill pitcher/batter specific stats
    df["era"] = df["era_pitcher"].fillna(4.50)
    df["whip"] = df["whip_pitcher"].fillna(1.35)
    df["babip"] = 0.300
    df["walksPer9Inn"] = 3.0
    df["strikeoutsPer9Inn"] = 8.0
    
    df["slg"] = df["slg_batter"].fillna(0.42)
    df["ops"] = df["ops_batter"].fillna(0.74)
    df["obp"] = df["obp_batter"].fillna(0.32)
    
    # Add delta/comparison stats (0 in pre-game since no historical comparison)
    df["hit_rate_delta"] = 0
    df["hr_rate_delta"] = 0
    
    # Add advanced feature ID placeholders
    df["batter_id_adv"] = df["batter_id"]
    df["pitcher_id_adv"] = df["pitcher_id"]
    
    # Add merge artifacts (for compatibility with training)
    df["player_id_x_x"] = df["batter_id"]
    df["player_id_y_y"] = df["pitcher_id"]
    df["player_id_x_y"] = df["batter_id"]
    df["player_id_y_x"] = df["pitcher_id"]
    
    return df


def get_explicit_features(mode="historical"):
    """
    Return the explicit feature set used for a given mode.
    
    Args:
        mode: "historical" or "scheduled"
        
    Returns:
        List of feature names expected in the model
    """
    base_features = [
        # Rolling batter features
        "batter_is_hit_roll7", "batter_is_hit_roll14", "batter_is_hit_roll30",
        "batter_is_hr_roll7", "batter_is_hr_roll14", "batter_is_hr_roll30",
        "batter_is_so_roll7", "batter_is_so_roll14", "batter_is_so_roll30",
        "batter_is_multi_roll7", "batter_is_multi_roll14", "batter_is_multi_roll30",
        
        # Rolling pitcher features
        "pitcher_is_hit_roll7", "pitcher_is_hit_roll14", "pitcher_is_hit_roll30",
        "pitcher_is_hr_roll7", "pitcher_is_hr_roll14", "pitcher_is_hr_roll30",
        "pitcher_is_so_roll7", "pitcher_is_so_roll14", "pitcher_is_so_roll30",
        
        # Season/career features
        "batter_avg", "batter_ops", "batter_slg", "batter_obp", "batter_hr_rate", "batter_so_rate",
        "pitcher_era", "pitcher_whip", "pitcher_avg", "pitcher_hr_rate", "pitcher_so_rate",
        
        # Park factors
        "park_hit_factor", "park_hr_factor", "park_multi_factor",
        
        # Career stats
        "career_avg", "career_hr_rate", "career_hit_rate", "career_obp", "career_slg", "career_ops",
        "career_era", "career_whip", "career_homeRunsPer9", "career_hitsPer9Inn", "career_strikeoutsPer9Inn", "career_walksPer9Inn",
        
        # Recent stats
        "recent_hit_rate", "recent_hr_rate", "recent_so_rate", "recent_so_rate_x", "recent_so_rate_y", "recent_pas",
        
        # Pitcher stats
        "era", "whip", "babip", "walksPer9Inn", "strikeoutsPer9Inn",
        
        # Batter stats
        "slg", "ops", "obp",
        
        # Deltas
        "hit_rate_delta", "hr_rate_delta",
    ]
    
    # Historical mode includes advanced features
    if mode == "historical":
        # These will be added from advanced_features if available
        pass
    
    return base_features


if __name__ == "__main__":
    print("Feature builder module loaded successfully")
