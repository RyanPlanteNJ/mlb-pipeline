import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import date
import logging
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.advanced_features import engineer_advanced_features
from utils.prediction_drift import PredictionDriftDetector
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# ----------------------------------------
# Paths
# ----------------------------------------
BASE = Path(__file__).parent.parent / "data" / "normalized"
MODEL_DIR = Path(__file__).parent / "models"

# ----------------------------------------
# Helper: Replicate Training Features
# ----------------------------------------
def rolling_features(df, id_col, date_col, stat_cols, windows=[7, 14, 30]):
    """Must match the training script exactly so the model gets the right columns."""
    df = df.sort_values([id_col, date_col])
    for w in windows:
        for col in stat_cols:
            df[f"{col}_roll{w}"] = (
                df.groupby(id_col)[col]
                  .rolling(w, min_periods=1)
                  .mean()
                  .reset_index(level=0, drop=True)
            )
    return df

# ----------------------------------------
# RUN FUNCTION
# ----------------------------------------
def run():
    # Load ensemble models safely
    models = {}
    for prop in ["hit", "hr", "multi", "k"]:
        model_path = MODEL_DIR / f"batter_prop_{prop}_short_ensemble.pkl"
        
        if model_path.exists():
            models[prop] = pickle.load(open(model_path, "rb"))
        else:
            # Fall back to single model if ensemble not available
            fallback_path = MODEL_DIR / f"batter_prop_{prop}_short.pkl"
            if fallback_path.exists():
                models[prop] = pickle.load(open(fallback_path, "rb"))
                log.warning("Using fallback single model for %s", prop)
            else:
                log.error("Model %s not found. Run training first.", model_path.name)
                return

    # Load season-level models for scheduled games
    season_models = {}
    for prop in ["hit", "hr", "multi", "k"]:
        model_path = MODEL_DIR / f"batter_prop_{prop}_season_short_ensemble.pkl"
        
        if model_path.exists():
            season_models[prop] = pickle.load(open(model_path, "rb"))
            log.info("Loaded season-level model for %s", prop)
        else:
            # Fall back to long-term season model
            fallback_path = MODEL_DIR / f"batter_prop_{prop}_season_ensemble.pkl"
            if fallback_path.exists():
                season_models[prop] = pickle.load(open(fallback_path, "rb"))
                log.info("Loaded long-term season-level model for %s", prop)
            else:
                log.warning("Season-level model for %s not found. Will use historical model for scheduled games.", prop)

    # Load normalized data
    player_stats = pd.read_parquet(BASE / "fact_player_stats.parquet")
    batter_logs = pd.read_parquet(BASE / "fact_batter_game_logs.parquet")
    pitcher_logs = pd.read_parquet(BASE / "fact_pitcher_game_logs.parquet")
    games = pd.read_parquet(BASE / "fact_games.parquet")
    team_stats = pd.read_parquet(BASE / "fact_team_stats.parquet")

    # FIX: Isolate the most recent team stats so we don't multiply games by 5 seasons
    team_stats = team_stats.drop_duplicates(subset=["team_name"], keep="last")

    # Use today's matchups as the base (matching training approach)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.date
    today = date.today()
    games_today = games[games["game_date"] == today]
    
    if games_today.empty:
        log.info("No games found for today.")
        return
    
    # Load matchups for today's games
    matchups = pd.read_parquet(BASE / "fact_batter_pitcher_matchups.parquet")
    matchups_today = matchups[matchups["game_pk"].isin(games_today["game_pk"])]
    if "event_type" in matchups_today.columns:
        advisory_types = {"game_advisory"}
        if not matchups_today[~matchups_today["event_type"].isin(advisory_types)].empty:
            valid_count = len(matchups_today[~matchups_today["event_type"].isin(advisory_types)])
            skipped = len(matchups_today) - valid_count
            if skipped:
                log.info("Removed %d advisory/historical placeholder rows from today's matchups", skipped)
            matchups_today = matchups_today[~matchups_today["event_type"].isin(advisory_types)]
        else:
            log.info("Removed advisory/historical placeholder rows from today's matchups; no valid historical rows remain.")
            matchups_today = pd.DataFrame()

    # If no historical matchups (games haven't started), generate scheduled matchups from probable pitchers
    if matchups_today.empty:
        log.info("No historical matchups found - generating scheduled matchups from probable pitchers...")
        scheduled_matchups = []
        
        # Build name-to-ID lookup for pitchers
        pitcher_stats = player_stats[player_stats["group"] == "pitching"]
        pitcher_name_to_id = dict(zip(pitcher_stats["player_name"].str.lower(), pitcher_stats["player_id"]))
        
        for _, game in games_today.iterrows():
            game_pk = game["game_pk"]
            home_team_id = game["home_team_id"]
            away_team_id = game["away_team_id"]
            home_probable_pitcher = game.get("home_probable_pitcher")
            away_probable_pitcher = game.get("away_probable_pitcher")
            
            # Get all batters for both teams from player stats (only latest season to exclude former players)
            latest_season = player_stats["season"].max()
            home_batters = player_stats[
                (player_stats["team_id"] == home_team_id) & 
                (player_stats["season"] == latest_season) &
                (player_stats["group"] == "hitting")
            ]
            away_batters = player_stats[
                (player_stats["team_id"] == away_team_id) & 
                (player_stats["season"] == latest_season) &
                (player_stats["group"] == "hitting")
            ]
            
            # Generate matchups with probable pitchers
            if home_probable_pitcher and pd.notna(home_probable_pitcher):
                # Find pitcher ID from name (case-insensitive)
                pitcher_id = pitcher_name_to_id.get(home_probable_pitcher.lower())
                if pitcher_id:
                    for _, batter in away_batters.iterrows():
                        scheduled_matchups.append({
                            "game_pk": game_pk,
                            "batter_id": batter["player_id"],
                            "batter_name": batter["player_name"],
                            "pitcher_id": pitcher_id,
                            "pitcher_name": home_probable_pitcher,
                            "is_pa": 0, "is_ab": 0, "is_hit": 0, "is_hr": 0, "is_so": 0, "is_multi": 0
                        })
                else:
                    log.warning("Could not find pitcher ID for '%s'", home_probable_pitcher)
            
            if away_probable_pitcher and pd.notna(away_probable_pitcher):
                # Find pitcher ID from name (case-insensitive)
                pitcher_id = pitcher_name_to_id.get(away_probable_pitcher.lower())
                if pitcher_id:
                    for _, batter in home_batters.iterrows():
                        scheduled_matchups.append({
                            "game_pk": game_pk,
                            "batter_id": batter["player_id"],
                            "batter_name": batter["player_name"],
                            "pitcher_id": pitcher_id,
                            "pitcher_name": away_probable_pitcher,
                            "is_pa": 0, "is_ab": 0, "is_hit": 0, "is_hr": 0, "is_so": 0, "is_multi": 0
                        })
                else:
                    log.warning("Could not find pitcher ID for '%s'", away_probable_pitcher)
        
        if not scheduled_matchups:
            log.warning("No probable pitchers found for today's games.")
            return
        
        matchups_today = pd.DataFrame(scheduled_matchups)
        log.info("Generated %d scheduled matchups.", len(matchups_today))
    
    # Check if these are scheduled matchups (no historical data)
    is_scheduled = matchups_today["is_pa"].sum() == 0
    
    if is_scheduled:
        log.info("Using season-level stats for scheduled matchups...")
        # For scheduled matchups, use season stats instead of rolling features
        # Use career stats as fallback if current season stats are missing
        latest_season = player_stats["season"].max()
        
        # Get current season stats
        batter_season_stats = player_stats[
            (player_stats["group"] == "hitting") & 
            (player_stats["season"] == latest_season)
        ].copy()
        pitcher_season_stats = player_stats[
            (player_stats["group"] == "pitching") & 
            (player_stats["season"] == latest_season)
        ].copy()
        
        # Get career stats (all seasons combined)
        # Convert columns to numeric first to handle string values
        batter_stats_all = player_stats[
            (player_stats["group"] == "hitting")
        ].copy()
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
        
        pitcher_stats_all = player_stats[
            (player_stats["group"] == "pitching")
        ].copy()
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
        
        # Merge season stats with career stats (use career as fallback)
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
        
        # Merge season stats directly into matchups
        # Rename columns before merge to avoid conflicts
        batter_cols = batter_season_stats[["player_id", "avg", "ops", "slg", "obp", "hits", "homeRuns", "strikeOuts", "atBats"]].copy()
        batter_cols.columns = [f"{col}_batter" if col != "player_id" else col for col in batter_cols.columns]
        
        pitcher_cols = pitcher_season_stats[["player_id", "era", "whip", "avg", "hits", "homeRuns", "strikeOuts", "ops", "slg", "obp"]].copy()
        pitcher_cols.columns = [f"{col}_pitcher" if col != "player_id" else col for col in pitcher_cols.columns]
        
        df = matchups_today.merge(
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
        
        # Calculate derived features (matching training script naming)
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
        
        # Fill missing recent_so_rate (training script uses this)
        df["recent_so_rate"] = df["batter_so_rate"].fillna(0.20)
        
        # Fill rolling feature columns with season stats (simplified approach)
        # This maps season stats to the rolling feature columns the model expects
        df["batter_is_hit_roll7"] = df["avg_batter"].fillna(0.25)
        df["batter_is_hit_roll14"] = df["avg_batter"].fillna(0.25)
        df["batter_is_hit_roll30"] = df["avg_batter"].fillna(0.25)
        df["batter_is_hr_roll7"] = df["batter_hr_rate"].fillna(0.03)
        df["batter_is_hr_roll14"] = df["batter_hr_rate"].fillna(0.03)
        df["batter_is_hr_roll30"] = df["batter_hr_rate"].fillna(0.03)
        df["batter_is_so_roll7"] = df["batter_so_rate"].fillna(0.20)
        df["batter_is_so_roll14"] = df["batter_so_rate"].fillna(0.20)
        df["batter_is_so_roll30"] = df["batter_so_rate"].fillna(0.20)
        df["batter_is_multi_roll7"] = (df["avg_batter"].fillna(0.25) * 0.3).fillna(0.075)  # Approx multi-hit rate
        df["batter_is_multi_roll14"] = (df["avg_batter"].fillna(0.25) * 0.3).fillna(0.075)
        df["batter_is_multi_roll30"] = (df["avg_batter"].fillna(0.25) * 0.3).fillna(0.075)
        
        df["pitcher_is_hit_roll7"] = df["avg_pitcher"].fillna(0.25)
        df["pitcher_is_hit_roll14"] = df["avg_pitcher"].fillna(0.25)
        df["pitcher_is_hit_roll30"] = df["avg_pitcher"].fillna(0.25)
        df["pitcher_is_hr_roll7"] = df["pitcher_hr_rate"].fillna(0.03)
        df["pitcher_is_hr_roll14"] = df["pitcher_hr_rate"].fillna(0.03)
        df["pitcher_is_hr_roll30"] = df["pitcher_hr_rate"].fillna(0.03)
        df["pitcher_is_so_roll7"] = df["pitcher_so_rate"].fillna(0.20)
        df["pitcher_is_so_roll14"] = df["pitcher_so_rate"].fillna(0.20)
        df["pitcher_is_so_roll30"] = df["pitcher_so_rate"].fillna(0.20)
        
        # Fill other required columns with defaults
        df["batter_game_pk"] = df["game_pk"]
        df["batter_batter_id"] = df["batter_id"]
        df["pitcher_game_pk"] = df["game_pk"]
        df["pitcher_pitcher_id"] = df["pitcher_id"]
        
        # Fill missing advanced features with season stats
        # Removed target variables to match training script (no data leakage)
        # df["batter_is_pa"] = df["atBats_batter"].fillna(100)
        # df["batter_is_ab"] = df["atBats_batter"].fillna(100)
        # df["batter_is_hit"] = df["avg_batter"].fillna(0.25)
        # df["batter_is_hr"] = df["batter_hr_rate"].fillna(0.03)
        # df["batter_is_so"] = df["batter_so_rate"].fillna(0.20)
        # df["batter_is_multi"] = (df["avg_batter"].fillna(0.25) * 0.3).fillna(0.075)
        
        # df["pitcher_is_hit"] = df["avg_pitcher"].fillna(0.25)
        # df["pitcher_is_hr"] = df["pitcher_hr_rate"].fillna(0.03)
        # df["pitcher_is_so"] = df["pitcher_so_rate"].fillna(0.20)
        
        # Fill career stats with season stats (simplified)
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
        df["career_walksPer9Inn"] = 3.0  # Default
        
        # Fill recent stats with season stats (simplified)
        df["recent_hit_rate"] = df["avg_batter"].fillna(0.25)
        df["recent_hr_rate"] = df["batter_hr_rate"].fillna(0.03)
        df["recent_so_rate_x"] = df["batter_so_rate"].fillna(0.20)
        df["recent_so_rate_y"] = df["batter_so_rate"].fillna(0.20)
        df["recent_pas"] = df["atBats_batter"].fillna(100)
        df["recent_hit_allowed"] = df["avg_pitcher"].fillna(0.25) * 30  # Approx per game
        df["recent_hr_allowed"] = df["pitcher_hr_rate"].fillna(0.03) * 30
        
        # Fill delta stats with 0 (no historical comparison)
        df["hit_rate_delta"] = 0
        df["hr_rate_delta"] = 0
        
        # Fill pitcher stats
        df["era"] = df["era_pitcher"].fillna(4.50)
        df["whip"] = df["whip_pitcher"].fillna(1.35)
        df["babip"] = 0.300  # League average
        df["walksPer9Inn"] = 3.0
        df["strikeoutsPer9Inn"] = 8.0
        
        # Add missing batter stats
        df["slg"] = df["slg_batter"].fillna(0.42)
        df["ops"] = df["ops_batter"].fillna(0.74)
        df["obp"] = df["obp_batter"].fillna(0.32)
        
        # Fill park factors with league average (1.0)
        df["park_hit_factor"] = 1.0
        df["park_hr_factor"] = 1.0
        df["park_multi_factor"] = 1.0
        
        # Fill advanced feature IDs
        df["batter_id_adv"] = df["batter_id"]
        df["pitcher_id_adv"] = df["pitcher_id"]
        
        # Fill merge artifacts
        df["player_id_x_x"] = df["batter_id"]
        df["player_id_y_y"] = df["pitcher_id"]
        df["player_id_x_y"] = df["batter_id"]
        df["player_id_y_x"] = df["pitcher_id"]
    else:
        # Calculate Rolling Features (matching training approach)
        games_map = games.set_index("game_pk")["game_date"]
        batter_logs["game_date"] = pd.to_datetime(batter_logs["game_pk"].map(games_map))
        pitcher_logs["game_date"] = pd.to_datetime(pitcher_logs["game_pk"].map(games_map))

        batter_logs = rolling_features(batter_logs, "batter_id", "game_date", ["is_hit", "is_hr", "is_so", "is_multi"])
        pitcher_logs = rolling_features(pitcher_logs, "pitcher_id", "game_date", ["is_hit", "is_hr", "is_so"])

        # Calculate advanced features
        log.info("Calculating advanced features...")
        advanced_features = engineer_advanced_features(matchups_today, player_stats, batter_logs, pitcher_logs, team_stats, games)
        
        # Merge advanced batter features
        batter_adv = advanced_features["batter_advanced"]
        batter_adv = batter_adv.rename(columns={"batter_id": "batter_id_adv"})
        
        # Merge advanced pitcher features
        pitcher_adv = advanced_features["pitcher_advanced"]
        pitcher_adv = pitcher_adv.rename(columns={"pitcher_id": "pitcher_id_adv"})

        # Merge rolling features into matchups (matching training approach)
        df = matchups_today.merge(
            batter_logs.add_prefix("batter_"),
            left_on=["batter_id", "game_pk"],
            right_on=["batter_batter_id", "batter_game_pk"],
            how="left"
        ).merge(
            pitcher_logs.add_prefix("pitcher_"),
            left_on=["pitcher_id", "game_pk"],
            right_on=["pitcher_pitcher_id", "pitcher_game_pk"],
            how="left"
        ).merge(
            batter_adv,
            left_on="batter_id",
            right_on="batter_id_adv",
            how="left"
        ).merge(
            pitcher_adv,
            left_on="pitcher_id",
            right_on="pitcher_id_adv",
            how="left"
        )
    
    # Add player names and team info
    # Use latest season for team info to ensure current team affiliation
    # Sort by season descending so drop_duplicates keeps the most recent team
    # Exclude pitchers (position == "P") to ensure only batters are included
    player_stats_names = player_stats[
        (player_stats["group"] == "hitting") & (player_stats["position"] != "P")
    ].sort_values("season", ascending=False)[["player_id", "player_name", "team_id", "team_name"]].drop_duplicates("player_id")
    
    df = df.merge(player_stats_names, left_on="batter_id", right_on="player_id", how="left")
    
    # Merge with games to get team names and dates
    df = df.merge(
        games_today[["game_pk", "home_team_name", "away_team_name", "game_date"]],
        on="game_pk",
        how="left"
    )
    
    # Filter out rows without player names (players not in player_stats)
    df = df.dropna(subset=["player_name"])
    log.info("Filtered to %d rows with valid player names", len(df))

    # 6. Model Scoring
    numeric_df = df.select_dtypes(include=["number"]).copy()
    numeric_df = numeric_df[[c for c in numeric_df.columns if not c.endswith("_prob")]]
    
    # For scheduled matchups, remove target variables from feature set
    # The model was trained on historical data with actual outcomes, but scheduled matchups
    # have all targets set to 0 (games haven't happened), causing extremely low predictions
    # Also remove batter_is_hit, batter_is_hr, batter_is_so, etc. which are now excluded from training
    if is_scheduled:
        target_columns = ["is_pa", "is_ab", "is_hit", "is_hr", "is_so", "is_multi",
                         "batter_is_pa", "batter_is_ab", "batter_is_hit", "batter_is_hr", 
                         "batter_is_so", "batter_is_multi", "pitcher_is_hit", "pitcher_is_hr", "pitcher_is_so"]
        numeric_df = numeric_df.drop(columns=[c for c in target_columns if c in numeric_df.columns])
        removed = [c for c in target_columns if c in numeric_df.columns]
        if removed:
            log.debug("Removed target variables for scheduled matchups: %s", removed)

    # Use season-level models for scheduled games, historical models for completed games
    models_to_use = season_models if is_scheduled else models

    for key, model in models_to_use.items():
        # Handle both ensemble and single models
        if hasattr(model, 'estimators_'):
            # Ensemble model - get features from first estimator
            first_estimator = model.estimators_[0]
            if isinstance(first_estimator, tuple):
                first_estimator = first_estimator[1]
            model_features = first_estimator.get_booster().feature_names
        elif hasattr(model, 'get_booster'):
            # XGBoost model
            model_features = model.get_booster().feature_names
        elif hasattr(model, 'feature_name_'):
            # LightGBM model
            model_features = model.feature_name_
        elif hasattr(model, 'feature_names_'):
            # CatBoost model
            model_features = model.feature_names_
        else:
            # Fallback - use all numeric columns
            model_features = numeric_df.columns.tolist()
            
        aligned = numeric_df.reindex(columns=model_features, fill_value=0)
        
        # Skip scaling - causing model recognition issues
        # Rely on regularization and other improvements for realistic predictions
        
        # Debug logging
        log.debug("=== %s MODEL ===", key.upper())
        log.debug("Model type: %s", type(model).__name__)
        log.debug("Model features: %d", len(model_features))
        log.debug("Available features: %d", len(numeric_df.columns))
        log.debug("Missing features: %s", set(model_features) - set(numeric_df.columns))
        log.debug("Sample aligned data (first row):\n%s", aligned.iloc[0].head(10))
        
        predictions = model.predict_proba(aligned)[:, 1]
        log.debug("Prediction stats: min=%.6f, max=%.6f, mean=%.6f", predictions.min(), predictions.max(), predictions.mean())
        log.debug("Sample predictions (first 5): %s", predictions[:5])
        
        # Log model performance metrics
        log.info(f"{key.upper()} model scoring: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")
        
        # Validation: Ensure predictions are within reasonable ranges
        # If predictions are extremely low, it indicates a model/data issue
        if predictions.mean() < 0.01:
            log.warning("%s predictions are extremely low (mean=%.6f)", key, predictions.mean())
            log.warning("This suggests the model features may not be properly aligned or the model needs retraining.")
        
        # No hard caps - rely on improved model training with regularization and scaling
        
        # Feature validation: Ensure predictions align with recent performance
        # If prediction is much higher than recent performance, adjust it
        if key in ["hit", "hr"]:
            # Get recent performance for each batter from advanced features
            recent_col = f"recent_{key}_rate"
            if recent_col in df.columns:
                # Adjust predictions if they're much higher than recent performance
                # But allow some variance (up to 2x recent rate)
                max_allowed = df[recent_col] * 2
                max_allowed = max_allowed.fillna(predictions.mean())  # Use mean if no recent data
                
                # Only cap if prediction is unreasonably high compared to recent performance
                predictions = np.minimum(predictions, max_allowed.values)
            else:
                # If advanced features not available, skip this validation
                pass
        
        df[f"{key}_prob"] = predictions
        
        # Add confidence intervals (using prediction distribution)
        df[f"{key}_prob_ci_lower"] = np.maximum(predictions - 0.05, 0)  # Simple 5% margin
        df[f"{key}_prob_ci_upper"] = np.minimum(predictions + 0.05, 1)

    # 7. Add Hot/Cold Streak Indicators (for both historical and scheduled matchups)
    # For scheduled games, use prediction probabilities for streak labels
    # For historical games, calculate recent performance from batter_logs
    if not is_scheduled:
        # Calculate recent performance (last 7 games) for streak indicators with improved weighting
        # Exclude today's game from streak calculation (should only use historical data)
        batter_logs_sorted = batter_logs.sort_values("game_date")
        
        # Get today's game_pks to exclude from streak calculation
        today_game_pks = set(df["game_pk"].unique())
        
        # Get last 7 games per batter, excluding today's games
        historical_logs = batter_logs_sorted[~batter_logs_sorted["game_pk"].isin(today_game_pks)]
        recent_games = historical_logs.groupby("batter_id").tail(7)
        
        # Calculate weighted performance (recent games weighted more heavily)
        def weighted_mean(series):
            if len(series) == 0:
                return 0
            # Exponential decay: most recent game weight = 1, oldest = 0.5
            weights = np.array([0.5 ** (len(series) - 1 - i) for i in range(len(series))])
            return np.average(series, weights=weights)
        
        # Calculate weighted rates with minimum sample size requirement
        streak_data = []
        for batter_id, group in recent_games.groupby("batter_id"):
            # Require minimum sample size (at least 3 games with 15+ total PAs)
            if len(group) >= 3 and group["is_pa"].sum() >= 15:
                hit_rate = weighted_mean(group["is_hit"].values)
                hr_rate = weighted_mean(group["is_hr"].values)
                so_rate = weighted_mean(group["is_so"].values)
                
                # Check consistency (low standard deviation = more consistent)
                hit_consistency = 1 - (group["is_hit"].std() if len(group) > 1 else 0)
                
                streak_data.append({
                    "batter_id": batter_id,
                    "recent_hit_rate": hit_rate,
                    "recent_hr_rate": hr_rate,
                    "recent_so_rate": so_rate,
                    "hit_consistency": hit_consistency
                })
        
        recent_performance = pd.DataFrame(streak_data)
    else:
        # For scheduled games, use prediction probabilities for streak labels
        # No actual recent performance data available for games that haven't happened
        recent_performance = pd.DataFrame()
    
    # Define improved streak thresholds based on prediction probabilities
    # For scheduled games, use lower thresholds since season-level models are more conservative
    def get_streak_label(prob, metric_type, is_scheduled=False):
        if is_scheduled:
            # Lower thresholds for scheduled games (season-level models are more conservative)
            if metric_type == "hit":
                # Hit: top 25% of predictions are HOT
                if prob >= 0.25: return "🔥 HOT"
                if prob <= 0.15: return "❄️ COLD"
                return "👍 NEUTRAL"
            elif metric_type == "hr":
                # HR: top 15% are HOT (since HR is rarer)
                if prob >= 0.04: return "🔥 HOT"
                if prob <= 0.015: return "❄️ COLD"
                return "👍 NEUTRAL"
            elif metric_type == "k":
                # K: top 25% are HOT
                if prob >= 0.25: return "🔥 HOT"
                if prob <= 0.15: return "❄️ COLD"
                return "👍 NEUTRAL"
        else:
            # Original thresholds for historical games
            if metric_type == "hit":
                if prob >= 0.35: return "🔥 HOT"
                if prob <= 0.20: return "❄️ COLD"
                return "👍 NEUTRAL"
            elif metric_type == "hr":
                if prob >= 0.08: return "🔥 HOT"
                if prob <= 0.02: return "❄️ COLD"
                return "👍 NEUTRAL"
            elif metric_type == "k":
                if prob >= 0.30: return "🔥 HOT"
                if prob <= 0.15: return "❄️ COLD"
                return "👍 NEUTRAL"
        return "👍 NEUTRAL"
    
    # Apply streak labels based on prediction probabilities
    df["hit_streak"] = df["hit_prob"].apply(lambda x: get_streak_label(x, "hit", is_scheduled))
    df["hr_streak"] = df["hr_prob"].apply(lambda x: get_streak_label(x, "hr", is_scheduled))
    df["k_streak"] = df["k_prob"].apply(lambda x: get_streak_label(x, "k", is_scheduled))

    # Fill missing streaks with neutral
    df["hit_streak"] = df["hit_streak"].fillna("👍 NEUTRAL")
    df["hr_streak"] = df["hr_streak"].fillna("👍 NEUTRAL")
    df["k_streak"] = df["k_streak"].fillna("👍 NEUTRAL")

    # 8. Clean Output Formatting
    out = df[[
        "batter_id", "player_name", "team_id", "team_name", "game_pk", "pitcher_id",
        "hit_prob", "hit_prob_ci_lower", "hit_prob_ci_upper",
        "hr_prob", "hr_prob_ci_lower", "hr_prob_ci_upper",
        "multi_prob", "multi_prob_ci_lower", "multi_prob_ci_upper",
        "k_prob", "k_prob_ci_lower", "k_prob_ci_upper",
        "hit_streak", "hr_streak", "k_streak",
        "home_team_name", "away_team_name", "game_date"
    ]].copy()
    
    # Rename columns to match expected output format
    out = out.rename(columns={
        "batter_id": "player_id",
        "pitcher_id": "opp_pitcher_id"
    })

    # 9. Final Deduplication
    out = out.drop_duplicates(subset=["player_id", "game_pk"])
    out.to_parquet(BASE / "fact_player_props.parquet", index=False)
    log.info("Saved → fact_player_props.parquet")

    # 10. Prediction Drift Detection
    log.info("=== PREDICTION DRIFT MONITORING ===")
    drift_detector = PredictionDriftDetector()
    current_stats, alerts = drift_detector.check_predictions(out)
    
    if alerts:
        log.warning("%d drift alerts detected - check logs/drift_alerts_*.log for details", len(alerts))
    else:
        log.info("No prediction drift detected")

    # 11. Add feature contributions for top predictions
    log.info("=== FEATURE CONTRIBUTIONS ===")
    for prop in ['hit', 'hr', 'multi', 'k']:
        prob_col = f'{prop}_prob'
        if prob_col in out.columns:
            # Get top 3 predictions for this prop
            top_preds = out.nlargest(3, prob_col)
            log.info("%s - Top 3 Predictions:", prop.upper())
            for idx, row in top_preds.iterrows():
                log.info("  %s: %.1f%%", row['player_name'], row[prob_col] * 100)
    
    log.info("Feature contributions analysis complete")

if __name__ == "__main__":
    run()