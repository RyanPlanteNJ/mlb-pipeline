import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import brier_score_loss, log_loss
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_versioning import ModelVersionTracker
from model.feature_builder import build_features_scheduled, get_explicit_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent / "data" / "normalized"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def filter_window(df, games, years):
    """Filter data to only include games from the last N years."""
    games["game_date"] = pd.to_datetime(games["game_date"])
    max_year = games["game_date"].dt.year.max()
    cutoff = max_year - (years - 1)
    valid = games[games["game_date"].dt.year >= cutoff].index
    return df[df["game_pk"].isin(valid)]


def train_model(df, target_col, feature_names=None, class_weight_ratio=None):
    """
    Train ensemble model for season-level predictions.
    
    Args:
        df: Feature matrix
        target_col: Target column name
        feature_names: Explicit list of feature names
        class_weight_ratio: Ratio for class weighting (for imbalanced targets)
    """
    # Select numeric columns
    df_numeric = df.select_dtypes(include=["number"]).copy()
    
    if target_col not in df_numeric.columns:
        log.error("Target column %s not found in data", target_col)
        return None, None, None, None, None, None
    
    # Remove NaN rows
    df_numeric = df_numeric.dropna(subset=[target_col])
    
    # Clip extreme values
    df_numeric[target_col] = df_numeric[target_col].clip(upper=13)
    
    # Drop all target columns to prevent data leakage
    target_columns = ["is_pa", "is_ab", "is_hit", "is_hr", "is_so", "is_multi"]
    
    if feature_names is None:
        # Use all numeric columns except targets
        X = df_numeric.drop(columns=target_columns, errors="ignore")
        feature_names = X.columns.tolist()
    else:
        # Use explicit feature list
        X = df_numeric[[c for c in feature_names if c in df_numeric.columns]].copy()
        missing = set(feature_names) - set(X.columns)
        if missing:
            log.warning("Missing %d features: %s", len(missing), list(missing)[:5])
    
    y = df_numeric[target_col]
    
    # Calculate class weights
    scale_pos_weight = None
    if class_weight_ratio is not None:
        positive_count = (y == 1).sum()
        negative_count = (y == 0).sum()
        if positive_count > 0:
            scale_pos_weight = negative_count / positive_count
            log.info("Class weight for %s: %.2f (pos=%d, neg=%d)", 
                    target_col, scale_pos_weight, positive_count, negative_count)
    
    # Train-test split (temporal order preserved)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # XGBoost with optional class weighting
    xgb_params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 0.1,
        "min_child_weight": 1,
        "gamma": 0.0,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    if scale_pos_weight is not None:
        xgb_params["scale_pos_weight"] = scale_pos_weight
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    
    # LightGBM with optional class weighting
    lgb_params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.01,
        "reg_lambda": 0.1,
        "min_child_samples": 20,
        "random_state": 42,
        "verbose": -1,
    }
    if scale_pos_weight is not None:
        lgb_params["is_unbalance"] = True
    
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    
    # CatBoost
    cb_model = cb.CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False,
        auto_class_weights="Balanced" if scale_pos_weight is not None else None,
    )
    
    # Ensemble
    ensemble = VotingClassifier(
        estimators=[
            ("xgb", xgb_model),
            ("lgb", lgb_model),
            ("catboost", cb_model)
        ],
        voting="soft"
    )
    
    ensemble.fit(X_train, y_train)
    
    # Train individual models for importance
    xgb_model.fit(X_train, y_train)
    lgb_model.fit(X_train, y_train)
    cb_model.fit(X_train, y_train)
    
    # Extract feature importance
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "xgb_importance": xgb_model.feature_importances_,
        "lgb_importance": lgb_model.feature_importances_,
        "catboost_importance": cb_model.feature_importances_
    })
    
    feature_importance["avg_importance"] = feature_importance[
        ["xgb_importance", "lgb_importance", "catboost_importance"]
    ].mean(axis=1)
    feature_importance = feature_importance.sort_values("avg_importance", ascending=False)
    
    log.info("=== TOP 10 FEATURE IMPORTANCE ===")
    for idx, row in feature_importance.head(10).iterrows():
        log.info("  %s: %.4f", row["feature"], row["avg_importance"])
    
    # K-fold cross-validation
    log.info("=== K-FOLD CROSS-VALIDATION (5 folds) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores_ensemble = cross_val_score(ensemble, X, y, cv=skf, scoring="neg_log_loss")
    log.info("Ensemble CV Log Loss: %.4f (+/- %.4f)", -cv_scores_ensemble.mean(), cv_scores_ensemble.std() * 2)
    
    cv_scores_xgb = cross_val_score(xgb_model, X, y, cv=skf, scoring="neg_log_loss")
    log.info("XGBoost CV Log Loss: %.4f (+/- %.4f)", -cv_scores_xgb.mean(), cv_scores_xgb.std() * 2)
    
    cv_scores_lgb = cross_val_score(lgb_model, X, y, cv=skf, scoring="neg_log_loss")
    log.info("LightGBM CV Log Loss: %.4f (+/- %.4f)", -cv_scores_lgb.mean(), cv_scores_lgb.std() * 2)
    
    cv_scores_cb = cross_val_score(cb_model, X, y, cv=skf, scoring="neg_log_loss")
    log.info("CatBoost CV Log Loss: %.4f (+/- %.4f)", -cv_scores_cb.mean(), cv_scores_cb.std() * 2)
    
    cv_results = {
        "ensemble": float(-cv_scores_ensemble.mean()),
        "xgb": float(-cv_scores_xgb.mean()),
        "lgb": float(-cv_scores_lgb.mean()),
        "catboost": float(-cv_scores_cb.mean()),
    }
    
    log.info("Trained ensemble: %d samples, %d features", len(X), len(X.columns))
    
    return ensemble, xgb_model, lgb_model, cb_model, feature_importance, cv_results


def run():
    """Train season-level models for pre-game batter prop predictions."""
    # Load data
    matchups = pd.read_parquet(BASE / "fact_batter_pitcher_matchups.parquet")
    games = pd.read_parquet(BASE / "fact_games.parquet").set_index("game_pk")
    player_stats = pd.read_parquet(BASE / "fact_player_stats.parquet")
    team_stats = pd.read_parquet(BASE / "fact_team_stats.parquet")
    
    # Deduplicate team stats (keep most recent season)
    team_stats = team_stats.drop_duplicates(subset=["team_name"], keep="last")
    
    # Get explicit feature list
    explicit_features = get_explicit_features(mode="scheduled")
    
    # Train models
    targets = {
        "batter_prop_hit_season": ("is_hit", None),
        "batter_prop_hr_season": ("is_hr", 0.1),  # Class weight for HR
        "batter_prop_multi_season": ("is_multi", None),
        "batter_prop_k_season": ("is_so", 0.15),  # Class weight for K
    }
    
    version_tracker = ModelVersionTracker()
    
    for key, (target_col, class_weight) in targets.items():
        log.info("=== TRAINING %s (SEASON-LEVEL) ===", key.upper())
        
        # Build season-level features
        df = build_features_scheduled(matchups, player_stats, team_stats, games)
        
        # Train long-term model
        log.info("Training long-term model...")
        result = train_model(df.copy(), target_col, feature_names=explicit_features, class_weight_ratio=class_weight)
        
        if result[0] is None:
            log.error("Failed to train long-term model for %s", key)
            continue
        
        ensemble_long, xgb_long, lgb_long, cb_long, importance_long, cv_long = result
        
        pickle.dump(ensemble_long, open(MODEL_DIR / f"{key}_ensemble.pkl", "wb"))
        pickle.dump(xgb_long, open(MODEL_DIR / f"{key}_xgb.pkl", "wb"))
        pickle.dump(lgb_long, open(MODEL_DIR / f"{key}_lgb.pkl", "wb"))
        pickle.dump(cb_long, open(MODEL_DIR / f"{key}_cb.pkl", "wb"))
        importance_long.to_parquet(MODEL_DIR / f"{key}_feature_importance.parquet", index=False)
        log.info("Saved → %s_ensemble.pkl", key)
        
        # Track with real CV scores
        version_tracker.register_model(
            key, "ensemble", MODEL_DIR / f"{key}_ensemble.pkl",
            cv_scores=cv_long,
            feature_importance=importance_long
        )
        
        # Train short-term model (last 3 seasons)
        log.info("Training short-term model...")
        df_short = filter_window(df.copy(), games.copy(), years=3)
        
        if len(df_short) > 100:
            result_short = train_model(df_short, target_col, feature_names=explicit_features, class_weight_ratio=class_weight)
            
            if result_short[0] is not None:
                ensemble_short, xgb_short, lgb_short, cb_short, importance_short, cv_short = result_short
                
                pickle.dump(ensemble_short, open(MODEL_DIR / f"{key}_short_ensemble.pkl", "wb"))
                pickle.dump(xgb_short, open(MODEL_DIR / f"{key}_short_xgb.pkl", "wb"))
                pickle.dump(lgb_short, open(MODEL_DIR / f"{key}_short_lgb.pkl", "wb"))
                pickle.dump(cb_short, open(MODEL_DIR / f"{key}_short_cb.pkl", "wb"))
                importance_short.to_parquet(MODEL_DIR / f"{key}_short_feature_importance.parquet", index=False)
                log.info("Saved → %s_short_ensemble.pkl", key)
                
                # Track with real CV scores
                version_tracker.register_model(
                    key, "ensemble_short", MODEL_DIR / f"{key}_short_ensemble.pkl",
                    cv_scores=cv_short,
                    feature_importance=importance_short
                )
        else:
            log.warning("Insufficient data for short-term model (%d samples)", len(df_short))


if __name__ == "__main__":
    run()
