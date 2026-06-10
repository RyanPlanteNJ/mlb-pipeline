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

from utils.advanced_features import engineer_advanced_features
from utils.model_versioning import ModelVersionTracker
from model.feature_builder import rolling_features, build_features_historical, get_explicit_features

# Setup logging first
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# Import hyperparameter tuning
try:
    from utils.hyperparameter_tuning import HyperparameterTuner, get_default_params
    TUNING_AVAILABLE = True
except ImportError:
    TUNING_AVAILABLE = False
    log.warning("Hyperparameter tuning not available - using default params")
    # Define fallback default params
    def get_default_params():
        return {
            'xgboost': {
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
            },
            'lightgbm': {
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
            },
            'catboost': {
                "iterations": 300,
                "depth": 5,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3,
                "random_seed": 42,
                "verbose": False,
            }
        }

# Import time-series cross-validation
try:
    from utils.time_series_cv import time_series_cv_score
    TS_CV_AVAILABLE = True
except ImportError:
    TS_CV_AVAILABLE = False
    log.warning("Time-series CV not available - using standard K-fold")

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


def train_model(df, target_col, feature_names=None, class_weight_ratio=None, tune_hyperparams=False, n_trials=20):
    """
    Train ensemble model with explicit features and optional class weighting.
    
    Args:
        df: Feature matrix
        target_col: Target column name
        feature_names: Explicit list of feature names to use (None = use all numeric)
        class_weight_ratio: Ratio of negative to positive examples for class weighting (for skewed targets)
        tune_hyperparams: Whether to run hyperparameter tuning
        n_trials: Number of trials for hyperparameter tuning
    """
    # Select numeric columns and target
    df_numeric = df.select_dtypes(include=["number"]).copy()
    
    if target_col not in df_numeric.columns:
        log.error("Target column %s not found in data", target_col)
        return None, None, None, None, None
    
    # Remove NaN rows in target
    df_numeric = df_numeric.dropna(subset=[target_col])
    
    # Clip extreme values in target
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
    
    # Calculate class weights for imbalanced targets
    scale_pos_weight = None
    if class_weight_ratio is not None:
        # scale_pos_weight = negative / positive
        positive_count = (y == 1).sum()
        negative_count = (y == 0).sum()
        if positive_count > 0:
            scale_pos_weight = negative_count / positive_count
            log.info("Class weight ratio for %s: %.2f (pos=%d, neg=%d)", 
                    target_col, scale_pos_weight, positive_count, negative_count)
    
    # Train-test split (temporal order preserved, no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Get hyperparameters
    if tune_hyperparams and TUNING_AVAILABLE:
        log.info(f"Running hyperparameter tuning for {target_col}...")
        tuner = HyperparameterTuner(n_trials=n_trials, timeout=1800)
        best_params = tuner.optimize_all(X_train, y_train, n_folds=5)
        xgb_params = best_params.get('xgboost', get_default_params()['xgboost'])
        lgb_params = best_params.get('lightgbm', get_default_params()['lightgbm'])
        cb_params = best_params.get('catboost', get_default_params()['catboost'])
    else:
        log.info(f"Using default hyperparameters for {target_col}")
        default_params = get_default_params()
        xgb_params = default_params['xgboost']
        lgb_params = default_params['lightgbm']
        cb_params = default_params['catboost']
    
    # XGBoost with optional class weighting
    if scale_pos_weight is not None:
        xgb_params["scale_pos_weight"] = scale_pos_weight
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    
    # LightGBM with optional class weighting
    if scale_pos_weight is not None:
        lgb_params["is_unbalance"] = True
    
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    
    # CatBoost
    cb_params["auto_class_weights"] = "Balanced" if scale_pos_weight is not None else None
    cb_model = cb.CatBoostClassifier(**cb_params)
    
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
    
    # Train individual models for feature importance
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
    
    if TS_CV_AVAILABLE and 'game_date' in df_numeric.columns:
        log.info("Using time-series cross-validation to prevent data leakage")
        cv_scores_ensemble = time_series_cv_score(ensemble, X_train, y_train, n_splits=5, scoring='neg_log_loss')
        log.info("Ensemble CV Log Loss: %.4f (+/- %.4f)", -cv_scores_ensemble['mean'], cv_scores_ensemble['std'] * 2)
        
        cv_scores_xgb = time_series_cv_score(xgb_model, X_train, y_train, n_splits=5, scoring='neg_log_loss')
        log.info("XGBoost CV Log Loss: %.4f (+/- %.4f)", -cv_scores_xgb['mean'], cv_scores_xgb['std'] * 2)
        
        cv_scores_lgb = time_series_cv_score(lgb_model, X_train, y_train, n_splits=5, scoring='neg_log_loss')
        log.info("LightGBM CV Log Loss: %.4f (+/- %.4f)", -cv_scores_lgb['mean'], cv_scores_lgb['std'] * 2)
        
        cv_scores_cb = time_series_cv_score(cb_model, X_train, y_train, n_splits=5, scoring='neg_log_loss')
        log.info("CatBoost CV Log Loss: %.4f (+/- %.4f)", -cv_scores_cb['mean'], cv_scores_cb['std'] * 2)
        
        cv_results = {
            "ensemble": float(-cv_scores_ensemble['mean']),
            "xgb": float(-cv_scores_xgb['mean']),
            "lgb": float(-cv_scores_lgb['mean']),
            "catboost": float(-cv_scores_cb['mean']),
        }
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_scores_ensemble = cross_val_score(ensemble, X_train, y_train, cv=skf, scoring="neg_log_loss")
        log.info("Ensemble CV Log Loss: %.4f (+/- %.4f)", -cv_scores_ensemble.mean(), cv_scores_ensemble.std() * 2)
        
        cv_scores_xgb = cross_val_score(xgb_model, X_train, y_train, cv=skf, scoring="neg_log_loss")
        log.info("XGBoost CV Log Loss: %.4f (+/- %.4f)", -cv_scores_xgb.mean(), cv_scores_xgb.std() * 2)
        
        cv_scores_lgb = cross_val_score(lgb_model, X_train, y_train, cv=skf, scoring="neg_log_loss")
        log.info("LightGBM CV Log Loss: %.4f (+/- %.4f)", -cv_scores_lgb.mean(), cv_scores_lgb.std() * 2)
        
        cv_scores_cb = cross_val_score(cb_model, X_train, y_train, cv=skf, scoring="neg_log_loss")
        log.info("CatBoost CV Log Loss: %.4f (+/- %.4f)", -cv_scores_cb.mean(), cv_scores_cb.std() * 2)
        
        cv_results = {
            "ensemble": float(-cv_scores_ensemble.mean()),
            "xgb": float(-cv_scores_xgb.mean()),
            "lgb": float(-cv_scores_lgb.mean()),
            "catboost": float(-cv_scores_cb.mean()),
        }
    
    log.info("Trained ensemble: %d samples, %d features", len(X), len(X.columns))
    
    return ensemble, xgb_model, lgb_model, cb_model, feature_importance, cv_results


def run(tune_hyperparams=False, n_trials=20):
    """Train hybrid long-term and short-term models for batter props.
    
    Args:
        tune_hyperparams: Whether to run hyperparameter tuning
        n_trials: Number of trials for hyperparameter tuning
    """
    # Load data
    matchups = pd.read_parquet(BASE / "fact_batter_pitcher_matchups.parquet")
    batter_logs = pd.read_parquet(BASE / "fact_batter_game_logs.parquet")
    pitcher_logs = pd.read_parquet(BASE / "fact_pitcher_game_logs.parquet")
    games = pd.read_parquet(BASE / "fact_games.parquet").set_index("game_pk")
    player_stats = pd.read_parquet(BASE / "fact_player_stats.parquet")
    team_stats = pd.read_parquet(BASE / "fact_team_stats.parquet")
    
    # Compute rolling features
    log.info("Computing rolling features...")
    batter_logs["game_date"] = pd.to_datetime(batter_logs["game_pk"].map(games["game_date"]))
    pitcher_logs["game_date"] = pd.to_datetime(pitcher_logs["game_pk"].map(games["game_date"]))
    
    batter_logs = rolling_features(batter_logs, "batter_id", "game_date", ["is_hit", "is_hr", "is_so", "is_multi"])
    pitcher_logs = rolling_features(pitcher_logs, "pitcher_id", "game_date", ["is_hit", "is_hr", "is_so"])
    
    # Compute advanced features
    log.info("Computing advanced features...")
    advanced_features = engineer_advanced_features(matchups, player_stats, batter_logs, pitcher_logs, team_stats, games)
    
    # Get explicit feature list
    explicit_features = get_explicit_features(mode="historical")
    
    # Train models
    targets = {
        "batter_prop_hit": ("is_hit", None),
        "batter_prop_hr": ("is_hr", 0.1),  # Class weight for HR (rare event)
        "batter_prop_multi": ("is_multi", None),
        "batter_prop_k": ("is_so", 0.15),  # Class weight for K (strikeout)
    }
    
    version_tracker = ModelVersionTracker()
    
    for key, (target_col, class_weight) in targets.items():
        log.info("=== TRAINING %s (HYBRID) ===", key.upper())
        
        # Build historical features
        df = build_features_historical(
            matchups,
            player_stats,
            batter_logs,
            pitcher_logs,
            team_stats,
            games,
            advanced_features=advanced_features
        )
        
        # Train long-term model
        log.info("Training long-term model...")
        result = train_model(
            df.copy(), target_col, feature_names=explicit_features, 
            class_weight_ratio=class_weight, tune_hyperparams=tune_hyperparams, n_trials=n_trials
        )
        
        if result[0] is None:
            log.error("Failed to train long-term model for %s", key)
            continue
        
        ensemble_long, xgb_long, lgb_long, cb_long, importance_long, cv_long = result
        
        pickle.dump(ensemble_long, open(MODEL_DIR / f"{key}_long_ensemble.pkl", "wb"))
        pickle.dump(xgb_long, open(MODEL_DIR / f"{key}_long_xgb.pkl", "wb"))
        pickle.dump(lgb_long, open(MODEL_DIR / f"{key}_long_lgb.pkl", "wb"))
        pickle.dump(cb_long, open(MODEL_DIR / f"{key}_long_cb.pkl", "wb"))
        importance_long.to_parquet(MODEL_DIR / f"{key}_long_feature_importance.parquet", index=False)
        log.info("Saved → %s_long_ensemble.pkl", key)
        
        # Track with real CV scores
        version_tracker.register_model(
            key, "ensemble", MODEL_DIR / f"{key}_long_ensemble.pkl",
            cv_scores=cv_long,
            feature_importance=importance_long
        )
        
        # Train short-term model (last 3 seasons)
        log.info("Training short-term model...")
        df_short = filter_window(df.copy(), games.copy(), years=3)
        
        if len(df_short) > 100:
            result_short = train_model(
                df_short, target_col, feature_names=explicit_features, 
                class_weight_ratio=class_weight, tune_hyperparams=tune_hyperparams, n_trials=n_trials
            )
            
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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    p.add_argument("--trials", type=int, default=20, help="Number of hyperparameter tuning trials")
    args = p.parse_args()
    run(tune_hyperparams=args.tune, n_trials=args.trials)

