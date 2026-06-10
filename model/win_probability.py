"""
win_probability.py — Hybrid XGBoost win-probability model.
Trains two models:
  • Long-term (all seasons)
  • Short-term (last 3 seasons)
Blends predictions: 70% short-term, 30% long-term
Outputs: fact_win_probability.parquet
"""

import argparse
import logging
import pickle
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import log_loss

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

# Import time-series cross-validation
try:
    from utils.time_series_cv import TimeSeriesSplit, time_series_cv_score, compare_cv_methods
    TS_CV_AVAILABLE = True
except ImportError:
    TS_CV_AVAILABLE = False
    log.warning("Time-series CV not available - using standard K-fold")

warnings.filterwarnings("ignore")

# ----------------------------------------
# Paths
# ----------------------------------------
NORM_DIR = Path(__file__).parent.parent / "data" / "normalized"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ----------------------------------------
# Feature Lists
# ----------------------------------------
HITTING_COLS = ["avg", "obp", "slg", "ops", "runs", "homeRuns", "strikeOuts", "baseOnBalls"]
PITCHING_COLS = ["era", "whip", "strikeoutsPer9", "walksPer9", "homeRunsPer9"]

# ----------------------------------------
# Load normalized tables
# ----------------------------------------
def load_tables():
    # Only load the exact tables needed for win probability
    needed_files = ["fact_games", "fact_team_stats", "fact_standings", "dim_teams"]
    tables = {}
    for name in needed_files:
        file_path = NORM_DIR / f"{name}.parquet"
        if file_path.exists():
            tables[name] = pd.read_parquet(file_path)
        else:
            log.warning(f"⚠️ Missing required table: {name}.parquet")
    return tables

# ----------------------------------------
# Pivot team stats
# ----------------------------------------
def pivot_stats(team_stats, group):
    cols = HITTING_COLS if group == "hitting" else PITCHING_COLS
    sub = team_stats[team_stats["group"] == group].copy()
    avail = [c for c in cols if c in sub.columns]
    # Convert string values to numeric
    for col in avail:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.groupby("team_id")[avail].mean().reset_index()
    sub.columns = ["team_id"] + [f"{group}_{c}" for c in avail]
    return sub

# ----------------------------------------
# Build feature matrix
# ----------------------------------------
def build_features(games, team_stats, standings, dim_teams):
    hit = pivot_stats(team_stats, "hitting")
    pit = pivot_stats(team_stats, "pitching")

    df = games.copy()

    for side in ("home", "away"):
        df = df.merge(
            hit.add_suffix(f"_{side}").rename(columns={f"team_id_{side}": f"{side}_team_id"}),
            on=f"{side}_team_id",
            how="left"
        )
        df = df.merge(
            pit.add_suffix(f"_{side}").rename(columns={f"team_id_{side}": f"{side}_team_id"}),
            on=f"{side}_team_id",
            how="left"
        )

    # Standings already has team_id - no need to merge with dim_teams
    # Just select the columns we need
    if "team_id" not in standings.columns:
        log.error(f"Standings columns: {standings.columns.tolist()}")
        raise KeyError("standings table missing 'team_id' column")
    
    stand = standings[["team_id", "win_pct", "run_diff"]].copy()
    for side in ("home", "away"):
        df = df.merge(
            stand.rename(columns={
                "team_id": f"{side}_team_id",
                "win_pct": f"{side}_wp",
                "run_diff": f"{side}_rd"
            }),
            on=f"{side}_team_id",
            how="left"
        )

    diffs = {}
    for base in [f"hitting_{c}" for c in HITTING_COLS] + [f"pitching_{c}" for c in PITCHING_COLS]:
        hc, ac = f"{base}_home", f"{base}_away"
        if hc in df.columns and ac in df.columns:
            diffs[f"diff_{base}"] = pd.to_numeric(df[hc], errors="coerce") - pd.to_numeric(df[ac], errors="coerce")

    diffs["diff_win_pct"] = pd.to_numeric(df["home_wp"], errors="coerce") - pd.to_numeric(df["away_wp"], errors="coerce")
    diffs["diff_run_diff"] = pd.to_numeric(df["home_rd"], errors="coerce") - pd.to_numeric(df["away_rd"], errors="coerce")

    return pd.concat([
        df[[
            "game_pk", "game_date",
            "home_team_id", "home_team_name",
            "away_team_id", "away_team_name",
            "home_is_winner"
        ]],
        pd.DataFrame(diffs, index=df.index)
    ], axis=1)

# ----------------------------------------
# Feature selector
# ----------------------------------------
def feat_cols(df):
    return [c for c in df.columns if c.startswith("diff_")]

# ----------------------------------------
# Train hybrid models with ensemble methods
# ----------------------------------------
def train_model(features, model_type="long", tune_hyperparams=False, n_trials=20):
    """Train ensemble model for win probability prediction
    
    Args:
        features: Feature dataframe
        model_type: 'long' for all seasons, 'short' for last 3 seasons
        tune_hyperparams: Whether to run hyperparameter tuning
        n_trials: Number of trials for hyperparameter tuning
    """
    features["game_date"] = pd.to_datetime(features["game_date"], errors="coerce")
    
    if model_type == "long":
        train_data = features.dropna(subset=["home_is_winner"]).copy()
    else:
        # Short-term: last 3 seasons
        max_year = features["game_date"].dt.year.max()
        cutoff_year = max_year - 2
        short = features[features["game_date"].dt.year >= cutoff_year]
        train_data = short.dropna(subset=["home_is_winner"]).copy()
    
    train_data["home_is_winner"] = train_data["home_is_winner"].astype(int)
    
    X = train_data[feat_cols(train_data)].fillna(0)
    y = train_data["home_is_winner"]
    
    # Get hyperparameters
    if tune_hyperparams and TUNING_AVAILABLE:
        log.info(f"Running hyperparameter tuning for {model_type} model...")
        tuner = HyperparameterTuner(n_trials=n_trials, timeout=1800)
        best_params = tuner.optimize_all(X, y, n_folds=5)
        xgb_params = best_params.get('xgboost', get_default_params()['xgboost'])
        lgb_params = best_params.get('lightgbm', get_default_params()['lightgbm'])
        cb_params = best_params.get('catboost', get_default_params()['catboost'])
    else:
        log.info(f"Using default hyperparameters for {model_type} model")
        default_params = get_default_params()
        xgb_params = default_params['xgboost']
        lgb_params = default_params['lightgbm']
        cb_params = default_params['catboost']
    
    # XGBoost model
    xgb_model = xgb.XGBClassifier(**xgb_params)
    
    # LightGBM model
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    
    # CatBoost model
    cb_model = cb.CatBoostClassifier(**cb_params)
    
    # Ensemble with soft voting
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('cb', cb_model)
        ],
        voting='soft'
    )
    
    # Fit ensemble
    ensemble.fit(X, y)
    
    # Fit individual models for feature importance
    xgb_model.fit(X, y)
    lgb_model.fit(X, y)
    cb_model.fit(X, y)
    
    # Extract feature importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'xgb_importance': xgb_model.feature_importances_,
        'lgb_importance': lgb_model.feature_importances_,
        'catboost_importance': cb_model.feature_importances_
    })
    importance_df['avg_importance'] = (
        importance_df['xgb_importance'] + 
        importance_df['lgb_importance'] + 
        importance_df['catboost_importance']
    ) / 3
    importance_df = importance_df.sort_values('avg_importance', ascending=False)
    
    # K-fold cross-validation
    log.info(f"\n=== K-FOLD CROSS-VALIDATION (5 folds) ===")
    
    if TS_CV_AVAILABLE and 'game_date' in train_data.columns:
        log.info("Using time-series cross-validation to prevent data leakage")
        cv_scores_ensemble = time_series_cv_score(ensemble, X, y, date_col='game_date', n_splits=5, scoring='neg_log_loss')
        log.info(f"Ensemble CV Log Loss: {-cv_scores_ensemble['mean']:.4f} (+/- {cv_scores_ensemble['std'] * 2:.4f})")
        
        # Evaluate individual models with time-series CV
        cv_scores_xgb = time_series_cv_score(xgb_model, X, y, date_col='game_date', n_splits=5, scoring='neg_log_loss')
        log.info(f"XGBoost CV Log Loss: {-cv_scores_xgb['mean']:.4f} (+/- {cv_scores_xgb['std'] * 2:.4f})")
        
        cv_scores_lgb = time_series_cv_score(lgb_model, X, y, date_col='game_date', n_splits=5, scoring='neg_log_loss')
        log.info(f"LightGBM CV Log Loss: {-cv_scores_lgb['mean']:.4f} (+/- {cv_scores_lgb['std'] * 2:.4f})")
        
        cv_scores_cb = time_series_cv_score(cb_model, X, y, date_col='game_date', n_splits=5, scoring='neg_log_loss')
        log.info(f"CatBoost CV Log Loss: {-cv_scores_cb['mean']:.4f} (+/- {cv_scores_cb['std'] * 2:.4f})")
    else:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Evaluate ensemble
        cv_scores_ensemble = cross_val_score(ensemble, X, y, cv=kf, scoring='neg_log_loss')
        log.info(f"Ensemble CV Log Loss: {-cv_scores_ensemble.mean():.4f} (+/- {cv_scores_ensemble.std() * 2:.4f})")
        
        # Evaluate individual models
        cv_scores_xgb = cross_val_score(xgb_model, X, y, cv=kf, scoring='neg_log_loss')
        log.info(f"XGBoost CV Log Loss: {-cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std() * 2:.4f})")
        
        cv_scores_lgb = cross_val_score(lgb_model, X, y, cv=kf, scoring='neg_log_loss')
        log.info(f"LightGBM CV Log Loss: {-cv_scores_lgb.mean():.4f} (+/- {cv_scores_lgb.std() * 2:.4f})")
        
        cv_scores_cb = cross_val_score(cb_model, X, y, cv=kf, scoring='neg_log_loss')
        log.info(f"CatBoost CV Log Loss: {-cv_scores_cb.mean():.4f} (+/- {cv_scores_cb.std() * 2:.4f})")
    
    log.info(f"Trained {model_type} ensemble model: {len(train_data)} games, {len(X.columns)} features")
    
    return ensemble, xgb_model, lgb_model, cb_model, importance_df


# ----------------------------------------
# Train hybrid models (legacy - kept for compatibility)
# ----------------------------------------
def train(features):
    features["game_date"] = pd.to_datetime(features["game_date"], errors="coerce")

    # Long-term model
    train_long = features.dropna(subset=["home_is_winner"]).copy()
    train_long["home_is_winner"] = train_long["home_is_winner"].astype(int)

    X_long = train_long[feat_cols(train_long)].fillna(0)
    y_long = train_long["home_is_winner"]

    model_long = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )
    model_long.fit(X_long, y_long)

    # Short-term model (last 3 seasons)
    max_year = features["game_date"].dt.year.max()
    cutoff_year = max_year - 2

    short = features[features["game_date"].dt.year >= cutoff_year]
    train_short = short.dropna(subset=["home_is_winner"]).copy()
    train_short["home_is_winner"] = train_short["home_is_winner"].astype(int)

    X_short = train_short[feat_cols(train_short)].fillna(0)
    y_short = train_short["home_is_winner"]

    model_short = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )
    model_short.fit(X_short, y_short)

    # Save models
    pickle.dump(model_long, open(MODEL_DIR / "win_prob_model_long.pkl", "wb"))
    pickle.dump(model_short, open(MODEL_DIR / "win_prob_model_short.pkl", "wb"))

    log.info(
        "Hybrid win-probability models trained → long: %d games, short: %d games",
        len(train_long), len(train_short)
    )

    return model_long, model_short

# ----------------------------------------
# RUN FUNCTION
# ----------------------------------------
def run(game_date=None, tune_hyperparams=False, n_trials=20):
    if game_date is None:
        game_date = str(date.today())

    tables = load_tables()
    feats = build_features(
        tables["fact_games"],
        tables["fact_team_stats"],
        tables["fact_standings"],
        tables["dim_teams"]
    )

    # Try to load existing ensemble models
    ensemble_long_path = MODEL_DIR / "win_prob_ensemble_long.pkl"
    ensemble_short_path = MODEL_DIR / "win_prob_ensemble_short.pkl"

    if ensemble_long_path.exists() and ensemble_short_path.exists() and not tune_hyperparams:
        log.info("Loading existing ensemble models...")
        ensemble_long = pickle.load(open(ensemble_long_path, "rb"))
        ensemble_short = pickle.load(open(ensemble_short_path, "rb"))
    else:
        log.info("Training new ensemble models...")
        
        # Train long-term ensemble
        ensemble_long, xgb_long, lgb_long, cb_long, importance_long = train_model(
            feats, "long", tune_hyperparams=tune_hyperparams, n_trials=n_trials
        )
        
        # Save ensemble and individual models
        pickle.dump(ensemble_long, open(ensemble_long_path, "wb"))
        pickle.dump(xgb_long, open(MODEL_DIR / "win_prob_xgb_long.pkl", "wb"))
        pickle.dump(lgb_long, open(MODEL_DIR / "win_prob_lgb_long.pkl", "wb"))
        pickle.dump(cb_long, open(MODEL_DIR / "win_prob_cb_long.pkl", "wb"))
        importance_long.to_parquet(MODEL_DIR / "win_prob_long_feature_importance.parquet", index=False)
        log.info("Saved → win_prob_ensemble_long.pkl")
        log.info("Saved → win_prob_long_feature_importance.parquet")
        
        # Train short-term ensemble
        ensemble_short, xgb_short, lgb_short, cb_short, importance_short = train_model(
            feats, "short", tune_hyperparams=tune_hyperparams, n_trials=n_trials
        )
        
        # Save ensemble and individual models
        pickle.dump(ensemble_short, open(ensemble_short_path, "wb"))
        pickle.dump(xgb_short, open(MODEL_DIR / "win_prob_xgb_short.pkl", "wb"))
        pickle.dump(lgb_short, open(MODEL_DIR / "win_prob_lgb_short.pkl", "wb"))
        pickle.dump(cb_short, open(MODEL_DIR / "win_prob_cb_short.pkl", "wb"))
        importance_short.to_parquet(MODEL_DIR / "win_prob_short_feature_importance.parquet", index=False)
        log.info("Saved → win_prob_ensemble_short.pkl")
        log.info("Saved → win_prob_short_feature_importance.parquet")
        
        # Display top features
        log.info("\n=== TOP 10 FEATURE IMPORTANCE (LONG-TERM) ===")
        for _, row in importance_long.head(10).iterrows():
            log.info(f"{row['feature']}: {row['avg_importance']:.4f}")
        
        log.info("\n=== TOP 10 FEATURE IMPORTANCE (SHORT-TERM) ===")
        for _, row in importance_short.head(10).iterrows():
            log.info(f"{row['feature']}: {row['avg_importance']:.4f}")

    X = feats[feat_cols(feats)].fillna(0)

    probs_long = ensemble_long.predict_proba(X)[:, 1]
    probs_short = ensemble_short.predict_proba(X)[:, 1]

    # Hybrid blend
    probs = 0.7 * probs_short + 0.3 * probs_long

    out = feats[[
        "game_pk", "game_date",
        "home_team_id", "home_team_name",
        "away_team_id", "away_team_name",
        "home_is_winner"
    ]].copy()

    out["home_win_prob"] = np.round(probs, 4)
    out["away_win_prob"] = np.round(1 - probs, 4)
    out["predicted_winner"] = np.where(
        probs >= 0.5,
        out["home_team_name"],
        out["away_team_name"]
    )

    out["confidence_label"] = pd.cut(
        np.abs(probs - 0.5),
        bins=[0, .10, .20, 1.0],
        labels=["Toss-Up", "Medium", "High"]
    )

    done = out["home_is_winner"].notna()
    out.loc[done, "prediction_correct"] = (
        (out.loc[done, "home_win_prob"] >= 0.5) ==
        out.loc[done, "home_is_winner"].astype(bool)
    )

    # Deduplicate BEFORE saving
    out = out.sort_values("game_pk").groupby("game_pk").tail(1)

    out.to_parquet(NORM_DIR / "fact_win_probability.parquet", index=False)
    log.info("Predictions written → fact_win_probability.parquet")

    # Log today's games
    today = out[out["game_date"] == game_date]
    today = today.sort_values("game_pk").groupby("game_pk").tail(1)

    for _, r in today.iterrows():
        log.info(
            "  %s @ %s  → home %.1f%%  (pick: %s)",
            r["away_team_name"], r["home_team_name"],
            r["home_win_prob"] * 100,
            r["predicted_winner"]
        )

# ----------------------------------------
# CLI
# ----------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=str(date.today()))
    p.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    p.add_argument("--trials", type=int, default=20, help="Number of hyperparameter tuning trials")
    args = p.parse_args()
    run(args.date, tune_hyperparams=args.tune, n_trials=args.trials)
