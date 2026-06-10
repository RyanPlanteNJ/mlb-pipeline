"""
performance_tracker.py - Track model performance over time
Logs prediction accuracy and model metrics
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
import json
import logging

log = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent / "data" / "normalized"
TRACKING_DIR = Path(__file__).parent.parent / "data" / "tracking"
TRACKING_DIR.mkdir(exist_ok=True)


def track_batter_predictions():
    """Track batter prop predictions vs actual results"""
    try:
        props = pd.read_parquet(BASE / "fact_player_props.parquet")
        matchups = pd.read_parquet(BASE / "fact_batter_pitcher_matchups.parquet")
        
        # Get completed games
        games = pd.read_parquet(BASE / "fact_games.parquet")
        games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.date
        
        # Filter for completed games
        completed_games = games[games["status"] == "Final"]
        
        if completed_games.empty:
            log.info("No completed games to track")
            return
        
        # Merge predictions with actual results
        tracked = props.merge(
            matchups[["game_pk", "batter_id", "is_hit", "is_hr", "is_so"]],
            left_on=["game_pk", "player_id"],
            right_on=["game_pk", "batter_id"],
            how="inner"
        )
        
        # Calculate accuracy metrics
        metrics = {}
        
        for prop in ["hit", "hr", "k"]:
            pred_col = f"{prop}_prob"
            actual_col = f"is_{prop}"
            
            if pred_col in tracked.columns and actual_col in tracked.columns:
                # Brier score (mean squared error of probabilities)
                brier_score = ((tracked[pred_col] - tracked[actual_col]) ** 2).mean()
                
                # Log loss
                eps = 1e-15
                log_loss = -(
                    tracked[actual_col] * np.log(tracked[pred_col] + eps) +
                    (1 - tracked[actual_col]) * np.log(1 - tracked[pred_col] + eps)
                ).mean()
                
                # Calibration (predicted vs actual rate)
                predicted_rate = tracked[pred_col].mean()
                actual_rate = tracked[actual_col].mean()
                calibration_error = abs(predicted_rate - actual_rate)
                
                metrics[prop] = {
                    "brier_score": float(brier_score),
                    "log_loss": float(log_loss),
                    "predicted_rate": float(predicted_rate),
                    "actual_rate": float(actual_rate),
                    "calibration_error": float(calibration_error),
                    "sample_size": len(tracked)
                }
        
        # Save metrics
        today = date.today()
        metrics_file = TRACKING_DIR / f"performance_{today}.json"
        
        with open(metrics_file, "w") as f:
            json.dump({
                "date": str(today),
                "metrics": metrics
            }, f, indent=2)
        
        log.info(f"Performance metrics saved to {metrics_file}")
        
        # Log summary
        for prop, m in metrics.items():
            log.info(f"{prop.upper()}: Brier={m['brier_score']:.4f}, Calibration Error={m['calibration_error']:.4f}")
        
        return metrics
        
    except Exception as e:
        log.error(f"Error tracking performance: {e}")
        return None


def track_win_probability():
    """Track win probability predictions vs actual results"""
    try:
        wp = pd.read_parquet(BASE / "fact_win_probability.parquet")
        
        # Filter for completed games with predictions
        completed = wp[wp["home_is_winner"].notna()]
        
        if completed.empty:
            log.info("No completed games to track win probability")
            return
        
        # Calculate accuracy
        accuracy = (completed["prediction_correct"] == True).mean()
        
        # Brier score
        brier_score = ((completed["home_win_prob"] - completed["home_is_winner"].astype(int)) ** 2).mean()
        
        metrics = {
            "accuracy": float(accuracy),
            "brier_score": float(brier_score),
            "sample_size": len(completed)
        }
        
        # Save metrics
        today = date.today()
        metrics_file = TRACKING_DIR / f"win_prob_performance_{today}.json"
        
        with open(metrics_file, "w") as f:
            json.dump({
                "date": str(today),
                "metrics": metrics
            }, f, indent=2)
        
        log.info(f"Win probability metrics saved: Accuracy={accuracy:.4f}, Brier={brier_score:.4f}")
        
        return metrics
        
    except Exception as e:
        log.error(f"Error tracking win probability: {e}")
        return None


def get_historical_performance(days=30):
    """Get performance metrics for the last N days"""
    metrics_history = []
    
    for file_path in TRACKING_DIR.glob("performance_*.json"):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                metrics_history.append(data)
        except:
            continue
    
    if not metrics_history:
        return None
    
    df = pd.DataFrame(metrics_history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(days)
    
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    track_batter_predictions()
    track_win_probability()
