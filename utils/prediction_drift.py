"""
prediction_drift.py - Monitor prediction drift and alert on anomalies
Detects when model predictions deviate from expected patterns
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)
BASE = Path(__file__).parent.parent / "data" / "normalized"
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class PredictionDriftDetector:
    """Detects prediction drift and alerts on anomalies"""
    
    def __init__(self, window_days: int = 7, alert_threshold: float = 0.15):
        """
        Initialize drift detector
        
        Args:
            window_days: Number of days to use for baseline comparison
            alert_threshold: Percentage change threshold for alerts (default 15%)
        """
        self.window_days = window_days
        self.alert_threshold = alert_threshold
        self.baseline_stats = {}
        self.load_baseline()
    
    def load_baseline(self):
        """Load baseline statistics from file"""
        baseline_file = LOGS_DIR / "prediction_baseline.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                self.baseline_stats = json.load(f)
    
    def save_baseline(self, stats: Dict):
        """Save baseline statistics to file"""
        baseline_file = LOGS_DIR / "prediction_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def calculate_statistics(self, predictions: pd.DataFrame) -> Dict:
        """Calculate prediction statistics for drift detection"""
        stats = {}
        
        for prop in ['hit', 'hr', 'multi', 'k']:
            if f'{prop}_prob' in predictions.columns:
                probs = predictions[f'{prop}_prob']
                stats[prop] = {
                    'mean': float(probs.mean()),
                    'std': float(probs.std()),
                    'min': float(probs.min()),
                    'max': float(probs.max()),
                    'median': float(probs.median()),
                    'q25': float(probs.quantile(0.25)),
                    'q75': float(probs.quantile(0.75))
                }
        
        stats['timestamp'] = datetime.now().isoformat()
        stats['sample_size'] = len(predictions)
        
        return stats
    
    def detect_drift(self, current_stats: Dict) -> List[Dict]:
        """
        Detect prediction drift compared to baseline
        
        Returns:
            List of drift alerts
        """
        alerts = []
        
        if not self.baseline_stats:
            # No baseline established, save current as baseline
            self.save_baseline(current_stats)
            return alerts
        
        for prop in ['hit', 'hr', 'multi', 'k']:
            if prop not in current_stats or prop not in self.baseline_stats:
                continue
            
            baseline = self.baseline_stats[prop]
            current = current_stats[prop]
            
            # Check for drift in mean prediction
            mean_change = abs(current['mean'] - baseline['mean']) / (baseline['mean'] + 1e-6)
            if mean_change > self.alert_threshold:
                alerts.append({
                    'type': 'mean_drift',
                    'prop': prop,
                    'baseline_mean': baseline['mean'],
                    'current_mean': current['mean'],
                    'percent_change': mean_change * 100,
                    'severity': 'high' if mean_change > 0.25 else 'medium'
                })
            
            # Check for drift in standard deviation
            std_change = abs(current['std'] - baseline['std']) / (baseline['std'] + 1e-6)
            if std_change > self.alert_threshold:
                alerts.append({
                    'type': 'std_drift',
                    'prop': prop,
                    'baseline_std': baseline['std'],
                    'current_std': current['std'],
                    'percent_change': std_change * 100,
                    'severity': 'high' if std_change > 0.25 else 'medium'
                })
            
            # Check for extreme values
            if current['max'] > 0.99 and baseline['max'] < 0.95:
                alerts.append({
                    'type': 'extreme_predictions',
                    'prop': prop,
                    'baseline_max': baseline['max'],
                    'current_max': current['max'],
                    'severity': 'high'
                })
        
        return alerts
    
    def update_baseline(self, current_stats: Dict):
        """Update baseline with current statistics"""
        self.baseline_stats = current_stats
        self.save_baseline(current_stats)
    
    def log_drift_alerts(self, alerts: List[Dict]):
        """Log drift alerts to file"""
        if not alerts:
            return
        
        alert_file = LOGS_DIR / f"drift_alerts_{datetime.now().strftime('%Y-%m-%d')}.log"
        with open(alert_file, 'a') as f:
            timestamp = datetime.now().isoformat()
            f.write(f"\n=== DRIFT ALERTS [{timestamp}] ===\n")
            for alert in alerts:
                f.write(f"ALERT: {alert}\n")
    
    def check_predictions(self, predictions: pd.DataFrame, force_baseline_update: bool = False) -> Tuple[Dict, List[Dict]]:
        """
        Check predictions for drift
        
        Args:
            predictions: DataFrame with prediction probabilities
            force_baseline_update: Force update baseline even if no drift detected
            
        Returns:
            Tuple of (current_stats, drift_alerts)
        """
        current_stats = self.calculate_statistics(predictions)
        alerts = self.detect_drift(current_stats)
        
        if alerts:
            self.log_drift_alerts(alerts)
            log.warning("Detected %d prediction drift alerts", len(alerts))
            for alert in alerts:
                log.warning("- %s %s: %.1f%% change", alert['prop'].upper(), alert['type'], alert['percent_change'])
        elif force_baseline_update or not self.baseline_stats:
            self.update_baseline(current_stats)
            log.info("Baseline statistics updated")
        
        return current_stats, alerts


def monitor_prediction_drift(predictions_file: str = None):
    """
    Monitor prediction drift from file
    
    Args:
        predictions_file: Path to predictions CSV file
    """
    if predictions_file is None:
        predictions_file = BASE / "fact_player_props.parquet"
    
    # Load predictions
    predictions = pd.read_parquet(predictions_file)
    
    # Initialize drift detector
    detector = PredictionDriftDetector()
    
    # Check for drift
    current_stats, alerts = detector.check_predictions(predictions)
    
    # Log summary
    log.info("Prediction statistics summary")
    for prop, stats in current_stats.items():
        if prop in ['hit', 'hr', 'multi', 'k']:
            log.info("%s: mean=%.4f, std=%.4f, min=%.4f, max=%.4f", prop.upper(), stats['mean'], stats['std'], stats['min'], stats['max'])
    
    return current_stats, alerts


if __name__ == "__main__":
    monitor_prediction_drift()
