"""
model_versioning.py - Track model versions and performance over time
Provides model versioning, performance tracking, and comparison
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import hashlib
from typing import Dict, List, Optional
import pickle

MODEL_DIR = Path(__file__).parent.parent / "model" / "models"
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class ModelVersionTracker:
    """Track model versions and performance metrics"""
    
    def __init__(self):
        self.version_file = LOGS_DIR / "model_versions.json"
        self.performance_file = LOGS_DIR / "model_performance.json"
        self.load_data()
    
    def load_data(self):
        """Load version and performance data"""
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                self.versions = json.load(f)
        else:
            self.versions = {}
        
        if self.performance_file.exists():
            with open(self.performance_file, 'r') as f:
                self.performance = json.load(f)
        else:
            self.performance = {}
    
    def save_data(self):
        """Save version and performance data"""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
        
        with open(self.performance_file, 'w') as f:
            json.dump(self.performance, f, indent=2)
    
    def generate_model_hash(self, model_file: Path) -> str:
        """Generate hash for model file"""
        with open(model_file, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def register_model(self, prop: str, model_type: str, model_file: Path, 
                      cv_scores: Dict = None, feature_importance: Dict = None):
        """
        Register a new model version
        
        Args:
            prop: Property type (hit, hr, multi, k)
            model_type: Model type (ensemble, xgb, lgb, cb)
            model_file: Path to model file
            cv_scores: Cross-validation scores
            feature_importance: Feature importance data
        """
        # Generate version ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_id = f"{prop}_{model_type}_{timestamp}"
        
        # Generate model hash
        model_hash = self.generate_model_hash(model_file)
        
        # Create version entry
        version_info = {
            'version_id': version_id,
            'prop': prop,
            'model_type': model_type,
            'model_file': str(model_file.name),
            'model_hash': model_hash,
            'timestamp': timestamp,
            'cv_scores': cv_scores or {},
            'feature_importance_top_features': self._get_top_features(feature_importance) if feature_importance is not None else []
        }
        
        # Store version
        if prop not in self.versions:
            self.versions[prop] = {}
        
        if model_type not in self.versions[prop]:
            self.versions[prop][model_type] = []
        
        self.versions[prop][model_type].append(version_info)
        
        # Keep only last 10 versions per model type
        if len(self.versions[prop][model_type]) > 10:
            self.versions[prop][model_type] = self.versions[prop][model_type][-10:]
        
        self.save_data()
        
        print(f"✅ Registered model version: {version_id}")
        return version_id
    
    def _get_top_features(self, feature_importance: pd.DataFrame, top_n: int = 5) -> List[str]:
        """Extract top features from feature importance dataframe"""
        if feature_importance is None or len(feature_importance) == 0:
            return []
        
        return feature_importance.head(top_n)['feature'].tolist()
    
    def log_performance(self, prop: str, predictions: pd.DataFrame, version_id: str = None):
        """
        Log model performance metrics
        
        Args:
            prop: Property type
            predictions: Predictions dataframe
            version_id: Model version ID
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Calculate performance metrics
        metrics = {}
        if f'{prop}_prob' in predictions.columns:
            probs = predictions[f'{prop}_prob']
            metrics = {
                'mean': float(probs.mean()),
                'std': float(probs.std()),
                'min': float(probs.min()),
                'max': float(probs.max()),
                'median': float(probs.median()),
                'sample_size': len(probs)
            }
        
        # Store performance
        if prop not in self.performance:
            self.performance[prop] = []
        
        performance_entry = {
            'timestamp': timestamp,
            'version_id': version_id,
            'metrics': metrics
        }
        
        self.performance[prop].append(performance_entry)
        
        # Keep only last 30 days of performance data
        if len(self.performance[prop]) > 30:
            self.performance[prop] = self.performance[prop][-30:]
        
        self.save_data()
        
        print(f"✅ Logged performance for {prop}: mean={metrics.get('mean', 0):.4f}")
    
    def get_latest_version(self, prop: str, model_type: str = 'ensemble') -> Optional[Dict]:
        """Get latest model version for a property"""
        if prop not in self.versions or model_type not in self.versions[prop]:
            return None
        
        versions = self.versions[prop][model_type]
        return versions[-1] if versions else None
    
    def compare_versions(self, prop: str, version_id1: str, version_id2: str) -> Dict:
        """Compare two model versions"""
        # Find versions
        version1 = None
        version2 = None
        
        for model_type in self.versions.get(prop, {}):
            for v in self.versions[prop][model_type]:
                if v['version_id'] == version_id1:
                    version1 = v
                if v['version_id'] == version_id2:
                    version2 = v
        
        if not version1 or not version2:
            return {'error': 'One or both versions not found'}
        
        comparison = {
            'version1': version1,
            'version2': version2,
            'cv_score_diff': {},
            'timestamp_diff': None
        }
        
        # Compare CV scores
        for metric in version1.get('cv_scores', {}):
            if metric in version2.get('cv_scores', {}):
                diff = version2['cv_scores'][metric] - version1['cv_scores'][metric]
                comparison['cv_score_diff'][metric] = diff
        
        # Calculate timestamp difference
        t1 = datetime.strptime(version1['timestamp'], '%Y%m%d_%H%M%S')
        t2 = datetime.strptime(version2['timestamp'], '%Y%m%d_%H%M%S')
        comparison['timestamp_diff'] = (t2 - t1).total_seconds() / 3600  # hours
        
        return comparison
    
    def get_performance_history(self, prop: str, days: int = 7) -> List[Dict]:
        """Get performance history for a property"""
        if prop not in self.performance:
            return []
        
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.strftime('%Y%m%d')
        
        history = []
        for entry in self.performance[prop]:
            if entry['timestamp'] >= cutoff_str:
                history.append(entry)
        
        return history
    
    def generate_report(self) -> str:
        """Generate a model versioning and performance report"""
        report_lines = ["=== MODEL VERSIONING & PERFORMANCE REPORT ==="]
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        for prop in ['hit', 'hr', 'multi', 'k']:
            report_lines.append(f"--- {prop.upper()} ---")
            
            # Latest versions
            for model_type in ['ensemble', 'xgb', 'lgb', 'cb']:
                latest = self.get_latest_version(prop, model_type)
                if latest:
                    report_lines.append(f"  {model_type.upper()}: {latest['version_id']}")
                    report_lines.append(f"    Timestamp: {latest['timestamp']}")
                    if latest.get('cv_scores'):
                        report_lines.append(f"    CV Scores: {latest['cv_scores']}")
                    if latest.get('feature_importance_top_features'):
                        report_lines.append(f"    Top Features: {latest['feature_importance_top_features'][:3]}")
            
            # Recent performance
            history = self.get_performance_history(prop, days=7)
            if history:
                report_lines.append(f"  Recent Performance (7 days):")
                for entry in history[-3:]:  # Last 3 entries
                    report_lines.append(f"    {entry['timestamp']}: mean={entry['metrics']['mean']:.4f}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


def track_model_training(prop: str, model_type: str, model_file: Path, 
                       cv_scores: Dict = None, feature_importance: pd.DataFrame = None):
    """
    Track model training and register version
    
    Args:
        prop: Property type
        model_type: Model type
        model_file: Path to model file
        cv_scores: Cross-validation scores
        feature_importance: Feature importance dataframe
    """
    tracker = ModelVersionTracker()
    version_id = tracker.register_model(prop, model_type, model_file, cv_scores, feature_importance)
    return version_id


def track_model_performance(prop: str, predictions: pd.DataFrame, version_id: str = None):
    """
    Track model performance
    
    Args:
        prop: Property type
        predictions: Predictions dataframe
        version_id: Model version ID
    """
    tracker = ModelVersionTracker()
    tracker.log_performance(prop, predictions, version_id)


def generate_model_report():
    """Generate and print model versioning report"""
    tracker = ModelVersionTracker()
    report = tracker.generate_report()
    print(report)
    return report


if __name__ == "__main__":
    generate_model_report()
