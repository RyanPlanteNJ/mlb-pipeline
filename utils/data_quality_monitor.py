"""
data_quality_monitor.py - Monitor data quality and freshness
Alerts on stale data or quality issues
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
import os

BASE = Path(__file__).parent.parent / "data" / "normalized"
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


class DataQualityMonitor:
    """Monitor data quality and freshness"""
    
    def __init__(self, max_age_hours: int = 24):
        """
        Initialize data quality monitor
        
        Args:
            max_age_hours: Maximum acceptable data age in hours
        """
        self.max_age_hours = max_age_hours
        self.quality_thresholds = {
            'max_null_percentage': 0.1,  # 10% nulls acceptable
            'max_duplicate_percentage': 0.05,  # 5% duplicates acceptable
            'min_sample_size': 100  # Minimum rows required
        }
    
    def check_file_freshness(self, file_path: Path) -> Dict:
        """
        Check if a file is fresh enough
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            Dict with freshness information
        """
        if not file_path.exists():
            return {
                'exists': False,
                'age_hours': None,
                'is_fresh': False,
                'message': f"File {file_path.name} does not exist"
            }
        
        # Get file modification time
        mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
        age = datetime.now() - mod_time
        age_hours = age.total_seconds() / 3600
        
        is_fresh = age_hours <= self.max_age_hours
        
        return {
            'exists': True,
            'age_hours': age_hours,
            'is_fresh': is_fresh,
            'message': f"File is {age_hours:.1f} hours old" + 
                     (f" (stale, max {self.max_age_hours}h)" if not is_fresh else " (fresh)")
        }
    
    def check_data_quality(self, df: pd.DataFrame, file_name: str) -> Dict:
        """
        Check data quality for a dataframe
        
        Args:
            df: DataFrame to check
            file_name: Name of the file (for reporting)
            
        Returns:
            Dict with quality information
        """
        issues = []
        
        # Check sample size
        if len(df) < self.quality_thresholds['min_sample_size']:
            issues.append({
                'type': 'low_sample_size',
                'severity': 'high',
                'message': f"Only {len(df)} rows (minimum {self.quality_thresholds['min_sample_size']})"
            })
        
        # Check for null values
        null_percentage = (df.isnull().sum() / len(df)).max()
        if null_percentage > self.quality_thresholds['max_null_percentage']:
            issues.append({
                'type': 'high_null_percentage',
                'severity': 'medium',
                'message': f"{null_percentage:.1%} null values (max {self.quality_thresholds['max_null_percentage']:.0%})"
            })
        
        # Check for duplicates
        duplicate_percentage = df.duplicated().sum() / len(df)
        if duplicate_percentage > self.quality_thresholds['max_duplicate_percentage']:
            issues.append({
                'type': 'high_duplicate_percentage',
                'severity': 'low',
                'message': f"{duplicate_percentage:.1%} duplicates (max {self.quality_thresholds['max_duplicate_percentage']:.0%})"
            })
        
        # Check for unreasonable values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].min() < -1e6 or df[col].max() > 1e6:
                issues.append({
                    'type': 'extreme_values',
                    'severity': 'medium',
                    'message': f"Column {col} has extreme values (min={df[col].min():.2f}, max={df[col].max():.2f})"
                })
        
        return {
            'file_name': file_name,
            'row_count': len(df),
            'column_count': len(df.columns),
            'null_percentage': null_percentage,
            'duplicate_percentage': duplicate_percentage,
            'issues': issues,
            'has_issues': len(issues) > 0
        }
    
    def monitor_all_data(self) -> Dict:
        """
        Monitor all data files in the normalized directory
        
        Returns:
            Dict with monitoring results for all files
        """
        results = {}
        critical_files = [
            'fact_player_stats.parquet',
            'fact_batter_game_logs.parquet',
            'fact_pitcher_game_logs.parquet',
            'fact_games.parquet',
            'fact_batter_pitcher_matchups.parquet',
            'fact_team_stats.parquet'
        ]
        
        for file_name in critical_files:
            file_path = BASE / file_name
            
            # Check freshness
            freshness = self.check_file_freshness(file_path)
            
            # Check quality if file exists
            quality = None
            if freshness['exists']:
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    quality = self.check_data_quality(df, file_name)
                except Exception as e:
                    quality = {
                        'file_name': file_name,
                        'error': str(e),
                        'has_issues': True,
                        'issues': [{
                            'type': 'read_error',
                            'severity': 'high',
                            'message': f"Could not read file: {str(e)}"
                        }]
                    }
            
            results[file_name] = {
                'freshness': freshness,
                'quality': quality,
                'overall_status': 'ok' if (freshness['is_fresh'] and (quality is None or not quality['has_issues'])) else 'alert'
            }
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """
        Generate a human-readable report from monitoring results
        
        Args:
            results: Monitoring results from monitor_all_data()
            
        Returns:
            Formatted report string
        """
        report_lines = ["=== DATA QUALITY MONITORING REPORT ==="]
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        alert_count = 0
        for file_name, result in results.items():
            status_emoji = "✅" if result['overall_status'] == 'ok' else "⚠️"
            report_lines.append(f"{status_emoji} {file_name}")
            
            # Freshness
            freshness = result['freshness']
            if freshness['exists']:
                report_lines.append(f"  Freshness: {freshness['message']}")
            else:
                report_lines.append(f"  Freshness: FILE MISSING")
                alert_count += 1
            
            # Quality
            quality = result['quality']
            if quality:
                if quality.get('error'):
                    report_lines.append(f"  Quality: ERROR - {quality['error']}")
                    alert_count += 1
                else:
                    report_lines.append(f"  Quality: {quality['row_count']} rows, {quality['column_count']} cols")
                    if quality['has_issues']:
                        for issue in quality['issues']:
                            severity_emoji = "🔴" if issue['severity'] == 'high' else "🟡" if issue['severity'] == 'medium' else "🟢"
                            report_lines.append(f"    {severity_emoji} {issue['message']}")
                            if issue['severity'] in ['high', 'medium']:
                                alert_count += 1
            report_lines.append("")
        
        report_lines.append(f"=== SUMMARY: {alert_count} alerts detected ===")
        
        return "\n".join(report_lines)
    
    def log_alerts(self, results: Dict):
        """Log alerts to file"""
        alert_file = LOGS_DIR / f"data_quality_alerts_{datetime.now().strftime('%Y-%m-%d')}.log"
        
        with open(alert_file, 'a') as f:
            timestamp = datetime.now().isoformat()
            f.write(f"\n=== DATA QUALITY ALERTS [{timestamp}] ===\n")
            
            for file_name, result in results.items():
                if result['overall_status'] == 'alert':
                    f.write(f"ALERT: {file_name}\n")
                    f.write(f"  Freshness: {result['freshness']['message']}\n")
                    
                    quality = result['quality']
                    if quality and quality['has_issues']:
                        for issue in quality['issues']:
                            f.write(f"  Issue: {issue['message']}\n")
    
    def run_monitoring(self) -> Tuple[Dict, str]:
        """
        Run full data quality monitoring
        
        Returns:
            Tuple of (results, report)
        """
        results = self.monitor_all_data()
        report = self.generate_report(results)
        
        # Log alerts if any
        for file_name, result in results.items():
            if result['overall_status'] == 'alert':
                self.log_alerts(results)
                break
        
        return results, report


def monitor_data_quality(max_age_hours: int = 24):
    """
    Monitor data quality and freshness
    
    Args:
        max_age_hours: Maximum acceptable data age in hours
    """
    monitor = DataQualityMonitor(max_age_hours=max_age_hours)
    results, report = monitor.run_monitoring()
    
    print(report)
    
    return results, report


if __name__ == "__main__":
    monitor_data_quality()
