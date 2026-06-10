"""
data_validation.py - Automated data validation checks for MLB pipeline
Validates data quality, detects anomalies, and logs issues
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
import logging

log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent.parent / "data" / "normalized"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "game_feeds"


def validate_raw_feeds():
    """Checks for file existence and minimal size before normalizing."""
    for file in RAW_DIR.glob("game_*.json"):
        if file.stat().st_size < 500: # Files smaller than 500 bytes are likely partials
            return False, f"Feed {file.name} is too small (likely partial download)"
    return True, []


def validate_player_stats():
    """Validate player stats data for quality issues"""
    try:
        df = pd.read_parquet(BASE / "fact_player_stats.parquet")
    except FileNotFoundError:
        log.warning("Player stats file not found")
        return False, ["File not found"]
    
    issues = []
    
    # Check for null critical fields
    critical_fields = ["player_id", "player_name", "team_id", "season"]
    for field in critical_fields:
        if field in df.columns and df[field].isnull().sum() > 0:
            issues.append(f"Null values in {field}: {df[field].isnull().sum()}")
    
    # Check for duplicate player_id/season/group
    if "player_id" in df.columns and "season" in df.columns and "group" in df.columns:
        dups = df.duplicated(subset=["player_id", "season", "group"]).sum()
        if dups > 0:
            issues.append(f"Duplicate player_id/season/group: {dups}")
    
    # Check for unreasonable stat values
    if "avg" in df.columns:
        avg_numeric = pd.to_numeric(df["avg"], errors="coerce")
        bad_avg = ((avg_numeric < 0) | (avg_numeric > 1)).sum()
        if bad_avg > 0:
            issues.append(f"Unreasonable avg values: {bad_avg}")
    
    if "era" in df.columns:
        era_numeric = pd.to_numeric(df["era"], errors="coerce")
        bad_era = ((era_numeric < 0) | (era_numeric > 20)).sum()
        if bad_era > 0:
            issues.append(f"Unreasonable era values: {bad_era}")
    
    if issues:
        log.warning(f"Player stats validation issues: {issues}")
        return False, issues
    else:
        log.info("Player stats validation passed")
        return True, []


def validate_matchups():
    """Validate matchups data for quality issues"""
    try:
        df = pd.read_parquet(BASE / "fact_batter_pitcher_matchups.parquet")
    except FileNotFoundError:
        log.warning("Matchups file not found")
        return False, ["File not found"]
    
    issues = []
    
    # Check for null critical fields
    critical_fields = ["game_pk", "batter_id", "pitcher_id"]
    for field in critical_fields:
        if field in df.columns and df[field].isnull().sum() > 0:
            issues.append(f"Null values in {field}: {df[field].isnull().sum()}")
    
    # Check is_multi values are reasonable (0 or 1)
    if "is_multi" in df.columns:
        bad_multi = df[(df["is_multi"] != 0) & (df["is_multi"] != 1)].shape[0]
        if bad_multi > 0:
            issues.append(f"Invalid is_multi values: {bad_multi}")
    
    # Check that is_multi has some positive values
    if "is_multi" in df.columns:
        multi_rate = df["is_multi"].mean()
        if multi_rate == 0:
            issues.append("is_multi column is all zeros - data pipeline issue")
    
    if issues:
        log.warning(f"Matchups validation issues: {issues}")
        return False, issues
    else:
        log.info("Matchups validation passed")
        return True, []


def validate_games():
    """Validate games data for quality issues"""
    try:
        df = pd.read_parquet(BASE / "fact_games.parquet")
    except FileNotFoundError:
        log.warning("Games file not found")
        return False, ["File not found"]
    
    issues = []
    
    # Check for null critical fields
    critical_fields = ["game_pk", "game_date", "home_team_id", "away_team_id"]
    for field in critical_fields:
        if field in df.columns and df[field].isnull().sum() > 0:
            issues.append(f"Null values in {field}: {df[field].isnull().sum()}")
    
    # Check for duplicate game_pk
    if "game_pk" in df.columns:
        dups = df.duplicated(subset=["game_pk"]).sum()
        if dups > 0:
            issues.append(f"Duplicate game_pk: {dups}")
    
    if issues:
        log.warning(f"Games validation issues: {issues}")
        return False, issues
    else:
        log.info("Games validation passed")
        return True, []


def run_all_validations():
    """Run all validation checks and return summary"""
    results = {
        "player_stats": validate_player_stats(),
        "matchups": validate_matchups(),
        "games": validate_games()
    }
    
    all_passed = all(result[0] for result in results.values())
    
    if all_passed:
        log.info("All data validations passed")
    else:
        log.warning("Some data validations failed")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all_validations()
