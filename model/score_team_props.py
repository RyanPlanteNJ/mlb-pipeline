import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
import os
import sys

sys.path.append(str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)
BASE = Path(__file__).parent.parent / "data" / "normalized"

# Import betting calculator
try:
    from utils.betting_calculator import (
        calculate_ev,
        calculate_kelly_criterion,
        get_betting_recommendation,
        calculate_team_betting_recommendations
    )
    BETTING_AVAILABLE = True
except ImportError:
    BETTING_AVAILABLE = False
    log.warning("Betting calculator not available - betting features disabled")

# Import odds fetcher (optional)
try:
    from utils.odds_fetcher import OddsFetcher
    ODDS_AVAILABLE = True
except ImportError:
    ODDS_AVAILABLE = False
    log.warning("Odds fetcher not available - will use implied odds")


def implied_moneyline(prob):
    if prob <= 0:
        return 500
    if prob >= 1:
        return -500
    if prob > 0.5:
        return -100 * prob / (1 - prob)
    else:
        return 100 * (1 - prob) / prob


def round_half(x):
    return np.round(x * 2) / 2


def confidence_label(p):
    if p >= 0.70:
        return "HIGH"
    if p >= 0.60:
        return "MEDIUM"
    if p >= 0.52:
        return "LEAN"
    return "LOW"


def run(fetch_odds=False, odds_api_key=None):
    games = pd.read_parquet(BASE / "fact_games.parquet")
    wp = pd.read_parquet(BASE / "fact_win_probability.parquet")
    team_stats = pd.read_parquet(BASE / "fact_team_stats.parquet")

    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.date
    wp["game_date"] = pd.to_datetime(wp["game_date"], errors="coerce").dt.date

    today = date.today()

    games_today = games[games["game_date"] == today]
    wp_today = wp[wp["game_date"] == today]

    if games_today.empty or wp_today.empty:
        log.info("No games found for today. Team props not generated.")
        return

    if "runsPerGame" not in team_stats.columns:
        team_stats["runsPerGame"] = team_stats["runs"] / team_stats["gamesPlayed"]

    league_rpg = team_stats["runsPerGame"].mean()

    merged = games_today.merge(
        wp_today[["game_pk", "home_win_prob", "away_win_prob", "predicted_winner"]],
        on="game_pk",
        how="left",
    )

    merged = merged.merge(
        team_stats[["team_id", "runsPerGame"]],
        left_on="home_team_id",
        right_on="team_id",
        how="left",
    ).rename(columns={"runsPerGame": "home_rpg"}).drop(columns=["team_id"])

    merged = merged.merge(
        team_stats[["team_id", "runsPerGame"]],
        left_on="away_team_id",
        right_on="team_id",
        how="left",
    ).rename(columns={"runsPerGame": "away_rpg"}).drop(columns=["team_id"])

    merged["home_rpg"] = merged["home_rpg"].fillna(league_rpg)
    merged["away_rpg"] = merged["away_rpg"].fillna(league_rpg)

    # Simple asymmetric model: home gets a bump, away gets a slight penalty
    HFA = 0.20
    merged["home_expected_runs"] = merged["home_rpg"] * 1.05 + HFA
    merged["away_expected_runs"] = merged["away_rpg"] * 0.95

    merged["home_team_total_runs_line"] = merged["home_expected_runs"].apply(round_half)
    merged["away_team_total_runs_line"] = merged["away_expected_runs"].apply(round_half)

    merged["game_total_runs_line"] = (
        merged["home_expected_runs"] + merged["away_expected_runs"]
    ).apply(round_half)

    merged["home_team_moneyline"] = (
        merged["home_win_prob"].apply(implied_moneyline).round().astype(int)
    )
    merged["away_team_moneyline"] = (
        merged["away_win_prob"].apply(implied_moneyline).round().astype(int)
    )

    # Fetch real odds from API if requested and available
    if fetch_odds and ODDS_AVAILABLE and odds_api_key:
        try:
            fetcher = OddsFetcher(odds_api_key)
            odds_data = fetcher.get_odds()
            if odds_data:
                odds_df = fetcher.parse_odds_to_dataframe(odds_data)
                merged = fetcher.match_odds_to_games(odds_df, merged)
                
                # Use real odds if available, otherwise use implied odds
                merged["home_odds"] = merged["home_odds"].fillna(merged["home_team_moneyline"])
                merged["away_odds"] = merged["away_odds"].fillna(merged["away_team_moneyline"])
                log.info("Using real odds from The Odds API")
            else:
                merged["home_odds"] = merged["home_team_moneyline"]
                merged["away_odds"] = merged["away_team_moneyline"]
                log.warning("No odds data from API, using implied odds")
        except Exception as e:
            log.error(f"Error fetching odds: {e}, using implied odds")
            merged["home_odds"] = merged["home_team_moneyline"]
            merged["away_odds"] = merged["away_team_moneyline"]
    else:
        # Use implied odds
        merged["home_odds"] = merged["home_team_moneyline"]
        merged["away_odds"] = merged["away_team_moneyline"]

    # Calculate betting recommendations if available
    if BETTING_AVAILABLE:
        merged = calculate_team_betting_recommendations(
            merged,
            home_prob_col='home_win_prob',
            away_prob_col='away_win_prob',
            home_odds_col='home_odds',
            away_odds_col='away_odds'
        )
        log.info("Calculated betting recommendations")
    else:
        # Add placeholder columns
        merged['home_recommendation'] = 'N/A'
        merged['away_recommendation'] = 'N/A'
        merged['home_ev'] = None
        merged['away_ev'] = None
        merged['home_edge'] = None
        merged['away_edge'] = None
        merged['home_kelly'] = None
        merged['away_kelly'] = None

    merged["home_confidence"] = merged["home_win_prob"].apply(confidence_label)
    merged["away_confidence"] = merged["away_win_prob"].apply(confidence_label)

    merged["home_ev_vs_coinflip"] = merged["home_win_prob"] - 0.5
    merged["away_ev_vs_coinflip"] = merged["away_win_prob"] - 0.5

    out = merged[
        [
            "game_pk",
            "game_date",
            "home_team_name",
            "away_team_name",
            "home_team_total_runs_line",
            "away_team_total_runs_line",
            "game_total_runs_line",
            "home_team_moneyline",
            "away_team_moneyline",
            "home_odds",
            "away_odds",
            "home_win_prob",
            "away_win_prob",
            "home_confidence",
            "away_confidence",
            "home_recommendation",
            "away_recommendation",
            "home_ev",
            "away_ev",
            "home_edge",
            "away_edge",
            "home_kelly",
            "away_kelly",
            "home_ev_vs_coinflip",
            "away_ev_vs_coinflip",
            "predicted_winner",
        ]
    ]

    out = out.groupby("game_pk").first().reset_index()

    out.to_parquet(BASE / "fact_team_props.parquet", index=False)
    log.info("Saved → fact_team_props.parquet")


if __name__ == "__main__":
    import argparse
    import os
    
    p = argparse.ArgumentParser()
    p.add_argument("--fetch-odds", action="store_true", help="Fetch real odds from The Odds API")
    p.add_argument("--odds-api-key", default=os.getenv("ODDS_API_KEY"), help="The Odds API key (or set ODDS_API_KEY env var)")
    args = p.parse_args()
    
    run(fetch_odds=args.fetch_odds, odds_api_key=args.odds_api_key)
