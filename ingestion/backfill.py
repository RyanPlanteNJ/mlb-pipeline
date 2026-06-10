"""
backfill.py — Fetches schedule + game feeds for a range of past dates.
Pulls teams/standings/stats once per season, then loops over dates and ensures
all game feeds are saved (for Statcast, defense, substitutions, etc.).

Usage:
    python ingestion/backfill.py --start 2024-04-01
    python ingestion/backfill.py --start 2024-04-01 --end 2024-06-03
    python ingestion/backfill.py --start 2022-04-01 --end 2026-10-01 --skip-existing
"""
import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ingestion.mlb_ingest import (
    fetch_schedule,
    fetch_teams,
    fetch_standings,
    fetch_team_stats,
    fetch_player_stats_bulk,
    fetch_game_feed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
SCHEDULE_DIR = RAW_DIR / "schedule"
GAMES_DIR = RAW_DIR / "game_feeds"
SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
GAMES_DIR.mkdir(parents=True, exist_ok=True)


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=str(date.today()), help="End date YYYY-MM-DD (default: today)")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip dates that already have a schedule file",
    )
    args = p.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    # 1. Generate the list of dates first
    dates = list(daterange(start, end))
    
    # 2. Extract every unique year present in the date range
    unique_years = sorted(list(set(d.year for d in dates)))

    log.info("Fetching active teams...")
    team_ids = fetch_teams()
    
    # 3. Loop through EACH unique year to check/fetch seasonal metadata
    for year in unique_years:
        log.info("=== Checking seasonal bulk data for the %d season ===", year)
        
        # Build paths mapping exactly to where mlb_ingest saves files
        standings_file = RAW_DIR / "standings" / f"standings_{year}.json"
        team_hit = RAW_DIR / "team_stats" / f"team_hitting_{year}.json"
        team_pitch = RAW_DIR / "team_stats" / f"team_pitching_{year}.json"
        player_hit = RAW_DIR / "player_stats" / f"player_hitting_{year}.json"
        player_pitch = RAW_DIR / "player_stats" / f"player_pitching_{year}.json"

        # Evaluate if ALL seasonal bulk files for this year are present
        seasonal_files_exist = all([
            standings_file.exists(),
            team_hit.exists(),
            team_pitch.exists(),
            player_hit.exists(),
            player_pitch.exists()
        ])

        # If --skip-existing is flagged and we have all files, bypass API requests
        if args.skip_existing and seasonal_files_exist:
            log.info("Skipping bulk data fetch for %d (all seasonal files already exist)", year)
            continue

        try:
            log.info("Fetching missing standings and stats for %d...", year)
            fetch_standings(year)
            fetch_team_stats(year, team_ids)
            fetch_player_stats_bulk(year, team_ids)
        except Exception as e:
            log.warning("Failed to fetch seasonal bulk data for %d: %s", year, e)

    # 4. Now handle daily loops for schedules and game boxscores
    log.info("Backfilling %d dates: %s → %s", len(dates), start, end)

    for d in dates:
        date_str = str(d)
        schedule_path = SCHEDULE_DIR / f"schedule_{date_str}.json"

        if args.skip_existing and schedule_path.exists():
            log.info("Skipping %s (schedule already exists)", date_str)
            continue

        try:
            # Fetch schedule for the date (writes schedule JSON internally)
            game_pks = fetch_schedule(date_str)
            log.info("Date %s: %d games", date_str, len(game_pks))

            # Fetch game feeds for each gamePk
            for pk in game_pks:
                game_path = GAMES_DIR / f"game_{pk}.json"
                if args.skip_existing and game_path.exists():
                    log.info("  Skipping game feed %s (already exists)", pk)
                    continue

                try:
                    fetch_game_feed(pk)
                    log.info("  Fetched game feed %s", pk)
                except Exception as e:
                    log.warning("  Failed to fetch game feed %s: %s", pk, e)

        except Exception as e:
            log.warning("Failed %s: %s", date_str, e)

    log.info("=== Backfill done. Now run normalize.py (including Statcast/defense/subs extraction) then retrain models ===")