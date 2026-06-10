"""
mlb_ingest.py — Fetches raw JSON from MLB Stats API endpoints.
Endpoints: schedule, game feed/live, people, teams, standings, stats
"""
import argparse, json, logging, os, time
from datetime import date
from pathlib import Path
from typing import Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL   = "https://statsapi.mlb.com"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RATE_LIMIT = 0.25

GAME_FEEDS_DIR = RAW_DIR / "game_feeds"
GAME_FEEDS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

def build_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=0.5,
                  status_forcelist=[429,500,502,503,504], allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"Accept": "application/json", "User-Agent": "mlb-powerbi/1.0"})
    return session

SESSION = build_session()

def get(path, params=None):
    resp = SESSION.get(f"{BASE_URL}{path}", params=params, timeout=15)
    resp.raise_for_status()
    time.sleep(RATE_LIMIT)
    return resp.json()

def save(data, subdir, filename):
    dest = RAW_DIR / subdir
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info("Saved → %s/%s", subdir, filename)

def fetch_schedule(game_date):
    data = get("/api/v1/schedule", params={
        "sportId": 1, "date": game_date,
        "hydrate": "team,linescore,game(content(summary)),probablePitcher(note)"
    })
    save(data, "schedule", f"schedule_{game_date}.json")
    pks = [g["gamePk"] for day in data.get("dates",[]) for g in day.get("games",[])]
    log.info("Schedule: %d game(s) on %s", len(pks), game_date)
    return pks

def fetch_game_feed(game_pk):
    """Fetches the comprehensive live feed (including Statcast data) and saves it once."""
    log.info("Fetching live game feed for PK: %s", game_pk)
    
    # 1. Hit the comprehensive live feed endpoint
    endpoint = f"/api/v1.1/game/{game_pk}/feed/live"
    data = get(endpoint)
    
    # 2. Save it to your single source of truth folder
    with open(GAME_FEEDS_DIR / f"game_{game_pk}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    log.info("Saved unified live feed → game_feeds/game_%s.json", game_pk)
    return data

def fetch_play_by_play(game_pk):
    data = get(f"/api/v1/game/{game_pk}/playByPlay")
    save(data, "play_by_play", f"pbp_{game_pk}.json")

def fetch_boxscore(game_pk):
    data = get(f"/api/v1/game/{game_pk}/boxscore")
    save(data, "boxscore", f"box_{game_pk}.json")

def fetch_context_metrics(game_pk):
    try:
        data = get(f"/api/v1/game/{game_pk}/contextMetrics")
        save(data, "context_metrics", f"context_{game_pk}.json")
    except Exception as e:
        log.warning("No contextMetrics for game %s — skipping (%s)", game_pk, e)

def fetch_teams():
    data = get("/api/v1/teams", params={"sportId": 1, "activeStatus": "Active"})
    save(data, "teams", "teams.json")
    return [t["id"] for t in data.get("teams", [])]

def fetch_standings(season):
    data = get("/api/v1/standings", params={
        "leagueId": "103,104", "season": season,
        "hydrate": "team,record(splitRecords),expectedRecords",
        "standingsTypes": "regularSeason"
    })
    save(data, "standings", f"standings_{season}.json")

def fetch_team_stats(season, team_ids=None):
    if team_ids:
        # Fetch per-team to guarantee all 30 are covered
        for group in ("hitting", "pitching"):
            all_splits = []
            for team_id in team_ids:
                data = get("/api/v1/stats", params={
                    "stats": "season", "group": group, "gameType": "R",
                    "season": season, "sportId": 1,
                    "teamId": team_id
                })
                splits = data.get("stats", [{}])[0].get("splits", [])
                all_splits.extend(splits)
                time.sleep(RATE_LIMIT)
            # Wrap back into same shape as original for normalize.py compatibility
            combined = {"stats": [{"splits": all_splits}]}
            save(combined, "team_stats", f"team_{group}_{season}.json")
    else:
        # Fallback: bulk with safe limit buffer
        for group in ("hitting", "pitching"):
            data = get("/api/v1/stats", params={
                "stats": "season", "group": group, "gameType": "R",
                "season": season, "sportId": 1, "order": "desc", "limit": 50
            })
            save(data, "team_stats", f"team_{group}_{season}.json")

def fetch_player_stats_bulk(season, team_ids=None):
    for group in ("hitting", "pitching"):
        all_splits = []
        if team_ids:
            for team_id in team_ids:
                data = get("/api/v1/stats", params={
                    "stats": "season", "group": group, "gameType": "R",
                    "season": season, "sportId": 1,
                    "playerPool": "All", "teamId": team_id
                })
                splits = data.get("stats", [{}])[0].get("splits", [])
                all_splits.extend(splits)
                time.sleep(RATE_LIMIT)
        else:
            # Fallback if no team_ids passed
            data = get("/api/v1/stats", params={
                "stats": "season", "group": group, "gameType": "R",
                "season": season, "sportId": 1, "playerPool": "All", "limit": 500
            })
            all_splits = data.get("stats", [{}])[0].get("splits", [])

        combined = {"stats": [{"splits": all_splits}]}
        save(combined, "player_stats", f"player_{group}_{season}.json")


def run(game_date=None, fetch_live=False):
    if game_date is None:
        game_date = str(date.today())

    season = int(game_date[:4])
    team_ids = fetch_teams()
    fetch_standings(season)
    fetch_team_stats(season, team_ids)
    fetch_player_stats_bulk(season, team_ids)
    pks = fetch_schedule(game_date)

    if fetch_live:
        for pk in pks:
            fetch_game_feed(pk)
            fetch_play_by_play(pk)
            fetch_boxscore(pk)
            fetch_context_metrics(pk)
    else:
        log.info("--live not specified; skipping live feed, play-by-play, boxscore, and context metrics")

    log.info("=== Ingestion complete ===")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=str(date.today()), help="Date to ingest in YYYY-MM-DD format")
    p.add_argument("--live", action="store_true", help="Fetch live game feed and related artifacts for the selected date")
    args = p.parse_args()
    run(args.date, args.live)
