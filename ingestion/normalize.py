"""
normalize.py — Flattens raw statsapi JSON → relational parquet tables.
Optimized Version 1.1M-Pro: Built for safe multi-season massive historical scaling.
Unified routing to game_feeds single source of truth.
Output: dim_teams, fact_games, fact_linescore,
        fact_team_stats, fact_player_stats, fact_standings,
        fact_batter_pitcher_matchups, fact_batter_game_logs,
        fact_pitcher_game_logs,
        fact_pitches, fact_statcast, fact_events, fact_runners,
        fact_pitch_locations, fact_hit_locations, fact_innings,
        fact_substitutions, fact_defense
"""
import orjson
import logging
from pathlib import Path
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import date
import os
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")
LOG_FILE = LOG_DIR / f"normalize_{today_str}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

log = logging.getLogger(__name__)
log.info("Loaded normalize.py from %s", __file__)
log.info("Current working directory: %s", os.getcwd())
log.info("Using normalize version 1.1M-PRO")

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "normalized"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Matchup rows that are advisory placeholders and should not count as real plate appearances.
# Also skip "unknown" and None event types as they likely indicate incomplete or corrupted data.
SKIP_MATCHUP_EVENT_TYPES = {"game_advisory", "unknown", None}


def load_json(path):
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def save_parquet(df, name):
    """Save dataframe to parquet with PyArrow optimizations"""
    out_path = OUT_DIR / name
    # Convert to PyArrow table for better performance
    table = pa.Table.from_pandas(df)
    # Write with compression for smaller file size
    pq.write_table(table, out_path, compression='snappy')
    log.info("Wrote %s  (%d rows × %d cols)", name, len(df), len(df.columns))


def should_build(output_name, raw_paths):
    out_path = OUT_DIR / output_name
    if not raw_paths:
        log.warning("No raw files found for %s — skipping", output_name)
        return False
    if not out_path.exists():
        log.info("%s does not exist — building", output_name)
        return True

    out_mtime = out_path.stat().st_mtime
    newest_raw = max(p.stat().st_mtime for p in raw_paths)
    if newest_raw > out_mtime:
        log.info("Raw data newer than %s — rebuilding", output_name)
        return True

    log.info("%s is up to date — skipping", output_name)
    return False


def should_build_derived(output_name, source_name):
    out_path = OUT_DIR / output_name
    src_path = OUT_DIR / source_name
    if not src_path.exists():
        log.warning("Source %s missing — skipping %s", source_name, output_name)
        return False
    if not out_path.exists():
        return True
    if src_path.stat().st_mtime > out_path.stat().st_mtime:
        log.info("%s is newer than %s — rebuilding", source_name, output_name)
        return True
    log.info("%s is up to date — skipping", output_name)
    return False


# PyArrow Schemas for Streaming Writers to prevent schema/type mismatch issues
PITCHES_SCHEMA = pa.schema([
    ("game_pk", pa.int64()),
    ("play_id", pa.string()),
    ("batter_id", pa.int64()),
    ("pitcher_id", pa.int64()),
    ("pitch_type", pa.string()),
    ("start_speed", pa.float64()),
    ("end_speed", pa.float64()),
    ("spin_rate", pa.float64()),
    ("spin_direction", pa.float64()),
    ("break_x", pa.float64()),
    ("break_z", pa.float64()),
    ("plate_x", pa.float64()),
    ("plate_z", pa.float64()),
    ("sz_top", pa.float64()),
    ("sz_bot", pa.float64()),
    ("zone", pa.float64()),
    ("call", pa.string()),
    ("inning", pa.int64()),
    ("is_top_inning", pa.bool_()),
    ("count_balls", pa.int64()),
    ("count_strikes", pa.int64()),
])

STATCAST_SCHEMA = pa.schema([
    ("game_pk", pa.int64()),
    ("batter_id", pa.int64()),
    ("pitcher_id", pa.int64()),
    ("exit_velocity", pa.float64()),
    ("launch_angle", pa.float64()),
    ("spray_angle", pa.float64()),
    ("hit_distance", pa.float64()),
    ("is_barrel", pa.bool_()),
    ("is_hard_hit", pa.int64()),
    ("xBA", pa.float64()),
    ("xSLG", pa.float64()),
    ("xWOBA", pa.float64()),
    ("event", pa.string()),
])

EVENTS_SCHEMA = pa.schema([
    ("game_pk", pa.int64()),
    ("play_id", pa.string()),
    ("batter_id", pa.int64()),
    ("pitcher_id", pa.int64()),
    ("event", pa.string()),
    ("event_type", pa.string()),
    ("description", pa.string()),
    ("rbi", pa.int64()),
    ("outs", pa.int64()),
    ("inning", pa.int64()),
    ("is_top_inning", pa.bool_()),
])

RUNNERS_SCHEMA = pa.schema([
    ("game_pk", pa.int64()),
    ("play_id", pa.string()),
    ("runner_id", pa.int64()),
    ("start_base", pa.string()),
    ("end_base", pa.string()),
    ("is_out", pa.bool_()),
    ("is_scored", pa.bool_()),
    ("movement_reason", pa.string()),
    ("responsible_pitcher", pa.int64()),
])

SUBSTITUTIONS_SCHEMA = pa.schema([
    ("game_pk", pa.int64()),
    ("play_id", pa.string()),
    ("player_in", pa.string()),
    ("player_out", pa.string()),
    ("position", pa.string()),
    ("inning", pa.int64()),
])

DEFENSE_SCHEMA = pa.schema([
    ("game_pk", pa.int64()),
    ("play_id", pa.string()),
    ("pitcher_id", pa.int64()),
    ("catcher_id", pa.int64()),
    ("first_id", pa.int64()),
    ("second_id", pa.int64()),
    ("third_id", pa.int64()),
    ("shortstop_id", pa.int64()),
    ("left_id", pa.int64()),
    ("center_id", pa.int64()),
    ("right_id", pa.int64()),
])



def build_dim_teams():
    raw_file = RAW_DIR / "teams" / "teams.json"
    if not should_build("dim_teams.parquet", [raw_file]):
        return
    data = load_json(raw_file)
    rows = [{
        "team_id": t.get("id"),
        "team_name": t.get("name"),
        "abbreviation": t.get("abbreviation"),
        "short_name": t.get("shortName"),
        "location_name": t.get("locationName"),
        "venue_id": t.get("venue", {}).get("id"),
        "venue_name": t.get("venue", {}).get("name"),
        "league_id": t.get("league", {}).get("id"),
        "league_name": t.get("league", {}).get("name"),
        "division_id": t.get("division", {}).get("id"),
        "division_name": t.get("division", {}).get("name"),
        "first_year": t.get("firstYearOfPlay"),
    } for t in data.get("teams", [])]
    save_parquet(pd.DataFrame(rows), "dim_teams.parquet")


def build_fact_games():
    raw_files = sorted((RAW_DIR / "schedule").glob("schedule_*.json"))
    if not should_build("fact_games.parquet", raw_files):
        return
    rows = []
    for path in raw_files:
        data = load_json(path)
        game_date = path.stem.replace("schedule_", "")
        for day in data.get("dates", []):
            for g in day.get("games", []):
                teams = g.get("teams", {})
                away, home = teams.get("away", {}), teams.get("home", {})
                ls = g.get("linescore", {})
                rows.append({
                    "game_pk": g.get("gamePk"),
                    "game_date": game_date,
                    "status": g.get("status", {}).get("detailedState"),
                    "away_team_id": away.get("team", {}).get("id"),
                    "away_team_name": away.get("team", {}).get("name"),
                    "away_wins": away.get("leagueRecord", {}).get("wins"),
                    "away_losses": away.get("leagueRecord", {}).get("losses"),
                    "away_win_pct": away.get("leagueRecord", {}).get("pct"),
                    "away_probable_pitcher": away.get("probablePitcher", {}).get("fullName"),
                    "away_score": away.get("score"),
                    "away_is_winner": away.get("isWinner"),
                    "home_team_id": home.get("team", {}).get("id"),
                    "home_team_name": home.get("team", {}).get("name"),
                    "home_wins": home.get("leagueRecord", {}).get("wins"),
                    "home_losses": home.get("leagueRecord", {}).get("losses"),
                    "home_win_pct": home.get("leagueRecord", {}).get("pct"),
                    "home_probable_pitcher": home.get("probablePitcher", {}).get("fullName"),
                    "home_score": home.get("score"),
                    "home_is_winner": home.get("isWinner"),
                    "innings_count": ls.get("currentInning"),
                    "venue_name": g.get("venue", {}).get("name"),
                })
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.drop_duplicates(subset=["game_pk"], keep="last")
    save_parquet(df, "fact_games.parquet")


def build_fact_linescore():
    raw_files = sorted((RAW_DIR / "schedule").glob("schedule_*.json"))
    if not should_build("fact_linescore.parquet", raw_files):
        return
    rows = []
    for path in raw_files:
        data = load_json(path)
        for day in data.get("dates", []):
            for g in day.get("games", []):
                for inning in g.get("linescore", {}).get("innings", []):
                    rows.append({
                        "game_pk": g.get("gamePk"),
                        "inning_num": inning.get("num"),
                        "away_runs": inning.get("away", {}).get("runs"),
                        "away_hits": inning.get("away", {}).get("hits"),
                        "home_runs": inning.get("home", {}).get("runs"),
                        "home_hits": inning.get("home", {}).get("hits"),
                    })
    save_parquet(pd.DataFrame(rows), "fact_linescore.parquet")


def build_fact_team_stats():
    raw_files = list((RAW_DIR / "team_stats").glob("team_*.json"))
    if not should_build("fact_team_stats.parquet", raw_files):
        return
    rows = []
    for path in raw_files:
        parts = path.stem.split("_")
        group, season = parts[1], parts[2]
        for sb in load_json(path).get("stats", []):
            for split in sb.get("splits", []):
                row = {
                    "group": group,
                    "season": season,
                    "team_id": split.get("team", {}).get("id"),
                    "team_name": split.get("team", {}).get("name"),
                }
                row.update(split.get("stat", {}))
                rows.append(row)
    df = pd.DataFrame(rows)
    # Convert string values to numeric to prevent TypeError in downstream processing
    numeric_cols = df.select_dtypes(include=['object']).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    save_parquet(df, "fact_team_stats.parquet")


def build_fact_player_stats():
    player_dir = RAW_DIR / "player_stats"
    if not player_dir.exists():
        log.warning("No player_stats directory found — skipping")
        return
    raw_files = list(player_dir.glob("player_*.json"))
    if not should_build("fact_player_stats.parquet", raw_files):
        return
    rows = []
    for path in raw_files:
        parts = path.stem.split("_")
        group = parts[1]
        season = parts[2]
        for sb in load_json(path).get("stats", []):
            for split in sb.get("splits", []):
                row = {
                    "group": group,
                    "season": season,
                    "player_id": split.get("player", {}).get("id"),
                    "player_name": split.get("player", {}).get("fullName"),
                    "team_id": split.get("team", {}).get("id"),
                    "team_name": split.get("team", {}).get("name"),
                    "position": split.get("position", {}).get("abbreviation"),
                }
                row.update(split.get("stat", {}))
                rows.append(row)
    df = pd.DataFrame(rows)
    # Convert string values to numeric to prevent TypeError in downstream processing
    numeric_cols = df.select_dtypes(include=['object']).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    save_parquet(df, "fact_player_stats.parquet")


def build_fact_batter_pitcher_matchups():
    feed_dir = RAW_DIR / "game_feeds"
    raw_files = sorted(feed_dir.glob("game_*.json"))
    if not should_build("fact_batter_pitcher_matchups.parquet", raw_files):
        return
    rows = []
    total_files = len(raw_files)
    for idx, path in enumerate(raw_files, 1):
        try:
            data = load_json(path)
            game_pk = int(path.stem.replace("game_", ""))
            all_plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])
            for play in all_plays:
                matchup = play.get("matchup", {})
                result = play.get("result", {})
                batter = matchup.get("batter", {})
                pitcher = matchup.get("pitcher", {})
                event_type = result.get("eventType")
                
                if not batter.get("id") or not pitcher.get("id"):
                    continue
                if event_type in SKIP_MATCHUP_EVENT_TYPES:
                    continue

                # Denominator math: Plate Appearances vs At Bats
                is_pa = 1
                not_an_ab = ["walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf"]
                is_ab = 0 if event_type in not_an_ab else 1

                rows.append({
                    "game_pk": game_pk,
                    "batter_id": batter.get("id"),
                    "batter_name": batter.get("fullName"),
                    "pitcher_id": pitcher.get("id"),
                    "pitcher_name": pitcher.get("fullName"),
                    "event": result.get("event"),
                    "event_type": event_type,
                    "is_pa": is_pa,
                    "is_ab": is_ab,
                    "is_hit": 1 if event_type in ["single", "double", "triple", "home_run"] else 0,
                    "is_hr": 1 if event_type == "home_run" else 0,
                    "is_so": 1 if event_type == "strikeout" else 0,
                    "is_multi": 0,  # Will be backfilled from game logs
                })
        except Exception as e:
            log.error(f"Error processing {path.name}: {e}")
            continue
        
        # Progress logging for large datasets
        if idx % 100 == 0 or idx == total_files:
            log.info(f"Processed {idx}/{total_files} game feed files")
    
    df = pd.DataFrame(rows)
    save_parquet(df, "fact_batter_pitcher_matchups.parquet")


def build_fact_batter_game_logs():
    if not should_build_derived("fact_batter_game_logs.parquet", "fact_batter_pitcher_matchups.parquet"):
        return
    src = OUT_DIR / "fact_batter_pitcher_matchups.parquet"
    df = pl.read_parquet(src)
    agg = df.group_by(["game_pk", "batter_id", "batter_name"]).agg([
        pl.col("is_pa").sum(),
        pl.col("is_ab").sum(),
        pl.col("is_hit").sum(),
        pl.col("is_hr").sum(),
        pl.col("is_so").sum(),
    ]).rename({"is_pa": "is_pa", "is_ab": "is_ab", "is_hit": "is_hit", "is_hr": "is_hr", "is_so": "is_so"})
    agg = agg.with_columns(
        (pl.col("is_hit") >= 2).cast(pl.Int32).alias("is_multi")
    )
    agg.write_parquet(OUT_DIR / "fact_batter_game_logs.parquet")
    log.info("Wrote fact_batter_game_logs.parquet (%d rows)", len(agg))


def build_fact_pitcher_game_logs():
    if not should_build_derived("fact_pitcher_game_logs.parquet", "fact_batter_pitcher_matchups.parquet"):
        return
    src = OUT_DIR / "fact_batter_pitcher_matchups.parquet"
    df = pl.read_parquet(src)
    agg = df.group_by(["game_pk", "pitcher_id", "pitcher_name"]).agg([
        pl.col("is_hit").sum(),
        pl.col("is_hr").sum(),
        pl.col("is_so").sum(),
    ]).rename({"is_hit": "is_hit", "is_hr": "is_hr", "is_so": "is_so"})
    agg.write_parquet(OUT_DIR / "fact_pitcher_game_logs.parquet")
    log.info("Wrote fact_pitcher_game_logs.parquet (%d rows)", len(agg))


def backfill_is_multi():
    """Backfill is_multi in matchups after game logs are built."""
    matchups_path = OUT_DIR / "fact_batter_pitcher_matchups.parquet"
    game_logs_path = OUT_DIR / "fact_batter_game_logs.parquet"
    
    if not matchups_path.exists() or not game_logs_path.exists():
        log.warning("Cannot backfill is_multi - required files missing")
        return
    
    matchups = pd.read_parquet(matchups_path)
    game_logs = pd.read_parquet(game_logs_path)
    
    # Create mapping of (game_pk, batter_id) -> is_multi
    multi_map = game_logs.set_index(["game_pk", "batter_id"])["is_multi"].to_dict()
    
    # Fill is_multi in matchups
    matchups["is_multi"] = (
        matchups.set_index(["game_pk", "batter_id"]).index
          .map(multi_map)
          .fillna(0)
          .astype(int)
    )
    
    # Save updated matchups
    matchups.to_parquet(matchups_path, index=False, engine='pyarrow')
    log.info("Backfilled is_multi from game logs for %d matchups", len(matchups))


def build_fact_standings():
    raw_files = list((RAW_DIR / "standings").glob("standings_*.json"))
    if not should_build("fact_standings.parquet", raw_files):
        return
    teams_data = load_json(RAW_DIR / "teams" / "teams.json")
    team_division = {
        t.get("id"): t.get("division", {}).get("name")
        for t in teams_data.get("teams", [])
    }
    rows = []
    for path in raw_files:
        season = path.stem.replace("standings_", "")
        for rec in load_json(path).get("records", []):
            div = rec.get("division", {})
            for tr in rec.get("teamRecords", []):
                splits = {s["type"]: s for s in tr.get("records", {}).get("splitRecords", [])}
                xwl = tr.get("expectedRecords", {}).get("xWinLoss", "0-0").split("-")
                team_id = tr.get("team", {}).get("id")
                rows.append({
                    "season": season,
                    "team_id": team_id,
                    "division_name": div.get("name") or team_division.get(team_id),
                    "team_name": tr.get("team", {}).get("name"),
                    "wins": tr.get("wins"),
                    "losses": tr.get("losses"),
                    "win_pct": tr.get("winningPercentage"),
                    "games_back": tr.get("gamesBack"),
                    "run_diff": tr.get("runDifferential"),
                    "runs_scored": tr.get("runsScored"),
                    "runs_allowed": tr.get("runsAllowed"),
                    "home_wins": splits.get("home", {}).get("wins"),
                    "away_wins": splits.get("away", {}).get("wins"),
                    "one_run_wins": splits.get("oneRun", {}).get("wins"),
                    "extra_inn_wins": splits.get("extraInning", {}).get("wins"),
                    "xwins": xwl[0],
                    "xlosses": xwl[-1],
                })
    save_parquet(pd.DataFrame(rows), "fact_standings.parquet")


# --- HEAVY STREAMING / BATCH-APPEND ENGINES ---

def build_fact_pitches():
    feed_dir = RAW_DIR / "game_feeds"
    raw_files = sorted(feed_dir.glob("game_*.json"))
    if not should_build("fact_pitches.parquet", raw_files):
        return

    out_file = OUT_DIR / "fact_pitches.parquet"
    if out_file.exists(): out_file.unlink()

    rows = []
    total_files = len(raw_files)

    for idx, path in enumerate(raw_files, 1):
        try:
            data = load_json(path)
            game_pk = int(path.stem.replace("game_", ""))
            all_plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])

            for play in all_plays:
                play_events = play.get("playEvents", [])
                batter = play.get("matchup", {}).get("batter", {})
                pitcher = play.get("matchup", {}).get("pitcher", {})

                for ev in play_events:
                    pitch = ev.get("pitchData")
                    if not pitch: continue

                    rows.append({
                        "game_pk": game_pk,
                        "play_id": play.get("playId"),
                        "batter_id": batter.get("id"),
                        "pitcher_id": pitcher.get("id"),
                        "pitch_type": ev.get("details", {}).get("type", {}).get("description"),
                        "start_speed": pitch.get("startSpeed"),
                        "end_speed": pitch.get("endSpeed"),
                        "spin_rate": pitch.get("breaks", {}).get("spinRate"),
                        "spin_direction": pitch.get("breaks", {}).get("spinDirection"),
                        "break_x": pitch.get("breaks", {}).get("breakHorizontal"),
                        "break_z": pitch.get("breaks", {}).get("breakVertical"),
                        "plate_x": pitch.get("coordinates", {}).get("pX"),
                        "plate_z": pitch.get("coordinates", {}).get("pZ"),
                        "sz_top": pitch.get("strikeZoneTop"),
                        "sz_bot": pitch.get("strikeZoneBottom"),
                        "zone": pitch.get("zone"),
                        "call": ev.get("details", {}).get("call", {}).get("description"),
                        "inning": play.get("about", {}).get("inning"),
                        "is_top_inning": play.get("about", {}).get("isTopInning"),
                        "count_balls": play.get("count", {}).get("balls"),
                        "count_strikes": play.get("count", {}).get("strikes"),
                    })
        except Exception as e:
            log.error(f"Error processing {path.name}: {e}")
            continue
        
        # Progress logging for large datasets
        if idx % 100 == 0 or idx == total_files:
            log.info(f"Processed {idx}/{total_files} game feed files for pitches")

    save_parquet(pd.DataFrame(rows), "fact_pitches.parquet")
    log.info("Wrote fact_pitches.parquet (Total: %d rows)", len(rows))


def build_fact_statcast():
    feed_dir = RAW_DIR / "game_feeds"
    raw_files = sorted(feed_dir.glob("game_*.json"))
    if not should_build("fact_statcast.parquet", raw_files):
        return

    out_file = OUT_DIR / "fact_statcast.parquet"
    if out_file.exists(): out_file.unlink()

    rows = []
    total_files = len(raw_files)

    for idx, path in enumerate(raw_files, 1):
        try:
            data = load_json(path)
            game_pk = int(path.stem.replace("game_", ""))
            all_plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])

            for play in all_plays:
                if not play: continue
                batter = play.get("matchup", {}).get("batter", {})
                pitcher = play.get("matchup", {}).get("pitcher", {})

                for ev in play.get("playEvents", []):
                    hit = ev.get("hitData")
                    if not hit: continue

                    rows.append({
                        "game_pk": game_pk,
                        "batter_id": batter.get("id"),
                        "pitcher_id": pitcher.get("id"),
                        "exit_velocity": hit.get("launchSpeed"),
                        "launch_angle": hit.get("launchAngle"),
                        "spray_angle": hit.get("sprayAngle"),
                        "hit_distance": hit.get("totalDistance"),
                        "is_barrel": hit.get("isBarrel"),
                        "is_hard_hit": 1 if (hit.get("launchSpeed") or 0) >= 95 else 0,
                        "xBA": hit.get("estimatedBA"),
                        "xSLG": hit.get("estimatedSLG"),
                        "xWOBA": hit.get("estimatedWOBA"),
                        "event": play.get("result", {}).get("event"),
                    })
        except Exception as e:
            log.error(f"Error processing {path.name}: {e}")
            continue
        
        # Progress logging for large datasets
        if idx % 100 == 0 or idx == total_files:
            log.info(f"Processed {idx}/{total_files} game feed files for statcast")

    save_parquet(pd.DataFrame(rows), "fact_statcast.parquet")
    log.info("Wrote fact_statcast.parquet (Total: %d rows)", len(rows))


def build_fact_events():
    feed_dir = RAW_DIR / "game_feeds"
    raw_files = sorted(feed_dir.glob("game_*.json"))
    if not should_build("fact_events.parquet", raw_files):
        return

    out_file = OUT_DIR / "fact_events.parquet"
    if out_file.exists(): out_file.unlink()

    rows = []
    total_files = len(raw_files)

    for idx, path in enumerate(raw_files, 1):
        try:
            data = load_json(path)
            game_pk = int(path.stem.replace("game_", ""))
            all_plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])

            for play in all_plays:
                result = play.get("result", {})
                matchup = play.get("matchup", {})
                batter = matchup.get("batter", {})
                pitcher = matchup.get("pitcher", {})

                rows.append({
                    "game_pk": game_pk,
                    "play_id": play.get("playId"),
                    "batter_id": batter.get("id"),
                    "pitcher_id": pitcher.get("id"),
                    "event": result.get("event"),
                    "event_type": result.get("eventType"),
                    "description": result.get("description"),
                    "rbi": result.get("rbi"),
                    "outs": result.get("outs"),
                    "inning": play.get("about", {}).get("inning"),
                    "is_top_inning": play.get("about", {}).get("isTopInning"),
                })
        except Exception as e:
            log.error(f"Error processing {path.name}: {e}")
            continue
        
        # Progress logging for large datasets
        if idx % 100 == 0 or idx == total_files:
            log.info(f"Processed {idx}/{total_files} game feed files for events")

    save_parquet(pd.DataFrame(rows), "fact_events.parquet")
    log.info("Wrote fact_events.parquet (Total: %d rows)", len(rows))


def build_fact_runners():
    feed_dir = RAW_DIR / "game_feeds"
    raw_files = sorted(feed_dir.glob("game_*.json"))
    if not should_build("fact_runners.parquet", raw_files):
        return

    out_file = OUT_DIR / "fact_runners.parquet"
    if out_file.exists(): out_file.unlink()

    rows = []
    total_files = len(raw_files)

    for idx, path in enumerate(raw_files, 1):
        try:
            data = load_json(path)
            game_pk = int(path.stem.replace("game_", ""))
            all_plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])

            for play in all_plays:
                if not play: continue
                for runner in play.get("runners", []):
                    if not runner: continue

                    details = runner.get("details") or {}
                    runner_obj = details.get("runner") or {}
                    resp_pitcher = details.get("responsiblePitcher") or {}

                    rows.append({
                        "game_pk": game_pk,
                        "play_id": play.get("playId"),
                        "runner_id": runner_obj.get("id"),
                        "start_base": runner.get("movement", {}).get("start"),
                        "end_base": runner.get("movement", {}).get("end"),
                        "is_out": runner.get("movement", {}).get("isOut"),
                        "is_scored": runner.get("movement", {}).get("isScored"),
                        "movement_reason": runner.get("movement", {}).get("movementReason"),
                        "responsible_pitcher": resp_pitcher.get("id"),
                    })
        except Exception as e:
            log.error(f"Error processing {path.name}: {e}")
            continue
        
        # Progress logging for large datasets
        if idx % 100 == 0 or idx == total_files:
            log.info(f"Processed {idx}/{total_files} game feed files for runners")

    save_parquet(pd.DataFrame(rows), "fact_runners.parquet")
    log.info("Wrote fact_runners.parquet (Total: %d rows)", len(rows))


def build_fact_pitch_locations():
    if not should_build_derived("fact_pitch_locations.parquet", "fact_pitches.parquet"):
        return
    src = OUT_DIR / "fact_pitches.parquet"
    df = pl.read_parquet(src)
    cols = [
        "game_pk", "play_id", "batter_id", "pitcher_id",
        "plate_x", "plate_z", "sz_top", "sz_bot", "zone"
    ]
    df.select(cols).write_parquet(OUT_DIR / "fact_pitch_locations.parquet")
    log.info("Wrote fact_pitch_locations.parquet (%d rows)", len(df))


def build_fact_hit_locations():
    if not should_build_derived("fact_hit_locations.parquet", "fact_statcast.parquet"):
        return
    src = OUT_DIR / "fact_statcast.parquet"
    df = pl.read_parquet(src)
    cols = [
        "game_pk", "batter_id", "pitcher_id",
        "spray_angle", "hit_distance", "launch_angle"
    ]
    df.select(cols).write_parquet(OUT_DIR / "fact_hit_locations.parquet")
    log.info("Wrote fact_hit_locations.parquet (%d rows)", len(df))


def build_fact_innings():
    feed_dir = RAW_DIR / "game_feeds"
    raw_files = sorted(feed_dir.glob("game_*.json"))
    if not should_build("fact_innings.parquet", raw_files):
        return
    rows = []
    total_files = len(raw_files)
    for idx, path in enumerate(raw_files, 1):
        try:
            data = load_json(path)
            game_pk = int(path.stem.replace("game_", ""))
            innings = data.get("liveData", {}).get("linescore", {}).get("innings", [])
            for inn in innings:
                rows.append({
                    "game_pk": game_pk,
                    "inning": inn.get("num"),
                    "away_runs": inn.get("away", {}).get("runs"),
                    "home_runs": inn.get("home", {}).get("runs"),
                    "away_hits": inn.get("away", {}).get("hits"),
                    "home_hits": inn.get("home", {}).get("hits"),
                    "away_errors": inn.get("away", {}).get("errors"),
                    "home_errors": inn.get("home", {}).get("errors"),
                })
        except Exception as e:
            log.error(f"Error processing {path.name}: {e}")
            continue
        
        # Progress logging for large datasets
        if idx % 100 == 0 or idx == total_files:
            log.info(f"Processed {idx}/{total_files} game feed files for innings")
    
    save_parquet(pd.DataFrame(rows), "fact_innings.parquet")


def build_fact_substitutions():
    feed_dir = RAW_DIR / "game_feeds"
    raw_files = sorted(feed_dir.glob("game_*.json"))
    if not should_build("fact_substitutions.parquet", raw_files):
        return

    out_file = OUT_DIR / "fact_substitutions.parquet"
    if out_file.exists(): out_file.unlink()

    rows = []
    total_files = len(raw_files)

    for idx, path in enumerate(raw_files, 1):
        try:
            data = load_json(path)
            game_pk = int(path.stem.replace("game_", ""))
            all_plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])

            for play in all_plays:
                play_id = play.get("playId")
                inning = play.get("about", {}).get("inning")

                for ev in play.get("playEvents", []):
                    for sub in ev.get("substitutions", []) or []:
                        rows.append({
                            "game_pk": game_pk,
                            "play_id": play_id,
                            "player_in": sub.get("playerIn", {}).get("fullName"),
                            "player_out": sub.get("playerOut", {}).get("fullName"),
                            "position": sub.get("position", {}).get("abbreviation"),
                            "inning": inning,
                        })
        except Exception as e:
            log.error(f"Error processing {path.name}: {e}")
            continue
        
        # Progress logging for large datasets
        if idx % 100 == 0 or idx == total_files:
            log.info(f"Processed {idx}/{total_files} game feed files for substitutions")

    save_parquet(pd.DataFrame(rows), "fact_substitutions.parquet")
    log.info("Wrote fact_substitutions.parquet (Total: %d rows)", len(rows))


def build_fact_defense():
    feed_dir = RAW_DIR / "game_feeds"
    raw_files = sorted(feed_dir.glob("game_*.json"))
    if not should_build("fact_play_defense.parquet", raw_files):
        return

    out_file = OUT_DIR / "fact_play_defense.parquet"
    if out_file.exists(): out_file.unlink()

    rows = []
    total_files = len(raw_files)

    for idx, path in enumerate(raw_files, 1):
        try:
            data = load_json(path)
            game_pk = int(path.stem.replace("game_", ""))
            all_plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])

            for play in all_plays:
                play_id = play.get("playId")
                defense = play.get("matchup", {}).get("defense", {})

                rows.append({
                    "game_pk": game_pk,
                    "play_id": play_id,
                    "pitcher_id": defense.get("pitcher", {}).get("id"),
                    "catcher_id": defense.get("catcher", {}).get("id"),
                    "first_id": defense.get("first", {}).get("id"),
                    "second_id": defense.get("second", {}).get("id"),
                    "third_id": defense.get("third", {}).get("id"),
                    "shortstop_id": defense.get("shortstop", {}).get("id"),
                    "left_id": defense.get("left", {}).get("id"),
                    "center_id": defense.get("center", {}).get("id"),
                    "right_id": defense.get("right", {}).get("id"),
                })
        except Exception as e:
            log.error(f"Error processing {path.name}: {e}")
            continue
        
        # Progress logging for large datasets
        if idx % 100 == 0 or idx == total_files:
            log.info(f"Processed {idx}/{total_files} game feed files for defense")

    save_parquet(pd.DataFrame(rows), "fact_play_defense.parquet")
    log.info("Wrote fact_play_defense.parquet (Total: %d rows)", len(rows))


def validate_raw_inputs(min_bytes=100):
    """Ensure no game_feeds are corrupted/empty before we start.
    
    Args:
        min_bytes: Minimum file size in bytes (default: 100)
    """
    for path in RAW_DIR.glob("game_feeds/*.json"):
        if path.stat().st_size < min_bytes:
            raise ValueError(f"CRITICAL: {path.name} is corrupted/empty ({path.stat().st_size} bytes < {min_bytes} bytes)!")

def run():
    # Validate raw inputs before processing
    validate_raw_inputs()
    
    build_dim_teams()
    build_fact_games()
    build_fact_linescore()
    build_fact_team_stats()
    build_fact_player_stats()
    build_fact_standings()
    build_fact_batter_pitcher_matchups()
    build_fact_batter_game_logs()
    build_fact_pitcher_game_logs()
    backfill_is_multi()

    build_fact_pitches()
    build_fact_statcast()
    build_fact_events()
    build_fact_runners()
    build_fact_pitch_locations()
    build_fact_hit_locations()
    build_fact_innings()
    build_fact_substitutions()
    build_fact_defense()

    # Run data validation checks
    try:
        from utils.data_validation import run_all_validations
        validation_results = run_all_validations()
        log.info("Data validation completed")
    except ImportError:
        log.warning("Data validation module not found - skipping validation")
    except Exception as e:
        log.error(f"Data validation failed: {e}")

    log.info("=== Normalization complete ===")


if __name__ == "__main__":
    run()