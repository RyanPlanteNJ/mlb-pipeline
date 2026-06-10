import requests
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

def get_all_game_pks_for_season(season: int) -> list[int]:
    params = {
        "sportId": 1,
        "season": season,
        "gameType": "R",
    }
    resp = requests.get(SCHEDULE_URL, params=params)
    resp.raise_for_status()
    data = resp.json()

    game_pks = []
    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            game_pks.append(game["gamePk"])
    return game_pks

def fetch_live_feed(game_pk: int) -> dict:
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def extract_subs_from_feed(game_pk: int, feed: dict) -> list[dict]:
    out = []
    plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])

    for play in plays:
        inning = play.get("about", {}).get("inning")
        event_index = play.get("playIndex")
        events = play.get("playEvents", [])

        for ev in events:
            subs = ev.get("substitutions", [])
            for s in subs:
                out.append({
                    "game_pk": game_pk,
                    "inning": inning,
                    "event_index": event_index,
                    "player_out_id": s.get("playerOut", {}).get("id"),
                    "player_out_name": s.get("playerOut", {}).get("fullName"),
                    "player_in_id": s.get("playerIn", {}).get("id"),
                    "player_in_name": s.get("playerIn", {}).get("fullName"),
                    "position_code": s.get("position", {}).get("code"),
                    "position_name": s.get("position", {}).get("name"),
                    "role": s.get("subType"),  # e.g. "pinch hitter", "defensive substitution"
                })

    return out

def run(season: int):
    game_pks = get_all_game_pks_for_season(season)
    rows = []

    for game_pk in game_pks:
        try:
            feed = fetch_live_feed(game_pk)
            rows.extend(extract_subs_from_feed(game_pk, feed))
        except Exception as e:
            print(f"[SUBS] Failed for game_pk={game_pk}: {e}")

    df = pd.DataFrame(rows)
    out_path = RAW_DIR / f"raw_substitutions_{season}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[SUBS] Wrote {len(df)} rows to {out_path}")

if __name__ == "__main__":
    run(2024)
