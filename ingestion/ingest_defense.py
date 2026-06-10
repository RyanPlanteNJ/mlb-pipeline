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
        "gameType": "R",  # regular season; add P, S, etc if you want
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

def extract_defense_from_feed(game_pk: int, feed: dict) -> list[dict]:
    out = []
    boxscore = feed.get("liveData", {}).get("boxscore", {})
    teams = boxscore.get("teams", {})

    for side in ["home", "away"]:
        team_block = teams.get(side, {})
        players = team_block.get("players", {})
        for pid, pdata in players.items():
            pos = pdata.get("position")
            if not pos:
                continue

            out.append({
                "game_pk": game_pk,
                "team_side": side,
                "player_id": pdata.get("person", {}).get("id"),
                "player_name": pdata.get("person", {}).get("fullName"),
                "position_code": pos.get("code"),
                "position_name": pos.get("name"),
            })

    return out

def run(season: int):
    game_pks = get_all_game_pks_for_season(season)
    rows = []

    for game_pk in game_pks:
        try:
            feed = fetch_live_feed(game_pk)
            rows.extend(extract_defense_from_feed(game_pk, feed))
        except Exception as e:
            print(f"[DEFENSE] Failed for game_pk={game_pk}: {e}")

    df = pd.DataFrame(rows)
    out_path = RAW_DIR / f"raw_defense_{season}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[DEFENSE] Wrote {len(df)} rows to {out_path}")

if __name__ == "__main__":
    run(2024)  # or pass via CLI later
