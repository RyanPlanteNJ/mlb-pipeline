import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
NORM_DIR = BASE_DIR / "data" / "normalized"
NORM_DIR.mkdir(parents=True, exist_ok=True)

def run(season: int):
    raw_path = RAW_DIR / f"raw_defense_{season}.csv"
    df = pd.read_csv(raw_path, encoding="utf-8", encoding_errors="ignore")

    # basic cleanup / schema
    df = df.rename(columns={
        "team_side": "team_side",
        "player_id": "player_id",
        "player_name": "player_name",
        "position_code": "position_code",
        "position_name": "position_name",
    })

    df["season"] = season

    out_path = NORM_DIR / "fact_defense.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[DEFENSE NORM] Wrote {len(df)} rows to {out_path}")

if __name__ == "__main__":
    run(2024)
