from ingestion.normalize import build_fact_substitutions, build_fact_defense
import os
from pathlib import Path

BASE = Path(__file__).parent.parent / "data" / "normalized"

# Force delete old CSVs so rebuild always happens
for fname in ["fact_substitutions.parquet", "fact_play_defense.parquet"]:
    f = BASE / fname
    if f.exists():
        print(f"Deleting old {fname}...")
        os.remove(f)

print("Rebuilding substitutions + defense only...")
build_fact_substitutions()
build_fact_defense()
print("Done.")
