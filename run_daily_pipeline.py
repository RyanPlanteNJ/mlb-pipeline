"""
run_daily_pipeline.py - Automated daily pipeline execution
Runs the full MLB pipeline: ingest → normalize → train → score → track
"""
import logging
from pathlib import Path
from datetime import date
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.mlb_ingest import run as ingest_run
from ingestion.normalize import run as normalize_run
from model.train_batter_props import run as train_batter_run
from model.win_probability import run as winprob_run
from model.score_batter_props import run as batterprops_run
from model.score_team_props import run as teamprops_run
from utils.performance_tracker import track_batter_predictions, track_win_probability

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(Path(__file__).parent / "logs" / f"pipeline_{date.today()}.log"),
        logging.StreamHandler()
    ]
)

log = logging.getLogger(__name__)


def run_daily_pipeline():
    """Execute the full daily pipeline"""
    log.info("=" * 60)
    log.info("STARTING DAILY MLB PIPELINE")
    log.info(f"Date: {date.today()}")
    log.info("=" * 60)
    
    try:
        today = str(date.today())

        # Step 1: Ingest data
        log.info("Step 1: Ingesting data from MLB API...")
        ingest_run(today)
        log.info("✅ Data ingestion complete")
        
        # Step 2: Normalize data
        log.info("Step 2: Normalizing data...")
        normalize_run()
        log.info("✅ Data normalization complete")
        
        # Step 3: Train models
        log.info("Step 3: Training models...")
        train_batter_run()
        log.info("✅ Model training complete")
        
        # Step 4: Score predictions
        log.info("Step 4: Scoring predictions...")
        batterprops_run()
        winprob_run()
        teamprops_run()
        log.info("✅ Prediction scoring complete")
        
        # Step 5: Track performance
        log.info("Step 5: Tracking model performance...")
        track_batter_predictions()
        track_win_probability()
        log.info("✅ Performance tracking complete")
        
        log.info("=" * 60)
        log.info("DAILY PIPELINE COMPLETED SUCCESSFULLY")
        log.info("=" * 60)
        
        return True
        
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_daily_pipeline()
    sys.exit(0 if success else 1)
