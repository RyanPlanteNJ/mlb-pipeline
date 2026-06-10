"""
run_production.py - Production pipeline orchestrator with validation gate
Runs the full MLB pipeline: ingest → normalize → validate → train → score → track
"""
import sys
import logging
from pathlib import Path
from datetime import date

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.mlb_ingest import run as ingest_run
from ingestion.normalize import run as normalize_run
from model.train_batter_props import run as train_batter_run
from model.win_probability import run as winprob_run
from model.score_batter_props import run as batterprops_run
from model.score_team_props import run as teamprops_run
from utils.performance_tracker import track_batter_predictions, track_win_probability
from utils.data_validation import run_all_validations

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"pipeline_{date.today()}.log"),
        logging.StreamHandler()
    ]
)

log = logging.getLogger(__name__)


def run_production_pipeline():
    """Execute the full production pipeline with validation gate"""
    log.info("=" * 60)
    log.info("STARTING PRODUCTION MLB PIPELINE")
    log.info(f"Date: {date.today()}")
    log.info("=" * 60)
    
    try:
        # Stage 1: Ingestion
        log.info("--- STAGE 1: INGESTION ---")
        ingest_run(str(date.today()))
        log.info("✅ Data ingestion complete")
        
        # Stage 2: Normalization
        log.info("--- STAGE 2: NORMALIZATION ---")
        normalize_run()
        log.info("✅ Data normalization complete")
        
        # Stage 3: Validation Gate (Circuit Breaker)
        log.info("--- STAGE 3: VALIDATION GATE ---")
        validation_results = run_all_validations()
        all_passed = all(result[0] for result in validation_results.values())
        
        if not all_passed:
            log.error("VALIDATION FAILED - Pipeline stopped before modeling")
            for table, (passed, issues) in validation_results.items():
                if not passed:
                    log.error(f"  {table}: {issues}")
            sys.exit(1)
        
        log.info("✅ All validations passed")
        
        # Stage 4: Model Training
        log.info("--- STAGE 4: MODEL TRAINING ---")
        train_batter_run()
        winprob_run()
        log.info("✅ Model training complete")
        
        # Stage 5: Prediction Scoring
        log.info("--- STAGE 5: PREDICTION SCORING ---")
        batterprops_run()
        teamprops_run()
        log.info("✅ Prediction scoring complete")
        
        # Stage 6: Performance Tracking
        log.info("--- STAGE 6: PERFORMANCE TRACKING ---")
        track_batter_predictions()
        track_win_probability()
        log.info("✅ Performance tracking complete")
        
        log.info("=" * 60)
        log.info("PRODUCTION PIPELINE COMPLETED SUCCESSFULLY")
        log.info("=" * 60)
        
        return True
        
    except Exception as e:
        log.critical(f"PIPELINE CRASHED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    success = run_production_pipeline()
    sys.exit(0 if success else 1)