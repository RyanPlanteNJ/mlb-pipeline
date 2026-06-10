"""
schedule_daily_retraining.py - Schedule automatic daily model retraining
Can be run with Windows Task Scheduler or cron for automated daily execution
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

BASE = Path(__file__).parent
LOGS_DIR = BASE / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def run_pipeline_step(step_name: str, script_path: Path) -> bool:
    """
    Run a single pipeline step
    
    Args:
        step_name: Name of the step for logging
        script_path: Path to the script to run
        
    Returns:
        True if successful, False otherwise
    """
    log_file = LOGS_DIR / f"daily_pipeline_{datetime.now().strftime('%Y%m%d')}.log"
    
    with open(log_file, 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"\n{'='*60}\n")
        f.write(f"STEP: {step_name}\n")
        f.write(f"TIME: {timestamp}\n")
        f.write(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=BASE,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per step
        )
        
        with open(log_file, 'a') as f:
            f.write(f"STDOUT:\n{result.stdout}\n")
            if result.stderr:
                f.write(f"STDERR:\n{result.stderr}\n")
            f.write(f"RETURN CODE: {result.returncode}\n")
        
        if result.returncode == 0:
            print(f"✅ {step_name} completed successfully")
            return True
        else:
            print(f"❌ {step_name} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        with open(log_file, 'a') as f:
            f.write(f"ERROR: Step timed out after 1 hour\n")
        print(f"❌ {step_name} timed out")
        return False
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"ERROR: {str(e)}\n")
        print(f"❌ {step_name} failed with exception: {str(e)}")
        return False


def run_daily_pipeline():
    """
    Run the complete daily pipeline:
    1. Data ingestion
    2. Normalization
    3. Model training
    4. Model scoring
    5. Performance tracking
    6. Data quality monitoring
    """
    print(f"\n{'='*60}")
    print(f"STARTING DAILY PIPELINE")
    print(f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    steps = [
        ("Data Ingestion", BASE / "ingestion" / "mlb_ingest.py"),
        ("Normalization", BASE / "ingestion" / "normalize.py"),
        ("Model Training", BASE / "model" / "train_batter_props.py"),
        ("Model Scoring", BASE / "model" / "score_batter_props.py"),
        ("Performance Tracking", BASE / "utils" / "performance_tracker.py"),
        ("Data Quality Monitoring", BASE / "utils" / "data_quality_monitor.py"),
    ]
    
    results = {}
    for step_name, script_path in steps:
        if not script_path.exists():
            print(f"⚠️  Skipping {step_name} - script not found: {script_path}")
            results[step_name] = "skipped"
            continue
        
        success = run_pipeline_step(step_name, script_path)
        results[step_name] = "success" if success else "failed"
        
        if not success:
            print(f"\n⚠️  Pipeline stopped due to failure in {step_name}")
            break
    
    # Generate summary
    print(f"\n{'='*60}")
    print(f"DAILY PIPELINE SUMMARY")
    print(f"TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    for step_name, status in results.items():
        emoji = "✅" if status == "success" else "❌" if status == "failed" else "⚠️"
        print(f"{emoji} {step_name}: {status.upper()}")
    
    # Save summary to file
    summary_file = LOGS_DIR / f"daily_summary_{datetime.now().strftime('%Y%m%d')}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    return all(status == "success" for status in results.values())


def create_windows_task_scheduler_instructions():
    """
    Generate instructions for setting up Windows Task Scheduler
    """
    instructions = """
# Windows Task Scheduler Setup Instructions

1. Open Task Scheduler (search for "Task Scheduler" in Windows)
2. Click "Create Task" in the right panel
3. General Tab:
   - Name: "MLB Daily Pipeline"
   - Description: "Run daily MLB data pipeline"
   - Select "Run whether user is logged on or not"
   - Check "Run with highest privileges"
4. Triggers Tab:
   - Click "New"
   - Select "Daily"
   - Set start time (e.g., 6:00 AM)
   - Click OK
5. Actions Tab:
   - Click "New"
   - Action: "Start a program"
   - Program/script: python
   - Add arguments: "E:\\MLB\\mlb_pipeline\\schedule_daily_retraining.py"
   - Start in: "E:\\MLB\\mlb_pipeline"
   - Click OK
6. Conditions Tab:
   - Uncheck "Start the task only if the computer is on AC power"
   - Check "Wake the computer to run this task" (optional)
7. Settings Tab:
   - Check "Allow task to be run on demand"
   - Check "Run task as soon as possible after a scheduled start is missed"
8. Click OK
9. Enter your Windows password when prompted
"""
    
    instructions_file = BASE / "task_scheduler_instructions.txt"
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    print(f"Task Scheduler instructions saved to: {instructions_file}")
    return instructions


def create_cron_instructions():
    """
    Generate instructions for setting up cron job (Linux/Mac)
    """
    instructions = """
# Cron Job Setup Instructions (Linux/Mac)

1. Open crontab:
   crontab -e

2. Add the following line to run daily at 6:00 AM:
   0 6 * * * cd /path/to/MLB/mlb_pipeline && /usr/bin/python3 schedule_daily_retraining.py >> logs/cron.log 2>&1

3. Save and exit (Ctrl+X, then Y, then Enter for nano; or :wq for vim)

4. Verify the cron job was added:
   crontab -l

Note: Replace /path/to/MLB/mlb_pipeline with your actual path
Note: Replace /usr/bin/python3 with your Python path (use `which python3` to find it)
"""
    
    instructions_file = BASE / "cron_instructions.txt"
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    print(f"Cron instructions saved to: {instructions_file}")
    return instructions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Schedule and run daily MLB pipeline")
    parser.add_argument("--run", action="store_true", help="Run the daily pipeline now")
    parser.add_argument("--setup-windows", action="store_true", help="Generate Windows Task Scheduler instructions")
    parser.add_argument("--setup-cron", action="store_true", help="Generate cron job instructions")
    
    args = parser.parse_args()
    
    if args.run:
        success = run_daily_pipeline()
        sys.exit(0 if success else 1)
    elif args.setup_windows:
        create_windows_task_scheduler_instructions()
    elif args.setup_cron:
        create_cron_instructions()
    else:
        print("Usage:")
        print("  python schedule_daily_retraining.py --run              # Run pipeline now")
        print("  python schedule_daily_retraining.py --setup-windows    # Generate Windows Task Scheduler instructions")
        print("  python schedule_daily_retraining.py --setup-cron       # Generate cron job instructions")
        print("\nExample: Run pipeline now")
        print("  python schedule_daily_retraining.py --run")
