# MLB Pipeline

A comprehensive MLB prediction pipeline with ensemble machine learning models, hyperparameter tuning, and betting integration.

## Features

- **Ensemble Models**: XGBoost + LightGBM + CatBoost for robust predictions
- **Hyperparameter Tuning**: Optuna-based automated hyperparameter optimization
- **Time-Series Cross-Validation**: Prevents data leakage in temporal data
- **Betting Integration**: Real-time odds from The Odds API with EV, Kelly Criterion calculations
- **Advanced Features**: Park factors, momentum metrics, composite ratings
- **Discord Bot**: Interactive bot for predictions and betting recommendations

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set up environment variables:

```bash
# The Odds API key (for betting integration)
$env:ODDS_API_KEY="your_api_key_here"
```

Get your free API key at https://the-odds-api.com/

## Usage

### Data Ingestion & Normalization

```bash
# Ingest MLB data
python ingestion/mlb_ingest.py

# Normalize data
python ingestion/normalize.py
```

### Model Training

```bash
# Train win probability model
python model/win_probability.py

# Train batter props models
python model/train_batter_props.py

# With hyperparameter tuning
python model/win_probability.py --tune --trials 20
python model/train_batter_props.py --tune --trials 20
```

### Scoring

```bash
# Score team props with betting integration
python model/score_team_props.py --fetch-odds

# Score batter props
python model/score_batter_props.py
```

### Discord Bot

```bash
# Start the Discord bot
python discord_bot.py
```

Available commands:
- `/team_props` - Team predictions with betting recommendations
- `/playerprops` - Player prop predictions
- `/matchups` - Game matchups
- `/player_history` - Historical player performance
- `/matchup` - Batter-pitcher matchup analysis
- `/hot_players` - Players on hot/cold streaks
- `/top_picks` - Top predictions
- `/performance` - Model performance metrics

## Model Architecture

### Win Probability Model
- **Features**: Team stats, standings, recent form, park factors
- **Models**: XGBoost, LightGBM, CatBoost ensemble
- **Output**: Home/away win probability

### Batter Props Models
- **Targets**: HIT, HR, MULTI-HIT, STRIKEOUT
- **Features**: Player stats, recent performance, pitcher matchups, park factors
- **Models**: XGBoost, LightGBM, CatBoost ensemble
- **Output**: Probability for each prop type

## Betting Integration

The pipeline integrates with The Odds API to provide:
- **Real-time odds** from multiple sportsbooks
- **Expected Value (EV)** calculations
- **Kelly Criterion** bet sizing
- **Edge detection** (model vs market)
- **Betting recommendations** (BET/PASS)

## Project Structure

```
mlb_pipeline/
├── ingestion/          # Data ingestion scripts
├── model/              # Model training and scoring
├── utils/              # Utility modules
├── matchups/           # Matchups UI
├── discord_bot.py      # Discord bot
├── requirements.txt    # Python dependencies
└── README.md
```

## Data Requirements

The pipeline requires MLB data from:
- MLB Statcast API
- MLB Gameday API
- The Odds API (optional, for betting)

## License

MIT License
