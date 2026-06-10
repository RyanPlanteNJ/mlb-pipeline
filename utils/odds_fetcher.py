"""
odds_fetcher.py - Fetch MLB odds from The Odds API
Integrates with The Odds API to get live betting odds from sportsbooks
"""
import requests
import pandas as pd
from datetime import datetime, date
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class OddsFetcher:
    """Fetch and cache MLB odds from The Odds API"""
    
    def __init__(self, api_key: str, cache_dir: str = None):
        """
        Initialize OddsFetcher.
        
        Args:
            api_key: The Odds API key
            cache_dir: Directory to cache odds data (default: data/odds)
        """
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent / "data" / "odds"
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_mlb_odds(
        self,
        date: str = None,
        regions: str = "us",
        markets: str = "h2h",
        bookmakers: str = "draftkings",
        odds_format: str = "american"
    ) -> List[Dict]:
        """
        Fetch MLB odds for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
            regions: Regions for odds (default: us)
            markets: Betting markets (default: h2h for moneyline)
            bookmakers: Sportsbooks to fetch (default: draftkings)
            odds_format: Odds format (default: american)
        
        Returns:
            List of game odds dictionaries
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        url = f"{self.base_url}/sports/baseball_mlb/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "bookmakers": bookmakers,
            "oddsFormat": odds_format,
            "date": date
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            log.info(f"Fetched {len(data)} MLB games from The Odds API for {date}")
            return data
        
        except requests.exceptions.RequestException as e:
            log.error(f"Error fetching odds from The Odds API: {e}")
            return []
    
    def cache_odds(self, odds_data: List[Dict], date: str = None) -> Path:
        """
        Cache odds data to local file.
        
        Args:
            odds_data: List of game odds dictionaries
            date: Date in YYYY-MM-DD format (default: today)
        
        Returns:
            Path to cached file
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        cache_file = self.cache_dir / f"mlb_odds_{date}.json"
        
        with open(cache_file, 'w') as f:
            json.dump(odds_data, f, indent=2)
        
        log.info(f"Cached odds to {cache_file}")
        return cache_file
    
    def load_cached_odds(self, date: str = None) -> Optional[List[Dict]]:
        """
        Load cached odds data from local file.
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
        
        Returns:
            List of game odds dictionaries or None if not found
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        cache_file = self.cache_dir / f"mlb_odds_{date}.json"
        
        if not cache_file.exists():
            log.warning(f"No cached odds found for {date}")
            return None
        
        with open(cache_file, 'r') as f:
            data = json.load(f)
        
        log.info(f"Loaded {len(data)} games from cached odds for {date}")
        return data
    
    def get_odds(self, date: str = None, use_cache: bool = True, force_refresh: bool = False) -> List[Dict]:
        """
        Get odds, using cache if available.
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
            use_cache: Whether to use cached data (default: True)
            force_refresh: Force refresh from API (default: False)
        
        Returns:
            List of game odds dictionaries
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        if force_refresh:
            odds_data = self.fetch_mlb_odds(date)
            if odds_data:
                self.cache_odds(odds_data, date)
            return odds_data
        
        if use_cache:
            cached = self.load_cached_odds(date)
            if cached:
                return cached
        
        # Fetch from API if cache not available
        odds_data = self.fetch_mlb_odds(date)
        if odds_data:
            self.cache_odds(odds_data, date)
        
        return odds_data
    
    def parse_odds_to_dataframe(self, odds_data: List[Dict]) -> pd.DataFrame:
        """
        Parse odds data into a pandas DataFrame.
        
        Args:
            odds_data: List of game odds dictionaries
        
        Returns:
            DataFrame with parsed odds data
        """
        rows = []
        
        for game in odds_data:
            game_id = game.get('id')
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            commence_time = game.get('commence_time')
            
            # Extract odds from bookmakers
            for bookmaker in game.get('bookmakers', []):
                bookmaker_key = bookmaker.get('key')
                
                for market in bookmaker.get('markets', []):
                    market_key = market.get('key')
                    
                    if market_key == 'h2h':  # Moneyline
                        outcomes = market.get('outcomes', [])
                        
                        home_odds = None
                        away_odds = None
                        
                        for outcome in outcomes:
                            name = outcome.get('name')
                            price = outcome.get('price')
                            
                            if name == home_team:
                                home_odds = price
                            elif name == away_team:
                                away_odds = price
                        
                        rows.append({
                            'game_id': game_id,
                            'home_team': home_team,
                            'away_team': away_team,
                            'commence_time': commence_time,
                            'bookmaker': bookmaker_key,
                            'home_odds': home_odds,
                            'away_odds': away_odds
                        })
        
        df = pd.DataFrame(rows)
        
        if not df.empty:
            df['commence_time'] = pd.to_datetime(df['commence_time'])
        
        return df
    
    def match_odds_to_games(
        self,
        odds_df: pd.DataFrame,
        games_df: pd.DataFrame,
        home_team_col: str = 'home_team_name',
        away_team_col: str = 'away_team_name'
    ) -> pd.DataFrame:
        """
        Match odds data to games dataframe by team names.
        
        Args:
            odds_df: DataFrame with odds data
            games_df: DataFrame with games data
            home_team_col: Column name for home team in games_df
            away_team_col: Column name for away team in games_df
        
        Returns:
            DataFrame with odds merged into games data
        """
        # Normalize team names for matching
        odds_df['home_team_normalized'] = odds_df['home_team'].str.lower().str.strip()
        odds_df['away_team_normalized'] = odds_df['away_team'].str.lower().str.strip()
        
        games_df['home_team_normalized'] = games_df[home_team_col].str.lower().str.strip()
        games_df['away_team_normalized'] = games_df[away_team_col].str.lower().str.strip()
        
        # Merge on normalized team names
        merged = games_df.merge(
            odds_df,
            on=['home_team_normalized', 'away_team_normalized'],
            how='left'
        )
        
        # Drop normalized columns
        merged = merged.drop(columns=['home_team_normalized', 'away_team_normalized'])
        
        log.info(f"Matched odds to {merged['home_odds'].notna().sum()} games")
        return merged


def fetch_odds_for_pipeline(api_key: str, date: str = None) -> pd.DataFrame:
    """
    Convenience function to fetch odds for pipeline use.
    
    Args:
        api_key: The Odds API key
        date: Date in YYYY-MM-DD format (default: today)
    
    Returns:
        DataFrame with odds data
    """
    fetcher = OddsFetcher(api_key)
    odds_data = fetcher.get_odds(date)
    
    if not odds_data:
        log.warning("No odds data available")
        return pd.DataFrame()
    
    return fetcher.parse_odds_to_dataframe(odds_data)


if __name__ == "__main__":
    # Test with a placeholder API key
    print("=== Odds Fetcher Test ===")
    print("To test, provide a valid API key:")
    print("python utils/odds_fetcher.py YOUR_API_KEY")
