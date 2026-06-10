"""
betting_calculator.py - Betting value calculations and recommendations
Calculates Expected Value (EV), Kelly criterion, and betting recommendations
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds.
    
    Args:
        american_odds: American odds (e.g., -150, +110)
    
    Returns:
        Decimal odds (e.g., 1.67, 2.10)
    """
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds.
    
    Args:
        decimal_odds: Decimal odds (e.g., 1.67, 2.10)
    
    Returns:
        American odds (e.g., -150, +110)
    """
    if decimal_odds >= 2:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))


def calculate_implied_probability(american_odds: int) -> float:
    """Calculate implied probability from American odds (without vig).
    
    Args:
        american_odds: American odds (e.g., -150, +110)
    
    Returns:
        Implied probability (0-1)
    """
    decimal_odds = american_to_decimal(american_odds)
    return 1 / decimal_odds


def calculate_vig(home_odds: int, away_odds: int) -> float:
    """Calculate the vig/juice from a moneyline.
    
    Args:
        home_odds: Home team American odds
        away_odds: Away team American odds
    
    Returns:
        Vig as percentage (e.g., 0.045 for 4.5%)
    """
    home_prob = calculate_implied_probability(home_odds)
    away_prob = calculate_implied_probability(away_odds)
    total = home_prob + away_prob
    return total - 1


def calculate_fair_odds(american_odds: int, vig: float) -> Tuple[float, float]:
    """Calculate fair odds after removing vig.
    
    Args:
        american_odds: American odds
        vig: Vig percentage (e.g., 0.045)
    
    Returns:
        Tuple of (fair_implied_prob, fair_decimal_odds)
    """
    implied_prob = calculate_implied_probability(american_odds)
    fair_prob = implied_prob / (1 + vig)
    fair_decimal = 1 / fair_prob
    return fair_prob, fair_decimal


def calculate_ev(model_prob: float, american_odds: int, stake: float = 100) -> Dict:
    """Calculate Expected Value for a bet.
    
    Args:
        model_prob: Model's predicted win probability (0-1)
        american_odds: American odds (e.g., -150, +110)
        stake: Bet amount (default: 100)
    
    Returns:
        Dictionary with EV calculation details
    """
    decimal_odds = american_to_decimal(american_odds)
    
    if american_odds > 0:
        # Underdog: win profit = odds * stake
        payout = decimal_odds * stake
    else:
        # Favorite: win profit = stake * (decimal_odds - 1)
        payout = stake * (decimal_odds - 1)
    
    win_ev = model_prob * payout
    lose_ev = (1 - model_prob) * stake
    ev = win_ev - lose_ev
    
    # Calculate edge vs market
    market_prob = calculate_implied_probability(american_odds)
    edge = model_prob - market_prob
    
    return {
        'model_prob': model_prob,
        'market_prob': market_prob,
        'edge': edge,
        'american_odds': american_odds,
        'decimal_odds': decimal_odds,
        'payout': payout,
        'stake': stake,
        'ev': ev,
        'ev_percent': (ev / stake) * 100
    }


def calculate_kelly_criterion(model_prob: float, american_odds: int) -> Dict:
    """Calculate Kelly criterion optimal bet size.
    
    Args:
        model_prob: Model's predicted win probability (0-1)
        american_odds: American odds (e.g., -150, +110)
    
    Returns:
        Dictionary with Kelly calculation details
    """
    decimal_odds = american_to_decimal(american_odds)
    
    # Kelly formula: (bp - q) / b
    # where b = decimal odds - 1, p = win prob, q = loss prob
    b = decimal_odds - 1
    p = model_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Fractional Kelly (half-Kelly is common for risk management)
    half_kelly = kelly / 2
    quarter_kelly = kelly / 4
    
    return {
        'kelly': max(0, kelly),  # Never bet on negative edge
        'half_kelly': max(0, half_kelly),
        'quarter_kelly': max(0, quarter_kelly),
        'decimal_odds': decimal_odds
    }


def get_betting_recommendation(model_prob: float, american_odds: int) -> Dict:
    """Get comprehensive betting recommendation.
    
    Args:
        model_prob: Model's predicted win probability (0-1)
        american_odds: American odds (e.g., -150, +110)
    
    Returns:
        Dictionary with full betting analysis
    """
    ev_result = calculate_ev(model_prob, american_odds)
    kelly_result = calculate_kelly_criterion(model_prob, american_odds)
    
    # Determine recommendation
    if ev_result['ev'] > 0:
        if ev_result['edge'] >= 0.05:
            strength = "STRONG"
        elif ev_result['edge'] >= 0.03:
            strength = "MODERATE"
        elif ev_result['edge'] >= 0.01:
            strength = "SMALL"
        else:
            strength = "MINIMAL"
    else:
        strength = "NO BET"
    
    return {
        'recommendation': strength,
        'ev': ev_result,
        'kelly': kelly_result,
        'should_bet': ev_result['ev'] > 0
    }


def calculate_team_betting_recommendations(
    df: pd.DataFrame,
    home_prob_col: str = 'home_win_prob',
    away_prob_col: str = 'away_win_prob',
    home_odds_col: str = 'home_odds',
    away_odds_col: str = 'away_odds'
) -> pd.DataFrame:
    """Calculate betting recommendations for all games in a dataframe.
    
    Args:
        df: DataFrame with game predictions and odds
        home_prob_col: Column name for home win probability
        away_prob_col: Column name for away win probability
        home_odds_col: Column name for home American odds
        away_odds_col: Column name for away American odds
    
    Returns:
        DataFrame with betting recommendations added
    """
    results = []
    
    for _, row in df.iterrows():
        home_prob = row[home_prob_col]
        away_prob = row[away_prob_col]
        
        # Skip if odds not available
        if pd.isna(row[home_odds_col]) or pd.isna(row[away_odds_col]):
            results.append({
                'home_recommendation': 'NO ODDS',
                'away_recommendation': 'NO ODDS',
                'home_ev': None,
                'away_ev': None,
                'home_edge': None,
                'away_edge': None,
                'home_kelly': None,
                'away_kelly': None
            })
            continue
        
        home_odds = row[home_odds_col]
        away_odds = row[away_odds_col]
        
        # Calculate recommendations
        home_rec = get_betting_recommendation(home_prob, home_odds)
        away_rec = get_betting_recommendation(away_prob, away_odds)
        
        results.append({
            'home_recommendation': home_rec['recommendation'],
            'away_recommendation': away_rec['recommendation'],
            'home_ev': home_rec['ev']['ev'],
            'away_ev': away_rec['ev']['ev'],
            'home_edge': home_rec['ev']['edge'],
            'away_edge': away_rec['ev']['edge'],
            'home_kelly': home_rec['kelly']['kelly'],
            'away_kelly': away_rec['kelly']['kelly']
        })
    
    # Add results to dataframe
    result_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), result_df], axis=1)


def format_betting_output(rec: Dict) -> str:
    """Format betting recommendation for display.
    
    Args:
        rec: Betting recommendation dictionary
    
    Returns:
        Formatted string
    """
    if not rec['should_bet']:
        return f"NO BET (EV: ${rec['ev']['ev']:.2f})"
    
    return (
        f"{rec['recommendation']} BET\n"
        f"  Edge: +{rec['ev']['edge']*100:.1f}%\n"
        f"  EV: ${rec['ev']['ev']:.2f} per $100\n"
        f"  Kelly: {rec['kelly']['kelly']*100:.1f}% of bankroll"
    )


if __name__ == "__main__":
    # Test calculations
    print("=== Betting Calculator Tests ===\n")
    
    # Test 1: Favorite with edge
    print("Test 1: Favorite with edge")
    rec = get_betting_recommendation(0.58, -120)
    print(format_betting_output(rec))
    print()
    
    # Test 2: Underdog with edge
    print("Test 2: Underdog with edge")
    rec = get_betting_recommendation(0.45, +130)
    print(format_betting_output(rec))
    print()
    
    # Test 3: No edge
    print("Test 3: No edge")
    rec = get_betting_recommendation(0.50, -110)
    print(format_betting_output(rec))
