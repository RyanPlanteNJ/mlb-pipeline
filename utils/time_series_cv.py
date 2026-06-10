"""
time_series_cv.py - Time-series cross-validation for temporal data
Prevents data leakage by respecting temporal order in cross-validation
"""
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, brier_score_loss

log = logging.getLogger(__name__)


class TimeSeriesSplit:
    """
    Time-series cross-validation splitter that respects temporal order.
    Unlike random K-fold, this ensures no future data leaks into training.
    """
    
    def __init__(self, n_splits=5, test_size=0.2):
        """
        Initialize time-series splitter.
        
        Args:
            n_splits: Number of splits
            test_size: Proportion of data to use as test set for each split
        """
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, X, y=None, groups=None, date_col=None):
        """
        Generate train/test splits respecting temporal order.
        
        Args:
            X: Feature matrix (must have date_col if date_col is provided)
            y: Target variable (not used in split generation)
            groups: Group labels (not used)
            date_col: Name of date column in X (if X is a DataFrame)
        
        Yields:
            (train_idx, test_idx) tuples
        """
        n_samples = len(X)
        
        # If date column is provided, sort by date
        if date_col is not None and hasattr(X, 'columns'):
            if date_col in X.columns:
                sorted_idx = X.sort_values(date_col).index
                X = X.loc[sorted_idx]
                if y is not None:
                    y = y.loc[sorted_idx]
        
        # Calculate test set size for each split
        test_samples = int(n_samples * self.test_size / self.n_splits)
        
        # Generate splits from earliest to latest
        for i in range(self.n_splits):
            # Test set is a contiguous block at the end
            test_start = n_samples - (self.n_splits - i) * test_samples
            test_end = n_samples - (self.n_splits - i - 1) * test_samples
            
            # Ensure test_end doesn't exceed n_samples
            test_end = min(test_end, n_samples)
            
            # Train set is everything before test set
            train_end = test_start
            
            # Train indices
            train_idx = np.arange(0, train_end)
            
            # Test indices
            test_idx = np.arange(test_start, test_end)
            
            # Skip if train set is too small
            if len(train_idx) < 10:
                continue
            
            yield train_idx, test_idx


def time_series_cv_score(model, X, y, date_col=None, n_splits=5, test_size=0.2, scoring='neg_log_loss'):
    """
    Perform time-series cross-validation and return scores.
    
    Args:
        model: Scikit-learn model
        X: Feature matrix
        y: Target variable
        date_col: Name of date column in X (if X is a DataFrame)
        n_splits: Number of splits
        test_size: Proportion of data to use as test set
        scoring: Scoring metric ('neg_log_loss', 'neg_brier_score')
    
    Returns:
        Dictionary of scores (mean, std, individual scores)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    scores = []
    
    for train_idx, test_idx in tscv.split(X, y, date_col=date_col):
        # Split data
        if hasattr(X, 'iloc'):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(X_test)[:, 1]
        else:
            y_pred = model.predict(X_test)
        
        # Calculate score
        if scoring == 'neg_log_loss':
            score = -log_loss(y_test, y_pred)
        elif scoring == 'neg_brier_score':
            score = -brier_score_loss(y_test, y_pred)
        else:
            raise ValueError(f"Unknown scoring metric: {scoring}")
        
        scores.append(score)
    
    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'scores': scores,
        'n_splits': len(scores)
    }


def expanding_window_cv(model, X, y, date_col=None, min_train_size=100, step_size=None):
    """
    Expanding window cross-validation for time series.
    Training set grows with each fold, test set is fixed size.
    
    Args:
        model: Scikit-learn model
        X: Feature matrix
        y: Target variable
        date_col: Name of date column in X (if X is a DataFrame)
        min_train_size: Minimum number of samples in training set
        step_size: Number of samples to advance each fold (default: test_size)
    
    Returns:
        Dictionary of scores
    """
    n_samples = len(X)
    
    # If date column is provided, sort by date
    if date_col is not None and hasattr(X, 'columns'):
        if date_col in X.columns:
            sorted_idx = X.sort_values(date_col).index
            X = X.loc[sorted_idx]
            y = y.loc[sorted_idx]
    
    # Test set size (20% of data)
    test_size = max(50, int(n_samples * 0.2))
    
    # Step size (default to test size)
    if step_size is None:
        step_size = test_size
    
    scores = []
    
    # Start from minimum training size
    train_end = min_train_size
    
    while train_end + test_size <= n_samples:
        # Train indices
        train_idx = np.arange(0, train_end)
        
        # Test indices
        test_idx = np.arange(train_end, train_end + test_size)
        
        # Split data
        if hasattr(X, 'iloc'):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(X_test)[:, 1]
        else:
            y_pred = model.predict(X_test)
        
        # Calculate score
        score = -log_loss(y_test, y_pred)
        scores.append(score)
        
        # Advance window
        train_end += step_size
    
    return {
        'mean': np.mean(scores) if scores else 0,
        'std': np.std(scores) if scores else 0,
        'scores': scores,
        'n_splits': len(scores)
    }


def compare_cv_methods(model, X, y, date_col=None, n_splits=5):
    """
    Compare random K-fold vs time-series CV to detect data leakage.
    
    Args:
        model: Scikit-learn model
        X: Feature matrix
        y: Target variable
        date_col: Name of date column in X (if X is a DataFrame)
        n_splits: Number of splits
    
    Returns:
        Dictionary comparing both methods
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    
    # Random K-fold (may have data leakage)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    kf_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_log_loss')
    
    # Time-series CV (no data leakage)
    tscv_scores = time_series_cv_score(model, X, y, date_col=date_col, n_splits=n_splits)
    
    log.info("=== Cross-Validation Comparison ===")
    log.info(f"Random K-Fold CV Log Loss: {-kf_scores.mean():.4f} (+/- {kf_scores.std() * 2:.4f})")
    log.info(f"Time-Series CV Log Loss: {-tscv_scores['mean']:.4f} (+/- {tscv_scores['std'] * 2:.4f})")
    
    # Check for significant difference (indicates data leakage)
    diff = abs(-kf_scores.mean() - -tscv_scores['mean'])
    if diff > 0.05:
        log.warning(f"⚠️ Significant CV difference ({diff:.4f}) - possible data leakage in random K-fold")
    else:
        log.info("✅ CV scores similar - no significant data leakage detected")
    
    return {
        'random_kfold': {
            'mean': float(-kf_scores.mean()),
            'std': float(kf_scores.std()),
            'scores': kf_scores.tolist()
        },
        'time_series': tscv_scores,
        'difference': diff
    }


if __name__ == "__main__":
    print("=== Time-Series Cross-Validation Module ===")
    print("This module provides time-series cross-validation for temporal data.")
    print("\nUsage:")
    print("  from utils.time_series_cv import TimeSeriesSplit, time_series_cv_score")
    print("  tscv = TimeSeriesSplit(n_splits=5, test_size=0.2)")
    print("  scores = time_series_cv_score(model, X, y, date_col='game_date')")
