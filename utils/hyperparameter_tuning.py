"""
hyperparameter_tuning.py - Automated hyperparameter optimization using Optuna
Optimizes XGBoost, LightGBM, and CatBoost hyperparameters for better model performance
"""
import optuna
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import log_loss
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from pathlib import Path
import json

log = logging.getLogger(__name__)


class HyperparameterTuner:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, model_type="ensemble", n_trials=50, timeout=3600):
        """
        Initialize hyperparameter tuner.
        
        Args:
            model_type: Type of model to tune (xgboost, lightgbm, catboost, ensemble)
            n_trials: Number of optimization trials
            timeout: Timeout in seconds for optimization
        """
        self.model_type = model_type
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = None
        self.study = None
    
    def optimize_xgboost(self, X, y, n_folds=5):
        """
        Optimize XGBoost hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_folds: Number of cross-validation folds
        
        Returns:
            Best parameters found
        """
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': 42,
                'n_jobs': -1
            }
            
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            model = xgb.XGBClassifier(**params)
            
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_log_loss', n_jobs=-1)
            return -scores.mean()  # Minimize negative log loss
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.study = study
        self.best_params = study.best_params
        log.info(f"Best XGBoost params: {self.best_params}")
        log.info(f"Best XGBoost CV Log Loss: {study.best_value:.4f}")
        
        return self.best_params
    
    def optimize_lightgbm(self, X, y, n_folds=5):
        """
        Optimize LightGBM hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_folds: Number of cross-validation folds
        
        Returns:
            Best parameters found
        """
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'objective': 'binary',
                'metric': 'binary_logloss',
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            model = lgb.LGBMClassifier(**params)
            
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_log_loss', n_jobs=-1)
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.study = study
        self.best_params = study.best_params
        log.info(f"Best LightGBM params: {self.best_params}")
        log.info(f"Best LightGBM CV Log Loss: {study.best_value:.4f}")
        
        return self.best_params
    
    def optimize_catboost(self, X, y, n_folds=5):
        """
        Optimize CatBoost hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_folds: Number of cross-validation folds
        
        Returns:
            Best parameters found
        """
        def objective(trial):
            params = {
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
                'iterations': trial.suggest_int('iterations', 100, 500),
                'random_state': 42,
                'verbose': False
            }
            
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            model = CatBoostClassifier(**params)
            
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_log_loss', n_jobs=-1)
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.study = study
        self.best_params = study.best_params
        log.info(f"Best CatBoost params: {self.best_params}")
        log.info(f"Best CatBoost CV Log Loss: {study.best_value:.4f}")
        
        return self.best_params
    
    def optimize_all(self, X, y, n_folds=5):
        """
        Optimize all three models (XGBoost, LightGBM, CatBoost).
        
        Args:
            X: Feature matrix
            y: Target variable
            n_folds: Number of cross-validation folds
        
        Returns:
            Dictionary of best parameters for each model
        """
        log.info("Starting hyperparameter optimization for all models...")
        
        results = {}
        
        log.info("Optimizing XGBoost...")
        results['xgboost'] = self.optimize_xgboost(X, y, n_folds)
        
        log.info("Optimizing LightGBM...")
        results['lightgbm'] = self.optimize_lightgbm(X, y, n_folds)
        
        log.info("Optimizing CatBoost...")
        results['catboost'] = self.optimize_catboost(X, y, n_folds)
        
        log.info("Hyperparameter optimization complete!")
        
        return results
    
    def save_params(self, filepath):
        """
        Save best parameters to JSON file.
        
        Args:
            filepath: Path to save parameters
        """
        if self.best_params is None:
            log.warning("No parameters to save")
            return
        
        with open(filepath, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        log.info(f"Saved parameters to {filepath}")
    
    def load_params(self, filepath):
        """
        Load parameters from JSON file.
        
        Args:
            filepath: Path to load parameters from
        
        Returns:
            Loaded parameters
        """
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        self.best_params = params
        log.info(f"Loaded parameters from {filepath}")
        return params


def get_default_params():
    """
    Get default hyperparameters for each model type.
    
    Returns:
        Dictionary of default parameters
    """
    return {
        'xgboost': {
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 0.1,
            'min_child_weight': 5,
            'n_estimators': 200,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        },
        'lightgbm': {
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
            'n_estimators': 200,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        },
        'catboost': {
            'depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bylevel': 0.8,
            'l2_leaf_reg': 3,
            'min_data_in_leaf': 20,
            'iterations': 200,
            'random_state': 42,
            'verbose': False
        }
    }


if __name__ == "__main__":
    print("=== Hyperparameter Tuning Module ===")
    print("This module provides automated hyperparameter optimization using Optuna.")
    print("\nUsage:")
    print("  from utils.hyperparameter_tuning import HyperparameterTuner")
    print("  tuner = HyperparameterTuner(n_trials=50)")
    print("  best_params = tuner.optimize_all(X, y, n_folds=5)")
