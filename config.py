"""
Configuration file for the ML Pipeline
Contains all hyperparameters, paths, and settings for training and inference
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class PathConfig:
    """File path configurations"""
    # Input paths (adjust for your environment)
    TRAIN_DATA_PATH: str = "/kaggle/input/playground-series-s5e12/train.csv"
    TEST_DATA_PATH: str = "/kaggle/input/playground-series-s5e12/test.csv"
    
    # Output paths
    OUTPUT_DIR: str = "/kaggle/working/"
    MODEL_DIR: str = "/kaggle/working/models/"
    SUBMISSION_FILE: str = "submission.csv"
    
    # Visualization outputs
    MODEL_COMPARISON_PLOT: str = "model_comparison.png"
    SHAP_SUMMARY_PLOT: str = "shap_summary_best_model.png"
    FEATURE_IMPORTANCE_PLOT: str = "feature_importance_best_model.png"


@dataclass
class ModelConfig:
    """Model training configurations"""
    # Target and ID columns
    TARGET_COL: str = "diagnosed_diabetes"
    ID_COL: str = "id"
    
    # Models to test
    MODELS_TO_TEST: List[str] = None
    
    # GPU settings
    USE_GPU: bool = False
    
    # Random seed for reproducibility
    RANDOM_SEED: int = 42
    
    # Optuna optimization settings
    N_FOLDS_OPT: int = 3  # Fewer folds for hyperparameter optimization
    N_TRIALS: int = 15  # Number of Optuna trials
    MAX_OPT_SAMPLES: int = 100000  # Sample size for Optuna tuning
    
    # Final training settings
    N_FOLDS_SUB: int = 5  # More folds for final model training
    
    # Early stopping
    EARLY_STOPPING_ROUNDS: int = 50
    
    # Model iterations
    MAX_ITERATIONS: int = 1500
    
    def __post_init__(self):
        if self.MODELS_TO_TEST is None:
            self.MODELS_TO_TEST = ['xgboost', 'lightgbm', 'catboost']


@dataclass
class FeatureConfig:
    """Feature engineering configurations"""
    # Columns to convert from numerical to categorical
    NUMERICAL_TO_BOOLEAN: List[str] = None
    
    # Columns to apply log transformation
    SKEWED_TO_GAUSS: List[str] = None
    
    def __post_init__(self):
        if self.NUMERICAL_TO_BOOLEAN is None:
            self.NUMERICAL_TO_BOOLEAN = [
                "cardiovascular_history",
                "hypertension_history",
                "family_history_diabetes"
            ]
        
        if self.SKEWED_TO_GAUSS is None:
            self.SKEWED_TO_GAUSS = ['physical_activity_minutes_per_week']


class Config:
    """Main configuration class that combines all configs"""
    def __init__(self):
        self.paths = PathConfig()
        self.model = ModelConfig()
        self.features = FeatureConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create output directories if they don't exist"""
        os.makedirs(self.paths.MODEL_DIR, exist_ok=True)
        os.makedirs(self.paths.OUTPUT_DIR, exist_ok=True)
    
    def get_hyperparameter_space(self, model_name: str, trial) -> Dict[str, Any]:
        """
        Define hyperparameter search space for each model
        
        Args:
            model_name: Name of the model
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        if model_name == 'xgboost':
            return {
                'n_estimators': self.model.MAX_ITERATIONS,
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'objective': 'binary:logistic',
                'tree_method': 'gpu_hist' if self.model.USE_GPU else 'hist',
                'enable_categorical': True,
                'n_jobs': -1,
                'random_state': self.model.RANDOM_SEED,
                'eval_metric': 'auc',
                'early_stopping_rounds': self.model.EARLY_STOPPING_ROUNDS
            }
        
        elif model_name == 'lightgbm':
            return {
                'n_estimators': self.model.MAX_ITERATIONS,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'objective': 'binary',
                'device': 'gpu' if self.model.USE_GPU else 'cpu',
                'n_jobs': -1,
                'verbosity': -1,
                'random_state': self.model.RANDOM_SEED
            }
        
        elif model_name == 'catboost':
            return {
                'iterations': self.model.MAX_ITERATIONS,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'loss_function': 'Logloss',
                'task_type': 'GPU' if self.model.USE_GPU else 'CPU',
                'thread_count': -1,
                'random_seed': self.model.RANDOM_SEED,
            }
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
