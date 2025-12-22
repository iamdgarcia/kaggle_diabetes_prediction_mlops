"""
Model Training Module
Contains functions for model training, hyperparameter optimization, and cross-validation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
import logging
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
import optuna
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and hyperparameter optimization"""
    
    def __init__(self, config, cat_features_indices: Optional[List[int]] = None):
        """
        Initialize trainer with configuration
        
        Args:
            config: Configuration object
            cat_features_indices: List of categorical feature indices
        """
        self.config = config
        self.cat_features_indices = cat_features_indices or []
        self.study_results = {}
        self.best_model_name = None
        self.best_params = None
        
    def get_model_class(self, model_name: str):
        """
        Get model class based on name
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model class
        """
        if model_name == 'xgboost':
            return xgb.XGBClassifier
        elif model_name == 'lightgbm':
            return lgb.LGBMClassifier
        elif model_name == 'catboost':
            return cb.CatBoostClassifier
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def run_cv_with_early_stopping(
        self, 
        model_name: str, 
        params: Dict[str, Any], 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_folds: int
    ) -> float:
        """
        Run cross-validation with early stopping for faster training
        
        Args:
            model_name: Name of the model
            params: Model hyperparameters
            X: Feature matrix
            y: Target vector
            n_folds: Number of CV folds
            
        Returns:
            Mean AUC score across folds
        """
        kf = StratifiedKFold(
            n_splits=n_folds, 
            shuffle=True, 
            random_state=self.config.model.RANDOM_SEED
        )
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            # Initialize model
            if model_name == 'xgboost':
                model = xgb.XGBClassifier(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                
            elif model_name == 'lightgbm':
                model = lgb.LGBMClassifier(**params)
                callbacks = [
                    lgb.early_stopping(
                        stopping_rounds=self.config.model.EARLY_STOPPING_ROUNDS, 
                        verbose=False
                    )
                ]
                model.fit(
                    X_tr, y_tr, 
                    eval_set=[(X_val, y_val)], 
                    eval_metric='auc', 
                    callbacks=callbacks
                )
                
            elif model_name == 'catboost':
                model = cb.CatBoostClassifier(**params)
                model.fit(
                    X_tr, y_tr, 
                    eval_set=(X_val, y_val), 
                    early_stopping_rounds=self.config.model.EARLY_STOPPING_ROUNDS, 
                    verbose=False
                )
            
            # Predict and score
            preds = model.predict_proba(X_val)[:, 1]
            
            try:
                score = roc_auc_score(y_val, preds)
                fold_scores.append(score)
            except Exception as e:
                logger.warning(f"Could not calculate AUC for fold {fold_idx}: {e}")
                fold_scores.append(0.5)  # Fallback score
        
        return np.mean(fold_scores)
    
    def optimize_hyperparameters(
        self, 
        model_name: str, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            model_name: Name of the model
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with best score and best parameters
        """
        logger.info(f"Optimizing hyperparameters for {model_name.upper()}")
        
        # Sample data if too large
        if len(X) > self.config.model.MAX_OPT_SAMPLES:
            logger.info(
                f"Dataset size ({len(X)}) exceeds limit. "
                f"Subsampling to {self.config.model.MAX_OPT_SAMPLES} rows."
            )
            X_opt, _, y_opt, _ = train_test_split(
                X, y, 
                train_size=self.config.model.MAX_OPT_SAMPLES, 
                stratify=y, 
                random_state=self.config.model.RANDOM_SEED
            )
        else:
            X_opt, y_opt = X, y
        
        # Define objective function
        def objective(trial):
            params = self.config.get_hyperparameter_space(model_name, trial)
            
            # Add cat_features for CatBoost
            if model_name == 'catboost' and self.cat_features_indices:
                params['cat_features'] = self.cat_features_indices
            
            avg_score = self.run_cv_with_early_stopping(
                model_name, params, X_opt, y_opt, 
                n_folds=self.config.model.N_FOLDS_OPT
            )
            return avg_score
        
        # Run optimization
        sampler = optuna.samplers.TPESampler(seed=self.config.model.RANDOM_SEED)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(objective, n_trials=self.config.model.N_TRIALS, show_progress_bar=True)
        
        logger.info(f"Best AUC for {model_name}: {study.best_value:.4f}")
        
        return {
            'best_score': study.best_value,
            'best_params': study.best_params,
            'study': study
        }
    
    def optimize_all_models(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """
        Optimize hyperparameters for all configured models
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with results for each model
        """
        logger.info("=" * 60)
        logger.info("Starting hyperparameter optimization for all models")
        logger.info("=" * 60)
        
        for model_name in self.config.model.MODELS_TO_TEST:
            result = self.optimize_hyperparameters(model_name, X, y)
            self.study_results[model_name] = result
        
        # Identify best model
        self.best_model_name = max(
            self.study_results, 
            key=lambda k: self.study_results[k]['best_score']
        )
        self.best_params = self.study_results[self.best_model_name]['best_params']
        
        logger.info("=" * 60)
        logger.info(f"Best model: {self.best_model_name.upper()}")
        logger.info(f"Best CV AUC: {self.study_results[self.best_model_name]['best_score']:.4f}")
        logger.info("=" * 60)
        
        return self.study_results
    
    def train_final_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[np.ndarray], List, np.ndarray]:
        """
        Train final model using K-Fold cross-validation.

        NOTE: This method no longer accepts or predicts on test data. Training
        and inference are separated in the MLOps flow — inference should be
        performed with the `InferencePipeline` using a saved preprocessor and
        the trained models.

        Args:
            X: Training features
            y: Training target
            model_name: Model to train (uses best if not specified)
            params: Model parameters (uses best if not specified)

        Returns:
            Tuple of (predictions (always None), fold_models, feature_importances)
        """
        # Use best model if not specified
        if model_name is None:
            model_name = self.best_model_name
        if params is None:
            params = self.best_params.copy()
        
        logger.info("=" * 60)
        logger.info(f"Training final model: {model_name.upper()}")
        logger.info("=" * 60)
        
        # Update parameters with model-specific settings
        if model_name == 'xgboost':
            params.update({
                'n_estimators': self.config.model.MAX_ITERATIONS,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'tree_method': 'gpu_hist' if self.config.model.USE_GPU else 'hist',
                'enable_categorical': True,
                'early_stopping_rounds': self.config.model.EARLY_STOPPING_ROUNDS
            })
        elif model_name == 'lightgbm':
            params.update({
                'n_estimators': self.config.model.MAX_ITERATIONS,
                'objective': 'binary',
                'device': 'gpu' if self.config.model.USE_GPU else 'cpu',
                'verbosity': -1
            })
        elif model_name == 'catboost':
            params.update({
                'iterations': self.config.model.MAX_ITERATIONS,
                'loss_function': 'Logloss',
                'task_type': 'GPU' if self.config.model.USE_GPU else 'CPU',
                'cat_features': self.cat_features_indices
            })
        
        # Cross-validation
        kf = StratifiedKFold(
            n_splits=self.config.model.N_FOLDS_SUB,
            shuffle=True,
            random_state=self.config.model.RANDOM_SEED
        )
        
        fold_models = []
        fold_preds = []
        feature_importances = np.zeros(len(X.columns))
        ModelClass = self.get_model_class(model_name)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            logger.info(f"Training Fold {fold+1}/{self.config.model.N_FOLDS_SUB}...")
            
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            model = ModelClass(**params)
            
            # Train with early stopping
            if model_name == 'lightgbm':
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric='auc',
                    callbacks=[
                        lgb.early_stopping(
                            stopping_rounds=self.config.model.EARLY_STOPPING_ROUNDS,
                            verbose=False
                        )
                    ]
                )
            elif model_name == 'catboost':
                model.fit(
                    X_tr, y_tr,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=self.config.model.EARLY_STOPPING_ROUNDS,
                    verbose=False
                )
            elif model_name == 'xgboost':
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            # Store model
            fold_models.append(model)
            
            # NOTE: Do not predict on any external test set here.
            # Inference is handled separately by the inference pipeline.
            
            # Accumulate feature importances
            try:
                feature_importances += model.feature_importances_ / self.config.model.N_FOLDS_SUB
            except AttributeError:
                pass
        
        # We intentionally do not produce test predictions here.
        avg_preds = None
        
        logger.info("Final model training completed")
        
        return avg_preds, fold_models, feature_importances


def create_trainer(config, cat_features_indices: Optional[List[int]] = None):
    """
    Factory function to create model trainer
    
    Args:
        config: Configuration object
        cat_features_indices: List of categorical feature indices
        
    Returns:
        ModelTrainer instance
    """
    return ModelTrainer(config, cat_features_indices)
