"""
Inference Pipeline Module
Contains functions for making predictions with trained models
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union
import logging
import pickle
import os

from data_preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferencePipeline:
    """Handles model inference and prediction generation"""
    
    def __init__(self, config):
        """
        Initialize inference pipeline
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.models = []
        self.preprocessor = None
        
    def load_models(self, model_dir: str) -> List:
        """
        Load trained models from directory
        
        Args:
            model_dir: Directory containing saved models
            
        Returns:
            List of loaded models
        """
        logger.info(f"Loading models from {model_dir}")
        
        models = []
        model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pkl')])
        
        for model_file in model_files:
            path = os.path.join(model_dir, model_file)
            with open(path, 'rb') as f:
                model = pickle.load(f)
                models.append(model)
            logger.info(f"Loaded {model_file}")
        
        self.models = models
        return models
    
    def predict_single_model(
        self, 
        model, 
        X: pd.DataFrame, 
        return_proba: bool = True
    ) -> np.ndarray:
        """
        Make predictions with a single model
        
        Args:
            model: Trained model
            X: Feature matrix
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions array
        """
        if return_proba:
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X)
    
    def predict_ensemble(
        self, 
        X: pd.DataFrame, 
        return_proba: bool = True
    ) -> np.ndarray:
        """
        Make predictions using ensemble of models (averaging)
        
        Args:
            X: Feature matrix
            return_proba: Whether to return probabilities
            
        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models loaded. Call load_models() first.")
        
        logger.info(f"Making predictions with {len(self.models)} models")
        
        predictions = []
        for model in self.models:
            preds = self.predict_single_model(model, X, return_proba)
            predictions.append(preds)
        
        # Average predictions
        ensemble_preds = np.mean(predictions, axis=0)
        
        return ensemble_preds
    
    def create_submission(
        self,
        test_df: pd.DataFrame,
        predictions: np.ndarray,
        id_col: Optional[str] = None,
        target_col: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create submission file
        
        Args:
            test_df: Test dataframe (must contain ID column)
            predictions: Prediction values
            id_col: Name of ID column
            target_col: Name of target column
            output_path: Path to save submission file
            
        Returns:
            Submission dataframe
        """
        if id_col is None:
            id_col = self.config.model.ID_COL
        if target_col is None:
            target_col = self.config.model.TARGET_COL
        
        submission = pd.DataFrame({
            id_col: test_df[id_col],
            target_col: predictions
        })
        
        if output_path is None:
            output_path = f"{self.config.paths.OUTPUT_DIR}{self.config.paths.SUBMISSION_FILE}"
        
        submission.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")
        
        return submission
    
    def run_inference(
        self,
        test_df: pd.DataFrame,
        preprocessor=None,
        model_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Complete inference pipeline
        
        Args:
            test_df: Test dataframe
            preprocessor: Data preprocessor (optional if already set)
            model_dir: Directory with saved models (optional if models already loaded)
            
        Returns:
            Submission dataframe
        """
        logger.info("=" * 60)
        logger.info("Starting inference pipeline")
        logger.info("=" * 60)
        
        # Load models if directory provided
        if model_dir:
            self.load_models(model_dir)
        
        # Preprocess data if preprocessor provided
        # Preprocessor can be:
        # - an instance of DataPreprocessor (already fitted),
        # - a path to a saved preprocessor artifact, or
        # - None (assume test_df already preprocessed)
        if preprocessor:
            if isinstance(preprocessor, str):
                # load saved artifact
                self.preprocessor = DataPreprocessor.load(preprocessor)
            else:
                self.preprocessor = preprocessor

            # Use inference transform (does not peek at labels)
            X_test = self.preprocessor.transform_for_inference(test_df)
        else:
            # Assume test_df is already preprocessed
            X_test = test_df.drop(columns=[self.config.model.ID_COL])
        
        # Make predictions
        predictions = self.predict_ensemble(X_test)
        
        # Create submission
        submission = self.create_submission(test_df, predictions)
        
        logger.info("Inference pipeline completed")
        
        return submission


def save_models(models: List, model_dir: str, prefix: str = "model"):
    """
    Save trained models to disk
    
    Args:
        models: List of trained models
        model_dir: Directory to save models
        prefix: Prefix for model filenames
    """
    os.makedirs(model_dir, exist_ok=True)
    
    for i, model in enumerate(models):
        path = os.path.join(model_dir, f"{prefix}_fold_{i}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model to {path}")


def create_inference_pipeline(config):
    """
    Factory function to create inference pipeline
    
    Args:
        config: Configuration object
        
    Returns:
        InferencePipeline instance
    """
    return InferencePipeline(config)
