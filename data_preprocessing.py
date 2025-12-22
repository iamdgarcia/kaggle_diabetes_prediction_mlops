"""
Data Preprocessing Module
Contains functions for loading data, feature engineering, and data validation
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing and feature engineering"""
    
    def __init__(self, config):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Configuration object containing all settings
        """
        self.config = config
        self.cat_features_indices = None
        self.feature_names = None
        
    def load_data(self, train_path: str, test_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load training and test data
        
        Args:
            train_path: Path to training data
            test_path: Path to test data (optional)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path)
        
        test_df = None
        if test_path:
            logger.info(f"Loading test data from {test_path}")
            test_df = pd.read_csv(test_path)
        
        logger.info(f"Training data shape: {train_df.shape}")
        if test_df is not None:
            logger.info(f"Test data shape: {test_df.shape}")
        
        return train_df, test_df
    
    def convert_numerical_to_categorical(
        self, 
        train_df: pd.DataFrame, 
        test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Convert specified numerical columns to categorical
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe (optional)
            
        Returns:
            Tuple of transformed dataframes
        """
        numerical_to_boolean = self.config.features.NUMERICAL_TO_BOOLEAN
        
        logger.info(f"Converting columns to categorical: {numerical_to_boolean}")
        train_df[numerical_to_boolean] = train_df[numerical_to_boolean].astype('category')
        
        if test_df is not None:
            test_df[numerical_to_boolean] = test_df[numerical_to_boolean].astype('category')
        
        return train_df, test_df
    
    def apply_log_transformation(
        self, 
        train_df: pd.DataFrame, 
        test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Apply log1p transformation to skewed features
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe (optional)
            
        Returns:
            Tuple of transformed dataframes
        """
        skewed_cols = self.config.features.SKEWED_TO_GAUSS
        
        logger.info(f"Applying log1p transformation to: {skewed_cols}")
        train_df[skewed_cols] = np.log1p(train_df[skewed_cols])
        
        if test_df is not None:
            test_df[skewed_cols] = np.log1p(test_df[skewed_cols])
        
        return train_df, test_df
    
    def align_categories(
        self, 
        train_df: pd.DataFrame, 
        test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Align categorical columns between train and test to have same categories
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe (optional)
            
        Returns:
            Tuple of aligned dataframes
        """
        if test_df is None:
            return train_df, None
        
        object_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Aligning categories for columns: {object_cols}")
        
        for col in object_cols:
            # Get union of unique values from both datasets
            unique_values = set(train_df[col].unique()) | set(test_df[col].unique())
            unique_values = sorted(list(unique_values))
            
            # Create categorical dtype with all possible values
            cat_type = pd.CategoricalDtype(categories=unique_values, ordered=False)
            
            # Apply to both dataframes
            train_df[col] = train_df[col].astype(cat_type)
            test_df[col] = test_df[col].astype(cat_type)
        
        return train_df, test_df
    
    def prepare_features(
        self, 
        train_df: pd.DataFrame, 
        test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], List[int]]:
        """
        Complete feature preparation pipeline
        
        Args:
            train_df: Training dataframe
            test_df: Test dataframe (optional)
            
        Returns:
            Tuple of (X_train, y_train, X_test, cat_features_indices)
        """
        # Apply all transformations
        train_df, test_df = self.convert_numerical_to_categorical(train_df, test_df)
        train_df, test_df = self.apply_log_transformation(train_df, test_df)
        train_df, test_df = self.align_categories(train_df, test_df)
        
        # Separate features and target
        X_train = train_df.drop(columns=[self.config.model.TARGET_COL, self.config.model.ID_COL])
        y_train = train_df[self.config.model.TARGET_COL]
        
        X_test = None
        if test_df is not None:
            X_test = test_df.drop(columns=[self.config.model.ID_COL])
        
        # Get categorical feature indices for models that need them
        cat_features_indices = [
            i for i, col in enumerate(X_train.columns) 
            if X_train[col].dtype.name == 'category'
        ]
        
        self.cat_features_indices = cat_features_indices
        self.feature_names = X_train.columns.tolist()
        
        logger.info(f"Features prepared. Shape: {X_train.shape}")
        logger.info(f"Number of categorical features: {len(cat_features_indices)}")
        
        return X_train, y_train, X_test, cat_features_indices
    
    def preprocess_pipeline(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        train_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame], List[int]]:
        """
        Complete preprocessing pipeline - can accept either paths or dataframes
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            train_df: Training dataframe (alternative to path)
            test_df: Test dataframe (alternative to path)
            
        Returns:
            Tuple of (X_train, y_train, X_test, cat_features_indices)
        """
        # Load data if paths provided
        if train_path:
            train_df, test_df = self.load_data(train_path, test_path)
        elif train_df is None:
            raise ValueError("Either train_path or train_df must be provided")
        
        # Prepare features
        return self.prepare_features(train_df, test_df)


def create_preprocessor(config):
    """
    Factory function to create preprocessor
    
    Args:
        config: Configuration object
        
    Returns:
        DataPreprocessor instance
    """
    return DataPreprocessor(config)
