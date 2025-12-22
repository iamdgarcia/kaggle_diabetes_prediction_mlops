"""
Utility Functions
Helper functions for common tasks across the pipeline
"""

import pandas as pd
import numpy as np
import json
import pickle
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_json(data: Dict, filepath: str):
    """
    Save dictionary to JSON file
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Load JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary with JSON contents
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {filepath}")
    return data


def save_pickle(obj: Any, filepath: str):
    """
    Save object using pickle
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved pickle to {filepath}")


def load_pickle(filepath: str) -> Any:
    """
    Load pickled object
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f"Loaded pickle from {filepath}")
    return obj


def create_timestamp() -> str:
    """
    Create timestamp string for versioning
    
    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(directory: str):
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Path to directory
    """
    os.makedirs(directory, exist_ok=True)


def get_file_size(filepath: str) -> str:
    """
    Get human-readable file size
    
    Args:
        filepath: Path to file
        
    Returns:
        File size string (e.g., "1.5 MB")
    """
    size_bytes = os.path.getsize(filepath)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def log_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Log comprehensive dataframe information
    
    Args:
        df: Dataframe to analyze
        name: Name for logging
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"{name} Information")
    logger.info(f"{'='*60}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"\nColumn types:")
    logger.info(f"{df.dtypes.value_counts()}")
    logger.info(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.info(f"{missing[missing > 0]}")
    else:
        logger.info("No missing values")
    logger.info(f"{'='*60}\n")


def calculate_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        y: Target variable
        
    Returns:
        Dictionary mapping class to weight
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    class_weights = {classes[i]: weights[i] for i in range(len(classes))}
    logger.info(f"Class weights: {class_weights}")
    
    return class_weights


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of dataframe by downcasting numeric types
    
    Args:
        df: Dataframe to optimize
        verbose: Whether to print memory reduction info
        
    Returns:
        Optimized dataframe
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        logger.info(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
                   f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df


def validate_submission(
    submission: pd.DataFrame,
    test_df: pd.DataFrame,
    id_col: str,
    target_col: str
) -> List[str]:
    """
    Validate submission file for common issues
    
    Args:
        submission: Submission dataframe
        test_df: Original test dataframe
        id_col: ID column name
        target_col: Target column name
        
    Returns:
        List of validation errors (empty if no errors)
    """
    errors = []
    
    # Check required columns
    if id_col not in submission.columns:
        errors.append(f"Missing ID column: {id_col}")
    if target_col not in submission.columns:
        errors.append(f"Missing target column: {target_col}")
    
    # Check for missing values
    if submission[target_col].isnull().any():
        errors.append("Submission contains null values")
    
    # Check ID alignment
    if not submission[id_col].equals(test_df[id_col]):
        errors.append("ID column does not match test data")
    
    # Check prediction range (for probabilities)
    if (submission[target_col] < 0).any() or (submission[target_col] > 1).any():
        errors.append("Predictions outside valid range [0, 1]")
    
    # Check for duplicates
    if submission[id_col].duplicated().any():
        errors.append("Duplicate IDs found in submission")
    
    if len(errors) == 0:
        logger.info("✓ Submission validation passed")
    else:
        logger.warning(f"✗ Submission validation failed with {len(errors)} errors")
        for error in errors:
            logger.warning(f"  - {error}")
    
    return errors


def print_training_summary(results: Dict):
    """
    Print formatted training summary
    
    Args:
        results: Results dictionary from training pipeline
    """
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    print(f"\nBest Model: {results['best_model_name'].upper()}")
    print(f"CV AUC: {results['study_results'][results['best_model_name']]['best_score']:.4f}")
    
    print("\n" + "-"*80)
    print("All Models Performance:")
    print("-"*80)
    for model_name, result in results['study_results'].items():
        print(f"  {model_name.upper():15s}: {result['best_score']:.4f}")
    
    print("\n" + "-"*80)
    print("Top 10 Most Important Features:")
    print("-"*80)
    fi_df = pd.DataFrame({
        'feature': results['feature_names'],
        'importance': results['feature_importances']
    })
    fi_df = fi_df.sort_values('importance', ascending=False).head(10)
    for idx, row in fi_df.iterrows():
        print(f"  {row['feature']:40s}: {row['importance']:.4f}")
    
    print("\n" + "="*80)


def merge_fold_predictions(
    fold_predictions: List[np.ndarray],
    method: str = 'mean'
) -> np.ndarray:
    """
    Merge predictions from multiple folds
    
    Args:
        fold_predictions: List of prediction arrays
        method: Merging method ('mean', 'median', 'geometric_mean')
        
    Returns:
        Merged predictions
    """
    if method == 'mean':
        return np.mean(fold_predictions, axis=0)
    elif method == 'median':
        return np.median(fold_predictions, axis=0)
    elif method == 'geometric_mean':
        return np.exp(np.mean(np.log(fold_predictions), axis=0))
    else:
        raise ValueError(f"Unknown merge method: {method}")
