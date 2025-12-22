"""
Main Training Pipeline
Complete end-to-end training pipeline with hyperparameter optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import logging
import warnings

# Import custom modules
from config import Config
from data_preprocessing import create_preprocessor
from model_training import create_trainer
from model_evaluation import create_evaluator
from inference_pipeline import save_models

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def training_pipeline(
    train_path: str = None,
    test_path: str = None,
    config: Config = None
):
    """
    Complete training pipeline with hyperparameter optimization
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        config: Configuration object (creates default if None)
        
    Returns:
        Dictionary containing:
            - predictions: Test predictions
            - models: List of trained models
            - study_results: Hyperparameter optimization results
            - feature_importances: Feature importance scores
            - best_model_name: Name of best performing model
    """
    # Initialize configuration
    if config is None:
        config = Config()
    
    if train_path is None:
        train_path = config.paths.TRAIN_DATA_PATH
    if test_path is None:
        test_path = config.paths.TEST_DATA_PATH
    
    logger.info("=" * 80)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 80)
    
    # ========================================
    # STEP 1: DATA PREPROCESSING
    # ========================================
    logger.info("\nSTEP 1: Data Preprocessing")
    logger.info("-" * 80)
    
    preprocessor = create_preprocessor(config)
    X_train, y_train, X_test, cat_features_indices = preprocessor.preprocess_pipeline(
        train_path=train_path,
        test_path=test_path
    )
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape if X_test is not None else 'N/A'}")
    logger.info(f"Target distribution:\n{y_train.value_counts(normalize=True)}")
    
    # ========================================
    # STEP 2: HYPERPARAMETER OPTIMIZATION
    # ========================================
    logger.info("\nSTEP 2: Hyperparameter Optimization")
    logger.info("-" * 80)
    
    trainer = create_trainer(config, cat_features_indices)
    study_results = trainer.optimize_all_models(X_train, y_train)
    
    # ========================================
    # STEP 3: TRAIN FINAL MODEL
    # ========================================
    logger.info("\nSTEP 3: Training Final Model with K-Fold CV")
    logger.info("-" * 80)
    
    predictions, fold_models, feature_importances = trainer.train_final_model(
        X_train, y_train, X_test
    )
    
    # Save trained models
    save_models(
        fold_models,
        config.paths.MODEL_DIR,
        prefix=f"model_{trainer.best_model_name}"
    )
    
    # ========================================
    # STEP 4: EVALUATION AND VISUALIZATION
    # ========================================
    logger.info("\nSTEP 4: Evaluation and Visualization")
    logger.info("-" * 80)
    
    evaluator = create_evaluator(config)
    
    # Get validation set for SHAP analysis (use last fold)
    kf = StratifiedKFold(
        n_splits=config.model.N_FOLDS_SUB,
        shuffle=True,
        random_state=config.model.RANDOM_SEED
    )
    
    for fold_idx, (_, val_idx) in enumerate(kf.split(X_train, y_train)):
        if fold_idx == config.model.N_FOLDS_SUB - 1:
            X_val = X_train.iloc[val_idx]
            break
    
    # Run evaluation
    evaluator.evaluate_and_visualize(
        study_results=study_results,
        final_model=fold_models[-1],  # Use last fold model for SHAP
        X_val=X_val,
        feature_names=X_train.columns.tolist(),
        feature_importances=feature_importances,
        model_name=trainer.best_model_name
    )
    
    # ========================================
    # STEP 5: CREATE SUBMISSION
    # ========================================
    if X_test is not None and predictions is not None:
        logger.info("\nSTEP 5: Creating Submission File")
        logger.info("-" * 80)
        
        test_df = pd.read_csv(test_path)
        submission = pd.DataFrame({
            config.model.ID_COL: test_df[config.model.ID_COL],
            config.model.TARGET_COL: predictions
        })
        
        submission_path = f"{config.paths.OUTPUT_DIR}{config.paths.SUBMISSION_FILE}"
        submission.to_csv(submission_path, index=False)
        logger.info(f"Submission saved to {submission_path}")
    
    # ========================================
    # SUMMARY
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Best Model: {trainer.best_model_name.upper()}")
    logger.info(f"Best CV AUC: {study_results[trainer.best_model_name]['best_score']:.4f}")
    logger.info(f"Models saved to: {config.paths.MODEL_DIR}")
    logger.info(f"Visualizations saved to: {config.paths.OUTPUT_DIR}")
    logger.info("=" * 80)
    
    return {
        'predictions': predictions,
        'models': fold_models,
        'study_results': study_results,
        'feature_importances': feature_importances,
        'best_model_name': trainer.best_model_name,
        'feature_names': X_train.columns.tolist(),
        'config': config
    }


def main():
    """Main execution function"""
    # Create config
    config = Config()
    
    # Run training pipeline
    results = training_pipeline(config=config)
    
    return results


if __name__ == "__main__":
    main()
