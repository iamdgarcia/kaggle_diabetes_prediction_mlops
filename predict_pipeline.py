"""
Main Inference Pipeline
Standalone inference pipeline for making predictions with trained models
"""

import os
import pandas as pd
import logging
import warnings

# Import custom modules
from config import Config
from data_preprocessing import DataPreprocessor
from inference_pipeline import create_inference_pipeline

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def inference_pipeline(
    test_path: str = None,
    model_dir: str = None,
    output_path: str = None,
    config: Config = None,
    preprocessor_path: str = None
):
    """
    Complete inference pipeline for making predictions
    
    Args:
        test_path: Path to test data
        model_dir: Directory containing saved models
        output_path: Path to save submission file
        config: Configuration object (creates default if None)
        
    Returns:
        Submission dataframe with predictions
    """
    # Initialize configuration
    if config is None:
        config = Config()
    
    if test_path is None:
        test_path = config.paths.TEST_DATA_PATH
    if model_dir is None:
        model_dir = config.paths.MODEL_DIR
    if output_path is None:
        output_path = f"{config.paths.OUTPUT_DIR}{config.paths.SUBMISSION_FILE}"
    
    logger.info("=" * 80)
    logger.info("STARTING INFERENCE PIPELINE")
    logger.info("=" * 80)
    
    # ========================================
    # STEP 1: LOAD AND PREPROCESS DATA
    # ========================================
    logger.info("\nSTEP 1: Loading and Preprocessing Data")
    logger.info("-" * 80)
    
    # Load test data
    test_df = pd.read_csv(test_path)
    logger.info(f"Test data loaded: {test_df.shape}")
    
    # Load the fitted preprocessor artifact created during training
    if preprocessor_path:
        preproc_path = preprocessor_path
    else:
        preproc_path = os.path.join(config.paths.MODEL_DIR, 'preprocessor.joblib')

    if not os.path.exists(preproc_path):
        raise FileNotFoundError(
            f"Preprocessor artifact not found at {preproc_path}. "
            "Run the training pipeline to create and save the fitted preprocessor, "
            "or pass a valid `preprocessor_path` to this function."
        )

    preprocessor = DataPreprocessor.load(preproc_path)
    logger.info(f"Loaded preprocessor from {preproc_path}")
    X_test = preprocessor.transform_for_inference(test_df)
    logger.info(f"Test features prepared: {X_test.shape}")
    
    # ========================================
    # STEP 2: LOAD MODELS AND MAKE PREDICTIONS
    # ========================================
    logger.info("\nSTEP 2: Loading Models and Making Predictions")
    logger.info("-" * 80)
    
    inference = create_inference_pipeline(config)
    inference.load_models(model_dir)
    
    # Make predictions
    predictions = inference.predict_ensemble(X_test)
    
    logger.info(f"Predictions generated for {len(predictions)} samples")
    logger.info(f"Prediction statistics:")
    logger.info(f"  Mean: {predictions.mean():.4f}")
    logger.info(f"  Std:  {predictions.std():.4f}")
    logger.info(f"  Min:  {predictions.min():.4f}")
    logger.info(f"  Max:  {predictions.max():.4f}")
    
    # ========================================
    # STEP 3: CREATE SUBMISSION
    # ========================================
    logger.info("\nSTEP 3: Creating Submission File")
    logger.info("-" * 80)
    
    submission = inference.create_submission(
        test_df=test_df,
        predictions=predictions,
        output_path=output_path
    )
    
    logger.info(f"Submission preview:")
    logger.info(f"\n{submission.head(10)}")
    
    # ========================================
    # SUMMARY
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("INFERENCE PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Number of predictions: {len(predictions)}")
    logger.info(f"Submission saved to: {output_path}")
    logger.info("=" * 80)
    
    return submission


def main():
    """Main execution function"""
    # Create config
    config = Config()
    
    # Run inference pipeline
    submission = inference_pipeline(config=config)
    
    return submission


if __name__ == "__main__":
    main()
