"""
Kaggle Training Script
Example script showing how to run the training pipeline in a Kaggle notebook
"""

# ============================================
# SETUP - Add this at the top of your Kaggle notebook
# ============================================

# Import all required modules (assuming they're in the same directory)
import sys
sys.path.append('/kaggle/working/')

from config import Config
from train_pipeline import training_pipeline

# ============================================
# CONFIGURATION
# ============================================

# Create configuration object
config = Config()

# Update paths for Kaggle environment
config.paths.TRAIN_DATA_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
config.paths.TEST_DATA_PATH = "/kaggle/input/playground-series-s5e12/test.csv"
config.paths.OUTPUT_DIR = "/kaggle/working/"
config.paths.MODEL_DIR = "/kaggle/working/models/"

# ============================================
# OPTIMIZATION SETTINGS
# ============================================

# GPU Configuration
# Set to True if you selected GPU accelerator in Kaggle
config.model.USE_GPU = False  # Change to True for GPU

# Models to test
config.model.MODELS_TO_TEST = ['xgboost', 'lightgbm', 'catboost']

# Hyperparameter optimization settings
# Faster settings for quick iteration
config.model.N_TRIALS = 15  # Increase for better results (e.g., 30)
config.model.N_FOLDS_OPT = 3  # Folds for optimization
config.model.MAX_OPT_SAMPLES = 100000  # Sample size for Optuna

# Final model training settings
# Higher quality settings for submission
config.model.N_FOLDS_SUB = 5  # Increase for more stable predictions (e.g., 10)
config.model.MAX_ITERATIONS = 1500
config.model.EARLY_STOPPING_ROUNDS = 50

# ============================================
# RUN TRAINING PIPELINE
# ============================================

print("Starting training pipeline...")
print("=" * 80)

results = training_pipeline(config=config)

# ============================================
# RESULTS SUMMARY
# ============================================

print("\n" + "=" * 80)
print("TRAINING COMPLETED")
print("=" * 80)
print(f"\nBest Model: {results['best_model_name'].upper()}")
print(f"CV AUC Score: {results['study_results'][results['best_model_name']]['best_score']:.4f}")

print("\nAll Model Scores:")
for model_name, result in results['study_results'].items():
    print(f"  {model_name.upper()}: {result['best_score']:.4f}")

print("\nTop 10 Features:")
import pandas as pd
fi_df = pd.DataFrame({
    'feature': results['feature_names'],
    'importance': results['feature_importances']
})
fi_df = fi_df.sort_values('importance', ascending=False).head(10)
print(fi_df.to_string(index=False))

print("\n" + "=" * 80)
print("OUTPUT ARTIFACTS:")
print("=" * 80)
print(f"✓ Models saved to: {config.paths.MODEL_DIR}")
print(f"✓ Fitted preprocessor artifact: {config.paths.MODEL_DIR}preprocessor.joblib")
print(f"✓ Visualizations saved to: {config.paths.OUTPUT_DIR}")
print("  - model_comparison.png")
print("  - feature_importance_best_model.png")
print("  - shap_summary_best_model.png")
print("\nNote: A submission CSV is not created by the training script.\n" \
    "To generate a submission file, run the standalone inference script `predict_pipeline.py` " \
    "which will load the saved preprocessor and models and write the submission.")

# ============================================
# OPTIONAL: PARAMETER TUNING TIPS
# ============================================

print("\n" + "=" * 80)
print("TUNING TIPS:")
print("=" * 80)
print("""
For better results, you can adjust these parameters:

1. Increase N_TRIALS (e.g., 30-50) for better hyperparameter search
2. Increase N_FOLDS_SUB (e.g., 10) for more stable predictions
3. Enable GPU if available for faster training
4. Adjust MAX_OPT_SAMPLES based on dataset size and time constraints

Example:
    config.model.N_TRIALS = 30
    config.model.N_FOLDS_SUB = 10
    config.model.USE_GPU = True
""")

print("=" * 80)
