"""
Kaggle Inference Script
Example script showing how to run the inference pipeline in a Kaggle notebook
"""

# ============================================
# SETUP - Add this at the top of your Kaggle notebook
# ============================================

# Import all required modules
import sys
sys.path.append('/kaggle/working/')

from config import Config
from predict_pipeline import inference_pipeline

# ============================================
# CONFIGURATION
# ============================================

# Create configuration object
config = Config()

# Update paths for Kaggle environment
config.paths.TEST_DATA_PATH = "/kaggle/input/playground-series-s5e12/test.csv"

# Path to saved models
# Option 1: If models are in a Kaggle dataset you created
config.paths.MODEL_DIR = "/kaggle/input/your-model-dataset/models/"

# Option 2: If models are in the working directory from training
# config.paths.MODEL_DIR = "/kaggle/working/models/"

# Output path
config.paths.OUTPUT_DIR = "/kaggle/working/"
config.paths.SUBMISSION_FILE = "submission.csv"

# ============================================
# RUN INFERENCE PIPELINE
# ============================================

print("Starting inference pipeline...")
print("=" * 80)

submission = inference_pipeline(config=config)

# ============================================
# RESULTS SUMMARY
# ============================================

print("\n" + "=" * 80)
print("INFERENCE COMPLETED")
print("=" * 80)

print("\nSubmission Statistics:")
print(f"  Number of predictions: {len(submission)}")
print(f"  Mean prediction: {submission[config.model.TARGET_COL].mean():.4f}")
print(f"  Std prediction: {submission[config.model.TARGET_COL].std():.4f}")
print(f"  Min prediction: {submission[config.model.TARGET_COL].min():.4f}")
print(f"  Max prediction: {submission[config.model.TARGET_COL].max():.4f}")

print("\nFirst 10 predictions:")
print(submission.head(10))

print("\nPrediction distribution:")
print(submission[config.model.TARGET_COL].describe())

print("\n" + "=" * 80)
print(f"✓ Submission saved to: {config.paths.OUTPUT_DIR}{config.paths.SUBMISSION_FILE}")
print("=" * 80)

# ============================================
# VALIDATION CHECKS
# ============================================

print("\n" + "=" * 80)
print("VALIDATION CHECKS:")
print("=" * 80)

# Check for any issues
issues = []

# Check for NaN values
if submission[config.model.TARGET_COL].isna().any():
    issues.append("⚠ Warning: NaN values detected in predictions!")

# Check prediction range
if (submission[config.model.TARGET_COL] < 0).any():
    issues.append("⚠ Warning: Negative predictions detected!")

if (submission[config.model.TARGET_COL] > 1).any():
    issues.append("⚠ Warning: Predictions > 1 detected!")

# Check ID column
import pandas as pd
test_df = pd.read_csv(config.paths.TEST_DATA_PATH)
if not submission[config.model.ID_COL].equals(test_df[config.model.ID_COL]):
    issues.append("⚠ Warning: ID column mismatch with test data!")

if len(issues) == 0:
    print("✓ All validation checks passed!")
else:
    for issue in issues:
        print(issue)

print("=" * 80)

# ============================================
# OPTIONAL: Quick Analysis
# ============================================

print("\n" + "=" * 80)
print("QUICK ANALYSIS:")
print("=" * 80)

# Distribution of predictions
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.histplot(submission[config.model.TARGET_COL], bins=50, kde=True)
plt.title('Distribution of Predictions')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.boxplot(submission[config.model.TARGET_COL])
plt.title('Prediction Box Plot')
plt.ylabel('Predicted Probability')

plt.tight_layout()
plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved prediction distribution plot")
plt.show()

print("=" * 80)
