# MLOps Pipeline for Diabetes Prediction

A production-ready machine learning pipeline for diabetes diagnosis prediction, following MLOps best practices with modular architecture, automated hyperparameter tuning, and comprehensive model evaluation.

[Visit reference Kaggle project](https://www.kaggle.com/code/danigarci1/s5e12-mlops)

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Components](#pipeline-components)
- [Configuration](#configuration)
- [Kaggle Execution](#kaggle-execution)

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for predicting diabetes diagnosis. It includes:
- **Automated hyperparameter optimization** using Optuna
- **Multiple model comparison** (XGBoost, LightGBM, CatBoost)
- **K-Fold cross-validation** for robust evaluation
- **Model interpretability** with SHAP analysis
- **Modular architecture** for easy deployment

## 📁 Project Structure

```
.
├── config.py                   # Central configuration management
├── data_preprocessing.py       # Data loading and feature engineering
├── model_training.py          # Model training and optimization
├── model_evaluation.py        # Evaluation and visualization
├── inference_pipeline.py      # Inference and prediction generation
├── train_pipeline.py          # Main training pipeline
├── predict_pipeline.py        # Main inference pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── examples/
    ├── kaggle_training.py     # Kaggle-specific training script
    └── kaggle_inference.py    # Kaggle-specific inference script
```

## ✨ Features

### Data Processing
- Automatic categorical feature alignment between train/test
- Log transformation for skewed features
- Type conversion for binary categorical variables
- Robust handling of missing values

### Model Training
- **Hyperparameter Optimization**: Optuna-based automated tuning
- **Multiple Models**: XGBoost, LightGBM, CatBoost
- **Early Stopping**: Efficient training with validation-based stopping
- **GPU Support**: Optional GPU acceleration
- **K-Fold CV**: Stratified cross-validation for stable estimates

### Model Evaluation
- Performance comparison plots
- Feature importance visualization
- SHAP summary plots for model interpretability
- Comprehensive logging

### Production Ready
- Modular, testable code structure
- Configurable via single config file
- Model serialization and versioning
- Inference pipeline for batch predictions

## 🚀 Installation

### Local Environment

```bash
# Clone or download the project files
cd diabetes-prediction-pipeline

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Kaggle Environment

All dependencies are pre-installed in Kaggle kernels. Simply upload the Python files to your notebook.

## 💻 Usage

### Training Pipeline

#### Basic Usage
```python
from config import Config
from train_pipeline import training_pipeline

# Create configuration
config = Config()

# Run training pipeline
results = training_pipeline(config=config)

# Access results
print(f"Best model: {results['best_model_name']}")
print(f"CV AUC: {results['study_results'][results['best_model_name']]['best_score']:.4f}")
```

#### Custom Configuration
```python
from config import Config
from train_pipeline import training_pipeline

# Create and customize config
config = Config()
config.model.N_TRIALS = 20  # More Optuna trials
config.model.N_FOLDS_SUB = 10  # More CV folds
config.model.USE_GPU = True  # Enable GPU

# Run with custom config
results = training_pipeline(
    train_path="/path/to/train.csv",
    test_path="/path/to/test.csv",
    config=config
)
```

### Inference Pipeline

```python
from config import Config
from predict_pipeline import inference_pipeline

# Run inference
submission = inference_pipeline(
    test_path="/path/to/test.csv",
    model_dir="/path/to/models/",
    output_path="submission.csv"
)
```

## 🔧 Pipeline Components

### 1. Configuration (`config.py`)
Central configuration management with three main components:
- **PathConfig**: File paths and directories
- **ModelConfig**: Model hyperparameters and training settings
- **FeatureConfig**: Feature engineering specifications

### 2. Data Preprocessing (`data_preprocessing.py`)
**Class**: `DataPreprocessor`

**Key Methods**:
- `load_data()`: Load train/test datasets
- `convert_numerical_to_categorical()`: Type conversion
- `apply_log_transformation()`: Handle skewed features
- `align_categories()`: Ensure consistent categorical levels
- `preprocess_pipeline()`: Complete preprocessing workflow

### 3. Model Training (`model_training.py`)
**Class**: `ModelTrainer`

**Key Methods**:
- `optimize_hyperparameters()`: Optuna-based tuning
- `run_cv_with_early_stopping()`: CV with early stopping
- `optimize_all_models()`: Compare multiple models
- `train_final_model()`: Train with best hyperparameters

### 4. Model Evaluation (`model_evaluation.py`)
**Class**: `ModelEvaluator`

**Key Methods**:
- `plot_model_comparison()`: Compare model performance
- `plot_feature_importance()`: Visualize feature importance
- `plot_shap_summary()`: SHAP-based interpretability
- `evaluate_and_visualize()`: Complete evaluation pipeline

### 5. Inference Pipeline (`inference_pipeline.py`)
**Class**: `InferencePipeline`

**Key Methods**:
- `load_models()`: Load saved models
- `predict_ensemble()`: Ensemble predictions
- `create_submission()`: Generate submission file
- `run_inference()`: Complete inference workflow

## ⚙️ Configuration

### Key Configuration Parameters

```python
# Model Selection
config.model.MODELS_TO_TEST = ['xgboost', 'lightgbm', 'catboost']

# Optimization Settings
config.model.N_TRIALS = 15  # Optuna trials
config.model.N_FOLDS_OPT = 3  # CV folds for optimization
config.model.MAX_OPT_SAMPLES = 100000  # Sample size for tuning

# Training Settings
config.model.N_FOLDS_SUB = 5  # CV folds for final training
config.model.MAX_ITERATIONS = 1500  # Max boosting rounds
config.model.EARLY_STOPPING_ROUNDS = 50

# GPU Settings
config.model.USE_GPU = False  # Set True for GPU acceleration

# Feature Engineering
config.features.NUMERICAL_TO_BOOLEAN = [
    "cardiovascular_history",
    "hypertension_history",
    "family_history_diabetes"
]
config.features.SKEWED_TO_GAUSS = ['physical_activity_minutes_per_week']
```

## 🎯 Kaggle Execution

### Training in Kaggle

```python
# In a Kaggle notebook
from config import Config
from train_pipeline import training_pipeline

# Configure for Kaggle paths
config = Config()
config.paths.TRAIN_DATA_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
config.paths.TEST_DATA_PATH = "/kaggle/input/playground-series-s5e12/test.csv"
config.paths.OUTPUT_DIR = "/kaggle/working/"
config.paths.MODEL_DIR = "/kaggle/working/models/"

# Enable GPU if available
config.model.USE_GPU = True

# Run training
results = training_pipeline(config=config)
```

### Inference in Kaggle

```python
# In a Kaggle notebook
from config import Config
from predict_pipeline import inference_pipeline

# Configure paths
config = Config()
config.paths.TEST_DATA_PATH = "/kaggle/input/playground-series-s5e12/test.csv"
config.paths.MODEL_DIR = "/kaggle/input/your-trained-models/models/"
config.paths.OUTPUT_DIR = "/kaggle/working/"

# Run inference
submission = inference_pipeline(config=config)
```

## 📊 Output Files

The pipeline generates the following outputs:

1. **Models**: Saved in `MODEL_DIR`
   - `model_{model_name}_fold_0.pkl`
   - `model_{model_name}_fold_1.pkl`
   - etc.

2. **Visualizations**:
   - `model_comparison.png`: Compare all models
   - `feature_importance_best_model.png`: Top features
   - `shap_summary_best_model.png`: SHAP analysis

3. **Predictions**:
   - `submission.csv`: Final predictions

## 🔬 Model Interpretability

The pipeline includes SHAP (SHapley Additive exPlanations) analysis for model interpretability:

```python
# SHAP analysis is automatically generated for the best model
# Find the plot at: shap_summary_best_model.png

# The plot shows:
# - Which features are most important
# - How features impact predictions (positive/negative)
# - Feature value distributions
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Error during Optuna**
   - Reduce `MAX_OPT_SAMPLES`
   - Reduce `N_TRIALS`
   - Use fewer `N_FOLDS_OPT`

2. **GPU Not Detected**
   - Ensure CUDA is installed
   - Set `USE_GPU = False` for CPU training

3. **Category Alignment Issues**
   - Ensure both train and test are processed together
   - Check for new categories in test data

## 📈 Performance Optimization

### Speed Improvements
1. **Reduce sample size for Optuna**: `MAX_OPT_SAMPLES = 50000`
2. **Fewer trials**: `N_TRIALS = 10`
3. **Fewer folds during optimization**: `N_FOLDS_OPT = 2`
4. **Enable GPU**: `USE_GPU = True`

### Quality Improvements
1. **More trials**: `N_TRIALS = 30`
2. **More CV folds**: `N_FOLDS_SUB = 10`
3. **Larger optimization sample**: Use full dataset
4. **More boosting rounds**: `MAX_ITERATIONS = 2000`

## 📝 Best Practices

1. **Always use configuration file**: Centralized settings
2. **Version your models**: Include timestamps in model names
3. **Log everything**: Use logging for debugging
4. **Validate preprocessing**: Check data distributions
5. **Monitor cross-validation**: Ensure stable performance
6. **Save intermediate results**: Don't lose training progress

## 🤝 Contributing

To extend this pipeline:

1. **Add new models**: Update `ModelTrainer.get_model_class()`
2. **Add features**: Extend `DataPreprocessor`
3. **Add metrics**: Extend `ModelEvaluator`
4. **Add transformations**: Create new preprocessing methods

## 📄 License

This project is provided as-is for educational and competition purposes.

## 🙏 Acknowledgments

- Optuna for hyperparameter optimization
- SHAP for model interpretability
- XGBoost, LightGBM, CatBoost teams
- Kaggle for hosting the competition
