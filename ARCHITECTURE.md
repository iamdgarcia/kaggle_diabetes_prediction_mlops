# MLOps Pipeline Architecture

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CONFIG LAYER                             │
│                        (config.py)                               │
│  - PathConfig: File paths and directories                        │
│  - ModelConfig: Training hyperparameters                         │
│  - FeatureConfig: Feature engineering rules                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                            │
│                   (train_pipeline.py)                            │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│     DATA     │    │    MODEL     │    │ EVALUATION   │
│ PREPROCESSING│───▶│   TRAINING   │───▶│              │
│              │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    Features          Trained Models        Visualizations
   (X, y, cat_idx)    (fold_models)    (plots + metrics)
         │                    │
         │                    │
         └────────┬───────────┘
                  ▼
         ┌──────────────┐
         │  INFERENCE   │
         │   PIPELINE   │
         │              │
         └──────────────┘
                  │
                  ▼
            Predictions
         (submission.csv)
```

## 📦 Module Breakdown

### 1. Configuration Module (`config.py`)
**Purpose**: Centralized configuration management

**Components**:
- `PathConfig`: Manages all file paths
- `ModelConfig`: Training parameters, model selection
- `FeatureConfig`: Feature engineering specifications
- `Config`: Main configuration class

**Key Features**:
- Single source of truth for all settings
- Easy environment switching (local/Kaggle)
- Type hints for better IDE support
- Validation and defaults

### 2. Data Preprocessing (`data_preprocessing.py`)
**Purpose**: Data loading and feature engineering

**Class**: `DataPreprocessor`

**Methods**:
```python
load_data()                           # Load CSV files
convert_numerical_to_categorical()    # Type conversion
apply_log_transformation()            # Handle skewed features
align_categories()                    # Sync train/test categories
prepare_features()                    # Complete feature prep
preprocess_pipeline()                 # End-to-end pipeline
```

**Features**:
- Handles missing values
- Type conversions
- Log transformations for skewed data
- Category alignment between train/test
- Automatic categorical feature detection

**Output**:
- `X_train`, `y_train`: Training features and labels
- `X_test`: Test features
- `cat_features_indices`: Indices of categorical features

### 3. Model Training (`model_training.py`)
**Purpose**: Model training and hyperparameter optimization

**Class**: `ModelTrainer`

**Methods**:
```python
optimize_hyperparameters()     # Optuna-based HPO
run_cv_with_early_stopping()   # Efficient CV
optimize_all_models()          # Compare multiple models
train_final_model()            # Train with best params
```

**Features**:
- Optuna TPE sampler for smart hyperparameter search
- Early stopping for efficiency
- K-Fold cross-validation
- GPU support (XGBoost, LightGBM, CatBoost)
- Ensemble through fold averaging

**Hyperparameter Spaces**:
- **XGBoost**: max_depth, learning_rate, subsample, colsample_bytree, min_child_weight
- **LightGBM**: learning_rate, num_leaves, max_depth, min_child_samples, subsample, colsample_bytree
- **CatBoost**: learning_rate, depth, l2_leaf_reg, subsample

**Output**:
- `study_results`: Optimization results for all models
- `fold_models`: List of trained models (one per fold)
- `predictions`: Test set predictions
- `feature_importances`: Feature importance scores

### 4. Model Evaluation (`model_evaluation.py`)
**Purpose**: Performance evaluation and visualization

**Class**: `ModelEvaluator`

**Methods**:
```python
plot_model_comparison()     # Compare all models
plot_feature_importance()   # Top features
plot_shap_summary()        # SHAP analysis
evaluate_and_visualize()   # Complete evaluation
```

**Visualizations**:
1. **Model Comparison**: Bar chart of CV AUC scores
2. **Feature Importance**: Top 20 features by magnitude
3. **SHAP Summary**: Directional feature impact

**Output Files**:
- `model_comparison.png`
- `feature_importance_best_model.png`
- `shap_summary_best_model.png`

### 5. Inference Pipeline (`inference_pipeline.py`)
**Purpose**: Production inference and prediction generation

**Class**: `InferencePipeline`

**Methods**:
```python
load_models()           # Load saved models
predict_single_model()  # Single model prediction
predict_ensemble()      # Average fold predictions
create_submission()     # Generate submission file
run_inference()        # Complete inference workflow
```

**Features**:
- Model serialization/deserialization
- Ensemble predictions (fold averaging)
- Submission file generation
- Validation checks

**Output**:
- `submission.csv`: Competition submission file

### 6. Utility Functions (`utils.py`)
**Purpose**: Reusable helper functions

**Functions**:
- JSON/Pickle I/O
- Timestamp generation
- DataFrame analysis
- Memory optimization
- Submission validation
- Class weight calculation

## 🔄 Data Flow

### Training Flow
```
Raw Data (CSV)
    │
    ▼
[Load Data]
    │
    ▼
[Feature Engineering]
    │
    ├─ Convert types
    ├─ Log transform
    └─ Align categories
    │
    ▼
[Hyperparameter Optimization]
    │
    ├─ XGBoost tuning
    ├─ LightGBM tuning
    └─ CatBoost tuning
    │
    ▼
[Select Best Model]
    │
    ▼
[Train Final Model]
    │
    ├─ K-Fold CV
    ├─ Early stopping
    └─ Save each fold
    │
    ▼
[Evaluate & Visualize]
    │
    ├─ Model comparison
    ├─ Feature importance
    └─ SHAP analysis
    │
    ▼
Trained Models + Predictions
```

### Inference Flow
```
Test Data (CSV)
    │
    ▼
[Load Data]
    │
    ▼
[Apply Same Transformations]
    │
    ├─ Convert types
    ├─ Log transform
    └─ Align categories
    │
    ▼
[Load Trained Models]
    │
    ├─ Load fold 0
    ├─ Load fold 1
    └─ ...
    │
    ▼
[Generate Predictions]
    │
    ├─ Predict with each fold
    └─ Average predictions
    │
    ▼
[Create Submission]
    │
    └─ Format & validate
    │
    ▼
Submission File
```

## 🎯 Design Principles

### 1. Modularity
Each module has a single responsibility and can be used independently:
- Preprocessing can run standalone
- Training can use pre-processed data
- Inference can use trained models

### 2. Configuration-Driven
All parameters in one place (`config.py`):
- Easy environment switching
- No hardcoded values
- Version-controlled settings

### 3. Reproducibility
Ensures consistent results:
- Fixed random seeds
- Same preprocessing pipeline
- Versioned models

### 4. Extensibility
Easy to add new components:
- New models: Update `ModelTrainer`
- New features: Extend `DataPreprocessor`
- New metrics: Extend `ModelEvaluator`

### 5. Production-Ready
Built for deployment:
- Error handling
- Logging throughout
- Model persistence
- Validation checks

## 🔧 Key Technologies

- **Data Processing**: Pandas, NumPy
- **ML Framework**: Scikit-learn
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- **Hyperparameter Tuning**: Optuna
- **Model Interpretation**: SHAP
- **Visualization**: Matplotlib, Seaborn

## 📈 Performance Characteristics

### Training Time (Approximate)
- **Fast Mode** (5 trials, 3 folds): ~15-30 minutes
- **Balanced Mode** (15 trials, 5 folds): ~45-90 minutes
- **Quality Mode** (30 trials, 10 folds): ~2-4 hours

*Times vary based on dataset size and hardware*

### Memory Usage
- **Preprocessing**: ~200-500 MB
- **Training**: ~1-2 GB (per model)
- **Inference**: ~500 MB

### Disk Space
- **Models**: ~50-200 MB (per fold)
- **Visualizations**: ~1-5 MB
- **Logs**: ~1-10 MB

## 🚀 Deployment Considerations

### For Kaggle
- Set paths to `/kaggle/input/` and `/kaggle/working/`
- Enable GPU if available
- Use appropriate time limits

### For Production API
1. Separate training and inference pipelines
2. Use model versioning
3. Implement batch prediction
4. Add monitoring and logging
5. Container deployment (Docker)

### For Scheduled Retraining
1. Automate data loading
2. Trigger on new data
3. Compare with previous models
4. Update if improved
5. Archive old models

## 🔐 Best Practices Implemented

✅ Separation of concerns
✅ Configuration management
✅ Error handling and logging
✅ Type hints for clarity
✅ Docstrings for all functions
✅ Reproducible results
✅ Model versioning
✅ Validation checks
✅ Memory optimization
✅ GPU support
