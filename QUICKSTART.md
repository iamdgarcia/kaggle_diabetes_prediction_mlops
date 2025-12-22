# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### For Kaggle Notebook

1. **Upload Files**: Upload all `.py` files to your Kaggle notebook

2. **Training**:
```python
from config import Config
from train_pipeline import training_pipeline

config = Config()
config.paths.TRAIN_DATA_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
config.paths.TEST_DATA_PATH = "/kaggle/input/playground-series-s5e12/test.csv"
config.model.USE_GPU = False  # Set True if GPU is enabled

results = training_pipeline(config=config)
```

3. **Done!** Your submission file is at `/kaggle/working/submission.csv`

### For Local Development

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Update Paths** in `config.py`:
```python
TRAIN_DATA_PATH = "path/to/train.csv"
TEST_DATA_PATH = "path/to/test.csv"
```

3. **Run Training**:
```python
python train_pipeline.py
```

## 📊 What You Get

After running the pipeline, you'll have:

✅ **Submission file** (`submission.csv`) - ready to submit
✅ **Trained models** (saved in `models/` directory)
✅ **Performance plots**:
- Model comparison chart
- Feature importance plot
- SHAP interpretability plot

## ⚙️ Key Configuration Options

### Fast Mode (Quick Testing)
```python
config.model.N_TRIALS = 5
config.model.N_FOLDS_OPT = 2
config.model.N_FOLDS_SUB = 3
```

### Quality Mode (Best Results)
```python
config.model.N_TRIALS = 30
config.model.N_FOLDS_OPT = 5
config.model.N_FOLDS_SUB = 10
config.model.USE_GPU = True
```

### Custom Model Selection
```python
config.model.MODELS_TO_TEST = ['xgboost']  # Test only one model
# OR
config.model.MODELS_TO_TEST = ['xgboost', 'lightgbm', 'catboost']  # Test all
```

## 🔄 Inference Only

If you already have trained models:

```python
from config import Config
from predict_pipeline import inference_pipeline

config = Config()
config.paths.TEST_DATA_PATH = "/path/to/test.csv"
config.paths.MODEL_DIR = "/path/to/models/"

submission = inference_pipeline(config=config)
```

## 🐛 Common Issues

**Issue**: Out of memory
**Solution**: Reduce `MAX_OPT_SAMPLES` or use fewer folds

**Issue**: Takes too long
**Solution**: Reduce `N_TRIALS` or test fewer models

**Issue**: GPU not working
**Solution**: Set `USE_GPU = False` or check CUDA installation

## 📚 File Structure

```
├── config.py                    # 🎛️ All settings here
├── train_pipeline.py           # 🏋️ Run this for training
├── predict_pipeline.py         # 🔮 Run this for inference
├── data_preprocessing.py       # 🔧 Data preparation
├── model_training.py          # 🤖 Model training logic
├── model_evaluation.py        # 📊 Evaluation & plots
├── inference_pipeline.py      # 🚀 Prediction generation
├── utils.py                   # 🛠️ Helper functions
└── requirements.txt           # 📦 Dependencies
```

## 💡 Pro Tips

1. **Always start with Fast Mode** to validate the pipeline works
2. **Enable GPU** for 2-3x speedup if available
3. **Use fewer models** (`['xgboost']`) for faster iteration
4. **Increase folds** (`N_FOLDS_SUB = 10`) for stable predictions
5. **Save your models** - they can be reused for inference

## 🎯 Next Steps

1. ✅ Run the pipeline with default settings
2. 📊 Check the visualizations
3. ⚙️ Tune hyperparameters for better results
4. 🚀 Submit to Kaggle
5. 🔁 Iterate and improve

## 📞 Need Help?

Check the detailed `README.md` for:
- Complete documentation
- Advanced configuration
- Troubleshooting guide
- MLOps best practices
