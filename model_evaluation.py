"""
Model Evaluation and Visualization Module
Contains functions for model evaluation, SHAP analysis, and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import logging
import shap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handles model evaluation and visualization"""
    
    def __init__(self, config):
        """
        Initialize evaluator with configuration
        
        Args:
            config: Configuration object
        """
        self.config = config
        
    def plot_model_comparison(
        self, 
        study_results: Dict[str, Dict[str, Any]], 
        save_path: Optional[str] = None
    ):
        """
        Create bar plot comparing model performance
        
        Args:
            study_results: Dictionary with results from all models
            save_path: Path to save the plot
        """
        logger.info("Generating model comparison plot")
        
        model_names = list(study_results.keys())
        scores = [study_results[m]['best_score'] for m in model_names]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=model_names, y=scores, palette='viridis')
        plt.title('Model Comparison: Best CV AUC Score', fontsize=15)
        plt.ylabel('AUC Score')
        plt.xlabel('Model')
        plt.ylim(min(scores) - 0.05, max(scores) + 0.05)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add score labels on bars
        for i, v in enumerate(scores):
            plt.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.config.paths.OUTPUT_DIR}{self.config.paths.MODEL_COMPARISON_PLOT}"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
        plt.close()
    
    def plot_feature_importance(
        self, 
        feature_names: List[str], 
        feature_importances: np.ndarray,
        model_name: str,
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            feature_importances: Array of feature importance scores
            model_name: Name of the model
            top_n: Number of top features to plot
            save_path: Path to save the plot
        """
        logger.info(f"Plotting top {top_n} feature importances")
        
        # Create dataframe and sort
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        })
        fi_df = fi_df.sort_values(by='importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=fi_df, palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - {model_name.upper()}', fontsize=14)
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.config.paths.OUTPUT_DIR}{self.config.paths.FEATURE_IMPORTANCE_PLOT}"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
        plt.close()
    
    def plot_shap_summary(
        self,
        model,
        X_val: pd.DataFrame,
        model_name: str,
        save_path: Optional[str] = None
    ):
        """
        Generate SHAP summary plot
        
        Args:
            model: Trained model
            X_val: Validation features
            model_name: Name of the model
            save_path: Path to save the plot
        """
        logger.info("Calculating SHAP values")
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_val)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                shap_matrix = shap_values[1]  # For binary classification
            else:
                shap_matrix = shap_values
            
            # Create plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_matrix, X_val, show=False)
            plt.title(f'SHAP Summary (Directional Importance) - {model_name.upper()}')
            plt.tight_layout()
            
            if save_path is None:
                save_path = f"{self.config.paths.OUTPUT_DIR}{self.config.paths.SHAP_SUMMARY_PLOT}"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate SHAP plot: {e}")
    
    def evaluate_and_visualize(
        self,
        study_results: Dict[str, Dict[str, Any]],
        final_model,
        X_val: pd.DataFrame,
        feature_names: List[str],
        feature_importances: np.ndarray,
        model_name: str
    ):
        """
        Complete evaluation pipeline with all visualizations
        
        Args:
            study_results: Results from hyperparameter optimization
            final_model: Trained final model
            X_val: Validation features for SHAP
            feature_names: List of feature names
            feature_importances: Feature importance scores
            model_name: Name of the model
        """
        logger.info("=" * 60)
        logger.info("Starting evaluation and visualization")
        logger.info("=" * 60)
        
        # Plot model comparison
        self.plot_model_comparison(study_results)
        
        # Plot feature importance
        self.plot_feature_importance(
            feature_names,
            feature_importances,
            model_name
        )
        
        # Plot SHAP summary
        self.plot_shap_summary(final_model, X_val, model_name)
        
        logger.info("Evaluation and visualization completed")


def create_evaluator(config):
    """
    Factory function to create evaluator
    
    Args:
        config: Configuration object
        
    Returns:
        ModelEvaluator instance
    """
    return ModelEvaluator(config)
