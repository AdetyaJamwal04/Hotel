"""
Model Evaluation Module
Evaluates models using multiple metrics and generates reports.
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates ML models and generates comprehensive reports."""
    
    def __init__(self):
        self.evaluation_results = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name='Model'):
        """
        Evaluate a single model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {model_name}")
        logger.info(f"{'='*60}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Log metrics
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {metrics['f1_score']:.4f}")
        if metrics['roc_auc']:
            logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        logger.info(f"\nConfusion Matrix:\n{np.array(metrics['confusion_matrix'])}")
        
        # Store results
        self.evaluation_results[model_name] = metrics
        
        return metrics
    
    def evaluate_all_models(self, models, X_test, y_test):
        """
        Evaluate multiple models.
        
        Args:
            models (dict): Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Evaluation results for all models
        """
        logger.info("\n" + "="*70)
        logger.info("EVALUATING ALL MODELS")
        logger.info("="*70)
        
        for model_name, model in models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
        
        return self.evaluation_results
    
    def get_best_model(self, metric='f1_score'):
        """
        Identify the best performing model based on a metric.
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            tuple: (best_model_name, best_score)
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        best_model = max(
            self.evaluation_results.items(),
            key=lambda x: x[1][metric] if x[1][metric] is not None else -np.inf
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Best model based on {metric}: {best_model[0]}")
        logger.info(f"{metric}: {best_model[1][metric]:.4f}")
        logger.info(f"{'='*60}")
        
        return best_model[0], best_model[1][metric]
    
    def plot_confusion_matrices(self, save_path='logs/confusion_matrices.png'):
        """
        Plot confusion matrices for all evaluated models.
        
        Args:
            save_path (str): Path to save the figure
        """
        n_models = len(self.evaluation_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.evaluation_results.items()):
            cm = np.array(results['confusion_matrix'])
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Canceled', 'Canceled'],
                yticklabels=['Not Canceled', 'Canceled'],
                ax=axes[idx]
            )
            
            axes[idx].set_title(f'{model_name}\nF1-Score: {results["f1_score"]:.4f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        # Save figure
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrices saved to {save_path}")
        
        plt.close()
    
    def plot_metrics_comparison(self, save_path='logs/metrics_comparison.png'):
        """
        Plot comparison of metrics across all models.
        
        Args:
            save_path (str): Path to save the figure
        """
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        data = []
        for model_name, results in self.evaluation_results.items():
            for metric in metrics_to_plot:
                if results.get(metric) is not None:
                    data.append({
                        'Model': model_name,
                        'Metric': metric.replace('_', ' ').title(),
                        'Score': results[metric]
                    })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 6))
        
        # Create grouped bar chart
        models = df['Model'].unique()
        metrics = df['Metric'].unique()
        x = np.arange(len(metrics))
        width = 0.25
        
        for idx, model in enumerate(models):
            model_data = df[df['Model'] == model]
            scores = [model_data[model_data['Metric'] == m]['Score'].values[0] 
                     if len(model_data[model_data['Metric'] == m]) > 0 else 0 
                     for m in metrics]
            
            plt.bar(x + idx*width, scores, width, label=model)
        
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width, metrics, rotation=15)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics comparison saved to {save_path}")
        
        plt.close()
    
    def extract_feature_importance(self, model, feature_names, model_name='Model', top_n=15):
        """
        Extract and visualize feature importance.
        
        Args:
            model: Trained model
            feature_names (list): List of feature names
            model_name (str): Name of the model
            top_n (int): Number of top features to show
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        logger.info(f"\nExtracting feature importance for {model_name}")
        
        # Get feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning(f"Cannot extract feature importance for {model_name}")
            return None
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Log top features
        logger.info(f"\nTop {top_n} most important features:")
        for idx, row in importance_df.head(top_n).iterrows():
            logger.info(f"  {row['feature']:30s}: {row['importance']:.6f}")
        
        return importance_df
    
    def plot_feature_importance(self, importance_df, model_name='Model', 
                                 top_n=15, save_path='logs/feature_importance.png'):
        """
        Plot feature importance.
        
        Args:
            importance_df (pd.DataFrame): Feature importance data
            model_name (str): Name of the model
            top_n (int): Number of top features to display
            save_path (str): Path to save the figure
        """
        if importance_df is None:
            return
        
        plt.figure(figsize=(10, 8))
        
        top_features = importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Save figure
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
    
    def generate_business_interpretation(self, importance_df, top_n=10):
        """
        Generate business interpretation of top features.
        
        Args:
            importance_df (pd.DataFrame): Feature importance data
            top_n (int): Number of features to interpret
            
        Returns:
            str: Business interpretation
        """
        if importance_df is None:
            return "Feature importance not available for this model."
        
        top_features = importance_df.head(top_n)['feature'].tolist()
        
        interpretations = {
            'lead_time': 'Booking lead time is crucial - longer advance bookings may have different cancellation patterns',
            'avg_price_per_room': 'Room price strongly influences cancellation - higher prices may increase cancellation likelihood',
            'no_of_special_requests': 'Special requests indicate customer commitment and reduce cancellation probability',
            'has_special_requests': 'Customers with special requests show higher booking commitment',
            'total_stay_nights': 'Length of stay affects cancellation - longer stays may have different behavior',
            'price_per_guest': 'Per-person cost perception impacts cancellation decisions',
            'market_segment_type': 'Customer acquisition channel affects cancellation patterns',
            'repeated_guest': 'Loyal customers are less likely to cancel',
            'no_of_previous_cancellations': 'Past cancellation history is a strong predictor',
            'booking_to_stay_ratio': 'Ratio of lead time to stay duration indicates planning behavior',
            'is_loyal_customer': 'Customers with booking history show more commitment',
            'peak_season': 'Peak season bookings have different cancellation dynamics'
        }
        
        interpretation = "\n" + "="*70 + "\n"
        interpretation += "BUSINESS INTERPRETATION OF KEY PREDICTORS\n"
        interpretation += "="*70 + "\n\n"
        
        for idx, feature in enumerate(top_features, 1):
            # Find matching interpretation (handle encoded features)
            base_feature = None
            for key in interpretations.keys():
                if key in feature:
                    base_feature = key
                    break
            
            if base_feature:
                interpretation += f"{idx}. {feature}\n"
                interpretation += f"   → {interpretations[base_feature]}\n\n"
            else:
                interpretation += f"{idx}. {feature}\n"
                interpretation += f"   → Important predictor of booking cancellations\n\n"
        
        logger.info(interpretation)
        
        return interpretation
    
    def save_evaluation_report(self, output_path='logs/evaluation_report.json'):
        """
        Save evaluation results to JSON file.
        
        Args:
            output_path (str): Path to save report
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        report = {}
        for model_name, results in self.evaluation_results.items():
            report[model_name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results.items()
            }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Evaluation report saved to {output_path}")


if __name__ == "__main__":
    print("Model evaluation module - run from main.py")
