"""
Model Training Module
Trains multiple ML models with hyperparameter tuning and handles class imbalance.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and manages multiple ML models for booking cancellation prediction."""
    
    def __init__(self, random_state=42):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.smote = None
        
    def apply_smote(self, X_train, y_train):
        """
        Apply SMOTE to balance the training dataset.
        
        Args:
            X_train (pd.DataFrame or np.array): Training features
            y_train (pd.Series or np.array): Training labels
            
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        logger.info("Applying SMOTE to handle class imbalance...")
        
        original_dist = pd.Series(y_train).value_counts().sort_index()
        logger.info(f"Original class distribution:\n{original_dist}")
        
        self.smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
        
        resampled_dist = pd.Series(y_resampled).value_counts().sort_index()
        logger.info(f"Resampled class distribution:\n{resampled_dist}")
        logger.info(f"SMOTE complete: {len(X_train)} → {len(X_resampled)} samples")
        
        return X_resampled, y_resampled
    
    def train_logistic_regression(self, X_train, y_train):
        """
        Train Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        logger.info("\n" + "="*60)
        logger.info("Training Logistic Regression (Baseline Model)")
        logger.info("="*60)
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            class_weight='balanced',
            solver='lbfgs'
        )
        
        model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        logger.info(f"5-Fold CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.models['Logistic Regression'] = model
        logger.info("✓ Logistic Regression training complete")
        
        return model
    
    def train_random_forest(self, X_train, y_train):
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        logger.info("\n" + "="*60)
        logger.info("Training Random Forest")
        logger.info("="*60)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        logger.info(f"5-Fold CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.models['Random Forest'] = model
        logger.info("✓ Random Forest training complete")
        
        return model
    
    def train_xgboost(self, X_train, y_train, tune_hyperparameters=False):
        """
        Train XGBoost model with optional hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            tune_hyperparameters (bool): Whether to run RandomizedSearchCV
            
        Returns:
            Trained model
        """
        logger.info("\n" + "="*60)
        logger.info("Training XGBoost")
        logger.info("="*60)
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        if tune_hyperparameters:
            logger.info("Starting hyperparameter tuning with RandomizedSearchCV...")
            
            # Define parameter distributions
            param_distributions = {
                'max_depth': [3, 5, 7, 9, 11],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2]
            }
            
            base_model = XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            random_search = RandomizedSearchCV(
                base_model,
                param_distributions=param_distributions,
                n_iter=20,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )
            
            random_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters: {random_search.best_params_}")
            logger.info(f"Best F1-Score: {random_search.best_score_:.4f}")
            
            model = random_search.best_estimator_
            
        else:
            # Train with default good parameters
            model = XGBClassifier(
                max_depth=7,
                learning_rate=0.1,
                n_estimators=200,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            logger.info(f"5-Fold CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        self.models['XGBoost'] = model
        logger.info("✓ XGBoost training complete")
        
        return model
    
    def train_all_models(self, X_train, y_train, use_smote=True, tune_xgboost=True):
        """
        Train all models in the pipeline.
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_smote (bool): Whether to apply SMOTE
            tune_xgboost (bool): Whether to tune XGBoost hyperparameters
            
        Returns:
            dict: All trained models
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("="*70)
        logger.info(f"Training set size: {X_train.shape}")
        logger.info(f"Use SMOTE: {use_smote}")
        logger.info(f"Tune XGBoost: {tune_xgboost}")
        
        # Apply SMOTE if requested
        if use_smote:
            X_train_balanced, y_train_balanced = self.apply_smote(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Train all models
        self.train_logistic_regression(X_train_balanced, y_train_balanced)
        self.train_random_forest(X_train_balanced, y_train_balanced)
        self.train_xgboost(X_train_balanced, y_train_balanced, tune_hyperparameters=tune_xgboost)
        
        logger.info("\n" + "="*70)
        logger.info(f"✓ All {len(self.models)} models trained successfully!")
        logger.info("="*70)
        
        return self.models
    
    def save_model(self, model, model_name, output_dir='models'):
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model object
            model_name (str): Name for the saved model
            output_dir (str): Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name.replace(' ', '_').lower()}_{timestamp}.pkl"
        filepath = output_path / filename
        
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    def save_all_models(self, output_dir='models'):
        """
        Save all trained models.
        
        Args:
            output_dir (str): Directory to save models
            
        Returns:
            list: Paths to saved models
        """
        saved_paths = []
        
        for model_name, model in self.models.items():
            filepath = self.save_model(model, model_name, output_dir)
            saved_paths.append(filepath)
        
        logger.info(f"\n✓ All {len(self.models)} models saved successfully")
        
        return saved_paths
    
    def set_best_model(self, model_name):
        """
        Set the best performing model.
        
        Args:
            model_name (str): Name of the best model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in trained models")
        
        self.best_model = self.models[model_name]
        self.best_model_name = model_name
        logger.info(f"Best model set to: {model_name}")
    
    def save_best_model(self, output_dir='models'):
        """
        Save the best model with a standard name for deployment.
        
        Args:
            output_dir (str): Directory to save model
            
        Returns:
            Path: Path to saved model
        """
        if self.best_model is None:
            raise ValueError("Best model not set. Call set_best_model() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / 'best_model.pkl'
        
        joblib.dump(self.best_model, filepath)
        logger.info(f"✓ Best model ({self.best_model_name}) saved to {filepath}")
        
        return filepath


def load_model(model_path):
    """
    Load a saved model from disk.
    
    Args:
        model_path (str): Path to saved model
        
    Returns:
        Loaded model
    """
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


if __name__ == "__main__":
    # Test model training
    from data_loader import DataLoader
    from preprocess import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    # Load and prepare data
    loader = DataLoader()
    data = loader.load_data()
    X_train, X_test, y_train, y_test = loader.create_train_test_split()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X_train_prep, y_train_prep = preprocessor.fit_transform(X_train, y_train)
    
    # Engineer features
    engineer = FeatureEngineer()
    X_train_eng = engineer.fit_transform(X_train_prep)
    
    # Train models
    trainer = ModelTrainer()
    models = trainer.train_all_models(X_train_eng, y_train_prep, use_smote=False, tune_xgboost=False)
    
    print(f"\nTrained {len(models)} models:")
    for name in models.keys():
        print(f"  - {name}")
