"""
Data Preprocessing Module
Handles data cleaning, encoding, and scaling.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles preprocessing of hotel booking data."""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None
        self.categorical_features = [
            'type_of_meal_plan', 'room_type_reserved', 'market_segment_type'
        ]
        self.binary_features = ['repeated_guest', 'required_car_parking_space']
        
    def fit_transform(self, X_train, y_train=None):
        """
        Fit preprocessor on training data and transform.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target (optional)
            
        Returns:
            tuple: (X_transformed, y_transformed) or X_transformed
        """
        logger.info("Fitting and transforming training data")
        
        X_processed = X_train.copy()
        
        # Handle any remaining data quality issues
        X_processed = self._validate_and_clean(X_processed)
        
        # Encode target if provided
        y_processed = None
        if y_train is not None:
            y_processed = self._encode_target(y_train, fit=True)
        
        # Store original feature names
        self.feature_names = X_processed.columns.tolist()
        
        logger.info(f"Preprocessing complete. Shape: {X_processed.shape}")
        
        if y_train is not None:
            return X_processed, y_processed
        return X_processed
    
    def transform(self, X_test, y_test=None):
        """
        Transform test data using fitted preprocessor.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target (optional)
            
        Returns:
            tuple: (X_transformed, y_transformed) or X_transformed
        """
        logger.info("Transforming test data")
        
        X_processed = X_test.copy()
        
        # Apply same cleaning
        X_processed = self._validate_and_clean(X_processed)
        
        # Encode target if provided
        y_processed = None
        if y_test is not None:
            y_processed = self._encode_target(y_test, fit=False)
        
        logger.info(f"Transform complete. Shape: {X_processed.shape}")
        
        if y_test is not None:
            return X_processed, y_processed
        return X_processed
    
    def _validate_and_clean(self, df):
        """
        Validate data and handle edge cases.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        df = df.copy()
        
        # Check for negative values in count columns
        count_cols = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 
                      'no_of_week_nights', 'lead_time']
        
        for col in count_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    logger.warning(f"Found {negative_count} negative values in {col}. Setting to 0.")
                    df[col] = df[col].clip(lower=0)
        
        # Handle any missing values that might have appeared
        if df.isnull().sum().sum() > 0:
            logger.warning(f"Found {df.isnull().sum().sum()} missing values. Filling...")
            
            # Fill numeric columns with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        logger.info("Data validation and cleaning complete")
        return df
    
    def _encode_target(self, y, fit=True):
        """
        Encode target variable (booking_status).
        
        Args:
            y (pd.Series): Target variable
            fit (bool): Whether to fit the encoder
            
        Returns:
            np.array: Encoded target
        """
        if fit:
            self.target_encoder = LabelEncoder()
            y_encoded = self.target_encoder.fit_transform(y)
            logger.info(f"Target encoding: {dict(zip(self.target_encoder.classes_, self.target_encoder.transform(self.target_encoder.classes_)))}")
        else:
            y_encoded = self.target_encoder.transform(y)
        
        return y_encoded
    
    def get_feature_names(self):
        """Get list of feature names after preprocessing."""
        return self.feature_names
    
    def inverse_transform_target(self, y_encoded):
        """
        Convert encoded target back to original labels.
        
        Args:
            y_encoded (np.array): Encoded labels
            
        Returns:
            np.array: Original labels
        """
        return self.target_encoder.inverse_transform(y_encoded)


def preprocess_data(X_train, X_test=None, y_train=None, y_test=None):
    """
    Convenience function for preprocessing.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features (optional)
        y_train (pd.Series): Training target (optional)
        y_test (pd.Series): Test target (optional)
        
    Returns:
        Preprocessed data and fitted preprocessor
    """
    preprocessor = DataPreprocessor()
    
    if y_train is not None:
        X_train_processed, y_train_processed = preprocessor.fit_transform(X_train, y_train)
    else:
        X_train_processed = preprocessor.fit_transform(X_train)
        y_train_processed = None
    
    if X_test is not None:
        if y_test is not None:
            X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)
        else:
            X_test_processed = preprocessor.transform(X_test)
            y_test_processed = None
        
        return X_train_processed, X_test_processed, y_train_processed, y_test_processed, preprocessor
    
    return X_train_processed, y_train_processed, preprocessor


if __name__ == "__main__":
    # Test preprocessing
    from data_loader import DataLoader
    
    loader = DataLoader()
    data = loader.load_data()
    X_train, X_test, y_train, y_test = loader.create_train_test_split()
    
    preprocessor = DataPreprocessor()
    X_train_proc, y_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc, y_test_proc = preprocessor.transform(X_test, y_test)
    
    print(f"Train shape after preprocessing: {X_train_proc.shape}")
    print(f"Test shape after preprocessing: {X_test_proc.shape}")
    print(f"Target encoded successfully: {y_train_proc[:5]}")
