"""
Feature Engineering Module
Creates new features to improve model performance.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates engineered features for hotel booking prediction."""
    
    def __init__(self):
        self.outlier_stats = {}
        self.scaler = StandardScaler()
        self.numeric_features = []
        
    def fit_transform(self, X_train):
        """
        Fit feature engineering on training data and transform.
        
        Args:
            X_train (pd.DataFrame): Training features
            
        Returns:
            pd.DataFrame: Transformed features
        """
        logger.info("Engineering features on training data")
        
        X_engineered = X_train.copy()
        
        # Create new features
        X_engineered = self._create_features(X_engineered)
        
        # Handle outliers (fit on train)
        X_engineered = self._handle_outliers(X_engineered, fit=True)
        
        # One-hot encode categorical features
        X_engineered = self._encode_categorical(X_engineered, fit=True)
        
        # Scale numeric features
        X_engineered = self._scale_features(X_engineered, fit=True)
        
        logger.info(f"Feature engineering complete. Final shape: {X_engineered.shape}")
        logger.info(f"Total features created: {X_engineered.shape[1]}")
        
        return X_engineered
    
    def transform(self, X_test):
        """
        Transform test data using fitted feature engineering.
        
        Args:
            X_test (pd.DataFrame): Test features
            
        Returns:
            pd.DataFrame: Transformed features
        """
        logger.info("Engineering features on test data")
        
        X_engineered = X_test.copy()
        
        # Apply same feature creation
        X_engineered = self._create_features(X_engineered)
        
        # Apply outlier treatment using training stats
        X_engineered = self._handle_outliers(X_engineered, fit=False)
        
        # Apply same encoding
        X_engineered = self._encode_categorical(X_engineered, fit=False)
        
        # Apply same scaling
        X_engineered = self._scale_features(X_engineered, fit=False)
        
        logger.info(f"Test feature engineering complete. Shape: {X_engineered.shape}")
        
        return X_engineered
    
    def _create_features(self, df):
        """
        Create all engineered features.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with new features
        """
        df = df.copy()
        
        # Feature 1: Total stay nights
        df['total_stay_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
        logger.info("✓ Created feature: total_stay_nights")
        
        # Feature 2: Total guests
        df['total_guests'] = df['no_of_adults'] + df['no_of_children']
        logger.info("✓ Created feature: total_guests")
        
        # Feature 3: Price per guest (handle division by zero)
        df['price_per_guest'] = df.apply(
            lambda row: row['avg_price_per_room'] / row['total_guests'] if row['total_guests'] > 0 
            else row['avg_price_per_room'], 
            axis=1
        )
        logger.info("✓ Created feature: price_per_guest")
        
        # Feature 4: Lead time category
        df['lead_time_category'] = pd.cut(
            df['lead_time'], 
            bins=[-1, 30, 90, df['lead_time'].max()],
            labels=['short', 'medium', 'long']
        )
        logger.info("✓ Created feature: lead_time_category")
        
        # Feature 5: Weekend booking flag
        df['is_weekend_booking'] = (df['no_of_weekend_nights'] > 0).astype(int)
        logger.info("✓ Created feature: is_weekend_booking")
        
        # Feature 6: Has special requests flag
        df['has_special_requests'] = (df['no_of_special_requests'] > 0).astype(int)
        logger.info("✓ Created feature: has_special_requests")
        
        # Feature 7: Peak season booking (June-Aug, December)
        df['peak_season'] = df['arrival_month'].apply(
            lambda x: 1 if x in [6, 7, 8, 12] else 0
        )
        logger.info("✓ Created feature: peak_season")
        
        # Additional useful features
        
        # Feature 8: Average price per night
        df['price_per_night'] = df.apply(
            lambda row: row['avg_price_per_room'] / row['total_stay_nights'] if row['total_stay_nights'] > 0 
            else row['avg_price_per_room'], 
            axis=1
        )
        logger.info("✓ Created feature: price_per_night")
        
        # Feature 9: Booking to arrival ratio (lead time vs stay)
        df['booking_to_stay_ratio'] = df.apply(
            lambda row: row['lead_time'] / row['total_stay_nights'] if row['total_stay_nights'] > 0 
            else row['lead_time'], 
            axis=1
        )
        logger.info("✓ Created feature: booking_to_stay_ratio")
        
        # Feature 10: Loyalty indicator (previous bookings)
        df['is_loyal_customer'] = (df['no_of_previous_bookings_not_canceled'] > 0).astype(int)
        logger.info("✓ Created feature: is_loyal_customer")
        
        return df
    
    def _handle_outliers(self, df, fit=True):
        """
        Detect and handle outliers using winsorization.
        
        Args:
            df (pd.DataFrame): Input data
            fit (bool): Whether to fit outlier boundaries
            
        Returns:
            pd.DataFrame: Data with outliers handled
        """
        df = df.copy()
        
        # Columns to check for outliers
        outlier_cols = ['avg_price_per_room', 'lead_time', 'price_per_guest', 
                        'price_per_night', 'booking_to_stay_ratio']
        
        for col in outlier_cols:
            if col in df.columns:
                if fit:
                    # Calculate percentiles on training data
                    lower_bound = df[col].quantile(0.01)
                    upper_bound = df[col].quantile(0.99)
                    self.outlier_stats[col] = {'lower': lower_bound, 'upper': upper_bound}
                    
                    # Count outliers before treatment
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    logger.info(f"Found {outliers} outliers in {col} (will cap at 1st and 99th percentiles)")
                
                # Apply winsorization
                df[col] = df[col].clip(
                    lower=self.outlier_stats[col]['lower'],
                    upper=self.outlier_stats[col]['upper']
                )
        
        return df
    
    def _encode_categorical(self, df, fit=True):
        """
        One-hot encode categorical features.
        
        Args:
            df (pd.DataFrame): Input data
            fit (bool): Whether to fit encoding
            
        Returns:
            pd.DataFrame: Data with encoded categoricals
        """
        df = df.copy()
        
        categorical_features = ['type_of_meal_plan', 'room_type_reserved', 
                                'market_segment_type', 'lead_time_category']
        
        if fit:
            self.categorical_columns = categorical_features
            self.encoded_columns = {}
        
        for col in categorical_features:
            if col in df.columns:
                # One-hot encode
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                
                if fit:
                    # Store the column names from training
                    self.encoded_columns[col] = dummies.columns.tolist()
                else:
                    # For test set, ensure exact same columns as training
                    # First, add any missing columns with zeros
                    for expected_col in self.encoded_columns[col]:
                        if expected_col not in dummies.columns:
                            dummies[expected_col] = 0
                    
                    # Then, keep only the columns that were in training (drop extras)
                    # This handles unseen categories in test set
                    dummies = dummies[self.encoded_columns[col]]
                
                # Drop original column first
                df = df.drop(col, axis=1)
                
                # Then concatenate the aligned dummy columns
                df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def _scale_features(self, df, fit=True):
        """
        Scale numeric features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input data
            fit (bool): Whether to fit scaler
            
        Returns:
            pd.DataFrame: Data with scaled features
        """
        df = df.copy()
        
        if fit:
            # Identify numeric columns (excluding binary flags)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclude binary features from scaling
            binary_cols = ['is_weekend_booking', 'has_special_requests', 'peak_season',
                           'repeated_guest', 'required_car_parking_space', 'is_loyal_customer']
            
            self.numeric_features = [col for col in numeric_cols if col not in binary_cols]
            
            if len(self.numeric_features) > 0:
                df[self.numeric_features] = self.scaler.fit_transform(df[self.numeric_features])
                logger.info(f"Scaled {len(self.numeric_features)} numeric features")
        else:
            # Use the exact same numeric features identified during fit 
            # This prevents issues with missing/extra columns in test set
            if len(self.numeric_features) > 0:
                # Verify all required columns exist
                missing_features = [f for f in self.numeric_features if f not in df.columns]
                if missing_features:
                    logger.warning(f"Missing numeric features in test set: {missing_features}")
                
                # Only transform columns that exist in both train and test
                available_features = [f for f in self.numeric_features if f in df.columns]
                df[available_features] = self.scaler.transform(df[available_features])
        
        return df
    
    def get_feature_names(self, df):
        """Get list of all feature names."""
        return df.columns.tolist()


def engineer_features(X_train, X_test=None):
    """
    Convenience function for feature engineering.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features (optional)
        
    Returns:
        Engineered data and fitted engineer
    """
    engineer = FeatureEngineer()
    
    X_train_eng = engineer.fit_transform(X_train)
    
    if X_test is not None:
        X_test_eng = engineer.transform(X_test)
        return X_train_eng, X_test_eng, engineer
    
    return X_train_eng, engineer


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import DataLoader
    from preprocess import DataPreprocessor
    
    loader = DataLoader()
    data = loader.load_data()
    X_train, X_test, y_train, y_test = loader.create_train_test_split()
    
    preprocessor = DataPreprocessor()
    X_train_prep, y_train_prep = preprocessor.fit_transform(X_train, y_train)
    X_test_prep, y_test_prep = preprocessor.transform(X_test, y_test)
    
    engineer = FeatureEngineer()
    X_train_eng = engineer.fit_transform(X_train_prep)
    X_test_eng = engineer.transform(X_test_prep)
    
    print(f"Train shape after feature engineering: {X_train_eng.shape}")
    print(f"Test shape after feature engineering: {X_test_eng.shape}")
    print(f"\nFeature names: {X_train_eng.columns.tolist()[:10]}...")
