"""
Data Loading Module
Handles data loading, validation, and train-test splitting.
"""

import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and validation of hotel booking data."""
    
    EXPECTED_COLUMNS = [
        'Booking_ID', 'no_of_adults', 'no_of_children', 'no_of_weekend_nights',
        'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space',
        'room_type_reserved', 'lead_time', 'arrival_year', 'arrival_month',
        'arrival_date', 'market_segment_type', 'repeated_guest',
        'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
        'avg_price_per_room', 'no_of_special_requests', 'booking_status'
    ]
    
    def __init__(self, data_path='data/raw/Hotel_Reservations.csv'):
        """
        Initialize DataLoader.
        
        Args:
            data_path (str): Path to the CSV file
        """
        self.data_path = Path(data_path)
        self.data = None
        
    def load_data(self):
        """
        Load data from CSV file with validation.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            logger.info(f"Loading data from {self.data_path}")
            self.data = pd.read_csv(self.data_path)
            
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            self._validate_schema()
            self._log_data_summary()
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _validate_schema(self):
        """Validate that all expected columns are present."""
        missing_cols = set(self.EXPECTED_COLUMNS) - set(self.data.columns)
        
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        
        logger.info("Schema validation passed")
    
    def _log_data_summary(self):
        """Log summary statistics of the loaded data."""
        logger.info(f"Total records: {len(self.data)}")
        logger.info(f"Total features: {len(self.data.columns)}")
        logger.info(f"Missing values:\n{self.data.isnull().sum().sum()} total")
        logger.info(f"Duplicate rows: {self.data.duplicated().sum()}")
        
        if 'booking_status' in self.data.columns:
            target_dist = self.data['booking_status'].value_counts()
            logger.info(f"Target distribution:\n{target_dist}")
    
    def create_train_test_split(self, test_size=0.2, random_state=42):
        """
        Create stratified train-test split.
        
        Args:
            test_size (float): Proportion of dataset for test set
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info(f"Creating train-test split (test_size={test_size})")
        
        # Separate features and target
        X = self.data.drop(['booking_status', 'Booking_ID'], axis=1)
        y = self.data['booking_status']
        
        # Stratified split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.info(f"Train target distribution:\n{y_train.value_counts()}")
        logger.info(f"Test target distribution:\n{y_test.value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, data, filename):
        """
        Save processed data to CSV.
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Output filename
        """
        output_path = Path('data/processed') / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")


def load_data(data_path='data/raw/Hotel_Reservations.csv'):
    """
    Convenience function to load data.
    
    Args:
        data_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    loader = DataLoader(data_path)
    return loader.load_data()


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    data = loader.load_data()
    print(f"\nData shape: {data.shape}")
    print(f"\nFirst few rows:\n{data.head()}")
    
    # Test train-test split
    X_train, X_test, y_train, y_test = loader.create_train_test_split()
    print(f"\nTrain-test split created successfully!")
