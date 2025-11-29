# API Reference Documentation

## Table of Contents

- [Data Loading Module](#data-loading-module)
- [Preprocessing Module](#preprocessing-module)
- [Feature Engineering Module](#feature-engineering-module)
- [Model Training Module](#model-training-module)
- [Model Evaluation Module](#model-evaluation-module)

---

## Data Loading Module

### `DataLoader` Class

**Module:** `src.data_loader`

Handles loading and validation of hotel booking data.

#### Constructor

```python
DataLoader(data_path='data/raw/Hotel_Reservations.csv')
```

**Parameters:**
- `data_path` (str): Path to the CSV file containing hotel booking data

**Attributes:**
- `data` (pd.DataFrame): Loaded dataset
- `EXPECTED_COLUMNS` (list): List of required column names

#### Methods

##### `load_data()`

Load data from CSV file with validation.

**Returns:**
- `pd.DataFrame`: Loaded and validated dataset

**Raises:**
- `FileNotFoundError`: If the data file doesn't exist
- `ValueError`: If expected columns are missing

**Example:**
```python
from src.data_loader import DataLoader

loader = DataLoader('data/raw/Hotel_Reservations.csv')
data = loader.load_data()
print(f"Loaded {len(data)} records")
```

##### `create_train_test_split(test_size=0.2, random_state=42)`

Create stratified train-test split.

**Parameters:**
- `test_size` (float): Proportion of dataset for test set (default: 0.2)
- `random_state` (int): Random seed for reproducibility (default: 42)

**Returns:**
- `tuple`: (X_train, X_test, y_train, y_test)

**Example:**
```python
X_train, X_test, y_train, y_test = loader.create_train_test_split(
    test_size=0.2,
    random_state=42
)
```

##### `save_processed_data(data, filename)`

Save processed data to CSV.

**Parameters:**
- `data` (pd.DataFrame): Data to save
- `filename` (str): Output filename

**Example:**
```python
loader.save_processed_data(processed_data, 'processed_bookings.csv')
```

---

## Preprocessing Module

### `DataPreprocessor` Class

**Module:** `src.preprocess`

Handles data cleaning, encoding, and validation.

#### Constructor

```python
DataPreprocessor()
```

**Attributes:**
- `label_encoders` (dict): Dictionary of fitted label encoders
- `target_encoder` (LabelEncoder): Encoder for target variable
- `categorical_features` (list): List of categorical feature names
- `binary_features` (list): List of binary feature names

#### Methods

##### `fit_transform(X_train, y_train=None)`

Fit preprocessor on training data and transform.

**Parameters:**
- `X_train` (pd.DataFrame): Training features
- `y_train` (pd.Series, optional): Training target

**Returns:**
- `tuple`: (X_transformed, y_transformed) if y_train provided
- `pd.DataFrame`: X_transformed if y_train not provided

**Example:**
```python
from src.preprocess import DataPreprocessor

preprocessor = DataPreprocessor()
X_train_prep, y_train_prep = preprocessor.fit_transform(X_train, y_train)
```

##### `transform(X_test, y_test=None)`

Transform test data using fitted preprocessor.

**Parameters:**
- `X_test` (pd.DataFrame): Test features
- `y_test` (pd.Series, optional): Test target

**Returns:**
- `tuple`: (X_transformed, y_transformed) if y_test provided
- `pd.DataFrame`: X_transformed if y_test not provided

**Example:**
```python
X_test_prep, y_test_prep = preprocessor.transform(X_test, y_test)
```

##### `inverse_transform_target(y_encoded)`

Convert encoded target back to original labels.

**Parameters:**
- `y_encoded` (np.array): Encoded labels (0 or 1)

**Returns:**
- `np.array`: Original labels ('Canceled' or 'Not_Canceled')

**Example:**
```python
original_labels = preprocessor.inverse_transform_target(predictions)
```

##### `get_feature_names()`

Get list of feature names after preprocessing.

**Returns:**
- `list`: Feature names

---

## Feature Engineering Module

### `FeatureEngineer` Class

**Module:** `src.feature_engineering`

Creates engineered features to improve model performance.

#### Constructor

```python
FeatureEngineer()
```

**Attributes:**
- `outlier_stats` (dict): Statistics for outlier detection
- `scaler` (StandardScaler): Fitted scaler for numeric features
- `numeric_features` (list): List of numeric feature names

#### Methods

##### `fit_transform(X_train)`

Fit feature engineering on training data and transform.

**Parameters:**
- `X_train` (pd.DataFrame): Training features

**Returns:**
- `pd.DataFrame`: Transformed features with engineered columns

**Features Created:**
1. `total_stay_nights`: Weekend nights + weekday nights
2. `total_guests`: Adults + children
3. `price_per_guest`: Average price per guest
4. `lead_time_category`: Categorized booking lead time
5. `is_weekend_booking`: Weekend stay indicator
6. `has_special_requests`: Special requests flag
7. `peak_season`: Peak season booking flag
8. `price_per_night`: Average price per night
9. `booking_to_stay_ratio`: Lead time to stay duration ratio
10. `is_loyal_customer`: Customer loyalty indicator

**Example:**
```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
X_train_eng = engineer.fit_transform(X_train_prep)
```

##### `transform(X_test)`

Transform test data using fitted feature engineering.

**Parameters:**
- `X_test` (pd.DataFrame): Test features

**Returns:**
- `pd.DataFrame`: Transformed features

**Example:**
```python
X_test_eng = engineer.transform(X_test_prep)
```

##### `get_feature_names(df)`

Get list of all feature names.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with features

**Returns:**
- `list`: All feature names

---

## Model Training Module

### `ModelTrainer` Class

**Module:** `src.train`

Trains and manages multiple ML models for booking cancellation prediction.

#### Constructor

```python
ModelTrainer(random_state=42)
```

**Parameters:**
- `random_state` (int): Random seed for reproducibility (default: 42)

**Attributes:**
- `models` (dict): Dictionary of trained models
- `best_model_name` (str): Name of the best performing model
- `random_state` (int): Random seed

#### Methods

##### `apply_smote(X_train, y_train)`

Apply SMOTE to balance the training dataset.

**Parameters:**
- `X_train` (pd.DataFrame or np.array): Training features
- `y_train` (pd.Series or np.array): Training labels

**Returns:**
- `tuple`: (X_resampled, y_resampled)

**Example:**
```python
trainer = ModelTrainer()
X_balanced, y_balanced = trainer.apply_smote(X_train, y_train)
```

##### `train_logistic_regression(X_train, y_train)`

Train Logistic Regression model.

**Parameters:**
- `X_train`: Training features
- `y_train`: Training labels

**Returns:**
- Trained LogisticRegression model

**Model Configuration:**
- Solver: liblinear
- Max iterations: 1000
- Class weight: balanced
- Random state: specified in constructor

##### `train_random_forest(X_train, y_train)`

Train Random Forest model.

**Parameters:**
- `X_train`: Training features
- `y_train`: Training labels

**Returns:**
- Trained RandomForestClassifier model

**Model Configuration:**
- Number of estimators: 200
- Max depth: 15
- Min samples split: 10
- Min samples leaf: 4
- Class weight: balanced
- Random state: specified in constructor

##### `train_xgboost(X_train, y_train, tune_hyperparameters=False)`

Train XGBoost model with optional hyperparameter tuning.

**Parameters:**
- `X_train`: Training features
- `y_train`: Training labels
- `tune_hyperparameters` (bool): Whether to run RandomizedSearchCV (default: False)

**Returns:**
- Trained XGBClassifier model

**Default Configuration:**
- Learning rate: 0.1
- Max depth: 7
- Number of estimators: 200
- Subsample: 0.8
- Colsample bytree: 0.8
- Objective: binary:logistic
- Eval metric: logloss

**Tuning Search Space (when tune_hyperparameters=True):**
- Learning rate: [0.01, 0.05, 0.1, 0.2]
- Max depth: [3, 5, 7, 9]
- Number of estimators: [100, 200, 300]
- Subsample: [0.6, 0.8, 1.0]
- Colsample bytree: [0.6, 0.8, 1.0]
- Min child weight: [1, 3, 5]

**Example:**
```python
# Without tuning
model = trainer.train_xgboost(X_train, y_train)

# With hyperparameter tuning
tuned_model = trainer.train_xgboost(X_train, y_train, tune_hyperparameters=True)
```

##### `train_all_models(X_train, y_train, use_smote=True, tune_xgboost=False)`

Train all models in the pipeline.

**Parameters:**
- `X_train`: Training features
- `y_train`: Training labels
- `use_smote` (bool): Whether to apply SMOTE (default: True)
- `tune_xgboost` (bool): Whether to tune XGBoost hyperparameters (default: False)

**Returns:**
- `dict`: Dictionary of all trained models
  - Keys: 'Logistic Regression', 'Random Forest', 'XGBoost'
  - Values: Trained model objects

**Example:**
```python
models = trainer.train_all_models(
    X_train_eng, 
    y_train_prep,
    use_smote=True,
    tune_xgboost=False
)
```

##### `save_model(model, model_name, output_dir='models')`

Save a trained model to disk.

**Parameters:**
- `model`: Trained model object
- `model_name` (str): Name for the saved model
- `output_dir` (str): Directory to save models (default: 'models')

**Output:**
- Saves model as `{output_dir}/{model_name}.pkl`

##### `save_all_models(output_dir='models')`

Save all trained models.

**Parameters:**
- `output_dir` (str): Directory to save models (default: 'models')

**Returns:**
- `list`: Paths to saved models

##### `set_best_model(model_name)`

Set the best performing model.

**Parameters:**
- `model_name` (str): Name of the best model

##### `save_best_model(output_dir='models')`

Save the best model with a standard name for deployment.

**Parameters:**
- `output_dir` (str): Directory to save model (default: 'models')

**Returns:**
- `Path`: Path to saved model

**Output:**
- Saves model as `{output_dir}/best_model.pkl`

---

## Model Evaluation Module

### `ModelEvaluator` Class

**Module:** `src.evaluate`

Evaluates ML models and generates comprehensive reports.

#### Constructor

```python
ModelEvaluator()
```

**Attributes:**
- `evaluation_results` (dict): Dictionary storing results for all evaluated models

#### Methods

##### `evaluate_model(model, X_test, y_test, model_name='Model')`

Evaluate a single model on test data.

**Parameters:**
- `model`: Trained model
- `X_test`: Test features
- `y_test`: Test labels
- `model_name` (str): Name of the model (default: 'Model')

**Returns:**
- `dict`: Evaluation metrics
  - `accuracy`: Overall accuracy
  - `precision`: Precision score
  - `recall`: Recall score
  - `f1_score`: F1 score
  - `roc_auc`: ROC-AUC score
  - `confusion_matrix`: Confusion matrix (2D array)
  - `classification_report`: Detailed classification report (dict)

**Example:**
```python
from src.evaluate import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_model(model, X_test, y_test, 'XGBoost')
print(f"Accuracy: {results['accuracy']:.4f}")
```

##### `evaluate_all_models(models, X_test, y_test)`

Evaluate multiple models.

**Parameters:**
- `models` (dict): Dictionary of trained models
- `X_test`: Test features
- `y_test`: Test labels

**Returns:**
- `dict`: Evaluation results for all models

**Example:**
```python
results = evaluator.evaluate_all_models(models, X_test_eng, y_test_prep)
```

##### `get_best_model(metric='f1_score')`

Identify the best performing model based on a metric.

**Parameters:**
- `metric` (str): Metric to use for comparison (default: 'f1_score')
  - Options: 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'

**Returns:**
- `tuple`: (best_model_name, best_score)

**Example:**
```python
best_name, best_score = evaluator.get_best_model(metric='f1_score')
print(f"Best model: {best_name} with F1={best_score:.4f}")
```

##### `plot_confusion_matrices(save_path='logs/confusion_matrices.png')`

Plot confusion matrices for all evaluated models.

**Parameters:**
- `save_path` (str): Path to save the figure (default: 'logs/confusion_matrices.png')

**Output:**
- Saves a figure with confusion matrices for all models

##### `plot_metrics_comparison(save_path='logs/metrics_comparison.png')`

Plot comparison of metrics across all models.

**Parameters:**
- `save_path` (str): Path to save the figure (default: 'logs/metrics_comparison.png')

**Output:**
- Saves a bar chart comparing multiple metrics across models

##### `extract_feature_importance(model, feature_names, model_name='Model', top_n=15)`

Extract and rank feature importance.

**Parameters:**
- `model`: Trained model
- `feature_names` (list): List of feature names
- `model_name` (str): Name of the model (default: 'Model')
- `top_n` (int): Number of top features to return (default: 15)

**Returns:**
- `pd.DataFrame`: Feature importance dataframe with columns:
  - `feature`: Feature name
  - `importance`: Importance score

**Note:** Only works with tree-based models (Random Forest, XGBoost)

##### `plot_feature_importance(importance_df, model_name='Model', top_n=15, save_path='logs/feature_importance.png')`

Plot feature importance.

**Parameters:**
- `importance_df` (pd.DataFrame): Feature importance data
- `model_name` (str): Name of the model (default: 'Model')
- `top_n` (int): Number of top features to display (default: 15)
- `save_path` (str): Path to save the figure (default: 'logs/feature_importance.png')

**Output:**
- Saves a horizontal bar chart of feature importances

##### `generate_business_interpretation(importance_df, top_n=10)`

Generate business interpretation of top features.

**Parameters:**
- `importance_df` (pd.DataFrame): Feature importance data
- `top_n` (int): Number of features to interpret (default: 10)

**Returns:**
- `str`: Business interpretation text

**Output:**
- Logs business-friendly interpretation of important features

##### `save_evaluation_report(output_path='logs/evaluation_report.json')`

Save evaluation results to JSON file.

**Parameters:**
- `output_path` (str): Path to save report (default: 'logs/evaluation_report.json')

**Output:**
- Saves comprehensive evaluation results in JSON format

---

## Convenience Functions

### `load_data(data_path)`

**Module:** `src.data_loader`

Quick function to load data.

**Parameters:**
- `data_path` (str): Path to CSV file

**Returns:**
- `pd.DataFrame`: Loaded dataset

### `preprocess_data(X_train, X_test=None, y_train=None, y_test=None)`

**Module:** `src.preprocess`

Quick function for preprocessing.

**Parameters:**
- `X_train` (pd.DataFrame): Training features
- `X_test` (pd.DataFrame, optional): Test features
- `y_train` (pd.Series, optional): Training target
- `y_test` (pd.Series, optional): Test target

**Returns:**
- Preprocessed data and fitted preprocessor

### `engineer_features(X_train, X_test=None)`

**Module:** `src.feature_engineering`

Quick function for feature engineering.

**Parameters:**
- `X_train` (pd.DataFrame): Training features
- `X_test` (pd.DataFrame, optional): Test features

**Returns:**
- Engineered data and fitted engineer

### `load_model(model_path)`

**Module:** `src.train`

Load a saved model from disk.

**Parameters:**
- `model_path` (str): Path to saved model

**Returns:**
- Loaded model object

**Example:**
```python
from src.train import load_model

model = load_model('models/best_model.pkl')
predictions = model.predict(X_new)
```

---

## Complete Pipeline Example

```python
from src.data_loader import DataLoader
from src.preprocess import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator

# 1. Load data
loader = DataLoader('data/raw/Hotel_Reservations.csv')
data = loader.load_data()
X_train, X_test, y_train, y_test = loader.create_train_test_split()

# 2. Preprocess
preprocessor = DataPreprocessor()
X_train_prep, y_train_prep = preprocessor.fit_transform(X_train, y_train)
X_test_prep, y_test_prep = preprocessor.transform(X_test, y_test)

# 3. Feature engineering
engineer = FeatureEngineer()
X_train_eng = engineer.fit_transform(X_train_prep)
X_test_eng = engineer.transform(X_test_prep)

# 4. Train models
trainer = ModelTrainer()
models = trainer.train_all_models(X_train_eng, y_train_prep, use_smote=True)

# 5. Evaluate
evaluator = ModelEvaluator()
results = evaluator.evaluate_all_models(models, X_test_eng, y_test_prep)
best_model_name, best_score = evaluator.get_best_model(metric='f1_score')

# 6. Save best model
trainer.set_best_model(best_model_name)
model_path = trainer.save_best_model()
```
