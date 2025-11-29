# Hotel Booking Cancellation Prediction 🏨

A complete end-to-end machine learning pipeline to predict hotel booking cancellations with production-ready code, CI/CD integration, and comprehensive documentation.

![Pipeline Status](https://img.shields.io/badge/pipeline-automated-brightgreen)
![Python](https://img.shields.io/badge/python-3.9-blue)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Deployment](#model-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Results](#results)

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[📖 Quick Start Guide](docs/QUICK_START.md)** - Get up and running in 5 minutes
- **[📋 API Reference](docs/API_REFERENCE.md)** - Complete API documentation for all modules
- **[👨‍💻 Developer Guide](docs/DEVELOPER_GUIDE.md)** - Development setup, architecture, and best practices
- **[🚀 Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Deploy to local, cloud, or production environments
- **[🤝 Contributing Guidelines](CONTRIBUTING.md)** - How to contribute to the project

### Quick Links

- **New to the project?** Start with the [Quick Start Guide](docs/QUICK_START.md)
- **Using the API?** Check the [API Reference](docs/API_REFERENCE.md)
- **Setting up for development?** Read the [Developer Guide](docs/DEVELOPER_GUIDE.md)
- **Ready to deploy?** Follow the [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- **Want to contribute?** See [Contributing Guidelines](CONTRIBUTING.md)

## 📊 Reports

Comprehensive analysis and results:

- **[📈 Exploratory Data Analysis](reports/EXPLORATORY_DATA_ANALYSIS.md)** - Complete EDA with insights and recommendations
- **[📋 Final Technical Report](reports/FINAL_TECHNICAL_REPORT.md)** - Model performance, results, and business impact

## 🎯 Project Overview

This project implements a complete machine learning workflow to predict whether a hotel booking will be canceled. The pipeline includes:

- Comprehensive EDA and data visualization
- Advanced feature engineering (10+ features)
- Multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Hyperparameter tuning with RandomizedSearchCV
- Class imbalance handling using SMOTE
- Outlier detection and treatment
- Model evaluation with multiple metrics
- Production-ready deployment code
- Automated CI/CD pipeline

**Dataset**: 36,275 hotel bookings with 19 features including guest information, booking details, and pricing.

**Target**: Predict `booking_status` (Canceled vs Not_Canceled)

## ✨ Features

### Data Pipeline
- ✅ Modular architecture with separate components
- ✅ Comprehensive data validation and error handling
- ✅ Stratified train-test splitting
- ✅ Advanced logging throughout pipeline

### Feature Engineering
Created 10 engineered features:
1. **total_stay_nights** - Combined weekend and weekday nights
2. **total_guests** - Total number of guests
3. **price_per_guest** - Average price per guest
4. **lead_time_category** - Categorized booking lead time
5. **is_weekend_booking** - Weekend stay indicator
6. **has_special_requests** - Special requests flag
7. **peak_season** - Peak season booking flag
8. **price_per_night** - Average price per night
9. **booking_to_stay_ratio** - Lead time to stay duration ratio
10. **is_loyal_customer** - Customer loyalty indicator

### Models
- **Logistic Regression** - Baseline linear model
- **Random Forest** - Ensemble decision tree model
- **XGBoost** - Gradient boosting with hyperparameter tuning

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrices
- Feature importance analysis
- Business interpretation of key predictors

## 📁 Project Structure

```
Hotel/
├── data/
│   ├── raw/                          # Original dataset
│   │   └── Hotel_Reservations.csv
│   └── processed/                    # Processed data (generated)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Data loading and splitting
│   ├── preprocess.py                 # Data cleaning and validation
│   ├── feature_engineering.py        # Feature creation and outlier handling
│   ├── train.py                      # Model training with SMOTE
│   └── evaluate.py                   # Model evaluation and reporting
├── models/                           # Saved models (generated)
│   └── best_model.pkl                # Best performing model
├── logs/                             # Execution logs (generated)
│   ├── pipeline_YYYYMMDD_HHMMSS.log
│   ├── evaluation_report.json
│   ├── confusion_matrices.png
│   ├── metrics_comparison.png
│   └── feature_importance_*.png
├── notebooks/
│   └── eda_analysis.ipynb            # Exploratory data analysis
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml           # CI/CD workflow
├── main.py                           # Pipeline orchestrator
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository** (or download the files)
   ```bash
   cd Hotel
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import pandas, sklearn, xgboost; print('All packages installed successfully!')"
   ```

## 💻 Usage

### Running the Complete Pipeline

**Basic usage** (recommended for first run):
```bash
python main.py
```

**With custom parameters**:
```bash
python main.py \
  --data-path data/raw/Hotel_Reservations.csv \
  --test-size 0.2 \
  --use-smote \
  --metric f1_score \
  --random-state 42
```

**With hyperparameter tuning** (slower, better performance):
```bash
python main.py --tune-xgboost
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | `data/raw/Hotel_Reservations.csv` | Path to dataset |
| `--test-size` | `0.2` | Test set proportion |
| `--use-smote` | `True` | Use SMOTE for class imbalance |
| `--tune-xgboost` | `False` | Run hyperparameter tuning |
| `--metric` | `f1_score` | Metric for best model selection |
| `--random-state` | `42` | Random seed |
| `--model-dir` | `models` | Model output directory |
| `--log-dir` | `logs` | Logs output directory |

### Running Individual Modules

**Data loading**:
```python
from src.data_loader import DataLoader

loader = DataLoader('data/raw/Hotel_Reservations.csv')
data = loader.load_data()
X_train, X_test, y_train, y_test = loader.create_train_test_split()
```

**Preprocessing**:
```python
from src.preprocess import DataPreprocessor

preprocessor = DataPreprocessor()
X_train_prep, y_train_prep = preprocessor.fit_transform(X_train, y_train)
X_test_prep, y_test_prep = preprocessor.transform(X_test, y_test)
```

**Feature engineering**:
```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
X_train_eng = engineer.fit_transform(X_train_prep)
X_test_eng = engineer.transform(X_test_prep)
```

**Model training**:
```python
from src.train import ModelTrainer

trainer = ModelTrainer()
models = trainer.train_all_models(X_train_eng, y_train_prep, use_smote=True)
```

## 🎯 Model Deployment

### Loading the Saved Model

```python
import joblib
import pandas as pd

# Load the best model
model = joblib.load('models/best_model.pkl')

# Load preprocessing and feature engineering pipelines
# (You would need to save these separately or include in the model pipeline)

# Make predictions
predictions = model.predict(X_new)
prediction_probabilities = model.predict_proba(X_new)
```

### Preparing New Data for Prediction

```python
from src.data_loader import DataLoader
from src.preprocess import DataPreprocessor
from src.feature_engineering import FeatureEngineer

# Load and prepare new booking data
new_data = pd.read_csv('new_bookings.csv')

# Apply same preprocessing steps
preprocessor = DataPreprocessor()  # Load fitted preprocessor
X_preprocessed = preprocessor.transform(new_data)

# Apply feature engineering
engineer = FeatureEngineer()  # Load fitted engineer
X_engineered = engineer.transform(X_preprocessed)

# Make predictions
predictions = model.predict(X_engineered)
probabilities = model.predict_proba(X_engineered)[:, 1]

# Interpret results
new_data['cancellation_probability'] = probabilities
new_data['predicted_status'] = ['Canceled' if p > 0.5 else 'Not_Canceled' 
                                  for p in probabilities]
```

### API Integration Example

```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and pipelines at startup
model = joblib.load('models/best_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')
engineer = joblib.load('models/engineer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get booking data from request
    booking_data = request.json
    
    # Convert to DataFrame
    df = pd.DataFrame([booking_data])
    
    # Preprocess and predict
    X_prep = preprocessor.transform(df)
    X_eng = engineer.transform(X_prep)
    probability = model.predict_proba(X_eng)[0, 1]
    
    return jsonify({
        'cancellation_probability': float(probability),
        'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'
    })

if __name__ == '__main__':
    app.run(debug=True)
```

## 🔄 CI/CD Pipeline

The project includes a GitHub Actions workflow (`.github/workflows/ml-pipeline.yml`) that automatically:

1. **Installs dependencies** from `requirements.txt`
2. **Validates dataset** schema and availability
3. **Runs the complete pipeline** end-to-end
4. **Verifies model** is saved and loadable
5. **Uploads artifacts** (models, logs, reports)
6. **Displays performance summary**

### Triggering the Workflow

The workflow runs automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main`
- Manual trigger via GitHub Actions UI

### Viewing Results

After workflow completion:
1. Go to GitHub Actions tab
2. Click on the latest workflow run
3. Download artifacts: `trained-models` and `pipeline-logs`

## 📊 Results

### Expected Performance

Based on model training (with SMOTE):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.75 | ~0.70 | ~0.65 | ~0.67 | ~0.80 |
| Random Forest | ~0.82 | ~0.78 | ~0.72 | ~0.75 | ~0.86 |
| **XGBoost** | **~0.85** | **~0.82** | **~0.78** | **~0.80** | **~0.89** |

*Note: Exact results may vary based on random state and hyperparameter tuning*

### Top Predictive Features

1. **lead_time** - Booking advance time
2. **avg_price_per_room** - Room pricing
3. **no_of_special_requests** - Customer commitment indicator
4. **market_segment_type** - Acquisition channel
5. **no_of_previous_cancellations** - Historical behavior

### Business Insights

- **Lead time matters**: Bookings made far in advance have different cancellation patterns
- **Special requests reduce risk**: Customers with special requests are less likely to cancel
- **Price sensitivity**: Higher room prices correlate with increased cancellation probability
- **Loyalty pays off**: Repeat customers show lower cancellation rates
- **Peak season dynamics**: Seasonal patterns affect cancellation behavior

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Built with ❤️ for production-ready ML deployments**
