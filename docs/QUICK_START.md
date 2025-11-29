# Quick Start Guide

Get up and running with the Hotel Booking Cancellation Prediction pipeline in minutes!

## Installation (5 minutes)

### 1. Prerequisites

Ensure you have Python 3.9+ installed:

```bash
python --version
```

### 2. Clone and Setup

```bash
# Navigate to the project directory
cd Hotel

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import pandas, sklearn, xgboost; print('✓ All packages installed successfully!')"
```

---

## Running Your First Prediction (2 minutes)

### Option 1: Run the Complete Pipeline

```bash
python main.py
```

This will:
- ✅ Load the hotel booking dataset
- ✅ Preprocess the data
- ✅ Engineer features
- ✅ Train 3 models (Logistic Regression, Random Forest, XGBoost)
- ✅ Evaluate and compare models
- ✅ Save the best model to `models/best_model.pkl`

**Expected Output:**
```
======================================================================
HOTEL BOOKING CANCELLATION PREDICTION PIPELINE
======================================================================
...
✓ Data loading complete
✓ Preprocessing complete
✓ Feature engineering complete
✓ Model training complete
✓ Model evaluation complete
======================================================================
Best Model: XGBoost
Best Score: 0.8012
======================================================================
```

**Duration:** ~5-10 minutes (without hyperparameter tuning)

### Option 2: Quick Prediction with Saved Model

```bash
python -c "
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('models/best_model.pkl')
print('✓ Model loaded successfully!')
print(f'Model type: {type(model).__name__}')
"
```

---

## Making Predictions (3 minutes)

### Single Booking Prediction

Create a file `predict_example.py`:

```python
import joblib
import pandas as pd
from src.data_loader import DataLoader
from src.preprocess import DataPreprocessor
from src.feature_engineering import FeatureEngineer

# Load pipeline components
preprocessor = joblib.load('models/preprocessor.pkl')
engineer = joblib.load('models/feature_engineer.pkl')
model = joblib.load('models/best_model.pkl')

# Example booking
booking = {
    'no_of_adults': 2,
    'no_of_children': 1,
    'no_of_weekend_nights': 1,
    'no_of_week_nights': 3,
    'type_of_meal_plan': 'Meal Plan 1',
    'required_car_parking_space': 0,
    'room_type_reserved': 'Room_Type 1',
    'lead_time': 150,
    'arrival_year': 2018,
    'arrival_month': 7,
    'arrival_date': 15,
    'market_segment_type': 'Online',
    'repeated_guest': 0,
    'no_of_previous_cancellations': 0,
    'no_of_previous_bookings_not_canceled': 0,
    'avg_price_per_room': 85.50,
    'no_of_special_requests': 2
}

# Convert to DataFrame
df = pd.DataFrame([booking])

# Preprocess and engineer features
X_prep = preprocessor.transform(df)
X_eng = engineer.transform(X_prep)

# Predict
prediction = model.predict(X_eng)[0]
probability = model.predict_proba(X_eng)[0, 1]

# Display results
print(f"\n{'='*50}")
print("BOOKING CANCELLATION PREDICTION")
print(f"{'='*50}")
print(f"Prediction: {'CANCELED' if prediction == 1 else 'NOT CANCELED'}")
print(f"Cancellation Probability: {probability:.2%}")
print(f"Risk Level: {'HIGH' if probability > 0.7 else 'MEDIUM' if probability > 0.4 else 'LOW'}")
print(f"{'='*50}\n")
```

Run it:

```bash
python predict_example.py
```

**Expected Output:**
```
==================================================
BOOKING CANCELLATION PREDICTION
==================================================
Prediction: NOT CANCELED
Cancellation Probability: 32.45%
Risk Level: LOW
==================================================
```

---

## Understanding the Output (2 minutes)

### Files Generated

After running the pipeline, you'll find:

```
Hotel/
├── models/
│   ├── best_model.pkl              # Best performing model (XGBoost)
│   ├── Logistic_Regression.pkl     # Logistic Regression model
│   ├── Random_Forest.pkl           # Random Forest model
│   └── XGBoost.pkl                 # XGBoost model
└── logs/
    ├── pipeline_YYYYMMDD_HHMMSS.log    # Detailed execution log
    ├── evaluation_report.json          # Model metrics in JSON
    ├── confusion_matrices.png          # Confusion matrices plot
    ├── metrics_comparison.png          # Metrics comparison chart
    └── feature_importance_XGBoost.png  # Feature importance plot
```

### Key Metrics

Open `logs/evaluation_report.json` to see:

```json
{
  "XGBoost": {
    "accuracy": 0.8523,
    "precision": 0.8234,
    "recall": 0.7856,
    "f1_score": 0.8012,
    "roc_auc": 0.8945
  },
  ...
}
```

**What these mean:**
- **Accuracy:** Overall correctness (85.23%)
- **Precision:** Of all predicted cancellations, how many were actually canceled (82.34%)
- **Recall:** Of all actual cancellations, how many did we catch (78.56%)
- **F1-Score:** Balance between precision and recall (80.12%)
- **ROC-AUC:** Overall model quality (89.45%)

---

## Common Use Cases

### Use Case 1: Batch Predictions

```python
import pandas as pd
import joblib

# Load components
preprocessor = joblib.load('models/preprocessor.pkl')
engineer = joblib.load('models/feature_engineer.pkl')
model = joblib.load('models/best_model.pkl')

# Load new bookings
new_bookings = pd.read_csv('new_bookings.csv')

# Process and predict
X_prep = preprocessor.transform(new_bookings.drop(['Booking_ID'], axis=1))
X_eng = engineer.transform(X_prep)
predictions = model.predict(X_eng)
probabilities = model.predict_proba(X_eng)[:, 1]

# Add to results
new_bookings['predicted_cancellation'] = predictions
new_bookings['cancellation_probability'] = probabilities

# Save
new_bookings.to_csv('predictions_output.csv', index=False)
print(f"✓ Predictions saved for {len(new_bookings)} bookings")
```

### Use Case 2: Model Comparison

```python
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator

# Train models
trainer = ModelTrainer()
models = trainer.train_all_models(X_train, y_train, use_smote=True)

# Evaluate all
evaluator = ModelEvaluator()
results = evaluator.evaluate_all_models(models, X_test, y_test)

# Find best
best_name, best_score = evaluator.get_best_model(metric='f1_score')
print(f"Best model: {best_name} (F1: {best_score:.4f})")

# Visualize
evaluator.plot_metrics_comparison(save_path='comparison.png')
```

### Use Case 3: Feature Importance Analysis

```python
from src.evaluate import ModelEvaluator

evaluator = ModelEvaluator()

# Get feature importance
importance_df = evaluator.extract_feature_importance(
    model,
    feature_names=X_train.columns.tolist(),
    model_name='XGBoost',
    top_n=10
)

# Visualize
evaluator.plot_feature_importance(
    importance_df,
    model_name='XGBoost',
    save_path='top_features.png'
)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10))
```

---

## Customizing the Pipeline

### Change Test Size

```bash
python main.py --test-size 0.3
```

### Run with Hyperparameter Tuning

```bash
python main.py --tune-xgboost
```

**Note:** This will take longer (~20-30 minutes) but may improve performance.

### Use Different Metric for Best Model Selection

```bash
python main.py --metric accuracy
# Options: accuracy, precision, recall, f1_score, roc_auc
```

### Run Without SMOTE

```bash
python main.py --use-smote False
```

---

## Troubleshooting

### Problem: "Module not found" error

**Solution:**
```bash
# Ensure virtual environment is activated
# Reinstall dependencies
pip install -r requirements.txt
```

### Problem: "Data file not found"

**Solution:**
```bash
# Verify data file exists
ls data/raw/Hotel_Reservations.csv

# If missing, ensure it's in the correct location
```

### Problem: Pipeline runs slowly

**Solutions:**
1. **Skip hyperparameter tuning:**
   ```bash
   python main.py
   ```

2. **Disable SMOTE:**
   ```bash
   python main.py --use-smote False
   ```

3. **Use smaller test size:**
   ```bash
   python main.py --test-size 0.3
   ```

### Problem: Out of memory

**Solution:**
```python
# Modify main.py to use a subset of data during development
# In data_loader.py, after loading:
data = data.sample(n=10000, random_state=42)
```

---

## Next Steps

### 1. Explore the Code

- **Data Loading:** `src/data_loader.py`
- **Preprocessing:** `src/preprocess.py`
- **Feature Engineering:** `src/feature_engineering.py`
- **Model Training:** `src/train.py`
- **Evaluation:** `src/evaluate.py`

### 2. Read the Documentation

- 📖 **[API Reference](docs/API_REFERENCE.md)** - Detailed API documentation
- 👨‍💻 **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Development setup and best practices
- 🚀 **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** - Deploy to production
- 🤝 **[Contributing](CONTRIBUTING.md)** - Contribute to the project

### 3. Experiment

Try:
- Adding new features to feature engineering
- Training with different models
- Tuning hyperparameters
- Creating visualizations
- Building a REST API

### 4. Deploy

Options:
- **Local:** Batch prediction script
- **Flask API:** Simple REST API
- **FastAPI:** Production-ready API
- **Docker:** Containerized deployment
- **Cloud:** AWS Lambda, Google Cloud Functions, Azure Functions

See **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)** for details.

---

## Advanced Usage

### Jupyter Notebook Analysis

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks/eda_analysis.ipynb
```

### Custom Model Training

```python
from src.train import ModelTrainer

trainer = ModelTrainer(random_state=42)

# Train only specific model
xgb_model = trainer.train_xgboost(
    X_train, 
    y_train,
    tune_hyperparameters=True  # Enable tuning
)

# Save custom model
trainer.save_model(xgb_model, 'custom_xgboost', output_dir='models')
```

### CI/CD Pipeline

The project includes automated testing via GitHub Actions.

**Local testing:**
```bash
# Validate project structure
python validate_structure.py

# Run full pipeline
python main.py
```

---

## Getting Help

- 📚 **Documentation:** Check the `docs/` folder
- 🐛 **Issues:** Open an issue on GitHub
- 💬 **Questions:** Tag with `question` label

---

## Quick Reference

### Essential Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py

# Run with tuning (slower, better)
python main.py --tune-xgboost

# Load and use model
python -c "import joblib; model = joblib.load('models/best_model.pkl'); print('Model loaded!')"

# Start Jupyter
jupyter notebook
```

### Key Files

- `main.py` - Pipeline orchestrator
- `requirements.txt` - Dependencies
- `models/best_model.pkl` - Trained model
- `logs/evaluation_report.json` - Performance metrics
- `README.md` - Project overview

### Important Directories

- `src/` - Source code modules
- `data/raw/` - Original dataset
- `models/` - Saved models
- `logs/` - Logs and reports
- `docs/` - Documentation

---

## Success!

You're now ready to:
✅ Run the ML pipeline
✅ Make predictions
✅ Understand the results
✅ Explore the code
✅ Deploy to production

**Happy predicting! 🎉**
