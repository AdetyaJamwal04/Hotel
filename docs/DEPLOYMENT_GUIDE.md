# Deployment Guide

## Table of Contents

- [Deployment Overview](#deployment-overview)
- [Local Deployment](#local-deployment)
- [Cloud Deployment Options](#cloud-deployment-options)
- [REST API Deployment](#rest-api-deployment)
- [Docker Deployment](#docker-deployment)
- [Production Considerations](#production-considerations)
- [Monitoring and Maintenance](#monitoring-and-maintenance)

---

## Deployment Overview

This guide covers different deployment strategies for the Hotel Booking Cancellation Prediction model, from local deployment to production-ready cloud solutions.

### Deployment Checklist

Before deploying, ensure you have:

- [ ] Trained model saved as `.pkl` file
- [ ] All preprocessing and feature engineering pipelines saved
- [ ] Dependencies documented in `requirements.txt`
- [ ] Model performance validated on test set
- [ ] API or inference code tested
- [ ] Environment variables configured
- [ ] Logging and monitoring set up

---

## Local Deployment

### Option 1: Python Script

**Use Case:** Batch predictions on new data

**Setup:**

1. **Save all pipeline components:**

```python
import joblib
from src.data_loader import DataLoader
from src.preprocess import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer

# After training
joblib.dump(preprocessor, 'models/preprocessor.pkl')
joblib.dump(engineer, 'models/feature_engineer.pkl')
joblib.dump(best_model, 'models/best_model.pkl')
```

2. **Create prediction script** (`predict.py`):

```python
import pandas as pd
import joblib

def load_pipeline_components():
    """Load all saved pipeline components."""
    preprocessor = joblib.load('models/preprocessor.pkl')
    engineer = joblib.load('models/feature_engineer.pkl')
    model = joblib.load('models/best_model.pkl')
    return preprocessor, engineer, model

def predict_cancellation(booking_data):
    """
    Predict cancellation probability for new bookings.
    
    Args:
        booking_data (pd.DataFrame): DataFrame with booking information
        
    Returns:
        pd.DataFrame: Original data with predictions
    """
    # Load components
    preprocessor, engineer, model = load_pipeline_components()
    
    # Preprocess
    X_prep = preprocessor.transform(booking_data.drop(['booking_status', 'Booking_ID'], 
                                                       axis=1, errors='ignore'))
    
    # Engineer features
    X_eng = engineer.transform(X_prep)
    
    # Predict
    predictions = model.predict(X_eng)
    probabilities = model.predict_proba(X_eng)[:, 1]
    
    # Add results to original data
    booking_data['predicted_cancellation'] = predictions
    booking_data['cancellation_probability'] = probabilities
    booking_data['risk_level'] = pd.cut(
        probabilities,
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    return booking_data

if __name__ == "__main__":
    # Example usage
    new_bookings = pd.read_csv('new_bookings.csv')
    results = predict_cancellation(new_bookings)
    results.to_csv('predictions.csv', index=False)
    print(f"Predictions saved for {len(results)} bookings")
```

3. **Run predictions:**

```bash
python predict.py
```

### Option 2: Jupyter Notebook

**Use Case:** Interactive exploration and predictions

Create `notebooks/make_predictions.ipynb`:

```python
import pandas as pd
import joblib

# Load components
preprocessor = joblib.load('../models/preprocessor.pkl')
engineer = joblib.load('../models/feature_engineer.pkl')
model = joblib.load('../models/best_model.pkl')

# Load new data
new_data = pd.read_csv('../data/new_bookings.csv')

# Prepare features
X_prep = preprocessor.transform(new_data.drop(['booking_status', 'Booking_ID'], 
                                               axis=1, errors='ignore'))
X_eng = engineer.transform(X_prep)

# Predict
predictions = model.predict(X_eng)
probabilities = model.predict_proba(X_eng)[:, 1]

# Display results
results_df = pd.DataFrame({
    'Booking_ID': new_data['Booking_ID'],
    'Prediction': predictions,
    'Probability': probabilities
})
results_df.head()
```

---

## Cloud Deployment Options

### AWS Deployment

#### Option 1: AWS Lambda + API Gateway

**Use Case:** Serverless, pay-per-request API

**Steps:**

1. **Create Lambda deployment package:**

```bash
# Create deployment directory
mkdir lambda_deployment
cd lambda_deployment

# Install dependencies
pip install -r requirements.txt -t .

# Copy model files
cp -r ../models .
cp -r ../src .

# Create lambda handler
cat > lambda_function.py << 'EOF'
import json
import joblib
import pandas as pd

# Load models at initialization (outside handler for reuse)
preprocessor = joblib.load('models/preprocessor.pkl')
engineer = joblib.load('models/feature_engineer.pkl')
model = joblib.load('models/best_model.pkl')

def lambda_handler(event, context):
    try:
        # Parse input
        booking_data = json.loads(event['body'])
        df = pd.DataFrame([booking_data])
        
        # Preprocess and predict
        X_prep = preprocessor.transform(df)
        X_eng = engineer.transform(X_prep)
        probability = float(model.predict_proba(X_eng)[0, 1])
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'cancellation_probability': probability,
                'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
EOF

# Create zip package
zip -r deployment.zip .
```

2. **Deploy to AWS Lambda:**
   - Create Lambda function in AWS Console
   - Upload `deployment.zip`
   - Set handler to `lambda_function.lambda_handler`
   - Configure memory (at least 512 MB)
   - Set timeout (30 seconds recommended)

3. **Create API Gateway:**
   - Create REST API
   - Add POST method
   - Integrate with Lambda function
   - Deploy to stage

#### Option 2: AWS SageMaker

**Use Case:** Managed ML deployment with scaling

```python
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# Create SageMaker model
sklearn_model = SKLearnModel(
    model_data='s3://your-bucket/model.tar.gz',
    role='arn:aws:iam::account:role/SageMakerRole',
    entry_point='inference.py',
    framework_version='1.0-1'
)

# Deploy endpoint
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)

# Make predictions
result = predictor.predict(new_booking_data)
```

### Google Cloud Platform (GCP)

#### Cloud Functions Deployment

1. **Create `main.py`:**

```python
import joblib
import pandas as pd
from flask import jsonify

# Load models
preprocessor = joblib.load('models/preprocessor.pkl')
engineer = joblib.load('models/feature_engineer.pkl')
model = joblib.load('models/best_model.pkl')

def predict_cancellation(request):
    """HTTP Cloud Function."""
    request_json = request.get_json()
    
    if request_json is None:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        df = pd.DataFrame([request_json])
        X_prep = preprocessor.transform(df)
        X_eng = engineer.transform(X_prep)
        probability = float(model.predict_proba(X_eng)[0, 1])
        
        return jsonify({
            'cancellation_probability': probability,
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

2. **Deploy:**

```bash
gcloud functions deploy predict-cancellation \
  --runtime python39 \
  --trigger-http \
  --allow-unauthenticated \
  --entry-point predict_cancellation
```

### Azure Deployment

#### Azure Functions

Similar to AWS Lambda, create a function app and deploy the model as an HTTP-triggered function.

---

## REST API Deployment

### Flask API

**Use Case:** Simple, lightweight API

**Setup:**

1. **Create `api.py`:**

```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models at startup
logger.info("Loading models...")
preprocessor = joblib.load('models/preprocessor.pkl')
engineer = joblib.load('models/feature_engineer.pkl')
model = joblib.load('models/best_model.pkl')
logger.info("Models loaded successfully")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict cancellation probability for a booking.
    
    Expected JSON body:
    {
        "no_of_adults": 2,
        "no_of_children": 0,
        "no_of_weekend_nights": 1,
        ...
    }
    """
    try:
        # Validate request
        if not request.json:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        booking_data = pd.DataFrame([request.json])
        
        # Validate required fields
        required_fields = [
            'no_of_adults', 'no_of_children', 'no_of_weekend_nights',
            'no_of_week_nights', 'type_of_meal_plan', 'required_car_parking_space',
            'room_type_reserved', 'lead_time', 'arrival_year', 'arrival_month',
            'arrival_date', 'market_segment_type', 'repeated_guest',
            'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
            'avg_price_per_room', 'no_of_special_requests'
        ]
        
        missing_fields = set(required_fields) - set(booking_data.columns)
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {list(missing_fields)}'
            }), 400
        
        # Preprocess
        X_prep = preprocessor.transform(booking_data)
        
        # Engineer features
        X_eng = engineer.transform(X_prep)
        
        # Predict
        prediction = int(model.predict(X_eng)[0])
        probability = float(model.predict_proba(X_eng)[0, 1])
        
        # Determine risk level
        if probability > 0.7:
            risk_level = 'High'
        elif probability > 0.4:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        response = {
            'prediction': 'Canceled' if prediction == 1 else 'Not Canceled',
            'cancellation_probability': round(probability, 4),
            'risk_level': risk_level,
            'confidence': round(max(probability, 1 - probability), 4)
        }
        
        logger.info(f"Prediction made: {response}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Predict for multiple bookings."""
    try:
        if not request.json or 'bookings' not in request.json:
            return jsonify({'error': 'No bookings data provided'}), 400
        
        bookings = pd.DataFrame(request.json['bookings'])
        
        # Preprocess and predict
        X_prep = preprocessor.transform(bookings)
        X_eng = engineer.transform(X_prep)
        
        predictions = model.predict(X_eng)
        probabilities = model.predict_proba(X_eng)[:, 1]
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'booking_index': i,
                'prediction': 'Canceled' if pred == 1 else 'Not Canceled',
                'cancellation_probability': round(float(prob), 4),
                'risk_level': 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low'
            })
        
        return jsonify({'predictions': results}), 200
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

2. **Run locally:**

```bash
python api.py
```

3. **Test the API:**

```bash
# Health check
curl http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "no_of_adults": 2,
    "no_of_children": 0,
    "no_of_weekend_nights": 1,
    "no_of_week_nights": 2,
    "type_of_meal_plan": "Meal Plan 1",
    "required_car_parking_space": 0,
    "room_type_reserved": "Room_Type 1",
    "lead_time": 224,
    "arrival_year": 2018,
    "arrival_month": 10,
    "arrival_date": 2,
    "market_segment_type": "Online",
    "repeated_guest": 0,
    "no_of_previous_cancellations": 0,
    "no_of_previous_bookings_not_canceled": 0,
    "avg_price_per_room": 65.0,
    "no_of_special_requests": 0
  }'
```

### FastAPI (Recommended for Production)

**Use Case:** High-performance API with automatic documentation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List

app = FastAPI(title="Hotel Cancellation Prediction API")

# Load models
preprocessor = joblib.load('models/preprocessor.pkl')
engineer = joblib.load('models/feature_engineer.pkl')
model = joblib.load('models/best_model.pkl')

class Booking(BaseModel):
    no_of_adults: int
    no_of_children: int
    no_of_weekend_nights: int
    no_of_week_nights: int
    type_of_meal_plan: str
    required_car_parking_space: int
    room_type_reserved: str
    lead_time: int
    arrival_year: int
    arrival_month: int
    arrival_date: int
    market_segment_type: str
    repeated_guest: int
    no_of_previous_cancellations: int
    no_of_previous_bookings_not_canceled: int
    avg_price_per_room: float
    no_of_special_requests: int

class PredictionResponse(BaseModel):
    prediction: str
    cancellation_probability: float
    risk_level: str
    confidence: float

@app.get("/")
def read_root():
    return {"message": "Hotel Cancellation Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(booking: Booking):
    try:
        df = pd.DataFrame([booking.dict()])
        X_prep = preprocessor.transform(df)
        X_eng = engineer.transform(X_prep)
        
        prediction = int(model.predict(X_eng)[0])
        probability = float(model.predict_proba(X_eng)[0, 1])
        
        return PredictionResponse(
            prediction='Canceled' if prediction == 1 else 'Not Canceled',
            cancellation_probability=round(probability, 4),
            risk_level='High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low',
            confidence=round(max(probability, 1 - probability), 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn api:app --reload
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY models/ models/
COPY src/ src/
COPY api.py .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "api.py"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
```

### Build and Run

```bash
# Build image
docker build -t hotel-cancellation-api .

# Run container
docker run -p 5000:5000 hotel-cancellation-api

# Using Docker Compose
docker-compose up -d
```

---

## Production Considerations

### 1. Model Versioning

```python
# Save with version
import datetime

version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
joblib.dump(model, f'models/model_v{version}.pkl')

# Keep version registry
with open('models/model_versions.json', 'w') as f:
    json.dump({
        'current_version': version,
        'metrics': {'f1_score': 0.85, 'accuracy': 0.87}
    }, f)
```

### 2. Input Validation

```python
def validate_booking_data(data):
    """Validate booking data before prediction."""
    errors = []
    
    if data['no_of_adults'] < 0:
        errors.append('no_of_adults must be non-negative')
    
    if data['lead_time'] < 0:
        errors.append('lead_time must be non-negative')
    
    valid_meal_plans = ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']
    if data['type_of_meal_plan'] not in valid_meal_plans:
        errors.append(f'Invalid meal plan. Must be one of {valid_meal_plans}')
    
    return errors
```

### 3. Caching

```python
from functools import lru_cache
import hashlib
import json

@lru_cache(maxsize=1000)
def cached_prediction(booking_json):
    """Cache predictions for identical inputs."""
    booking_data = json.loads(booking_json)
    # ... prediction logic
    return result

# Usage
booking_hash = json.dumps(booking_dict, sort_keys=True)
result = cached_prediction(booking_hash)
```

### 4. Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # ... prediction logic
    pass
```

### 5. Security

- Use HTTPS in production
- Implement authentication (API keys, JWT tokens)
- Sanitize inputs
- Don't expose stack traces to clients
- Use environment variables for secrets

```python
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')

@app.before_request
def check_api_key():
    if request.headers.get('X-API-Key') != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401
```

---

## Monitoring and Maintenance

### Logging

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler('logs/api.log', maxBytes=10000000, backupCount=5)
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)

app.logger.addHandler(handler)

# Log predictions
app.logger.info(f"Prediction: {prediction}, Probability: {probability}")
```

### Performance Monitoring

Track key metrics:
- Response time
- Request rate
- Error rate
- Prediction distribution

```python
import time
from prometheus_client import Counter, Histogram

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions made')
prediction_time = Histogram('prediction_duration_seconds', 'Prediction duration')

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    # ... prediction logic
    
    prediction_counter.inc()
    prediction_time.observe(time.time() - start_time)
    
    return result
```

### Model Retraining

Schedule periodic retraining:

```python
# retrain_schedule.py
import schedule
import time

def retrain_model():
    """Retrain model with new data."""
    print("Retraining model...")
    # Load new data
    # Retrain pipeline
    # Evaluate performance
    # Deploy if improved
    
schedule.every().week.do(retrain_model)

while True:
    schedule.run_pending()
    time.sleep(1)
```

---

## Rollback Strategy

### Version Control

Maintain multiple model versions:

```
models/
├── current -> v20231128_143022/
├── v20231128_143022/
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── feature_engineer.pkl
└── v20231120_091530/
    ├── best_model.pkl
    ├── preprocessor.pkl
    └── feature_engineer.pkl
```

### Quick Rollback

```bash
# Rollback to previous version
cd models
rm current
ln -s v20231120_091530 current

# Restart service
docker-compose restart api
```

---

## Summary

This deployment guide covers local, cloud, and production deployments. Key takeaways:

1. **Start simple:** Begin with local batch predictions
2. **Build an API:** Use Flask or FastAPI for serving predictions
3. **Containerize:** Use Docker for consistency
4. **Monitor:** Track performance and errors
5. **Secure:** Implement authentication and validation
6. **Maintain:** Plan for updates and rollbacks
