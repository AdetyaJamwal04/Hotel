"""
Quick validation test for pipeline modules (without dependencies)
Tests the module imports and structure
"""

import sys
from pathlib import Path

print("="*70)
print("HOTEL BOOKING CANCELLATION PIPELINE - VALIDATION TEST")
print("="*70)

# Test project structure
print("\n1. Testing Project Structure...")
required_dirs = ['src', 'models', 'logs', 'notebooks', 'data/raw', '.github/workflows']
required_files = [
    'main.py',
    'requirements.txt',
    'README.md',
    'src/__init__.py',
    'src/data_loader.py',
    'src/preprocess.py',
    'src/feature_engineering.py',
    'src/train.py',
    'src/evaluate.py',
    '.github/workflows/ml-pipeline.yml',
    'notebooks/eda_analysis.ipynb'
]

all_exist = True
for dir_path in required_dirs:
    if Path(dir_path).exists():
        print(f"  [OK] {dir_path}")
    else:
        print(f"  [MISSING] {dir_path}")
        all_exist = False

for file_path in required_files:
    if Path(file_path).exists():
        print(f"  [OK] {file_path}")
    else:
        print(f"  [MISSING] {file_path}")
        all_exist = False

if all_exist:
    print("\n[OK] All required files and directories exist!")
else:
    print("\n[ERROR] Some files or directories are missing")


# Test data file
print("\n2. Testing Data File...")
data_file = Path('data/raw/Hotel_Reservations.csv')
if data_file.exists():
    size_mb = data_file.stat().st_size / (1024 * 1024)
    print(f"  [OK] Dataset found: {size_mb:.2f} MB")
else:
    print(f"  [ERROR] Dataset not found")

# Test module documentation
print("\n3. Testing Module Documentation...")
modules_to_check = [
    'src/data_loader.py',
    'src/preprocess.py',
    'src/feature_engineering.py',
    'src/train.py',
    'src/evaluate.py',
    'main.py'
]

for module in modules_to_check:
    with open(module, 'r') as f:
        content = f.read()
        has_docstring = '"""' in content
        has_logging = 'logging' in content
        has_class_or_func = 'class ' in content or 'def ' in content
        
        status = "[OK]" if all([has_docstring, has_logging, has_class_or_func]) else "[WARN]"
        print(f"  {status} {module} - Docstring: {has_docstring}, Logging: {has_logging}")

print("\n4. Testing Feature Engineering Implementation...")
with open('src/feature_engineering.py', 'r') as f:
    content = f.read()
    features = [
        'total_stay_nights',
        'total_guests',
        'price_per_guest',
        'lead_time_category',
        'is_weekend_booking',
        'has_special_requests',
        'peak_season'
    ]
    
    print(f"  Checking for engineered features...")
    for feat in features:
        status = "[OK]" if feat in content else "[MISSING]"
        print(f"    {status} {feat}")

print("\n5. Testing Model Implementations...")
with open('src/train.py', 'r') as f:
    content = f.read()
    models = ['LogisticRegression', 'RandomForestClassifier', 'XGBClassifier']
    techniques = ['SMOTE', 'RandomizedSearchCV']
    
    for model in models:
        status = "[OK]" if model in content else "[MISSING]"
        print(f"  {status} {model}")
    
    for tech in techniques:
        status = "[OK]" if tech in content else "[MISSING]"
        print(f"  {status} {tech}")

print("\n6. Testing Evaluation Metrics...")
with open('src/evaluate.py', 'r') as f:
    content = f.read()
    metrics = ['accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'roc_auc_score']
    
    for metric in metrics:
        status = "[OK]" if metric in content else "[MISSING]"
        print(f"  {status} {metric}")

print("\n7. Testing CI/CD Workflow...")
with open('.github/workflows/ml-pipeline.yml', 'r') as f:
    content = f.read()
    checks = ['pip install -r requirements.txt', 'python main.py', 'joblib', 'upload-artifact']
    
    for check in checks:
        status = "[OK]" if check in content else "[MISSING]"
        print(f"  {status} {check}")

print("\n" + "="*70)
print("VALIDATION COMPLETE")
print("="*70)
print("\nTo run the complete pipeline after installing dependencies:")
print("  1. pip install -r requirements.txt")
print("  2. python main.py")
print("\nFor hyperparameter tuning:")
print("  python main.py --tune-xgboost")
print("="*70)
