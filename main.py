"""
Main Pipeline Orchestrator
Runs the complete hotel booking cancellation prediction pipeline.
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import DataLoader
from src.preprocess import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator


def setup_logging(log_dir='logs'):
    """
    Set up logging configuration.
    
    Args:
        log_dir (str): Directory for log files
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*70)
    logger.info("HOTEL BOOKING CANCELLATION PREDICTION PIPELINE")
    logger.info("="*70)
    logger.info(f"Log file: {log_file}")
    
    return logger


def main(args):
    """
    Main pipeline execution function.
    
    Args:
        args: Command line arguments
    """
    logger = setup_logging(args.log_dir)
    
    try:
        # ============================================================
        # STAGE 1: DATA LOADING
        # ============================================================
        logger.info("\n" + "="*70)
        logger.info("STAGE 1: DATA LOADING")
        logger.info("="*70)
        
        data_loader = DataLoader(args.data_path)
        data = data_loader.load_data()
        
        X_train, X_test, y_train, y_test = data_loader.create_train_test_split(
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        logger.info(f"✓ Data loading complete")
        logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        
        # ============================================================
        # STAGE 2: DATA PREPROCESSING
        # ============================================================
        logger.info("\n" + "="*70)
        logger.info("STAGE 2: DATA PREPROCESSING")
        logger.info("="*70)
        
        preprocessor = DataPreprocessor()
        X_train_prep, y_train_prep = preprocessor.fit_transform(X_train, y_train)
        X_test_prep, y_test_prep = preprocessor.transform(X_test, y_test)
        
        logger.info(f"✓ Preprocessing complete")
        
        # ============================================================
        # STAGE 3: FEATURE ENGINEERING
        # ============================================================
        logger.info("\n" + "="*70)
        logger.info("STAGE 3: FEATURE ENGINEERING")
        logger.info("="*70)
        
        engineer = FeatureEngineer()
        X_train_eng = engineer.fit_transform(X_train_prep)
        X_test_eng = engineer.transform(X_test_prep)
        
        logger.info(f"✓ Feature engineering complete")
        logger.info(f"  Features: {X_train_eng.shape[1]}")
        
        # ============================================================
        # STAGE 4: MODEL TRAINING
        # ============================================================
        logger.info("\n" + "="*70)
        logger.info("STAGE 4: MODEL TRAINING")
        logger.info("="*70)
        
        trainer = ModelTrainer(random_state=args.random_state)
        models = trainer.train_all_models(
            X_train_eng, 
            y_train_prep,
            use_smote=args.use_smote,
            tune_xgboost=args.tune_xgboost
        )
        
        logger.info(f"✓ Model training complete")
        logger.info(f"  Models trained: {len(models)}")
        
        # ============================================================
        # STAGE 5: MODEL EVALUATION
        # ============================================================
        logger.info("\n" + "="*70)
        logger.info("STAGE 5: MODEL EVALUATION")
        logger.info("="*70)
        
        evaluator = ModelEvaluator()
        results = evaluator.evaluate_all_models(models, X_test_eng, y_test_prep)
        
        # Generate visualizations
        evaluator.plot_confusion_matrices(
            save_path=f'{args.log_dir}/confusion_matrices.png'
        )
        evaluator.plot_metrics_comparison(
            save_path=f'{args.log_dir}/metrics_comparison.png'
        )
        
        # Get best model
        best_model_name, best_score = evaluator.get_best_model(metric=args.metric)
        
        # Extract and visualize feature importance for best model
        best_model = models[best_model_name]
        importance_df = evaluator.extract_feature_importance(
            best_model,
            X_train_eng.columns.tolist(),
            model_name=best_model_name,
            top_n=15
        )
        
        if importance_df is not None:
            evaluator.plot_feature_importance(
                importance_df,
                model_name=best_model_name,
                top_n=15,
                save_path=f'{args.log_dir}/feature_importance_{best_model_name.replace(" ", "_")}.png'
            )
            
            # Generate business interpretation
            evaluator.generate_business_interpretation(importance_df, top_n=10)
        
        # Save evaluation report
        evaluator.save_evaluation_report(
            output_path=f'{args.log_dir}/evaluation_report.json'
        )
        
        logger.info(f"✓ Model evaluation complete")
        
        # ============================================================
        # STAGE 6: MODEL SAVING
        # ============================================================
        logger.info("\n" + "="*70)
        logger.info("STAGE 6: MODEL SAVING")
        logger.info("="*70)
        
        # Save all models
        trainer.save_all_models(output_dir=args.model_dir)
        
        # Save best model
        trainer.set_best_model(best_model_name)
        best_model_path = trainer.save_best_model(output_dir=args.model_dir)
        
        logger.info(f"✓ Model saving complete")
        logger.info(f"  Best model saved: {best_model_path}")
        
        # ============================================================
        # PIPELINE SUMMARY
        # ============================================================
        logger.info("\n" + "="*70)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*70)
        
        logger.info(f"✓ Data loaded: {len(data)} records")
        logger.info(f"✓ Features engineered: {X_train_eng.shape[1]}")
        logger.info(f"✓ Models trained: {len(models)}")
        logger.info(f"✓ Best model: {best_model_name}")
        logger.info(f"✓ Best {args.metric}: {best_score:.4f}")
        logger.info(f"✓ Model saved: {best_model_path}")
        
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        
        return {
            'best_model_name': best_model_name,
            'best_score': best_score,
            'evaluation_results': results,
            'model_path': str(best_model_path)
        }
        
    except Exception as e:
        logger.error(f"\n{'='*70}")
        logger.error(f"PIPELINE FAILED: {str(e)}")
        logger.error(f"{'='*70}")
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hotel Booking Cancellation Prediction Pipeline'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/raw/Hotel_Reservations.csv',
        help='Path to the dataset CSV file'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for test set (default: 0.2)'
    )
    
    parser.add_argument(
        '--use-smote',
        action='store_true',
        default=True,
        help='Use SMOTE for handling class imbalance'
    )
    
    parser.add_argument(
        '--tune-xgboost',
        action='store_true',
        default=False,
        help='Run hyperparameter tuning for XGBoost (slower)'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        default='f1_score',
        choices=['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
        help='Metric to use for selecting best model (default: f1_score)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for log files (default: logs)'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    results = main(args)
    
    print("\n" + "="*70)
    print("PIPELINE RESULTS")
    print("="*70)
    print(f"Best Model: {results['best_model_name']}")
    print(f"Best Score: {results['best_score']:.4f}")
    print(f"Model Path: {results['model_path']}")
    print("="*70)
