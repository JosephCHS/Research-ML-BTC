#!/usr/bin/env python3

import sys
from model.machine_learning import MachineLearning

def run_sklearn_models(ml):
    """Run and evaluate sklearn models"""
    print("\n=== Running sklearn models ===")
    models_sklearn = [
        ('Logistic Regression', ml.model_logistic_regression()),
        ('SVM', ml.model_svm()),
    ]
    
    for name, model in models_sklearn:
        print(f"\nResults for {name}:")
        ml.display_result(model)
    
    ml.display_report_sklearn()

def run_deep_learning_models(ml):
    """Run and evaluate deep learning models"""
    print("\n=== Running deep learning models ===")
    
    print("\nTraining LSTM model...")
    ml.model_lstm()
    
    print("\nTraining CNN model...")
    ml.model_cnn()
    
    print("\nTraining BNN model...")
    ml.model_bnn()

def main():
    try:
        # Initialize machine learning pipeline
        print("Initializing machine learning models...")
        ml = MachineLearning()
        ml.display_information()

        # Run traditional ML models
        run_sklearn_models(ml)

        # Run deep learning models
        run_deep_learning_models(ml)

    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
