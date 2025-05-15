#!/usr/bin/env python3

import os
# Configure TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = ''    # Disable GPU for now

import tensorflow as tf
# Enable eager execution and set memory growth
tf.compat.v1.enable_eager_execution()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Set TF logging level
tf.get_logger().setLevel('ERROR')

# Usage example
from source.model.predictor import BitcoinPricePredictor


if __name__ == "__main__":
    # Create predictor
    predictor = BitcoinPricePredictor(use_btc1=True)
    
    # Run full pipeline
    results = predictor.run_full_pipeline(optimize_classical=True)
    
    # Access individual models if needed
    # lstm_model = predictor.models['lstm']
    # Make predictions with specific model
    # predictor.models['transformer'].predict(new_data)

    

# import sys
# from model.machine_learning import MachineLearning

# def run_sklearn_models(ml):
#     """Run and evaluate sklearn models"""
#     print("\n=== Running sklearn models ===")
#     models_sklearn = [
#         ('Logistic Regression', ml.model_logistic_regression()),
#         ('SVM', ml.model_svm()),
#     ]
    
#     for name, model in models_sklearn:
#         print(f"\nResults for {name}:")
#         ml.display_result(model)
    
#     ml.display_report_sklearn()

# def run_deep_learning_models(ml):
#     """Run and evaluate deep learning models"""
#     print("\n=== Running deep learning models ===")
    
#     print("\nTraining LSTM model...")
#     ml.model_lstm()
    
#     print("\nTraining CNN model...")
#     ml.model_cnn()
    
#     print("\nTraining BNN model...")
#     ml.model_bnn()

# def main():
#     try:
#         # Initialize machine learning pipeline
#         print("Initializing machine learning models...")
#         ml = MachineLearning()
#         ml.display_information()

#         # Run traditional ML models
#         run_sklearn_models(ml)

#         # Run deep learning models
#         run_deep_learning_models(ml)

#     except Exception as e:
#         print(f"An error occurred: {str(e)}", file=sys.stderr)
#         return 1

#     return 0

# if __name__ == '__main__':
#     sys.exit(main())
