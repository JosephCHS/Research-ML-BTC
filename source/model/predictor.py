from typing import Dict, List
import os
from matplotlib import pyplot as plt
import numpy
import pandas
from source.model.model_builder import ModelBuilder
from source.model.data_processor import DataProcessor
from source.model.model_evaluator import ModelEvaluator


class BitcoinPricePredictor:
    """Main class for Bitcoin price prediction"""
    
    def __init__(self, data_path: str = None, use_btc1: bool = False):
        """Initialize Bitcoin price predictor
        
        Args:
            data_path: Path to the dataset
            use_btc1: Whether to use BTC1 2017-2021 data
        """
        # Set default data path if not provided
        if data_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            if use_btc1:
                data_path = os.path.join(script_dir, "../fetch/data/dataset_with_future.csv")
            else:
                data_path = os.path.join(script_dir, "../fetch/data/dataset.csv")
        
        # Initialize components
        self.data_processor = DataProcessor(data_path, use_btc1)
        self.model_builder = ModelBuilder(save_path="./models")
        self.evaluator = ModelEvaluator()
        
        # Initialize data
        self.data = None
        self.models = {}
        
    def prepare_data(self, engineer_features: bool = True) -> Dict:
        """Prepare data for modeling
        
        Args:
            engineer_features: Whether to perform feature engineering
            
        Returns:
            Dictionary with prepared data
        """
        # Load and process data
        self.data_processor.load_data()
        self.data_processor.normalize_data()
        
        if engineer_features:
            self.data_processor.engineer_features()
        
        # Prepare both tabular and sequence data
        self.data = self.data_processor.prepare_deep_learning_data()
        
        # Get class weights for handling imbalance
        unique, counts = numpy.unique(self.data['tabular']['y_train'], return_counts=True)
        total = sum(counts)
        class_weights = {
            int(unique[i]): total / (len(unique) * counts[i])
            for i in range(len(unique))
        }
        
        self.data['class_weights'] = class_weights
        
        return self.data
    
    def train_classical_models(self, optimize: bool = True) -> Dict:
        """Train classical machine learning models
        
        Args:
            optimize: Whether to optimize hyperparameters
            
        Returns:
            Dictionary with trained models
        """
        if self.data is None:
            self.prepare_data()
            
        data = self.data['tabular']
        
        # Train logistic regression
        print("Training Logistic Regression...")
        lr_model = self.model_builder.train_logistic_regression(
            data['X_train'], data['y_train'], optimize
        )
        
        # Train SVM
        print("Training SVM...")
        svm_model = self.model_builder.train_svm(
            data['X_train'], data['y_train'], optimize
        )
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_model = self.model_builder.train_random_forest(
            data['X_train'], data['y_train'], optimize
        )
        
        # Store models
        self.models.update({
            'logistic_regression': lr_model,
            'svm': svm_model,
            'random_forest': rf_model
        })
        
        return self.models
    
    def train_deep_learning_models(self) -> Dict:
        """Train deep learning models
        
        Args:
            None
            
        Returns:
            Dictionary with trained models
        """
        if self.data is None:
            self.prepare_data()
            
        # Get sequence data for deep learning
        seq_data = self.data['sequence']
        
        # Split validation data from training data
        val_split_idx = int(len(seq_data['X_train']) * 0.8)
        X_train = seq_data['X_train'][:val_split_idx]
        y_train = seq_data['y_train'][:val_split_idx]
        X_val = seq_data['X_train'][val_split_idx:]
        y_val = seq_data['y_train'][val_split_idx:]
        
        # Get input shapes
        lstm_input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_len, features)
        cnn_input_shape = (X_train.shape[1], X_train.shape[2])   # (seq_len, features)
        
        # Build and train LSTM model
        print("Training LSTM model...")
        lstm_model = self.model_builder.build_lstm_model(lstm_input_shape)
        trained_lstm = self.model_builder.train_deep_learning_model(
            lstm_model, X_train, y_train, X_val, y_val,
            model_name="lstm_bitcoin",
            epochs=100,
            batch_size=32,
            class_weights=self.data['class_weights']
        )
        
        # Build and train CNN model
        print("Training CNN model...")
        cnn_model = self.model_builder.build_cnn_model(cnn_input_shape)
        trained_cnn = self.model_builder.train_deep_learning_model(
            cnn_model, X_train, y_train, X_val, y_val,
            model_name="cnn_bitcoin",
            epochs=100,
            batch_size=32,
            class_weights=self.data['class_weights']
        )
        
        # Build and train Transformer model
        print("Training Transformer model...")
        transformer_model = self.model_builder.build_transformer_model(lstm_input_shape)
        trained_transformer = self.model_builder.train_deep_learning_model(
            transformer_model, X_train, y_train, X_val, y_val,
            model_name="transformer_bitcoin",
            epochs=100,
            batch_size=32,
            class_weights=self.data['class_weights']
        )
        
        # Train Bayesian Neural Network with tabular data
        print("Training Bayesian Neural Network...")
        tabular_data = self.data['tabular']
        bayes_model = self.model_builder.build_bayesian_nn(tabular_data['X_train'].shape[1])
        trained_bayes = self.model_builder.train_deep_learning_model(
            bayes_model, 
            tabular_data['X_train'], tabular_data['y_train'],
            tabular_data['X_test'][:len(tabular_data['X_test'])//2], 
            tabular_data['y_test'][:len(tabular_data['y_test'])//2],
            model_name="bayesian_nn_bitcoin",
            epochs=50,
            batch_size=64,
            class_weights=self.data['class_weights']
        )
        
        # Store models
        self.models.update({
            'lstm': trained_lstm,
            'cnn': trained_cnn,
            'transformer': trained_transformer,
            'bayesian_nn': trained_bayes
        })
        
        return self.models
    
    def evaluate_all_models(self) -> Dict:
        """Evaluate all trained models
        
        Returns:
            Dictionary with evaluation results for all models
        """
        results = {}
        
        # Evaluate classical models
        if any(model in self.models for model in ['logistic_regression', 'svm', 'random_forest']):
            tabular_data = self.data['tabular']
            for name, model in self.models.items():
                if name in ['logistic_regression', 'svm', 'random_forest']:
                    print(f"Evaluating {name}...")
                    # Make predictions
                    y_pred = model.predict(tabular_data['X_test'])
                    
                    # Get probabilities if available
                    try:
                        y_proba = model.predict_proba(tabular_data['X_test'])[:, 1]
                    except:
                        y_proba = None
                    
                    # Evaluate
                    results[name] = self.evaluator.evaluate_classification(
                        tabular_data['y_test'], y_pred, y_proba
                    )
                    
                    # Plot confusion matrix
                    self.evaluator.plot_confusion_matrix(tabular_data['y_test'], y_pred)
                    
                    # Plot ROC curve if probabilities are available
                    if y_proba is not None:
                        self.evaluator.plot_roc_curve(tabular_data['y_test'], y_proba)
        
        # Evaluate deep learning models
        if any(model in self.models for model in ['lstm', 'cnn', 'transformer', 'bayesian_nn']):
            seq_data = self.data['sequence']
            tabular_data = self.data['tabular']
            
            for name, model in self.models.items():
                if name in ['lstm', 'cnn', 'transformer']:
                    print(f"Evaluating {name}...")
                    # Make predictions
                    y_pred_proba = model.predict(seq_data['X_test'])
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    
                    # Evaluate
                    results[name] = self.evaluator.evaluate_classification(
                        seq_data['y_test'], y_pred, y_pred_proba
                    )
                    
                    # Plot confusion matrix and ROC curve
                    self.evaluator.plot_confusion_matrix(seq_data['y_test'], y_pred)
                    self.evaluator.plot_roc_curve(seq_data['y_test'], y_pred_proba)
                    
                    # Plot learning curves
                    if hasattr(model, 'history'):
                        self.evaluator.plot_learning_curves(model.history)
                
                elif name == 'bayesian_nn':
                    print(f"Evaluating {name}...")
                    # For Bayesian NN, make multiple predictions to estimate uncertainty
                    n_samples = 10
                    predictions = []
                    
                    for _ in range(n_samples):
                        y_pred_proba = model.predict(tabular_data['X_test'])
                        predictions.append(y_pred_proba)
                    
                    # Average predictions
                    y_pred_proba_mean = numpy.mean(predictions, axis=0)
                    y_pred_std = numpy.std(predictions, axis=0)
                    y_pred = (y_pred_proba_mean > 0.5).astype(int)
                    
                    # Evaluate
                    results[name] = self.evaluator.evaluate_classification(
                        tabular_data['y_test'], y_pred, y_pred_proba_mean
                    )
                    
                    # Add uncertainty information
                    results[name]['uncertainty_mean'] = y_pred_std.mean()
                    
                    # Plot confusion matrix and ROC curve
                    self.evaluator.plot_confusion_matrix(tabular_data['y_test'], y_pred)
                    self.evaluator.plot_roc_curve(tabular_data['y_test'], y_pred_proba_mean)
                    
                    # Plot learning curves
                    if hasattr(model, 'history'):
                        self.evaluator.plot_learning_curves(model.history)
        
        return results
    
    def compare_models(self, results: Dict) -> None:
        """Compare performance metrics across models"""
        # Extract performance metrics
        models = []
        accuracy = []
        precision = []
        recall = []
        f1_score = []
        auc = []
        
        for model_name, metrics in results.items():
            models.append(model_name)
            accuracy.append(metrics['accuracy'])
            precision.append(metrics['precision'])
            recall.append(metrics['recall'])
            # Fix the metric name to match what's returned by evaluate_classification
            f1_score.append(metrics['f1'])  # Changed from 'f1_score' to 'f1'
            auc.append(metrics.get('auc', 0))
        
        # Create DataFrame for comparison
        comparison = pandas.DataFrame({
            'Model': models,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score,
            'AUC': auc
        })
        
        # Sort by F1 score
        comparison = comparison.sort_values('F1 Score', ascending=False).reset_index(drop=True)
        
        # Print comparison
        print("\nModel Comparison:")
        print(comparison.to_string(index=False))
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        x = numpy.arange(len(models))
        width = 0.15
        multiplier = 0
        
        for metric in metrics_to_plot:
            offset = width * multiplier
            plt.bar(x + offset, comparison[metric], width, label=metric)
            multiplier += 1
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Comparison')
        plt.xticks(x + width * 2, comparison['Model'], rotation=45)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
        plt.tight_layout()
        plt.show()
    
    def make_ensemble_prediction(self, X_test: numpy.ndarray, models: List[str] = None) -> numpy.ndarray:
        """Make predictions using an ensemble of models
        
        Args:
            X_test: Test data for prediction
            models: List of model names to include in ensemble
            
        Returns:
            Array of ensemble predictions
        """
        if models is None:
            # Use all available models
            models = list(self.models.keys())
        
        # Container for predictions
        predictions = []
        
        # Get predictions from each model
        for model_name in models:
            if model_name not in self.models:
                continue
                
            model = self.models[model_name]
            
            # Handle different input requirements for different models
            if model_name in ['lstm', 'cnn', 'transformer']:
                # These models require sequence data
                if len(X_test.shape) == 2:  # If tabular data provided
                    # Need to create sequences - use a simplified approach
                    X_test_seq = numpy.array([X_test])  # Create single sequence
                    X_test_seq = numpy.transpose(X_test_seq, (1, 0, 2))  # Reshape to (samples, seq_len, features)
                    
                    if model_name == 'cnn':
                        # CNN might need reshaping
                        # This is a simplification - in practice, ensure X_test has right format
                        pass
                else:
                    X_test_seq = X_test
                    
                y_pred = model.predict(X_test_seq)
            else:
                # For classical models and bayesian_nn
                y_pred = model.predict(X_test)
                
                # Convert to probabilities if it's not already
                if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
                    try:
                        y_pred = model.predict_proba(X_test)[:, 1].reshape(-1, 1)
                    except:
                        # If model doesn't support predict_proba, use existing predictions
                        y_pred = y_pred.reshape(-1, 1)
            
            predictions.append(y_pred)
        
        # Average predictions
        ensemble_preds = numpy.mean(predictions, axis=0)
        
        # Convert to binary predictions
        binary_preds = (ensemble_preds > 0.5).astype(int)
        
        return binary_preds
    
    def run_full_pipeline(self, optimize_classical: bool = True) -> Dict:
        """Run the full prediction pipeline
        
        Args:
            optimize_classical: Whether to optimize classical models
            
        Returns:
            Dictionary with results
        """
        # Prepare data
        print("Preparing data...")
        self.prepare_data(engineer_features=True)
        
        # Train classical models
        print("\nTraining classical models...")
        self.train_classical_models(optimize=optimize_classical)
        
        # Train deep learning models
        print("\nTraining deep learning models...")
        self.train_deep_learning_models()
        
        # Evaluate all models
        print("\nEvaluating models...")
        results = self.evaluate_all_models()
        
        # Compare models
        self.compare_models(results)
        
        # Create ensemble prediction
        print("\nCreating ensemble prediction...")
        if self.data is not None:
            ensemble_pred = self.make_ensemble_prediction(self.data['tabular']['X_test'])
            ensemble_results = self.evaluator.evaluate_classification(
                self.data['tabular']['y_test'], ensemble_pred
            )
            results['ensemble'] = ensemble_results
            
            print("Ensemble Model Performance:")
            print(f"Accuracy: {ensemble_results['accuracy']:.4f}")
            print(f"Precision: {ensemble_results['precision']:.4f}")
            print(f"Recall: {ensemble_results['recall']:.4f}")
            print(f"F1 Score: {ensemble_results['f1']:.4f}")
        
        return results