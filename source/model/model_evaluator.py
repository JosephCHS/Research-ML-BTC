from typing import Dict, List, Optional
import numpy as np
from sklearn import metrics
# Data processing
import numpy
# Visualization
import matplotlib.pyplot as plt
import seaborn
# ML preprocessing
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)


class ModelEvaluator:
    """Class for evaluating model performance"""
    
    @staticmethod
    def evaluate_classification(y_true, y_pred, model_name=""):
        """Evaluate classification model performance with proper handling of edge cases"""
        # Ensure predictions are flattened and binary
        y_pred = np.asarray(y_pred).flatten()
        y_true = np.asarray(y_true).flatten()
        
        # Ensure binary predictions
        y_pred = (y_pred > 0.5).astype(int)
        y_true = y_true.astype(int)
        
        results = {}
        
        # Handle case where all predictions are the same
        unique_preds = np.unique(y_pred)
        if len(unique_preds) == 1:
            print(f"Warning: {model_name} predicted all samples as {unique_preds[0]}")
        
        # Calculate metrics with zero_division parameter
        results['precision'] = metrics.precision_score(
            y_true, y_pred, zero_division=0, average='binary'
        )
        results['recall'] = metrics.recall_score(
            y_true, y_pred, zero_division=0, average='binary'
        )
        results['accuracy'] = metrics.accuracy_score(y_true, y_pred)
        results['f1'] = metrics.f1_score(
            y_true, y_pred, zero_division=0, average='binary'
        )
        
        # Get confusion matrix
        cm = metrics.confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm
        
        # Calculate class distribution safely
        def get_class_dist(y):
            counts = np.bincount(y, minlength=2)  # Ensure at least 2 classes
            return counts / len(y)
        
        try:
            class_dist = get_class_dist(y_true)
            pred_dist = get_class_dist(y_pred)
            
            # Print detailed evaluation
            print(f"\nEvaluation for {model_name}:")
            print(f"Precision: {results['precision']:.6f}")
            print(f"Recall: {results['recall']:.6f}")
            print(f"Accuracy: {results['accuracy']:.6f}")
            print(f"F1 Score: {results['f1']:.6f}")
            print("\nConfusion Matrix:")
            print(cm)
            print("\nClass Distribution:")
            print(f"True  - Class 0: {class_dist[0]:.2%}, Class 1: {class_dist[1]:.2%}")
            print(f"Pred  - Class 0: {pred_dist[0]:.2%}, Class 1: {pred_dist[1]:.2%}")
            
        except Exception as e:
            print(f"Warning: Could not calculate class distribution: {str(e)}")
            print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
            print(f"y_true dtype: {y_true.dtype}, y_pred dtype: {y_pred.dtype}")
            print(f"Unique values in y_true: {np.unique(y_true)}")
            print(f"Unique values in y_pred: {np.unique(y_pred)}")
            
        return results
    
    @staticmethod
    def evaluate_regression(y_true: numpy.ndarray, y_pred: numpy.ndarray) -> Dict:
        """Evaluate regression model performance
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        return {
            'rmse': numpy.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
    
    @staticmethod
    def plot_confusion_matrix(y_true: numpy.ndarray, y_pred: numpy.ndarray, class_names: List[str] = None) -> None:
        """Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of the classes
        """
        if class_names is None:
            class_names = ["DOWN", "UP"]
            
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true: numpy.ndarray, y_proba: numpy.ndarray) -> None:
        """Plot ROC curve
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()
    
    @staticmethod
    def plot_learning_curves(history) -> None:
        """Plot learning curves from Keras history
        
        Args:
            history: Keras training history
        """
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy if available
        if 'accuracy' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def get_classification_report(y_true: numpy.ndarray, y_pred: numpy.ndarray) -> str:
        """Generate classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        return classification_report(y_true, y_pred)

