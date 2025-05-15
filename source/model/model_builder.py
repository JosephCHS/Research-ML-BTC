import os
from typing import Tuple, Dict, Optional
# Data processing
import numpy
# ML preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# ML models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, BatchNormalization, Conv1D, MaxPooling1D,
    Flatten, Input, MultiHeadAttention, LayerNormalization, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)

import tensorflow_probability as tfp

from source.model.model_evaluator import ModelEvaluator


class ModelBuilder:
    """Class for building and training models"""
    
    def __init__(self, save_path: str = "./models"):
        """Initialize model builder
        
        Args:
            save_path: Path to save trained models
        """
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.evaluator = ModelEvaluator()
    
    def train_logistic_regression(self, X_train: numpy.ndarray, y_train: numpy.ndarray,
                                 optimize: bool = True) -> LogisticRegression:
        """Train logistic regression model
        
        Args:
            X_train: Training features
            y_train: Training targets
            optimize: Whether to optimize hyperparameters
            
        Returns:
            Trained logistic regression model
        """
        if optimize:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
                'class_weight': [None, 'balanced']
            }
            
            grid_search = GridSearchCV(
                LogisticRegression(max_iter=1000),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        else:
            model = LogisticRegression(solver='lbfgs', max_iter=1000)
            model.fit(X_train, y_train)
            return model
    
    def train_svm(self, X_train: numpy.ndarray, y_train: numpy.ndarray,
                 optimize: bool = True) -> SVC:
        """Train SVM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            optimize: Whether to optimize hyperparameters
            
        Returns:
            Trained SVM model
        """
        if optimize:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
            
            grid_search = GridSearchCV(
                SVC(probability=True),
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        else:
            model = SVC(kernel="rbf", probability=True)
            model.fit(X_train, y_train)
            return model
    
    def train_random_forest(self, X_train: numpy.ndarray, y_train: numpy.ndarray,
                          optimize: bool = True) -> RandomForestClassifier:
        """Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training targets
            optimize: Whether to optimize hyperparameters
            
        Returns:
            Trained Random Forest model
        """
        if optimize:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': [None, 'balanced']
            }
            
            grid_search = RandomizedSearchCV(
                RandomForestClassifier(),
                param_grid,
                n_iter=20,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                random_state=42
            )
            
            grid_search.fit(X_train, y_train)
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best score: {grid_search.best_score_:.4f}")
            
            return grid_search.best_estimator_
        else:
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, y_train)
            return model
    
    def build_lstm_model(self, input_shape: Tuple, output_units: int = 1,
                       is_classification: bool = True) -> Model:
        """Build LSTM model architecture"""
        # Create input layer
        inputs = Input(shape=input_shape)
        
        # First LSTM block
        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        # Second LSTM block
        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        # Third LSTM for final sequence processing
        x = Bidirectional(LSTM(16))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        # Dense layers for classification
        x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        if is_classification:
            outputs = Dense(output_units, activation='sigmoid')(x)
        else:
            outputs = Dense(output_units, activation='linear')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=True
        )
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy' if is_classification else 'mse',
            metrics=['accuracy', tf.keras.metrics.AUC()] if is_classification else ['mae', 'mse']
        )
        
        return model
    
    def build_cnn_model(self, input_shape: Tuple, output_units: int = 1,
                      is_classification: bool = True) -> Model:
        """Build CNN model architecture"""
        # Validate input shape
        if len(input_shape) != 2:
            raise ValueError(f"Expected input_shape to be (sequence_length, features), got {input_shape}")
        
        sequence_length, n_features = input_shape
        if sequence_length < 3:
            raise ValueError(f"Sequence length must be at least 3, got {sequence_length}")

        # Create model using functional API instead of Sequential
        inputs = Input(shape=input_shape)
        
        # 1D CNN layers
        x = Conv1D(
            filters=32,
            kernel_size=min(3, sequence_length - 1),
            activation='relu',
            padding='same'
        )(inputs)
        x = BatchNormalization()(x)
        
        x = Conv1D(
            filters=64,
            kernel_size=min(3, sequence_length - 1),
            activation='relu',
            padding='same'
        )(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Conv1D(
            filters=128,
            kernel_size=min(3, sequence_length - 1),
            activation='relu',
            padding='same'
        )(x)
        x = BatchNormalization()(x)
        
        # Global pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = Dropout(0.5)(x)
        x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        outputs = Dense(output_units, activation='sigmoid' if is_classification else 'linear')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy' if is_classification else 'mse',
            metrics=['accuracy', tf.keras.metrics.AUC()] if is_classification else ['mae', 'mse']
        )
        
        model.summary()
        return model
    
    def build_transformer_model(self, input_shape: Tuple, output_units: int = 1,
                             is_classification: bool = True) -> Model:
        """Build Transformer model architecture for time series
        
        Args:
            input_shape: Shape of input data
            output_units: Number of output units
            is_classification: Whether this is a classification problem
            
        Returns:
            Compiled Transformer model
        """
        # Input layer
        inputs = Input(shape=input_shape)
        
        # Add positional encoding (simplified)
        x = inputs
        
        # Transformer block
        x = LayerNormalization(epsilon=1e-6)(x)
        attention_output = MultiHeadAttention(
            num_heads=2, key_dim=64
        )(x, x)
        x = x + attention_output  # Residual connection
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(64)(x)
        x = x[:, -1, :]  # Take the last output for sequence classification
        
        # Output layer
        outputs = Dense(output_units, activation='sigmoid' if is_classification else 'linear')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile with appropriate loss function
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy' if is_classification else 'mse',
            metrics=['accuracy'] if is_classification else []
        )
        
        return model
    
    def build_bayesian_nn(self, input_shape: int, output_units: int = 1) -> Model:
        """Build Bayesian Neural Network with TensorFlow Probability
        
        Args:
            input_shape: Number of input features
            output_units: Number of output units
            
        Returns:
            Compiled Bayesian NN model
        """
        model = Sequential([
            # Input layer with explicit input shape
            Input(shape=(input_shape,)),
            
            # First dense layer with dropout for uncertainty
            Dense(128, activation='relu', 
                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Second dense layer
            Dense(64, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Output layer
            Dense(32, activation='relu',
                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.3),
            Dense(output_units, activation='sigmoid')
        ])

        # Use Monte Carlo Dropout for Bayesian approximation
        def bayesian_loss(y_true, y_pred):
            # Standard binary crossentropy loss
            standard_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            
            # Add KL divergence terms for the dropout layers
            kl_loss = sum(model.losses)
            
            return standard_loss + kl_loss

        # Compile model with custom loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=bayesian_loss,
            metrics=['accuracy']
        )
        
        return model
    
    def train_deep_learning_model(self, model: Model, X_train: numpy.ndarray, y_train: numpy.ndarray,
                                X_val: numpy.ndarray, y_val: numpy.ndarray,
                                model_name: str, epochs: int = 100,
                                batch_size: int = 32,
                                class_weights: Optional[Dict] = None) -> Model:
        """Train deep learning model with callbacks"""
        # Ensure the model directory exists
        os.makedirs(self.save_path, exist_ok=True)
        model_path = os.path.join(self.save_path, f"{model_name}.keras")
        # Adjust batch size based on data size
        batch_size = min(32, len(X_train) // 20)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                min_delta=1e-4
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=7,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                save_best_only=True,
                save_weights_only=False,  # Save full model
                monitor='val_loss',
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(self.save_path, 'logs', model_name),
                histogram_freq=1,
                update_freq='epoch'
            )
        ]
        
        try:
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weights,
                shuffle=True,
                verbose=1
            )
            
            # Save the entire model (not just weights)
            model.save(model_path)
            model.history = history
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
        
        return model


