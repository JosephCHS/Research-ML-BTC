from typing import Tuple, Dict
# Data processing
import pandas
import numpy
# ML preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


class DataProcessor:
    """Class responsible for data loading, preprocessing and feature engineering"""
    
    def __init__(self, data_path: str, use_btc1: bool = False):
        """Initialize the data processor
        
        Args:
            data_path: Path to the dataset CSV file
            use_btc1: Whether to use BTC1 2017-2021 data (True) or 2012-2021 data (False)
        """
        self.data_path = data_path
        self.use_btc1 = use_btc1
        self.raw_data = None
        self.processed_data = None
        self.target_column = "Position"
        
    def load_data(self) -> pandas.DataFrame:
        """Load and perform initial cleaning of the dataset
        
        Returns:
            Cleaned DataFrame
        """
        # Load data
        df = pandas.read_csv(self.data_path)
        
        # Convert boolean values to integers
        df[self.target_column] = df[self.target_column].astype(int)
        
        # Handle missing values
        df = df.replace("?", numpy.nan).infer_objects(copy=False)
        df = df.ffill().bfill().fillna(0)  # Impute missing values
        
        # Process date column
        df["date"] = pandas.to_datetime(df["date"])
        df["date"] = df["date"].astype(numpy.int64) // 10**9  # Convert to Unix timestamp
        
        # Remove unnecessary columns for BTC1 data
        if self.use_btc1:
            df = df.drop(columns=['Value-BCDDY', 'low', "high", "Value-MKTCP", "close", "EMA"])
        
        self.raw_data = df
        return df
    
    def normalize_data(self, scaler_type: str = "minmax") -> pandas.DataFrame:
        """Normalize the dataset features
        
        Args:
            scaler_type: Type of scaler to use ("minmax" or "standard")
            
        Returns:
            Normalized DataFrame
        """
        # Separate target and features
        target = self.raw_data[self.target_column]
        features = self.raw_data.drop(columns=[self.target_column])
        
        # Apply scaling based on the chosen method
        if scaler_type == "standard":
            scaler = StandardScaler()
        else:  # default to MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            
        # Fit and transform the features
        scaled_features = scaler.fit_transform(features)
        
        # Create new DataFrame with scaled features
        df_scaled = pandas.DataFrame(scaled_features, columns=features.columns)
        df_scaled[self.target_column] = target
        
        self.processed_data = df_scaled
        return df_scaled
    
    def engineer_features(self) -> pandas.DataFrame:
        """Create additional features that may improve model performance
        
        Returns:
            DataFrame with engineered features
        """
        if self.processed_data is None:
            self.processed_data = self.normalize_data()
            
        df = self.processed_data.copy()
        
        # Example feature engineering for time series data
        if 'close' in df.columns:
            # Add rolling window statistics
            df['rolling_mean_5'] = df['close'].rolling(window=5).mean()
            df['rolling_std_5'] = df['close'].rolling(window=5).std()
            
            # Add momentum indicators
            df['momentum_5'] = df['close'] - df['close'].shift(5)
            df['roc_5'] = df['close'].pct_change(periods=5) * 100
            
            # Add volatility indicator
            df['volatility_5'] = df['close'].rolling(window=5).std() / df['close'].rolling(window=5).mean()
            
        # Handle missing values created by rolling windows
        df = df.ffill().bfill().fillna(0)
        
        self.processed_data = df
        return df
    
    def create_sequences(self, seq_length: int = 50) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Create time series sequences for deep learning models
        
        Args:
            seq_length: Length of each sequence
            
        Returns:
            Tuple of (X, y) arrays with sequences
        """
        if self.processed_data is None:
            self.processed_data = self.normalize_data()
            
        # Convert DataFrame to numpy array
        data = self.processed_data.copy()
        target = data.pop(self.target_column).values
        features = data.values
        
        X, y = [], []
        for i in range(len(features) - seq_length):
            X.append(features[i:i+seq_length])
            y.append(target[i+seq_length])
            
        return numpy.array(X), numpy.array(y)
    
    def train_test_split(self, test_size: float = 0.3, random_state: int = 42) -> Tuple:
        """Split data into training and testing sets
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.processed_data is None:
            self.processed_data = self.normalize_data()
            
        data = self.processed_data.copy()
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def prepare_deep_learning_data(self, seq_length: int = 50) -> Dict:
        """Prepare data specifically for deep learning models
        
        Args:
            seq_length: Length of sequences for time series models
            
        Returns:
            Dictionary containing train and test data for deep learning
        """
        # For time-based split instead of random split
        if self.processed_data is None:
            self.processed_data = self.normalize_data()
            
        # Determine split point (70/30)
        n_samples = len(self.processed_data)
        n_train = int(n_samples * 0.7)
        
        # Split data
        train_data = self.processed_data.iloc[:n_train]
        test_data = self.processed_data.iloc[n_train:]
        
        # Extract targets
        train_target = train_data.pop(self.target_column).values
        test_target = test_data.pop(self.target_column).values
        
        # Create sequences for recurrent models
        X_sequences, y_sequences = self.create_sequences(seq_length)
        seq_split = int(len(X_sequences) * 0.7)
        
        return {
            'tabular': {
                'X_train': train_data.values,
                'y_train': train_target,
                'X_test': test_data.values,
                'y_test': test_target
            },
            'sequence': {
                'X_train': X_sequences[:seq_split],
                'y_train': y_sequences[:seq_split],
                'X_test': X_sequences[seq_split:],
                'y_test': y_sequences[seq_split:]
            }
        }

