import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelecomDataPreprocessor:
    """
    Advanced data preprocessing pipeline for telecom churn data.
    Implements ADASYN sampling, feature selection, and encoding.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.numeric_features = None
        self.categorical_features = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates, handle missing values, and standardize entries."""
        logger.info("Starting data cleaning...")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values (remove rows with missing data)
        initial_shape = df.shape[0]
        df = df.dropna()
        logger.info(f"Removed {initial_shape - df.shape[0]} rows with missing values")
        
        # Standardize categorical values
        if 'InternetService' in df.columns:
            df['InternetService'] = df['InternetService'].replace(
                ['No internet service'], 'No'
            )
        if 'PhoneService' in df.columns:
            df['PhoneService'] = df['PhoneService'].replace(
                ['No phone service'], 'No'
            )
        
        return df
    
    def select_features(self, df: pd.DataFrame, target_col: str = 'Churn',
                       correlation_threshold: float = 0.05) -> pd.DataFrame:
        """Select features based on correlation with target variable."""
        logger.info("Performing feature selection...")
        
        # Separate numeric and categorical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Calculate correlations for numeric features
        correlations = df[numeric_cols + [target_col]].corr()[target_col].abs()
        
        # Select features above threshold
        selected_features = correlations[
            correlations >= correlation_threshold
        ].index.tolist()
        selected_features.remove(target_col)
        
        # Add all categorical features (they'll be encoded)
        categorical_cols = df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        # Remove ID columns
        id_cols = [col for col in categorical_cols if 'id' in col.lower()]
        categorical_cols = [col for col in categorical_cols if col not in id_cols]
        
        selected_features.extend(categorical_cols)
        
        logger.info(f"Selected {len(selected_features)} features")
        return df[selected_features + [target_col]]
    
    def encode_features(self, df: pd.DataFrame, target_col: str = 'Churn',
                       fit: bool = True) -> pd.DataFrame:
        """Encode categorical features and scale numerical features."""
        logger.info("Encoding features...")
        
        df_encoded = df.copy()
        
        # Identify feature types
        numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_encoded.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Encode categorical features
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(
                    df_encoded[col].astype(str)
                )
            else:
                df_encoded[col] = self.label_encoders[col].transform(
                    df_encoded[col].astype(str)
                )
        
        # Scale numerical features
        if numeric_cols:
            if fit:
                df_encoded[numeric_cols] = self.scaler.fit_transform(
                    df_encoded[numeric_cols]
                )
            else:
                df_encoded[numeric_cols] = self.scaler.transform(
                    df_encoded[numeric_cols]
                )
        
        # Encode target variable
        if target_col in df_encoded.columns:
            if fit:
                self.label_encoders[target_col] = LabelEncoder()
                df_encoded[target_col] = self.label_encoders[target_col].fit_transform(
                    df_encoded[target_col].astype(str)
                )
            else:
                df_encoded[target_col] = self.label_encoders[target_col].transform(
                    df_encoded[target_col].astype(str)
                )
        
        self.numeric_features = numeric_cols
        self.categorical_features = categorical_cols
        
        return df_encoded
    
    def balance_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance dataset using ADASYN oversampling."""
        logger.info("Balancing dataset with ADASYN...")
        
        initial_dist = np.bincount(y)
        logger.info(f"Initial class distribution: {initial_dist}")
        
        adasyn = ADASYN(random_state=self.config.get('random_state', 42))
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        
        final_dist = np.bincount(y_resampled)
        logger.info(f"Final class distribution: {final_dist}")
        
        return X_resampled, y_resampled
    
    def preprocess_pipeline(self, df: pd.DataFrame, target_col: str = 'Churn',
                          balance: bool = True) -> Tuple:
        """Complete preprocessing pipeline."""
        # Clean data
        df_clean = self.clean_data(df)
        
        # Select features
        df_selected = self.select_features(
            df_clean, target_col,
            self.config.get('correlation_threshold', 0.05)
        )
        
        # Encode features
        df_encoded = self.encode_features(df_selected, target_col, fit=True)
        
        # Split features and target
        X = df_encoded.drop(columns=[target_col]).values
        y = df_encoded[target_col].values
        
        # Balance data
        if balance:
            X, y = self.balance_data(X, y)
        
        # Store feature names
        self.feature_columns = df_encoded.drop(columns=[target_col]).columns.tolist()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.get('test_size', 0.2),
            random_state=self.config.get('random_state', 42),
            stratify=y
        )
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
