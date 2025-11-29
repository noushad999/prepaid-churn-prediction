import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelecomDataLoader:
    """Load and validate telecom churn data."""

    def __init__(self, config: dict):
        self.config = config
        self.raw_path = Path(config['data']['raw_path'])

    def load_customer_data(self, filename: str) -> pd.DataFrame:
        """Load customer demographic and usage data."""
        filepath = self.raw_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        logger.info(f"Loading customer data from {filepath}")

        # Support multiple file formats
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filename}")

        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        return df

    def load_call_records(self, filename: str) -> pd.DataFrame:
        """Load call detail records (CDR) for graph construction."""
        filepath = self.raw_path / filename

        if not filepath.exists():
            logger.warning(f"Call records not found: {filepath}")
            return None

        logger.info(f"Loading call records from {filepath}")
        df = pd.read_csv(filepath)

        # Validate required columns
        required_cols = ['caller_id', 'receiver_id']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns in call records: {missing_cols}")

        logger.info(f"Loaded {len(df)} call records")
        return df

    def load_text_data(self, filename: str) -> pd.DataFrame:
        """Load customer text data (support tickets, complaints)."""
        filepath = self.raw_path / filename

        if not filepath.exists():
            logger.warning(f"Text data not found: {filepath}")
            return None

        logger.info(f"Loading text data from {filepath}")
        df = pd.read_csv(filepath)

        # Validate required columns
        if 'customer_id' not in df.columns or 'text' not in df.columns:
            raise ValueError("Text data must contain 'customer_id' and 'text' columns")

        logger.info(f"Loaded {len(df)} text records")
        return df

    def validate_data(self, df: pd.DataFrame, target_col: str = 'Churn') -> bool:
        """Validate data quality and structure."""
        logger.info("Validating data...")

        # Check for empty dataframe
        if df.empty:
            logger.error("Dataframe is empty!")
            return False

        # Check for target column
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found!")
            return False

        # Check data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        logger.info(f"Numeric features: {len(numeric_cols)}")
        logger.info(f"Categorical features: {len(categorical_cols)}")

        # Check for excessive missing values
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 50]

        if len(high_missing) > 0:
            logger.warning(f"Columns with >50% missing values: {list(high_missing.index)}")

        # Check target distribution
        target_dist = df[target_col].value_counts()
        logger.info(f"Target distribution:\n{target_dist}")

        logger.info("✅ Data validation complete!")
        return True

    def create_sample_dataset(self, output_path: str, n_samples: int = 5000):
        """Create a sample dataset for testing."""
        logger.info(f"Creating sample dataset with {n_samples} records...")

        np.random.seed(42)

        data = {
            'customer_id': [f'CUST_{i:06d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.randint(0, 72, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check', 
                'Bank transfer', 'Credit card'
            ], n_samples),
            'MonthlyCharges': np.random.uniform(18, 118, n_samples).round(2),
            'TotalCharges': np.random.uniform(18, 8684, n_samples).round(2),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
        }

        df = pd.DataFrame(data)

        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

        logger.info(f"✅ Sample dataset created: {output_file}")
        return df
