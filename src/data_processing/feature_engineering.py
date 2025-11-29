import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Create advanced features for churn prediction."""

    def __init__(self):
        pass

    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral and interaction features."""
        logger.info("Creating behavioral features...")

        df = df.copy()

        # Tenure-based features
        if 'tenure' in df.columns:
            df['tenure_years'] = df['tenure'] / 12
            df['tenure_group'] = pd.cut(
                df['tenure'], bins=[0, 12, 24, 48, 72],
                labels=['0-1yr', '1-2yr', '2-4yr', '4+yr']
            )

        # Revenue features
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            df['total_revenue'] = df['MonthlyCharges'] * df['tenure']
            df['avg_monthly_revenue'] = df['total_revenue'] / (df['tenure'] + 1)

        if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns:
            df['charge_ratio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)

        # Service count
        service_cols = [
            'PhoneService', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies'
        ]
        available_service_cols = [col for col in service_cols if col in df.columns]

        if available_service_cols:
            df['total_services'] = df[available_service_cols].apply(
                lambda row: sum([1 for val in row if val == 'Yes' or val == 1]),
                axis=1
            )
            df['service_adoption_rate'] = df['total_services'] / len(available_service_cols)

        # Contract engagement score
        if 'Contract' in df.columns:
            contract_scores = {
                'Month-to-month': 1,
                'One year': 2,
                'Two year': 3
            }
            df['contract_score'] = df['Contract'].map(contract_scores).fillna(1)

        # Payment method risk score
        if 'PaymentMethod' in df.columns:
            payment_scores = {
                'Electronic check': 3,
                'Mailed check': 2,
                'Bank transfer (automatic)': 1,
                'Credit card (automatic)': 1
            }
            df['payment_risk_score'] = df['PaymentMethod'].map(payment_scores).fillna(2)

        logger.info(f"Created {len(df.columns)} total features")
        return df

    def create_lifecycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer lifecycle stage features."""
        logger.info("Creating lifecycle features...")

        df = df.copy()

        if 'tenure' in df.columns:
            # Lifecycle stage
            def get_lifecycle_stage(tenure):
                if tenure <= 3:
                    return 'new'
                elif tenure <= 12:
                    return 'growing'
                elif tenure <= 36:
                    return 'mature'
                else:
                    return 'loyal'

            df['lifecycle_stage'] = df['tenure'].apply(get_lifecycle_stage)

        if 'MonthlyCharges' in df.columns:
            # Revenue tier
            df['revenue_tier'] = pd.qcut(
                df['MonthlyCharges'], q=4,
                labels=['low', 'medium', 'high', 'premium'],
                duplicates='drop'
            )

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables."""
        logger.info("Creating interaction features...")

        df = df.copy()

        # Contract * Tenure
        if 'Contract' in df.columns and 'tenure' in df.columns:
            df['contract_tenure_interaction'] = (
                df['Contract'].astype(str) + '_' + 
                df['tenure'].apply(lambda x: 'short' if x < 12 else 'long')
            )

        # Senior * Dependents
        if 'SeniorCitizen' in df.columns and 'Dependents' in df.columns:
            df['senior_with_dependents'] = (
                (df['SeniorCitizen'] == 1) & 
                (df['Dependents'] == 'Yes')
            ).astype(int)

        # Internet * Streaming
        if 'InternetService' in df.columns:
            streaming_cols = ['StreamingTV', 'StreamingMovies']
            available_streaming = [col for col in streaming_cols if col in df.columns]

            if available_streaming:
                df['has_streaming'] = df[available_streaming].apply(
                    lambda row: 1 if any(val == 'Yes' for val in row) else 0,
                    axis=1
                )
                df['internet_streaming_combo'] = (
                    (df['InternetService'] != 'No') & 
                    (df['has_streaming'] == 1)
                ).astype(int)

        return df
