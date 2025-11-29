# predict.py - Make predictions on new customers
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load the same model architecture
class ChurnPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.network(x)

def predict_churn(customer_data_path='data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """Predict churn probability for customers."""
    
    print("="*70)
    print("  CUSTOMER CHURN PREDICTION")
    print("="*70)
    
    # Load and preprocess data
    print("\n1. Loading customer data...")
    df = pd.read_csv(customer_data_path)
    
    # Keep customer IDs
    customer_ids = df['customerID'].values if 'customerID' in df.columns else None
    
    # Preprocess
    df = df.drop(columns=['customerID'], errors='ignore')
    if 'Churn' in df.columns:
        df = df.drop(columns=['Churn'])
    df = df.dropna()
    
    # Encode categorical
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    # Scale
    X = StandardScaler().fit_transform(df.values)
    print(f"   ✓ Loaded {len(X)} customers")
    
    # Load model
    print("\n2. Loading trained model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChurnPredictor(X.shape[1])
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    print(f"   ✓ Model loaded on {device}")
    
    # Predict
    print("\n3. Making predictions...")
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)
        churn_probs = probs[:, 1].cpu().numpy()  # Probability of churn
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    
    # Create results dataframe
    results = pd.DataFrame({
        'CustomerID': customer_ids[:len(predictions)] if customer_ids is not None else range(len(predictions)),
        'Churn_Prediction': ['Will Churn' if p == 1 else 'Will Stay' for p in predictions],
        'Churn_Probability': churn_probs,
        'Risk_Level': pd.cut(churn_probs, bins=[0, 0.3, 0.7, 1.0], 
                            labels=['Low Risk', 'Medium Risk', 'High Risk'])
    })
    
    # Save results
    results.to_csv('predictions.csv', index=False)
    print(f"   ✓ Predictions saved to: predictions.csv")
    
    # Summary
    print("\n" + "="*70)
    print("  PREDICTION SUMMARY")
    print("="*70)
    print(f"\n  Total Customers: {len(results)}")
    print(f"  Will Churn: {(predictions == 1).sum()} ({(predictions == 1).mean()*100:.1f}%)")
    print(f"  Will Stay: {(predictions == 0).sum()} ({(predictions == 0).mean()*100:.1f}%)")
    
    print(f"\n  Risk Distribution:")
    print(results['Risk_Level'].value_counts().to_string())
    
    # Show high-risk customers
    high_risk = results[results['Churn_Probability'] > 0.8].head(10)
    if len(high_risk) > 0:
        print(f"\n  Top 10 High-Risk Customers (Churn Probability > 80%):")
        print(high_risk.to_string(index=False))
    
    print("\n" + "="*70)
    
    return results

if __name__ == "__main__":
    predictions = predict_churn()
