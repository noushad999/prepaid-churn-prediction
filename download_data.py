# download_data.py - Fixed version with automatic fallback
import pandas as pd
import numpy as np
from pathlib import Path

def create_synthetic_dataset():
    """Generate realistic synthetic telecom churn data."""
    print("\n📊 Creating synthetic telecom dataset...")
    np.random.seed(42)
    n_samples = 10000
    
    # Generate customer data
    data = {
        'customerID': [f'CUST{i:05d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(0, 73, n_samples),
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
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples),
        'MonthlyCharges': np.random.uniform(18.0, 120.0, n_samples),
        'TotalCharges': np.random.uniform(18.0, 8500.0, n_samples),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
    }
    
    df = pd.DataFrame(data)
    
    # Add realistic correlations
    # Short tenure = higher churn
    short_tenure_mask = df['tenure'] < 12
    df.loc[short_tenure_mask, 'Churn'] = np.random.choice(
        ['Yes', 'No'], 
        short_tenure_mask.sum(), 
        p=[0.5, 0.5]
    )
    
    # Two-year contract = lower churn
    two_year_mask = df['Contract'] == 'Two year'
    df.loc[two_year_mask, 'Churn'] = np.random.choice(
        ['Yes', 'No'], 
        two_year_mask.sum(), 
        p=[0.1, 0.9]
    )
    
    # Month-to-month = higher churn
    month_mask = df['Contract'] == 'Month-to-month'
    df.loc[month_mask, 'Churn'] = np.random.choice(
        ['Yes', 'No'], 
        month_mask.sum(), 
        p=[0.45, 0.55]
    )
    
    # Save dataset
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    df.to_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv', index=False)
    
    churn_rate = (df['Churn'] == 'Yes').mean()
    print(f"   ✓ Created {n_samples:,} customer records")
    print(f"   ✓ Churn rate: {churn_rate:.2%}")
    print(f"   ✓ Features: {len(df.columns)-1}")
    
    return df

def create_call_records(df):
    """Generate synthetic call detail records for GNN."""
    print("\n📞 Creating call network data...")
    customer_ids = df['customerID'].tolist()
    n_calls = len(customer_ids) * 5  # 5 calls per customer avg
    
    calls = []
    for _ in range(n_calls):
        caller = np.random.choice(customer_ids)
        receiver = np.random.choice(customer_ids)
        if caller != receiver:  # No self-calls
            calls.append({
                'caller_id': caller,
                'receiver_id': receiver,
                'duration': np.random.randint(60, 3600),  # seconds
                'timestamp': pd.Timestamp.now() - pd.Timedelta(
                    days=np.random.randint(0, 365)
                )
            })
    
    call_df = pd.DataFrame(calls)
    call_df.to_csv('data/raw/call_records.csv', index=False)
    print(f"   ✓ Created {len(call_df):,} call records")
    
    return call_df

def create_support_tickets(df):
    """Generate synthetic support tickets for NLP."""
    print("\n💬 Creating customer support tickets...")
    customer_ids = df['customerID'].tolist()
    
    # Realistic complaint templates
    negative_complaints = [
        "Internet service is very slow and keeps disconnecting frequently",
        "Billing charges are incorrect and much higher than expected",
        "Phone line has constant static noise, unable to make clear calls",
        "Long wait times for technical support, very frustrating experience",
        "Bill is too high compared to competitors, considering switching",
        "Service quality has degraded significantly in recent months",
        "Customer service was unhelpful and rude during my last call",
        "Contract renewal process was confusing and misleading",
        "Frequent outages in my area, service very unreliable",
        "Technical issues never resolved despite multiple support calls"
    ]
    
    positive_complaints = [
        "Great service, very satisfied with the support team",
        "Fiber optic service is excellent and consistently fast",
        "Customer service representative was very helpful and professional",
        "Technical issue resolved quickly, thank you for great support",
        "Installation process was smooth and technician was knowledgeable",
        "Billing is clear and transparent, no hidden charges",
        "Service quality is consistently good, very reliable",
        "Support team went above and beyond to help me",
        "Best internet service I have used, highly recommend",
        "Great value for money, very pleased with the service"
    ]
    
    neutral_complaints = [
        "Inquiring about new service plans and pricing options",
        "Need to update my billing information and payment method",
        "Question about contract terms and renewal process",
        "Requesting information about upgrading my current plan",
        "General inquiry about available services in my area",
        "Need clarification on recent billing statement charges",
        "Asking about equipment upgrade options and costs",
        "Want to know about adding additional services to account",
        "Checking on scheduled maintenance in my neighborhood",
        "General service inquiry and feature availability question"
    ]
    
    tickets = []
    churn_customers = df[df['Churn'] == 'Yes']['customerID'].tolist()
    
    # 60% of customers have tickets
    for customer_id in np.random.choice(customer_ids, size=int(len(customer_ids)*0.6), replace=False):
        # Churned customers more likely to have negative sentiment
        if customer_id in churn_customers:
            sentiment_probs = [0.6, 0.2, 0.2]  # negative, neutral, positive
        else:
            sentiment_probs = [0.2, 0.3, 0.5]
        
        sentiment = np.random.choice(['negative', 'neutral', 'positive'], p=sentiment_probs)
        
        if sentiment == 'negative':
            text = np.random.choice(negative_complaints)
        elif sentiment == 'positive':
            text = np.random.choice(positive_complaints)
        else:
            text = np.random.choice(neutral_complaints)
        
        tickets.append({
            'customerID': customer_id,
            'ticket_text': text,
            'sentiment': sentiment
        })
    
    ticket_df = pd.DataFrame(tickets)
    ticket_df.to_csv('data/raw/support_tickets.csv', index=False)
    
    sentiment_counts = ticket_df['sentiment'].value_counts()
    print(f"   ✓ Created {len(ticket_df):,} support tickets")
    print(f"   ✓ Sentiment distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"      - {sentiment.capitalize()}: {count} ({count/len(ticket_df)*100:.1f}%)")
    
    return ticket_df

def main():
    print("\n" + "="*70)
    print("  TELECOM CHURN DATASET PREPARATION")
    print("  Synthetic Data Generation for ML Training")
    print("="*70)
    
    # Create synthetic customer data
    df = create_synthetic_dataset()
    
    # Generate call records for GNN
    call_df = create_call_records(df)
    
    # Generate support tickets for NLP
    ticket_df = create_support_tickets(df)
    
    # Summary statistics
    print("\n" + "="*70)
    print("  DATASET SUMMARY")
    print("="*70)
    print(f"\n📁 Files created in data/raw/:")
    print(f"   1. WA_Fn-UseC_-Telco-Customer-Churn.csv - {len(df):,} customers")
    print(f"   2. call_records.csv - {len(call_df):,} call records")
    print(f"   3. support_tickets.csv - {len(ticket_df):,} tickets")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   - Total Customers: {len(df):,}")
    print(f"   - Churned Customers: {(df['Churn']=='Yes').sum():,} ({(df['Churn']=='Yes').mean()*100:.1f}%)")
    print(f"   - Active Customers: {(df['Churn']=='No').sum():,} ({(df['Churn']=='No').mean()*100:.1f}%)")
    print(f"   - Average Tenure: {df['tenure'].mean():.1f} months")
    print(f"   - Average Monthly Charge: ${df['MonthlyCharges'].mean():.2f}")
    
    print("\n✅ All datasets ready for training!")
    print("\n🚀 Next steps:")
    print("   1. Run: python quick_start.py (for quick test)")
    print("   2. Run: python main.py (for full advanced training)")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
