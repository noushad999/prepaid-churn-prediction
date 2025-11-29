# retention_strategy.py - Generate targeted retention campaigns
import pandas as pd
import numpy as np

def generate_retention_strategies():
    """Generate personalized retention strategies based on churn risk."""
    
    print("="*70)
    print("  CUSTOMER RETENTION STRATEGY GENERATOR")
    print("="*70)
    
    # Load predictions
    predictions = pd.read_csv('predictions.csv')
    customers = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Merge data
    if 'customerID' in customers.columns:
        customers = customers.rename(columns={'customerID': 'CustomerID'})
    
    df = predictions.merge(customers, on='CustomerID', how='left')
    
    # Segment high-risk customers
    high_risk = df[df['Risk_Level'] == 'High Risk'].copy()
    
    print(f"\n📊 Analysis Summary:")
    print(f"   Total Customers: {len(df):,}")
    print(f"   High Risk: {len(high_risk):,} ({len(high_risk)/len(df)*100:.1f}%)")
    
    # Calculate potential revenue loss
    if 'MonthlyCharges' in df.columns:
        avg_monthly_revenue = high_risk['MonthlyCharges'].mean()
        annual_revenue_at_risk = avg_monthly_revenue * 12 * len(high_risk)
        
        print(f"\n💰 Financial Impact:")
        print(f"   Avg Monthly Charge: ${avg_monthly_revenue:.2f}")
        print(f"   Annual Revenue at Risk: ${annual_revenue_at_risk:,.2f}")
        print(f"   If we save 50%: ${annual_revenue_at_risk * 0.5:,.2f}")
    
    # Retention strategies by segment
    strategies = []
    
    for idx, row in high_risk.iterrows():
        customer_id = row['CustomerID']
        churn_prob = row['Churn_Probability']
        
        # Initialize strategy
        strategy = {
            'CustomerID': customer_id,
            'Churn_Probability': f"{churn_prob:.1%}",
            'Priority': 'CRITICAL' if churn_prob > 0.85 else 'HIGH',
            'Actions': [],
            'Discount_Offer': None,
            'Contact_Method': 'Phone Call'
        }
        
        # Contract-based strategies
        if 'Contract' in row and pd.notna(row['Contract']):
            if row['Contract'] == 'Month-to-month':
                strategy['Actions'].append('Offer 1-year contract with 15% discount')
                strategy['Discount_Offer'] = '15% for 12-month commitment'
            elif row['Contract'] == 'One year':
                strategy['Actions'].append('Upgrade to 2-year contract with premium benefits')
                strategy['Discount_Offer'] = '20% + premium support'
        
        # Tenure-based strategies
        if 'tenure' in row and pd.notna(row['tenure']):
            if row['tenure'] < 12:
                strategy['Actions'].append('Assign dedicated onboarding specialist')
                strategy['Actions'].append('Provide free setup assistance')
            elif row['tenure'] > 36:
                strategy['Actions'].append('Loyalty reward: 10% lifetime discount')
                strategy['Discount_Offer'] = '10% loyalty discount'
        
        # Service-based strategies
        if 'InternetService' in row and pd.notna(row['InternetService']):
            if row['InternetService'] == 'Fiber optic':
                strategy['Actions'].append('Offer free speed upgrade')
            elif row['InternetService'] == 'DSL':
                strategy['Actions'].append('Promote fiber optic upgrade with discount')
        
        # Support-based strategies
        if 'TechSupport' in row and row['TechSupport'] == 'No':
            strategy['Actions'].append('Include free tech support for 6 months')
        
        # Payment-based strategies
        if 'PaymentMethod' in row and row['PaymentMethod'] == 'Electronic check':
            strategy['Actions'].append('Incentivize automatic payment: $5/month credit')
        
        # Monthly charges strategy
        if 'MonthlyCharges' in row and pd.notna(row['MonthlyCharges']):
            if row['MonthlyCharges'] > 70:
                strategy['Actions'].append('Review plan: possible downgrade to save cost')
                strategy['Discount_Offer'] = '$10/month discount for 12 months'
        
        # Default actions if none assigned
        if not strategy['Actions']:
            strategy['Actions'].append('Personal call from retention specialist')
            strategy['Actions'].append('Custom plan review and optimization')
            strategy['Discount_Offer'] = '10% discount for next 6 months'
        
        # Convert actions list to string
        strategy['Actions'] = ' | '.join(strategy['Actions'])
        
        strategies.append(strategy)
    
    # Create retention campaign dataframe
    retention_df = pd.DataFrame(strategies)
    
    # Sort by priority and churn probability
    retention_df = retention_df.sort_values('Churn_Probability', ascending=False)
    
    # Save to CSV
    retention_df.to_csv('retention_campaign.csv', index=False)
    
    print(f"\n📋 Retention Campaign Details:")
    print(f"   Campaign File: retention_campaign.csv")
    print(f"   Total Actions: {len(retention_df)}")
    
    # Priority breakdown
    priority_counts = retention_df['Priority'].value_counts()
    print(f"\n   Priority Breakdown:")
    for priority, count in priority_counts.items():
        print(f"      {priority}: {count} customers")
    
    # Show sample strategies
    print(f"\n📝 Sample Retention Strategies (Top 5):")
    print("="*70)
    for idx, row in retention_df.head(5).iterrows():
        print(f"\nCustomer: {row['CustomerID']} | Risk: {row['Churn_Probability']} | Priority: {row['Priority']}")
        print(f"Actions: {row['Actions']}")
        print(f"Offer: {row['Discount_Offer']}")
    
    print("\n" + "="*70)
    print("✅ Retention campaign ready for execution!")
    print("   Next: Import 'retention_campaign.csv' into your CRM system")
    print("="*70)
    
    return retention_df

if __name__ == "__main__":
    strategies = generate_retention_strategies()
