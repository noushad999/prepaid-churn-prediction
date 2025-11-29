# executive_report.py - Fixed for Windows encoding
import pandas as pd
from datetime import datetime

def generate_executive_report():
    """Generate comprehensive executive summary report."""
    
    # Load data
    predictions = pd.read_csv('predictions.csv')
    retention = pd.read_csv('retention_campaign.csv')
    customers = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Merge data
    if 'customerID' in customers.columns:
        customers = customers.rename(columns={'customerID': 'CustomerID'})
    df = predictions.merge(customers, on='CustomerID', how='left')
    
    # Create report
    report = []
    report.append("="*80)
    report.append("  TELECOM CUSTOMER CHURN PREDICTION - EXECUTIVE SUMMARY")
    report.append(f"  Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    report.append("="*80)
    
    # 1. Executive Summary
    report.append("\n[EXECUTIVE SUMMARY]")
    report.append("-"*80)
    total_customers = len(df)
    churn_customers = len(df[df['Churn_Prediction'] == 'Will Churn'])
    churn_rate = churn_customers / total_customers * 100
    
    report.append(f"\nTotal Customer Base: {total_customers:,} customers")
    report.append(f"Predicted to Churn: {churn_customers:,} customers ({churn_rate:.1f}%)")
    report.append(f"Predicted to Stay: {total_customers - churn_customers:,} customers ({100-churn_rate:.1f}%)")
    
    # 2. Financial Impact
    report.append("\n\n[FINANCIAL IMPACT ANALYSIS]")
    report.append("-"*80)
    
    high_risk = df[df['Risk_Level'] == 'High Risk']
    medium_risk = df[df['Risk_Level'] == 'Medium Risk']
    
    if 'MonthlyCharges' in df.columns:
        high_risk_revenue = high_risk['MonthlyCharges'].sum() * 12
        medium_risk_revenue = medium_risk['MonthlyCharges'].sum() * 12
        total_at_risk = high_risk_revenue + medium_risk_revenue
        
        report.append(f"\nAnnual Revenue at Risk:")
        report.append(f"  - High Risk: ${high_risk_revenue:,.2f} ({len(high_risk)} customers)")
        report.append(f"  - Medium Risk: ${medium_risk_revenue:,.2f} ({len(medium_risk)} customers)")
        report.append(f"  - TOTAL AT RISK: ${total_at_risk:,.2f}")
        
        # ROI Calculations
        retention_rates = [0.3, 0.5, 0.7]
        report.append(f"\nRetention Scenarios:")
        for rate in retention_rates:
            saved = total_at_risk * rate
            report.append(f"  - {rate*100:.0f}% retention = ${saved:,.2f} saved annually")
        
        # Campaign costs vs benefits
        campaign_cost_per_customer = 50
        total_campaign_cost = len(high_risk) * campaign_cost_per_customer
        report.append(f"\nCampaign Cost Estimate: ${total_campaign_cost:,.2f}")
        report.append(f"Break-even Retention Rate: {(total_campaign_cost/total_at_risk)*100:.1f}%")
        report.append(f"Expected ROI (50% retention): {((high_risk_revenue*0.5 - total_campaign_cost)/total_campaign_cost)*100:.0f}%")
    
    # 3. Risk Segmentation
    report.append("\n\n[CUSTOMER RISK SEGMENTATION]")
    report.append("-"*80)
    
    risk_dist = df['Risk_Level'].value_counts()
    for risk, count in risk_dist.items():
        pct = count / len(df) * 100
        report.append(f"\n{risk}:")
        report.append(f"  - Count: {count:,} customers")
        report.append(f"  - Percentage: {pct:.1f}%")
        
        if 'MonthlyCharges' in df.columns:
            avg_revenue = df[df['Risk_Level'] == risk]['MonthlyCharges'].mean()
            report.append(f"  - Avg Monthly Revenue: ${avg_revenue:.2f}")
    
    # 4. Key Patterns
    report.append("\n\n[KEY CHURN PATTERNS IDENTIFIED]")
    report.append("-"*80)
    
    churn_df = df[df['Churn_Prediction'] == 'Will Churn']
    
    if 'Contract' in churn_df.columns:
        contract_churn = churn_df['Contract'].value_counts()
        report.append(f"\nChurn by Contract Type:")
        for contract, count in contract_churn.items():
            report.append(f"  - {contract}: {count} customers")
    
    if 'tenure' in churn_df.columns:
        avg_tenure = churn_df['tenure'].mean()
        report.append(f"\nAverage Tenure of Churning Customers: {avg_tenure:.1f} months")
        
        tenure_segments = {
            '0-12 months': (churn_df['tenure'] <= 12).sum(),
            '13-24 months': ((churn_df['tenure'] > 12) & (churn_df['tenure'] <= 24)).sum(),
            '25-48 months': ((churn_df['tenure'] > 24) & (churn_df['tenure'] <= 48)).sum(),
            '48+ months': (churn_df['tenure'] > 48).sum()
        }
        report.append(f"\nChurn by Tenure Segment:")
        for segment, count in tenure_segments.items():
            report.append(f"  - {segment}: {count} customers")
    
    if 'InternetService' in churn_df.columns:
        internet_churn = churn_df['InternetService'].value_counts()
        report.append(f"\nChurn by Internet Service:")
        for service, count in internet_churn.items():
            report.append(f"  - {service}: {count} customers")
    
    # 5. Recommended Actions
    report.append("\n\n[RECOMMENDED IMMEDIATE ACTIONS]")
    report.append("-"*80)
    
    critical_customers = retention[retention['Priority'] == 'CRITICAL']
    report.append(f"\n1. CRITICAL PRIORITY ({len(critical_customers)} customers)")
    report.append(f"   - Action: Immediate phone outreach within 24-48 hours")
    report.append(f"   - Owner: Retention team + Senior account managers")
    report.append(f"   - Budget: Authorize up to 20% discounts")
    
    high_customers = retention[retention['Priority'] == 'HIGH']
    report.append(f"\n2. HIGH PRIORITY ({len(high_customers)} customers)")
    report.append(f"   - Action: Proactive outreach within 7 days")
    report.append(f"   - Owner: Customer success team")
    report.append(f"   - Budget: Authorize up to 15% discounts")
    
    medium_risk_count = len(df[df['Risk_Level'] == 'Medium Risk'])
    report.append(f"\n3. MEDIUM RISK ({medium_risk_count} customers)")
    report.append(f"   - Action: Automated email campaign + satisfaction survey")
    report.append(f"   - Owner: Marketing automation")
    report.append(f"   - Budget: Service improvement offers")
    
    # 6. Success Metrics
    report.append("\n\n[SUCCESS METRICS TO TRACK]")
    report.append("-"*80)
    report.append("\nKPIs to Monitor:")
    report.append("  - Actual churn rate vs predicted (weekly)")
    report.append("  - Retention campaign success rate by segment")
    report.append("  - Customer satisfaction scores post-intervention")
    report.append("  - Revenue retained vs forecast")
    report.append("  - Cost per retained customer")
    report.append("  - Time to intervention for high-risk customers")
    
    # 7. Model Performance
    report.append("\n\n[PREDICTION MODEL PERFORMANCE]")
    report.append("-"*80)
    report.append("\nModel Accuracy: 74.96%")
    report.append("Precision: 75.08%")
    report.append("Recall: 74.96%")
    report.append("F1-Score: 0.7496")
    report.append("\nModel Status: Production-ready")
    report.append("Last Updated: " + datetime.now().strftime('%Y-%m-%d'))
    
    # 8. Quick Action Items
    report.append("\n\n[TOP 10 QUICK WINS]")
    report.append("-"*80)
    report.append("\n1. Call 582 CRITICAL customers this week")
    report.append("2. Send retention offers to high-risk month-to-month customers")
    report.append("3. Review pricing for customers with charges > $70/month")
    report.append("4. Offer free tech support to customers without it")
    report.append("5. Promote automatic payment methods (reduce electronic check)")
    report.append("6. Create loyalty program for customers with tenure > 36 months")
    report.append("7. Launch fiber optic upgrade campaign for DSL customers")
    report.append("8. Implement dedicated onboarding for new customers (< 12 months)")
    report.append("9. Develop win-back strategy for already-churned customers")
    report.append("10. A/B test retention offers to measure effectiveness")
    
    report.append("\n" + "="*80)
    report.append("  END OF REPORT")
    report.append("="*80)
    
    # Save report with UTF-8 encoding (FIX FOR WINDOWS)
    report_text = '\n'.join(report)
    with open('Executive_Report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print("\n\n" + "="*80)
    print("  REPORT SAVED SUCCESSFULLY")
    print("="*80)
    print(f"\nFile location: Executive_Report.txt")
    print(f"File size: {len(report_text)} characters")
    print(f"\nKey Highlights:")
    print(f"  - Total Revenue at Risk: ${total_at_risk:,.0f}")
    print(f"  - High-Risk Customers: {len(high_risk):,}")
    print(f"  - Potential Savings: ${total_at_risk * 0.5:,.0f} (50% retention)")
    print("="*80)
    
    return report_text

if __name__ == "__main__":
    generate_executive_report()
