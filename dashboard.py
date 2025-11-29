# dashboard.py - Interactive web dashboard
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

st.title("📊 Telecom Customer Churn Prediction Dashboard")

# Load data
@st.cache_data
def load_data():
    predictions = pd.read_csv('predictions.csv')
    customers = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    if 'customerID' in customers.columns:
        customers = customers.rename(columns={'customerID': 'CustomerID'})
    return predictions.merge(customers, on='CustomerID', how='left')

df = load_data()

# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", f"{len(df):,}")
with col2:
    churn_rate = (df['Churn_Prediction'] == 'Will Churn').mean() * 100
    st.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
with col3:
    high_risk = (df['Risk_Level'] == 'High Risk').sum()
    st.metric("High Risk Customers", f"{high_risk:,}")
with col4:
    if 'MonthlyCharges' in df.columns:
        revenue_at_risk = df[df['Risk_Level'] == 'High Risk']['MonthlyCharges'].sum() * 12
        st.metric("Annual Revenue at Risk", f"${revenue_at_risk:,.0f}")

# Visualizations
st.subheader("📈 Risk Distribution")

col1, col2 = st.columns(2)

with col1:
    # Risk level pie chart
    risk_counts = df['Risk_Level'].value_counts()
    fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                 title="Customer Risk Segmentation",
                 color_discrete_map={'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Churn probability distribution
    fig = px.histogram(df, x='Churn_Probability', nbins=50,
                      title="Churn Probability Distribution",
                      labels={'Churn_Probability': 'Churn Probability'})
    st.plotly_chart(fig, use_container_width=True)

# High-risk customers table
st.subheader("🚨 High Risk Customers (Top 20)")
high_risk_df = df[df['Risk_Level'] == 'High Risk'].sort_values('Churn_Probability', ascending=False).head(20)
st.dataframe(high_risk_df[['CustomerID', 'Churn_Probability', 'Contract', 'tenure', 'MonthlyCharges']], 
             use_container_width=True)

# Download button
st.download_button(
    label="📥 Download Full Report",
    data=df.to_csv(index=False),
    file_name="churn_predictions_full.csv",
    mime="text/csv"
)
