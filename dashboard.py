import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.stats import linregress

# Page Config
st.set_page_config(
    page_title="Maternal Health Dashboard",
    page_icon="🤱",
    layout="wide"
)

# Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load('best_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    history = pd.read_csv('engineered_maternal_mortality.csv')
    history = history.sort_values('year').reset_index(drop=True)
    return model, feature_names, history

model, feature_names, historical_df = load_assets()

# Helper Functions (Same logic as app.py)
def calculate_slope(series):
    if len(series.dropna()) < 3:
        return np.nan
    valid_series = series.dropna()
    x = np.arange(len(valid_series))
    y = valid_series.values
    slope, _, _, _, _ = linregress(x, y)
    return slope

def predict_mmr(year, sba, anc, health_spend):
    # Input DataFrame
    input_data = pd.DataFrame({
        'year': [year],
        'MMR': [np.nan],
        'skilled_birth_attendance': [sba],
        'antenatal_care_coverage': [anc],
        'health_spending': [health_spend]
    })
    
    # Combine & Feature Engineer
    combined_df = pd.concat([historical_df, input_data], ignore_index=True)
    combined_df = combined_df.sort_values('year').reset_index(drop=True)
    
    # Lag Features
    combined_df['mmr_lag_1'] = combined_df['MMR'].shift(1)
    combined_df['mmr_lag_2'] = combined_df['MMR'].shift(2)
    combined_df['mmr_3yr_avg'] = combined_df['MMR'].rolling(window=3).mean()
    combined_df['mmr_5yr_avg'] = combined_df['MMR'].rolling(window=5).mean()
    
    # Slope
    combined_df['trend_slope'] = combined_df['MMR'].rolling(window=5, min_periods=3).apply(lambda x: calculate_slope(x), raw=False)
    combined_df['risk_flag'] = 0
    
    # Extract prediction row
    new_row = combined_df[combined_df['year'] == year]
    features = new_row[feature_names]
    
    if features[['mmr_lag_1', 'mmr_lag_2']].isnull().any().any():
        return None, "Insufficient data for this year "
        
    prediction = float(model.predict(features)[0])
    return prediction, None

# --- UI Layout ---

st.title("🤱 Maternal Mortality Prediction Dashboard")
st.markdown("Forecasting maternal health outcomes using AI and socioeconomic indicators.")

# Sidebar Controls
with st.sidebar:
    st.header("Prediction Parameters")
    
    input_year = st.slider("Target Year", min_value=2024, max_value=2050, value=2025)
    
    st.subheader("Health Indicators")
    input_sba = st.slider("Skilled Birth Attendance (%)", 0.0, 100.0, 80.0, help="Percentage of births attended by skilled health personnel")
    input_anc = st.slider("Antenatal Care Coverage (%)", 0.0, 100.0, 75.0, help="Percentage of women receiving antenatal care")
    input_health_spend = st.number_input("Health Spending (USD)", min_value=0.0, value=100.0, step=10.0)
    
    if st.button("Generate Forecast", type="primary"):
        run_prediction = True
    else:
        run_prediction = True # Auto-run on load

# Main Dashboard
col1, col2 = st.columns([2, 1])

if run_prediction:
    prediction, error = predict_mmr(input_year, input_sba, input_anc, input_health_spend)
    
    if error:
        st.error(error)
    else:
        # Determine Risk
        if prediction > 1000:
            risk_level = "High Risk"
            risk_color = "red"
            risk_icon = "🚨"
        elif prediction > 500:
            risk_level = "Medium Risk"
            risk_color = "orange"
            risk_icon = "⚠️"
        else:
            risk_level = "Low Risk"
            risk_color = "green"
            risk_icon = "✅"

        with col1:
            st.subheader("📉 Historical Trends & Forecast")
            
            # Prepare chart data
            chart_data = historical_df[['year', 'MMR']].dropna()
            chart_data['Type'] = 'Historical'
            
            # Add prediction point
            pred_row = pd.DataFrame({
                'year': [input_year], 
                'MMR': [prediction], 
                'Type': ['Forecast']
            })
            
            # Combine for plotting
            plot_df = pd.concat([chart_data, pred_row], ignore_index=True)
            
            # Line Chart
            st.line_chart(plot_df, x='year', y='MMR', color='Type')
            
            st.info(f"Visualizing data from 2000 to {input_year}. The orange line/point represents your forecasted scenario.")

        with col2:
            st.subheader("Prediction Results")
            
            st.metric(label=f"Projected MMR for {input_year}", value=f"{prediction:.2f}")
            
            st.markdown(f"""
            <div style="
                padding: 15px; 
                border-radius: 10px; 
                background-color: rgba(255, 255, 255, 0.1); 
                border-left: 5px solid {risk_color};
                margin-top: 10px;">
                <h3 style="margin:0; color: {risk_color};">{risk_icon} {risk_level}</h3>
                <p style="margin:5px 0 0 0;">Requires immediate attention.</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Feature Contribution Analysis"):
                st.write("Impact of your inputs:")
                st.write(f"- **SBA**: {input_sba}%")
                st.write(f"- **ANC**: {input_anc}%")
                st.write(f"- **Spending**: ${input_health_spend}")
