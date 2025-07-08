import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import fredapi as fa
import os

# Set page config
st.set_page_config(
    page_title="US Treasury Yield Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("US Treasury Yield Predictor")
st.markdown("""
This application predicts the 10-Year US Treasury Yield using machine learning models.
The model takes into account various economic indicators and market data to make predictions.
""")

# Sidebar for FRED API key input
with st.sidebar:
    st.header("Configuration")
    fred_api_key = st.text_input("FRED API Key", type="password")
    if not fred_api_key:
        st.warning("Please enter your FRED API key to fetch latest data")
    else:
        st.success("FRED API key configured!")

# Function to fetch historical data
def fetch_historical_data(api_key, days=365):
    try:
        fred = fa.Fred(api_key=api_key)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Define series to fetch
        series_dict = {
            'DGS3': '3_year_yield',
            'DGS5': '5_year_yield',
            'DGS7': '7_year_yield',
            'DGS10': '10_year_yield',
            'DGS20': '20_year_yield',
            'DGS2': '2_year_yield',
            'CPILFESL': 'core_cpi',
            'WALCL': 'fed_assets',
            'DTWEXBGS': 'usd_index',
            'FEDFUNDS': 'fed_funds_rate'
        }
        
        # Fetch historical data
        historical_data = {}
        for series_id, series_name in series_dict.items():
            data = fred.get_series(series_id, start_date, end_date)
            if not data.empty:
                historical_data[series_name] = data
        
        return pd.DataFrame(historical_data)
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return None

# Function to fetch latest data
def fetch_latest_data(api_key):
    try:
        fred = fa.Fred(api_key=api_key)
        
        # Define series to fetch
        series_dict = {
            'DGS3': '3_year_yield',
            'DGS5': '5_year_yield',
            'DGS7': '7_year_yield',
            'DGS10': '10_year_yield',
            'DGS20': '20_year_yield',
            'DGS2': '2_year_yield',
            'CPILFESL': 'core_cpi',
            'WALCL': 'fed_assets',
            'DTWEXBGS': 'usd_index',
            'FEDFUNDS': 'fed_funds_rate'
        }
        
        # Fetch latest data
        latest_data = {}
        for series_id, series_name in series_dict.items():
            data = fred.get_series(series_id, limit=1)
            if not data.empty:
                latest_data[series_name] = data.iloc[-1]
        
        return pd.Series(latest_data)
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Function to create features
def create_features_from_latest(latest_data):
    # Create a DataFrame with the latest data
    df = pd.DataFrame([latest_data])
    
    # Calculate spreads
    df['spread_2y_10y'] = df['10_year_yield'] - df['2_year_yield']
    df['spread_10y_3y'] = df['10_year_yield'] - df['3_year_yield']
    df['spread_20y_10y'] = df['20_year_yield'] - df['10_year_yield']
    df['spread_20y_3y'] = df['20_year_yield'] - df['3_year_yield']
    
    # Calculate YoY change for fed assets
    df['fed_assets_yoy'] = df['fed_assets'].pct_change() * 100
    
    return df

# Function to plot yield curve
def plot_yield_curve(latest_data):
    tenors = ['2_year_yield', '3_year_yield', '5_year_yield', '7_year_yield', '10_year_yield', '20_year_yield']
    yields = [latest_data[tenor] for tenor in tenors]
    years = [2, 3, 5, 7, 10, 20]
    
    plt.figure(figsize=(10, 6))
    plt.plot(years, yields, 'b-o', linewidth=2)
    plt.title('Current Yield Curve')
    plt.xlabel('Tenor (Years)')
    plt.ylabel('Yield (%)')
    plt.grid(True)
    return plt

# Function to plot historical yields
def plot_historical_yields(historical_data):
    plt.figure(figsize=(12, 6))
    for col in ['2_year_yield', '10_year_yield', '20_year_yield']:
        plt.plot(historical_data.index, historical_data[col], label=col.replace('_yield', 'Y'))
    plt.title('Historical Treasury Yields')
    plt.xlabel('Date')
    plt.ylabel('Yield (%)')
    plt.legend()
    plt.grid(True)
    return plt

# Function to plot feature importance
def plot_feature_importance(model):
    if hasattr(model, 'named_estimators_') and 'xgb' in model.named_estimators_:
        xgb_model = model.named_estimators_['xgb']
        importance = xgb_model.get_booster().get_score(importance_type='gain')
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
        plt.figure(figsize=(10, 6))
        plt.barh(list(importance.keys()), list(importance.values()))
        plt.title('Top 10 Feature Importance')
        plt.xlabel('Gain')
        plt.gca().invert_yaxis()
        return plt
    return None

# Main content
if fred_api_key:
    # Fetch latest and historical data
    latest_data = fetch_latest_data(fred_api_key)
    historical_data = fetch_historical_data(fred_api_key)
    
    if latest_data is not None and historical_data is not None:
        # Display current market data
        st.header("Current Market Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("10-Year Yield", f"{latest_data['10_year_yield']:.2f}%")
            st.metric("2-Year Yield", f"{latest_data['2_year_yield']:.2f}%")
        
        with col2:
            st.metric("Fed Funds Rate", f"{latest_data['fed_funds_rate']:.2f}%")
            st.metric("Core CPI", f"{latest_data['core_cpi']:.2f}")
        
        with col3:
            st.metric("USD Index", f"{latest_data['usd_index']:.2f}")
            st.metric("Fed Assets", f"{latest_data['fed_assets']:.2f}")
        
        # Create features for prediction
        features_df = create_features_from_latest(latest_data)
        
        # Load model and make prediction
        try:
            model = joblib.load('treasury_yield_ensemble_model.pkl')
            prediction = model.predict(features_df)[0]
            
            # Display prediction
            st.header("Yield Prediction")
            st.markdown(f"""
            ### Predicted 10-Year Treasury Yield for tomorrow:
            # <div style='text-align: center; font-size: 48px; color: #1f77b4;'>{prediction:.2f}%</div>
            """, unsafe_allow_html=True)
            
            # Display prediction details
            st.subheader("Prediction Details")
            current_yield = latest_data['10_year_yield']
            change = prediction - current_yield
            direction = "↑" if change > 0 else "↓"
            color = "green" if change > 0 else "red"
            
            st.markdown(f"""
            - Current Yield: {current_yield:.2f}%
            - Predicted Change: <span style='color: {color}'>{direction} {abs(change):.2f}%</span>
            - Prediction Date: {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}
            """, unsafe_allow_html=True)
            
            # Visualizations
            st.header("Market Visualizations")
            
            # Yield Curve
            st.subheader("Current Yield Curve")
            yield_curve = plot_yield_curve(latest_data)
            st.pyplot(yield_curve)
            plt.close()
            
            # Historical Yields
            st.subheader("Historical Yields")
            hist_yields = plot_historical_yields(historical_data)
            st.pyplot(hist_yields)
            plt.close()
            
            # Feature Importance
            st.subheader("Feature Importance")
            feature_importance = plot_feature_importance(model)
            if feature_importance:
                st.pyplot(feature_importance)
                plt.close()
            
            # Correlation Analysis
            st.subheader("Correlation Analysis")
            yield_cols = ['2_year_yield', '3_year_yield', '5_year_yield', '7_year_yield', '10_year_yield', '20_year_yield']
            corr_matrix = historical_data[yield_cols].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Yield Correlations')
            st.pyplot(plt)
            plt.close()
            
        except Exception as e:
            st.error(f"Error loading model or making prediction: {str(e)}")
            st.info("Please ensure the model file 'treasury_yield_ensemble_model.pkl' is in the same directory as this app.")
    
    # Model Performance Section
    st.header("Model Performance")
    st.markdown("""
    The model's performance metrics:
    - Training RMSE: 0.0413
    - Test RMSE: 0.1208
    - Training Directional Accuracy: 64.95%
    - Test Directional Accuracy: 48.33%
    """)
    
    # Disclaimer
    st.markdown("""
    ---
    ### Disclaimer
    This prediction model is for informational purposes only and should not be used as financial advice.
    Past performance is not indicative of future results.
    """)

else:
    st.info("Please enter your FRED API key in the sidebar to start making predictions.") 