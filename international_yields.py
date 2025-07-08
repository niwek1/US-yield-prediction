import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from config import FRED_API_KEY, FRED_SERIES

st.set_page_config(page_title="International Yields Analysis", layout="wide")
st.title("International Government Bond Yields Analysis")

# Initialize Fred object with shared API key
fred = Fred(api_key=FRED_API_KEY)

# Date range selection
st.sidebar.subheader("Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", pd.Timestamp('2010-01-01'))
with col2:
    end_date = st.date_input("End Date", pd.Timestamp.now())

# Analysis parameters
st.sidebar.subheader("Analysis Parameters")
rolling_window = st.sidebar.slider("Rolling Window Size (months)", 12, 120, 60)
var_maxlags = st.sidebar.slider("Maximum VAR Lags", 1, 24, 12)
irf_periods = st.sidebar.slider("IRF Periods", 12, 48, 24)
forecast_periods = st.sidebar.slider("Forecast Periods (months)", 1, 24, 12)

# --- Data Loading Function ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_yield_data():
    yield_data = pd.DataFrame()
    for country, series_id in FRED_SERIES.items():
        try:
            data = fred.get_series(series_id, start_date=start_date, end_date=end_date)
            if data is not None:
                yield_data[country] = data
            else:
                st.warning(f"No data found for {country} with series ID {series_id}")
        except Exception as e:
            st.error(f"Error fetching data for {country}: {e}")

    # Process data
    yield_data.index = pd.to_datetime(yield_data.index)
    yield_data = yield_data.resample('M').last()
    yield_data.fillna(method='ffill', inplace=True)
    yield_data.dropna(inplace=True)
    
    return yield_data

# --- US Yield Prediction Function ---
def predict_us_yield(yield_data, periods=12):
    us_data = yield_data['US10Y'].dropna()
    
    # Fit ARIMA model
    model = ARIMA(us_data, order=(2,1,2))
    model_fit = model.fit()
    
    # Generate forecast
    forecast = model_fit.forecast(steps=periods)
    forecast_index = pd.date_range(start=us_data.index[-1], periods=periods+1, freq='M')[1:]
    forecast_series = pd.Series(forecast, index=forecast_index)
    
    return forecast_series

# --- Main Analysis Function ---
def run_analysis():
    with st.spinner("Loading and analyzing data..."):
        # Load data
        yield_data = load_yield_data()
        
        if yield_data.empty:
            st.error("No data available for the selected period.")
            return
            
        st.success(f"Successfully loaded {len(yield_data)} months of data.")
        
        # US Yield Prediction
        st.subheader("US 10-Year Treasury Yield Prediction")
        us_forecast = predict_us_yield(yield_data, forecast_periods)
        
        # Plot US yield and forecast
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(yield_data.index, yield_data['US10Y'], label='Historical', color='blue')
        ax.plot(us_forecast.index, us_forecast, label='Forecast', color='red', linestyle='--')
        ax.fill_between(us_forecast.index, 
                       us_forecast - 0.5, 
                       us_forecast + 0.5, 
                       color='red', alpha=0.1)
        ax.set_title('US 10-Year Treasury Yield: Historical and Forecast')
        ax.set_xlabel('Date')
        ax.set_ylabel('Yield (%)')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display forecast values
        st.write("Forecast Values:")
        st.dataframe(us_forecast.to_frame('Forecasted Yield'))
        
        # Display data summary
        st.subheader("Data Summary")
        st.dataframe(yield_data.describe())
        
        # Correlation Analysis
        st.subheader("Correlation Analysis")
        correlation_matrix = yield_data.corr()
        
        # Plot correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title('Historical Correlation of 10-Year Government Bond Yields')
        st.pyplot(fig)
        
        # Rolling Correlations
        st.subheader("Rolling Correlations")
        countries = list(yield_data.columns)
        
        for country1 in countries:
            fig, ax = plt.subplots(figsize=(12, 6))
            for country2 in countries:
                if country1 != country2:
                    rolling_corr = yield_data[country1].rolling(window=rolling_window).corr(yield_data[country2])
                    ax.plot(rolling_corr.index, rolling_corr, label=f'{country2}')
            
            ax.set_title(f'Rolling {rolling_window}-Month Correlation of {country1} with Other Countries')
            ax.set_xlabel('Date')
            ax.set_ylabel('Correlation')
            ax.legend(title=f'Correlated with {country1}', loc='center left', bbox_to_anchor=(1, 0.5))
            ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig)
        
        # VAR Analysis
        st.subheader("Vector Autoregression (VAR) Analysis")
        
        # Stationarity Tests
        st.write("Augmented Dickey-Fuller Test Results (Original Series)")
        adf_results = []
        for col in yield_data.columns:
            result = adfuller(yield_data[col].dropna())
            adf_results.append({
                'Country': col,
                'ADF Statistic': result[0],
                'p-value': result[1],
                'Stationary': result[1] <= 0.05
            })
        
        st.dataframe(pd.DataFrame(adf_results))
        
        # Difference the data
        yield_diff = yield_data.diff().dropna()
        
        if not yield_diff.empty:
            # VAR Model
            model = VAR(yield_diff)
            lag_selection_results = model.select_order(maxlags=var_maxlags)
            optimal_lags = lag_selection_results.bic
            
            st.write(f"Optimal lag order (BIC): {optimal_lags}")
            
            if optimal_lags is not None and optimal_lags > 0:
                # Fit VAR model
                var_model = model.fit(optimal_lags)
                
                # Impulse Response Functions
                st.subheader("Impulse Response Functions")
                irf = var_model.irf(periods=irf_periods)
                
                # Plot IRFs
                fig = irf.plot(figsize=(15, 10), orth=True)
                plt.suptitle('Orthogonalized Impulse Response Functions', y=1.02)
                plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                st.pyplot(fig)
                
                # Individual country shock plots
                var_names = var_model.model.data.ynames
                nvar = len(var_names)
                
                for shock_country in var_names:
                    shock_index = var_names.index(shock_country)
                    
                    fig, axes = plt.subplots(nrows=nvar, ncols=1, figsize=(10, 2 * nvar), sharex=True)
                    fig.suptitle(f'Orthogonalized Impulse Response to {shock_country} Shock', y=1.02, fontsize=16)
                    
                    for i, response_country in enumerate(var_names):
                        response_index = i
                        response_data = irf.orth_irfs[:, response_index, shock_index]
                        
                        ax = axes[i]
                        ax.plot(range(irf_periods), response_data, color='blue')
                        ax.axhline(0, color='grey', linestyle='--')
                        ax.set_title(f'Response of {response_country}', loc='left', fontsize=12)
                        ax.grid(True, linestyle=':', alpha=0.7)
                        
                        try:
                            if hasattr(irf, 'orth_lower') and hasattr(irf, 'orth_upper'):
                                lower_bound = irf.orth_lower[:, response_index, shock_index]
                                upper_bound = irf.orth_upper[:, response_index, shock_index]
                                ax.fill_between(range(irf_periods), lower_bound, upper_bound, color='blue', alpha=0.2)
                        except Exception as e:
                            st.warning(f"Could not plot confidence intervals for {shock_country} on {response_country}")
                    
                    plt.xlabel('Periods')
                    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                    st.pyplot(fig)
            else:
                st.warning("Optimal lag order is 0 or None. VAR model cannot be reliably fitted.")
        else:
            st.error("Differenced data is empty. Cannot perform VAR analysis.")

# --- Run Analysis Button ---
if st.button("Run Analysis"):
    run_analysis() 