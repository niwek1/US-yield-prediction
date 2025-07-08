import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import requests
import io
import warnings
from config import FRED_API_KEY

# Suppress warnings
warnings.filterwarnings("ignore", message="The use_label_encoder parameter is deprecated")
warnings.filterwarnings("ignore", message="`use_label_encoder` is deprecated")
warnings.filterwarnings("ignore", message="XGBoost is expecting ADA instead of `alpha`")
warnings.filterwarnings("ignore", message="No frequency information was provided")
warnings.filterwarnings("ignore", message="A date index must have a frequency set")

st.set_page_config(page_title="Backtesting", layout="wide")
st.title("Treasury Yield Backtesting")

# Date range selection
st.sidebar.subheader("Test Periods")
test_set_size = st.sidebar.number_input("Test Set Size (days)", 1, 500, 252)

# Date range selection
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", pd.Timestamp('2020-01-01'))
with col2:
    end_date = st.date_input("End Date", pd.Timestamp.now())

# Generate test periods using YE (Year End) instead of Y
test_periods = pd.date_range(start=start_date, end=end_date, freq='YE')
test_periods = [d.strftime('%Y-%m-%d') for d in test_periods]
if end_date not in pd.to_datetime(test_periods):
    test_periods.append(end_date.strftime('%Y-%m-%d'))

# Model parameters
st.sidebar.subheader("Model Parameters")
arima_order = st.sidebar.selectbox("ARIMA Order", [(2,0,0), (1,0,1), (2,0,1)], format_func=lambda x: f"ARIMA{x}")
base_nudge_amount = st.sidebar.slider("Base Nudge Amount", 0.001, 0.05, 0.01, 0.001)
dns_change_threshold = st.sidebar.slider("DNS Change Threshold", 0.0001, 0.01, 0.001, 0.0001)
nudge_amplification_factor = st.sidebar.slider("Nudge Amplification Factor", 1.0, 5.0, 2.5, 0.1)

# --- Data Loading Function ---
def load_data_from_fred(series_id='DGS10', start_date='2000-01-01', end_date=None):
    if end_date is None:
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    url = (f"https://api.stlouisfed.org/fred/series/observations?"
           f"series_id={series_id}&"
           f"api_key={FRED_API_KEY}&"
           f"file_type=json&"
           f"observation_start={start_date}&"
           f"observation_end={end_date}")

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()
    observations = data['observations']
    df = pd.DataFrame(observations)

    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df.set_index('date', inplace=True)
    df = df[['value']].rename(columns={'value': 'Actual 10Y Yield'})

    df.ffill(inplace=True)
    df.dropna(inplace=True)

    return df

# --- Feature Engineering Function ---
def engineer_features(df):
    initial_rows = len(df)
    df_processed = df.copy()

    df_processed['DNS Forecast'] = df_processed['Actual 10Y Yield'].shift(1) + np.random.randn(len(df_processed)) * 0.005
    df_processed['DNS Residual'] = df_processed['Actual 10Y Yield'] - df_processed['DNS Forecast']
    df_processed['Direction'] = (df_processed['Actual 10Y Yield'].diff().shift(-1) > 0).astype(int)

    num_dummy_features = 190
    dummy_features_data = np.random.rand(len(df_processed), num_dummy_features) * 10
    dummy_features_df = pd.DataFrame(dummy_features_data,
                                   index=df_processed.index,
                                   columns=[f'Feature_{i}' for i in range(1, num_dummy_features + 1)])

    df_processed = pd.concat([df_processed, dummy_features_df], axis=1)
    df_processed.dropna(inplace=True)

    feature_cols = [col for col in df_processed.columns if 'Feature_' in col]
    if 'DNS Forecast' in df_processed.columns:
        feature_cols.append('DNS Forecast')

    aligned_features = df_processed[feature_cols].copy()
    aligned_target = df_processed['Actual 10Y Yield'].copy()
    aligned_dns_forecast = df_processed['DNS Forecast'].copy()
    aligned_dns_residuals = df_processed['DNS Residual'].copy()
    aligned_directional_target = df_processed['Direction'].copy()

    common_index = aligned_features.index.intersection(aligned_target.index)\
                                  .intersection(aligned_dns_forecast.index)\
                                  .intersection(aligned_dns_residuals.index)\
                                  .intersection(aligned_directional_target.index)

    aligned_features = aligned_features.loc[common_index]
    aligned_target = aligned_target.loc[common_index]
    aligned_dns_forecast = aligned_dns_forecast.loc[common_index]
    aligned_dns_residuals = aligned_dns_residuals.loc[common_index]
    aligned_directional_target = aligned_directional_target.loc[common_index]

    return aligned_features, aligned_target, aligned_dns_forecast, aligned_dns_residuals, aligned_directional_target

# --- ARIMA Model Function ---
def train_arima_model(residuals_train, residuals_test_size, order=(2,0,0)):
    try:
        arima_model = ARIMA(residuals_train, order=order)
        arima_fit = arima_model.fit()
        arima_residual_forecasts = arima_fit.predict(start=len(residuals_train), 
                                                   end=len(residuals_train) + residuals_test_size - 1)
        return arima_residual_forecasts
    except Exception as e:
        st.error(f"Error training ARIMA model: {e}")
        return None

# --- Directional Classifier Function ---
def train_directional_classifier(X_classifier_train, y_classifier_train, X_dir_test, y_dir_test):
    try:
        estimators = [
            ('rf', RandomForestClassifier(max_depth=3, n_estimators=50, random_state=42)),
            ('xgb', XGBClassifier(learning_rate=0.05, max_depth=2, n_estimators=50, random_state=42))
        ]

        meta_learner = LogisticRegression(random_state=42, solver='liblinear')

        classifier_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('stack', StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=3,
                passthrough=False
            ))
        ])

        classifier_pipeline.fit(X_classifier_train, y_classifier_train)

        directional_probabilities = classifier_pipeline.predict_proba(X_dir_test)
        directional_predictions = classifier_pipeline.predict(X_dir_test)

        dir_confidence = np.array([
            directional_probabilities[i, pred]
            for i, pred in enumerate(directional_predictions)
        ])

        dir_test_accuracy = classifier_pipeline.score(X_dir_test, y_dir_test) * 100

        return directional_predictions, dir_confidence, dir_test_accuracy

    except Exception as e:
        st.error(f"Error training directional classifier: {e}")
        return None, None, None

# --- Forecast Generation and Evaluation ---
def generate_and_evaluate_forecasts(dns_forecast_test, actual_yield_test,
                                  arima_residual_forecasts,
                                  directional_predictions, dir_confidence,
                                  dir_test_accuracy,
                                  base_nudge_amount=0.01,
                                  dns_change_threshold=0.001,
                                  nudge_amplification_factor=2.5):
    
    if not arima_residual_forecasts.index.equals(dns_forecast_test.index):
        arima_residual_forecasts = arima_residual_forecasts.reindex(dns_forecast_test.index)
        arima_residual_forecasts.fillna(0, inplace=True)

    combined_forecast = dns_forecast_test + arima_residual_forecasts

    dns_predicted_change = dns_forecast_test.diff().fillna(0)
    dns_predicted_change_abs = dns_predicted_change.abs()

    nudge_direction = np.where(directional_predictions == 1, 1, -1)

    nudge_effect = pd.Series(0.0, index=combined_forecast.index)

    dns_predicted_change_abs_series = pd.Series(dns_predicted_change_abs, index=combined_forecast.index)
    dir_confidence_series = pd.Series(dir_confidence, index=combined_forecast.index)
    nudge_direction_series = pd.Series(nudge_direction, index=combined_forecast.index)

    for i in range(len(combined_forecast)):
        idx = combined_forecast.index[i]
        if dns_predicted_change_abs_series.loc[idx] < dns_change_threshold:
            if dir_confidence_series.loc[idx] > 0.5:
                scaled_nudge_amount = base_nudge_amount * (nudge_amplification_factor * (dir_confidence_series.loc[idx] - 0.5))
                nudge_effect.loc[idx] = scaled_nudge_amount * nudge_direction_series.loc[idx]

    combined_forecast_nudge = combined_forecast + nudge_effect

    def calculate_rmse(predictions, actuals):
        common_index = predictions.index.intersection(actuals.index)
        aligned_predictions = predictions.loc[common_index]
        aligned_actuals = actuals.loc[common_index]
        return np.sqrt(np.mean((aligned_predictions - aligned_actuals)**2))

    def calculate_mae(predictions, actuals):
        common_index = predictions.index.intersection(actuals.index)
        aligned_predictions = predictions.loc[common_index]
        aligned_actuals = actuals.loc[common_index]
        return np.mean(np.abs(aligned_predictions - aligned_actuals))

    def calculate_directional_accuracy(predictions, actuals):
        actual_direction = (actuals.diff().dropna() > 0).astype(int)
        predicted_direction = (predictions.diff().dropna() > 0).astype(int)

        common_index = actual_direction.index.intersection(predicted_direction.index)
        aligned_actual_direction = actual_direction.loc[common_index]
        aligned_predicted_direction = predicted_direction.loc[common_index]

        if len(aligned_actual_direction) == 0:
            return 0.0

        correct_predictions = (aligned_predicted_direction == aligned_actual_direction).sum()
        total_predictions = len(aligned_actual_direction)
        return (correct_predictions / total_predictions) * 100

    dns_rmse = calculate_rmse(dns_forecast_test, actual_yield_test)
    dns_mae = calculate_mae(dns_forecast_test, actual_yield_test)
    dns_dir_acc = calculate_directional_accuracy(dns_forecast_test, actual_yield_test)

    combined_rmse = calculate_rmse(combined_forecast_nudge, actual_yield_test)
    combined_mae = calculate_mae(combined_forecast_nudge, actual_yield_test)
    combined_dir_acc = calculate_directional_accuracy(combined_forecast_nudge, actual_yield_test)

    rmse_improvement = ((dns_rmse - combined_rmse) / dns_rmse) * 100 if dns_rmse != 0 else 0

    return {
        'dns_rmse': dns_rmse,
        'dns_mae': dns_mae,
        'dns_dir_acc': dns_dir_acc,
        'combined_rmse': combined_rmse,
        'combined_mae': combined_mae,
        'combined_dir_acc': combined_dir_acc,
        'dir_classifier_test_acc': dir_test_accuracy,
        'rmse_improvement': rmse_improvement
    }, actual_yield_test, dns_forecast_test, combined_forecast_nudge

# --- Plotting Function ---
def plot_forecasts(actuals, dns_forecast, combined_forecast, title_suffix=""):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(actuals.index, actuals, label='Actual 10Y Yield', color='blue', linewidth=2)
    ax.plot(dns_forecast.index, dns_forecast, label='DNS Forecast', color='orange', linestyle='--', linewidth=1.5)
    ax.plot(combined_forecast.index, combined_forecast, label='Combined Forecast', color='green', linewidth=1.5)
    ax.set_title(f'10Y Treasury Yield Forecasts {title_suffix}')
    ax.set_xlabel('Date')
    ax.set_ylabel('10Y Yield (%)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

# --- Main Backtesting Execution ---
def run_backtest():
    with st.spinner("Running backtest..."):
        results_summary_list = []
        
        # Load all data upfront
        max_date = pd.to_datetime(test_periods).max()
        min_date = pd.to_datetime(test_periods).min() - pd.DateOffset(years=5)
        
        try:
            df_raw_full = load_data_from_fred(start_date=min_date.strftime('%Y-%m-%d'),
                                            end_date=max_date.strftime('%Y-%m-%d'))
            
            if df_raw_full.empty:
                st.error("No data retrieved from FRED API. Please check your date range.")
                return
                
            st.success(f"Successfully loaded {len(df_raw_full)} observations from FRED.")
            
        except Exception as e:
            st.error(f"Error loading data from FRED: {e}")
            return

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, test_end_date_str in enumerate(test_periods):
            test_end_date = pd.to_datetime(test_end_date_str)
            status_text.text(f"Processing period {i+1}/{len(test_periods)} (Ending: {test_end_date.strftime('%Y-%m-%d')})")
            
            try:
                # Find test period indices
                test_end_idx = df_raw_full.index.get_loc(test_end_date)
                test_start_idx = test_end_idx - test_set_size + 1
                
                if test_start_idx < 0:
                    st.warning(f"Not enough data for test period ending {test_end_date.strftime('%Y-%m-%d')}. Skipping.")
                    continue
                
                # Extract test and training data
                df_raw_test_segment = df_raw_full.iloc[test_start_idx:test_end_idx + 1]
                df_raw_train_segment = df_raw_full.loc[:df_raw_test_segment.index[0] - pd.Timedelta(days=1)]
                
                if len(df_raw_train_segment) < 252:
                    st.warning(f"Training data too small for period ending {test_end_date.strftime('%Y-%m-%d')}. Skipping.")
                    continue
                
                # Feature engineering
                df_segment_for_features = df_raw_full.loc[df_raw_train_segment.index[0]:test_end_date]
                
                aligned_features, aligned_target, aligned_dns_forecast, aligned_dns_residuals, aligned_directional_target = \
                    engineer_features(df_segment_for_features.copy())
                
                # Prepare data splits
                current_test_start_date = df_raw_test_segment.index[0]
                current_test_end_date = df_raw_test_segment.index[-1]
                
                actual_test_segment = aligned_features.loc[current_test_start_date:current_test_end_date].copy()
                
                if len(actual_test_segment) < test_set_size:
                    st.warning(f"Test segment too small for period ending {test_end_date.strftime('%Y-%m-%d')}. Skipping.")
                    continue
                
                train_segment_end_date = actual_test_segment.index[0] - pd.Timedelta(days=1)
                
                # Prepare data for models
                residuals_train = aligned_dns_residuals.loc[:train_segment_end_date]
                arima_test_steps = len(actual_test_segment)
                
                X_classifier_train = aligned_features.loc[:train_segment_end_date]
                y_classifier_train = aligned_directional_target.loc[:train_segment_end_date]
                X_dir_test = actual_test_segment[X_classifier_train.columns]
                y_dir_test = aligned_directional_target.loc[actual_test_segment.index]
                
                actual_yield_test = aligned_target.loc[actual_test_segment.index]
                dns_forecast_test = aligned_dns_forecast.loc[actual_test_segment.index]
                
                # Train models and generate forecasts
                arima_residual_forecasts = train_arima_model(residuals_train, arima_test_steps, order=arima_order)
                if arima_residual_forecasts is None:
                    continue
                    
                arima_residual_forecasts.index = actual_yield_test.index
                
                directional_predictions, dir_confidence, dir_test_accuracy = \
                    train_directional_classifier(X_classifier_train, y_classifier_train, X_dir_test, y_dir_test)
                
                if directional_predictions is None:
                    continue
                
                # Generate and evaluate forecasts
                metrics, plot_actuals, plot_dns_forecast, plot_combined_forecast = \
                    generate_and_evaluate_forecasts(dns_forecast_test, actual_yield_test,
                                                  arima_residual_forecasts,
                                                  directional_predictions, dir_confidence,
                                                  dir_test_accuracy,
                                                  base_nudge_amount,
                                                  dns_change_threshold,
                                                  nudge_amplification_factor)
                
                results_summary_list.append(metrics)
                
                # Plot results
                st.pyplot(plot_forecasts(plot_actuals, plot_dns_forecast, plot_combined_forecast,
                                       title_suffix=f" (Period Ending: {test_end_date.strftime('%Y-%m-%d')})"))
                
                # Update progress
                progress_bar.progress((i + 1) / len(test_periods))
                
            except Exception as e:
                st.error(f"Error processing period ending {test_end_date.strftime('%Y-%m-%d')}: {e}")
                continue
        
        # Display final results
        if results_summary_list:
            df_results = pd.DataFrame(results_summary_list)
            
            st.subheader("Individual Period Results")
            st.dataframe(df_results)
            
            st.subheader("Average Performance Across All Periods")
            st.dataframe(df_results.mean().to_frame('Average'))
            
            # Plot summary metrics
            fig, ax = plt.subplots(figsize=(12, 6))
            metrics_to_plot = ['dns_rmse', 'combined_rmse', 'dns_dir_acc', 'combined_dir_acc']
            df_results[metrics_to_plot].mean().plot(kind='bar', ax=ax)
            ax.set_title('Average Performance Metrics')
            ax.set_ylabel('Value')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No successful test periods were processed.")

# --- Run Backtest Button ---
if st.button("Run Backtest"):
    run_backtest() 