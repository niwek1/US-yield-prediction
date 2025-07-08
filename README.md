# US Treasury Yield Predictor

This application predicts the 10-Year US Treasury Yield using machine learning models trained on historical data and various economic indicators.

## Features

- Real-time data fetching from FRED (Federal Reserve Economic Data)
- Current market data visualization
- Next-day yield predictions
- Model performance metrics
- Interactive user interface

## Prerequisites

- Python 3.8 or higher
- FRED API key (get one at https://fred.stlouisfed.org/docs/api/api_key.html)

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Enter your FRED API key in the sidebar
3. View current market data and predictions

## Model Details

The prediction model uses an ensemble of XGBoost and Random Forest algorithms, trained on:
- Treasury yields (2Y, 3Y, 5Y, 7Y, 10Y, 20Y)
- Federal Funds Rate
- Core CPI
- Federal Reserve Assets
- USD Index
- Various yield spreads and derived features

## Performance Metrics

- Training RMSE: 0.0413
- Test RMSE: 0.1208
- Training Directional Accuracy: 64.95%
- Test Directional Accuracy: 48.33%

## Disclaimer

This prediction model is for informational purposes only and should not be used as financial advice. Past performance is not indicative of future results. 