import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def _convert_to_json_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {key: _convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return str(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    else:
        return obj


# Import our data pipeline
from data_collector import fetch_wti_futures, fetch_brent_futures, fetch_usd_index, collect_all_data
from feature_engineering import OilFeatureEngineer
from streamlit_calendar import calendar
from eda import (
    plot_correlation_matrix,
    plot_scatter,
    adf_test,
    plot_rolling_stats,
    plot_acf_pacf,
    plot_returns_heatmap
)
from arima_forecasting import ARIMAForecaster
from garch_forecasting import GARCHForecaster
from hedge_ratios import HedgeRatioCalculator
from backtesting import DynamicHedgingBacktest
from robustness_checks import RobustnessChecker
from reporting import OilHedgingReporter

# Page configuration
st.set_page_config(
    page_title="oil futures hedging dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Clean Professional Color Palette */
    :root {
        --background: #fbfcf8;      /* Clean light background */
        --primary: #e26d5c;         /* Green for titles and buttons */
        --text: #000000;            /* Black text */
        --white: #ffffff;           /* White */
    }
    
    /* Apply San Francisco font and background to everything */
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }
    
    /* Main background - multiple selectors to ensure coverage */
    .main .block-container {
        background-color: var(--background) !important;
    }
    
    /* Streamlit main background */
    .stApp {
        background-color: var(--background) !important;
    }
    
    /* Main content area */
    .main {
        background-color: var(--background) !important;
    }
    
    /* Block container */
    .block-container {
        background-color: var(--background) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--background) !important;
    }
    
    /* Sidebar background */
    .css-1lcbmhc {
        background-color: var(--background) !important;
    }
    
    /* Additional sidebar selectors */
    [data-testid="stSidebar"] {
        background-color: var(--background) !important;
    }
    
    /* Body background */
    body {
        background-color: var(--background) !important;
    }
    
    /* Additional Streamlit elements */
    .stMarkdown {
        background-color: var(--background) !important;
    }
    
    /* Report view */
    .reportview-container {
        background-color: var(--background) !important;
    }
    
    /* Main report view */
    .reportview-container .main .block-container {
        background-color: var(--background) !important;
    }
    
    /* Force background on all elements */
    div {
        background-color: var(--background) !important;
    }
    
    /* Override any dark themes */
    .stApp[data-theme="dark"] {
        background-color: var(--background) !important;
    }
    
    /* Main header styling */
    h1.main-header {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: var(--primary) !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Section headers */
    h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        color: var(--primary) !important;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: var(--white) !important;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid var(--primary);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar headers */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: var(--primary);
        margin-bottom: 1rem;
    }
    
    /* Streamlit widgets styling */
    .stSelectbox > div > div {
        background-color: var(--white) !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    .stButton > button {
        background-color: var(--primary) !important;
        color: #fbfcf8 !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
    }
    /* Ensure button text is always the correct color and no background */
    .stButton > button *,
    .stButton > button > div,
    .stButton > button > span,
    .stButton > button > div > span,
    .stButton > button > div > div,
    .stButton > button > div > div > span {
        color: #fbfcf8 !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    /* Override any Streamlit default button styling */
    button[data-testid="baseButton-primary"] {
        background-color: var(--primary) !important;
        color: #fbfcf8 !important;
    }
    button[data-testid="baseButton-primary"] *,
    button[data-testid="baseButton-primary"] > span,
    button[data-testid="baseButton-primary"] > div,
    button[data-testid="baseButton-primary"] > div > span {
        color: #fbfcf8 !important;
        background: transparent !important;
        box-shadow: none !important;
    }
    
    .stButton > button:hover {
        background-color: #5a8f6f !important;
    }
    

    
    /* Metrics styling */
    .stMetric > div {
        background-color: var(--white) !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Chart styling */
    .js-plotly-plot {
        background-color: var(--white) !important;
    }
    
    /* Dataframe styling */
    
    /* Calendar/Date Input Styling - Aggressive overrides */
    /* Target the date input container */
    div[data-testid="stDateInput"] {
        background-color: #fbfcf8 !important;
    }
    
    /* Target the calendar popup */
    div[data-baseweb="calendar"] {
        background-color: #fbfcf8 !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
        padding: 8px !important;
    }
    
    /* Calendar header (month/year) */
    div[data-baseweb="calendar"] div[role="presentation"] div {
        background-color: #fbfcf8 !important;
        color: #000000 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }
    
    /* Calendar day buttons */
    div[data-baseweb="calendar"] button {
        background-color: #fbfcf8 !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        transition: all 0.2s !important;
    }
    
    /* Selected day and today */
    div[data-baseweb="calendar"] button[aria-selected="true"],
    div[data-baseweb="calendar"] button[aria-label*="today"] {
        background: #e26d5c !important;
        color: #fbfcf8 !important;
        font-weight: 600 !important;
        border-radius: 50% !important;
        border: 2px solid #e26d5c !important;
    }
    div[data-baseweb="calendar"] button[aria-selected="true"] span,
    div[data-baseweb="calendar"] button[aria-label*="today"] span {
        color: #fbfcf8 !important;
    }
    
    /* Navigation buttons (Previous/Next) */
    div[data-baseweb="calendar"] button[aria-label*="Previous"],
    div[data-baseweb="calendar"] button[aria-label*="Next"] {
        background-color: #000000 !important;
        color: #fbfcf8 !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        width: 28px !important;
        height: 28px !important;
    }
    
    /* Hover effect for day buttons */
    div[data-baseweb="calendar"] button:hover:not([aria-selected="true"]) {
        background-color: #e8f5e8 !important;
        color: #000000 !important;
    }
    
    /* Outside month days */
    div[data-baseweb="calendar"] button[aria-label*="outside"] {
        color: #cccccc !important;
        background-color: #fbfcf8 !important;
    }
    
    /* Weekday headers */
    div[data-baseweb="calendar"] div[role="grid"] div[role="row"]:first-child button {
        background-color: #fbfcf8 !important;
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 12px !important;
    }
    
    /* Additional calendar selectors for maximum coverage */
    [data-baseweb="calendar"] {
        background-color: #fbfcf8 !important;
    }
    
    [data-baseweb="calendar"] * {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
    }
    .stDataFrame {
        background-color: var(--white) !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Text elements */
    p, span, div {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
        color: var(--text) !important;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: rgba(102, 161, 130, 0.1) !important;
        border-left: 4px solid var(--primary) !important;
    }
    
    /* Error messages */
    .stError {
        background-color: rgba(220, 53, 69, 0.1) !important;
        border-left: 4px solid #dc3545 !important;
    }
    
    /* Info messages */
    .stInfo {
        background-color: rgba(102, 161, 130, 0.1) !important;
        border-left: 4px solid var(--primary) !important;
    }

    /* DataFrame/Table text black */
    .stDataFrame, .stTable, .stDataFrame * {
        color: #000 !important;
        background: #fff !important;
    }

    /* Metric number: set background to white, make text black */
    .stMetric .stMetricValue, .stMetric [data-testid="stMetricValue"] {
        background: #ffffff !important;
        color: #000 !important;
        border-radius: 0.5rem !important;
        padding: 0.25em 0.75em !important;
        box-shadow: none !important;
    }
    
    /* Additional selectors to ensure white background on metric numbers */
    .stMetric [data-testid="stMetricValue"] *,
    .stMetric .stMetricValue *,
    .stMetric [data-testid="stMetricValue"] > div,
    .stMetric .stMetricValue > div {
        background: #ffffff !important;
        color: #000 !important;
    }
    
    /* Delta (change indicator) styling */
    .stMetric [data-testid="stMetricDelta"] {
        background: #ffffff !important;
        border-radius: 0.5rem !important;
        padding: 0.25em 0.75em !important;
        box-shadow: none !important;
    }
    /* Up delta: arrow and all text green */
    .stMetric [data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Up"] {
        color: #66A182 !important;
    }
    .stMetric [data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Up"]) span,
    .stMetric [data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Up"]) div {
        color: #66A182 !important;
    }
    /* Down delta: arrow and all text red */
    .stMetric [data-testid="stMetricDelta"] svg[data-testid="stMetricDeltaIcon-Down"] {
        color: #dc3545 !important;
    }
    .stMetric [data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Down"]) span,
    .stMetric [data-testid="stMetricDelta"]:has(svg[data-testid="stMetricDeltaIcon-Down"]) div {
        color: #dc3545 !important;
    }

    /* Download button styling */
    .download-btn {
        display: inline-block;
        background: #e26d5c;
        color: #fbfcf8;
        padding: 8px 16px;
        text-decoration: none;
        border-radius: 6px;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 14px;
        transition: background-color 0.2s;
    }
    
    .download-btn:hover {
        background: #5a8f6f;
        color: #fbfcf8;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Force date input text color to black in sidebar */
    [data-testid="stSidebar"] input[type="text"] {
        color: #000 !important;
        font-weight: 500 !important;
        background: #fff !important;
        caret-color: #000 !important;
    }
    [data-testid="stSidebar"] label {
        color: #000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Black background, white text, and black grid lines for Streamlit dataframes */
    .stDataFrame table {
        background: #000 !important;
        color: #fff !important;
        border: 1px solid #000 !important;
        border-collapse: collapse !important;
    }
    .stDataFrame th, .stDataFrame td {
        background: #000 !important;
        color: #fff !important;
        border: 1px solid #fff !important;
    }
    .stDataFrame thead th {
        border-bottom: 2px solid #fff !important;
    }
    .stDataFrame tbody tr {
        border-top: 1px solid #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


class OilDataVisualizer:
    def __init__(self):
        self.data = {}
        
    def load_processed_data(self):
        """Load processed data from feature engineering pipeline."""
        try:
            # Load engineered features
            features = pd.read_csv('data/processed/engineered_features.csv')
            features['Date'] = pd.to_datetime(features['Date'])
            features.set_index('Date', inplace=True)
            
            # Load raw data for comparison
            wti_raw = pd.read_csv('data/raw/wti_futures.csv')
            wti_raw['Date'] = pd.to_datetime(wti_raw['Date'])
            wti_raw.set_index('Date', inplace=True)
            
            brent_raw = pd.read_csv('data/raw/brent_futures.csv')
            brent_raw['Date'] = pd.to_datetime(brent_raw['Date'])
            brent_raw.set_index('Date', inplace=True)
            
            usd_raw = pd.read_csv('data/raw/usd_index.csv')
            usd_raw['Date'] = pd.to_datetime(usd_raw['Date'])
            usd_raw.set_index('Date', inplace=True)
            
            return features, wti_raw, brent_raw, usd_raw
            
        except Exception as e:
            st.error(f"Error loading processed data: {e}")
            return None, None, None, None
    
    def fetch_data_direct(self, symbol, start_date, end_date):
        """Fetch data directly for quick preview."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                st.error(f"No data found for {symbol}")
                return None
                
            data = data.reset_index()
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock_Splits']
            return data
            
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def create_price_chart(self, data, symbol):
        """Create an interactive price chart with candlesticks and moving averages."""
        fig = go.Figure()
        
        # Handle date column - check if it exists, otherwise use index
        if 'Date' in data.columns.tolist():
            x_data = data['Date']
        elif 'date' in data.columns.tolist():
            x_data = data['date']
        elif isinstance(data.index, pd.DatetimeIndex):
            x_data = data.index
        else:
            # If no date column found, create a simple range
            x_data = range(len(data))
        
        # Candlestick chart with default green/red
        fig.add_trace(go.Candlestick(
            x=x_data,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=f'{symbol} Price',
            increasing_line_color='#e26d5c',  # Green for gains
            decreasing_line_color='#dc3545'   # Red for losses
        ))
        # Moving averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        color_sma20 = '#1f77b4'  # Blue
        color_sma50 = '#ff7f0e'  # Orange
        fig.add_trace(go.Scatter(
            x=x_data, y=data['SMA_20'], mode='lines', name='20-Day MA', line=dict(color=color_sma20, width=2)
        ))
        fig.add_trace(go.Scatter(
            x=x_data, y=data['SMA_50'], mode='lines', name='50-Day MA', line=dict(color=color_sma50, width=2)
        ))
        # Optionally add Bollinger Bands for more color
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        color_bb = '#9467bd'  # Purple
        fig.add_trace(go.Scatter(
            x=x_data, y=data['BB_Upper'], mode='lines', name='BB Upper', line=dict(color=color_bb, dash='dash'), showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=x_data, y=data['BB_Lower'], mode='lines', name='BB Lower', line=dict(color=color_bb, dash='dash'), fill='tonexty', fillcolor='rgba(148,103,189,0.1)', showlegend=True
        ))
        fig.update_layout(
            title=f'{symbol} Price Chart with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            showlegend=True,
            template='plotly_white',
            plot_bgcolor='#fbfcf8',  # Clean background
            paper_bgcolor='#fbfcf8',
            font=dict(color='#000000'),  # Black text
            legend=dict(font=dict(color='#000'), bgcolor='#fbfcf8'),
        )
        fig.update_xaxes(color='#000', tickfont=dict(color='#000'), title_font=dict(color='#000'))
        fig.update_yaxes(color='#000', tickfont=dict(color='#000'), title_font=dict(color='#000'))
        return fig
    
    def create_returns_chart(self, data, symbol):
        """Create returns distribution and time series charts."""
        # Calculate returns
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Handle date column - check if it exists, otherwise use index
        if 'Date' in data.columns.tolist():
            x_data = data['Date']
        elif 'date' in data.columns.tolist():
            x_data = data['date']
        elif isinstance(data.index, pd.DatetimeIndex):
            x_data = data.index
        else:
            # If no date column found, create a simple range
            x_data = range(len(data))
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Returns', 'Returns Distribution', 'Cumulative Returns', 'Volatility'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Daily returns
        fig.add_trace(
            go.Scatter(x=x_data, y=data['Returns'], mode='lines', name='Daily Returns'),
            row=1, col=1
        )
        
        # Returns distribution
        fig.add_trace(
            go.Histogram(x=data['Returns'].dropna(), nbinsx=50, name='Returns Distribution'),
            row=1, col=2
        )
        
        # Cumulative returns
        cumulative_returns = (1 + data['Returns']).cumprod()
        fig.add_trace(
            go.Scatter(x=x_data, y=cumulative_returns, mode='lines', name='Cumulative Returns'),
            row=2, col=1
        )
        
        # Rolling volatility
        volatility = data['Returns'].rolling(window=20).std() * np.sqrt(252)
        fig.add_trace(
            go.Scatter(x=x_data, y=volatility, mode='lines', name='20-Day Volatility'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f'{symbol} Returns Analysis',
            height=600,
            showlegend=True,
            template='plotly_white',
            plot_bgcolor='#fbfcf8',  # Clean background
            paper_bgcolor='#fbfcf8',
            font=dict(color='#000000'),  # Black text
            legend=dict(font=dict(color='#000'), bgcolor='#fbfcf8'),
        )
        # Set all axes and subplot titles to black
        fig.update_xaxes(color='#000', tickfont=dict(color='#000'), title_font=dict(color='#000'))
        fig.update_yaxes(color='#000', tickfont=dict(color='#000'), title_font=dict(color='#000'))
        for i in range(1, 5):
            fig['layout'][f'annotations'][i-1]['font'] = dict(color='#000')
        return fig
    
    def create_technical_indicators(self, data, symbol):
        """Create technical indicators chart."""
        # Calculate technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Handle date column - check if it exists, otherwise use index
        if 'Date' in data.columns.tolist():
            x_data = data['Date']
        elif 'date' in data.columns.tolist():
            x_data = data['date']
        elif isinstance(data.index, pd.DatetimeIndex):
            x_data = data.index
        else:
            # If no date column found, create a simple range
            x_data = range(len(data))
        
        # Color palette
        color_price = '#e26d5c'
        color_sma20 = '#1f77b4'
        color_sma50 = '#ff7f0e'
        color_bb = '#9467bd'
        color_macd = '#d62728'
        color_macd_signal = '#2ca02c'
        color_rsi = '#8c564b'
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price with Moving Averages', 'MACD', 'RSI'),
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2]
        )
        # Price and moving averages
        fig.add_trace(
            go.Scatter(x=x_data, y=data['Close'], mode='lines', name='Price', line=dict(color=color_price)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_data, y=data['SMA_20'], mode='lines', name='SMA 20', line=dict(color=color_sma20)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_data, y=data['SMA_50'], mode='lines', name='SMA 50', line=dict(color=color_sma50)),
            row=1, col=1
        )
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=x_data, y=data['BB_Upper'], mode='lines', name='BB Upper', line=dict(color=color_bb, dash='dash'), showlegend=True),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_data, y=data['BB_Lower'], mode='lines', name='BB Lower', line=dict(color=color_bb, dash='dash'), fill='tonexty', fillcolor='rgba(148,103,189,0.1)', showlegend=True),
            row=1, col=1
        )
        # MACD
        fig.add_trace(
            go.Scatter(x=x_data, y=data['MACD'], mode='lines', name='MACD', line=dict(color=color_macd)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=x_data, y=data['MACD_Signal'], mode='lines', name='MACD Signal', line=dict(color=color_macd_signal)),
            row=2, col=1
        )
        # RSI
        fig.add_trace(
            go.Scatter(x=x_data, y=data['RSI'], mode='lines', name='RSI', line=dict(color=color_rsi)),
            row=3, col=1
        )
        # Add horizontal lines for RSI overbought/oversold levels
        fig.add_hline(y=70, line_dash="dash", line_color="#dc3545")  # Red
        fig.add_hline(y=30, line_dash="dash", line_color="#e26d5c")  # Green
        fig.update_layout(
            title=f'{symbol} Technical Indicators',
            height=700,
            showlegend=True,
            template='plotly_white',
            plot_bgcolor='#fbfcf8',  # Clean background
            paper_bgcolor='#fbfcf8',
            font=dict(color='#000000'),  # Black text
            legend=dict(font=dict(color='#000'), bgcolor='#fbfcf8'),
        )
        fig.update_xaxes(color='#000', tickfont=dict(color='#000'), title_font=dict(color='#000'))
        fig.update_yaxes(color='#000', tickfont=dict(color='#000'), title_font=dict(color='#000'))
        for i in range(1, 4):
            fig['layout'][f'annotations'][i-1]['font'] = dict(color='#000')
        return fig
    
    def create_feature_analysis(self, features):
        """Create feature analysis charts from engineered features."""
        if features is None or features.empty:
            return None
            
        # Select key features for visualization
        key_features = [
            'wti_close', 'wti_returns', 'wti_volatility_20', 'wti_rsi',
            'brent_close', 'brent_returns', 'brent_volatility_20',
            'wti_brent_spread', 'usd_index', 'usd_returns'
        ]
        
        available_features = [col for col in key_features if col in features.columns]
        
        if not available_features:
            return None
            
        # Create subplots
        n_features = len(available_features)
        n_cols = 2
        n_rows = (n_features + 1) // 2
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=available_features,
            shared_xaxes=True
        )
        
        for i, feature in enumerate(available_features):
            row = (i // n_cols) + 1
            col = (i % n_cols) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=features.index, 
                    y=features[feature], 
                    mode='lines', 
                    name=feature
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Engineered Features Analysis',
            height=300 * n_rows,
            showlegend=False,
            template='plotly_white',
            plot_bgcolor='#fbfcf8',  # Clean background
            paper_bgcolor='#fbfcf8',
            font=dict(color='#000000'),  # Black text
            xaxis=dict(color='#000', tickfont=dict(color='#000'), title=dict(font=dict(color='#000'))),
            yaxis=dict(color='#000', tickfont=dict(color='#000'), title=dict(font=dict(color='#000'))),
            legend_font_color='#000'
        )
        
        return fig
    
    def create_spread_analysis(self, wti_data, brent_data):
        """Create WTI-Brent spread analysis."""
        if wti_data is None or brent_data is None:
            return None
            
        # Handle date column for merging - check if it exists, otherwise use index
        wti_copy = wti_data.copy()
        brent_copy = brent_data.copy()
        
        # If no Date column, try to create one from index
        if 'Date' not in wti_copy.columns.tolist():
            if isinstance(wti_copy.index, pd.DatetimeIndex):
                wti_copy['Date'] = wti_copy.index
            else:
                wti_copy['Date'] = range(len(wti_copy))
                
        if 'Date' not in brent_copy.columns.tolist():
            if isinstance(brent_copy.index, pd.DatetimeIndex):
                brent_copy['Date'] = brent_copy.index
            else:
                brent_copy['Date'] = range(len(brent_copy))
        
        # Align data by date
        merged_data = pd.merge(wti_copy, brent_copy, on='Date', suffixes=('_WTI', '_Brent'))
        
        # Calculate spread
        merged_data['Spread'] = merged_data['Close_WTI'] - merged_data['Close_Brent']
        merged_data['Spread_Pct'] = (merged_data['Close_WTI'] - merged_data['Close_Brent']) / merged_data['Close_Brent']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('WTI vs Brent Prices', 'WTI-Brent Spread', 'Spread Distribution', 'Spread Over Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # WTI vs Brent prices
        fig.add_trace(
            go.Scatter(x=merged_data['Date'], y=merged_data['Close_WTI'], mode='lines', name='WTI'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=merged_data['Date'], y=merged_data['Close_Brent'], mode='lines', name='Brent'),
            row=1, col=1
        )
        
        # Spread
        fig.add_trace(
            go.Scatter(x=merged_data['Date'], y=merged_data['Spread'], mode='lines', name='WTI-Brent Spread'),
            row=1, col=2
        )
        
        # Spread distribution
        fig.add_trace(
            go.Histogram(x=merged_data['Spread'].dropna(), nbinsx=50, name='Spread Distribution'),
            row=2, col=1
        )
        
        # Spread percentage over time
        fig.add_trace(
            go.Scatter(x=merged_data['Date'], y=merged_data['Spread_Pct'], mode='lines', name='Spread %'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='WTI-Brent Spread Analysis',
            height=600,
            showlegend=True,
            template='plotly_white',
            plot_bgcolor='#fbfcf8',  # Clean background
            paper_bgcolor='#fbfcf8',
            font=dict(color='#000000'),  # Black text
            xaxis=dict(color='#000', tickfont=dict(color='#000'), title=dict(font=dict(color='#000'))),
            yaxis=dict(color='#000', tickfont=dict(color='#000'), title=dict(font=dict(color='#000'))),
            legend_font_color='#000'
        )
        
        return fig

def main():
    # Header
    st.markdown('<h1 class="main-header"> oil futures hedging dashboard</h1>', unsafe_allow_html=True)
    
    # Instructions and symbol explanations
    st.markdown("""
    <div style="background-color: #ffffff; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #e26d5c; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 2rem;">
        <h3 style="color: #e26d5c; margin-bottom: 1rem;">How to Start</h3>
        <p style="margin-bottom: 1rem; line-height: 1.6;">
            Welcome to the Oil Futures Hedging Dashboard! This tool provides the analysis of oil futures data 
            for hedging strategies. Follow these steps to begin your analysis:
        </p>
        <ol style="margin-bottom: 1rem; line-height: 1.6;">
            <li><strong>Select a Symbol:</strong> Choose from WTI Crude Oil, Brent Crude Oil, or USD Index.</li>
            <li><strong>Set Date Range:</strong> Pick your analysis period using the date selectors.</li>
            <li><strong>Load Data:</strong> Click "Load Data & Analyze" to fetch and process your data.</li>
            <li><strong>Explore Results:</strong> View real-time analysis, technical indicators, and hedging insights.</li>
        </ol>
        
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown('<div class="sidebar-header">Data Analysis Settings</div>', unsafe_allow_html=True)
    
    # Symbol selection and date range
    st.sidebar.header("Select Symbol and Date Range")
    symbol_options = {
        'WTI Crude Oil': 'CL=F',
        'Brent Crude Oil': 'BZ=F',
        'USD Index': 'DX-Y.NYB'
    }
    
    selected_symbol_name = st.sidebar.selectbox("Symbol", list(symbol_options.keys()))
    selected_symbol = symbol_options[selected_symbol_name]

    # Date range selection
    st.sidebar.markdown("**Select Start and End Date:**")
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=365),
        max_value=datetime.now()
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.now(),
        min_value=start_date,
        max_value=datetime.now()
    )
    
    # Load data button
    if st.sidebar.button('Load Data & Analyze', type='primary'):
        with st.spinner('Loading data and preparing analysis...'):
            try:
                # Load real-time data
                visualizer = OilDataVisualizer()
                realtime_data = yf.Ticker(selected_symbol).history(start=start_date, end=end_date)
                
                if realtime_data.empty or 'Close' not in realtime_data.columns or realtime_data['Close'].isnull().all():
                    st.error("No valid real-time data found for selected symbol and date range.")
                    return
                
                # Store real-time data
                st.session_state['realtime_data'] = realtime_data
                st.session_state['symbol'] = selected_symbol
                st.session_state['symbol_name'] = selected_symbol_name
                
                # Load processed data
                features, wti_raw, brent_raw, usd_raw = visualizer.load_processed_data()
                
                if features is not None:
                    st.session_state['processed_data'] = {
                        'features': features,
                        'wti_raw': wti_raw,
                        'brent_raw': brent_raw,
                        'usd_raw': usd_raw
                    }
                    st.success(f'Data loaded successfully! Real-time data for {selected_symbol_name} and processed data available.')
                else:
                    st.warning('Real-time data loaded, but no processed data found.')
                    
            except Exception as e:
                st.error(f'Error loading data: {e}')
    
    # GitHub Repository Link
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
<div style="
    text-align: left;
    padding: 1rem 0.5rem;
    background-color: #fbfcf8;
    border-radius: 0.5rem;
">
    <a href="https://github.com/valeriacartagena/quant-hedging-app" 
       target="_blank" 
       style="
           display: inline-flex;
           align-items: center;
           justify-content: flex-start;
           background-color: #333;
           color: white;
           text-decoration: none;
           padding: 0.75rem 1rem;
           border-radius: 0.5rem;
           font-weight: 500;
           transition: background-color 0.3s ease;
           box-shadow: 0 2px 4px rgba(0,0,0,0.1);
       "
       onmouseover="this.style.backgroundColor='#555'"
       onmouseout="this.style.backgroundColor='#333'">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" style="margin-right: 0.5rem;">
            <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 
                     8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416
                     -.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729
                     1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 
                     3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334
                     -5.467-5.931 0-1.311.469-2.381 1.236-3.221
                     -.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 
                     1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 
                     3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 
                     2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 
                     5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 
                     .319.192.694.801.576 4.765-1.589 8.199-6.086 
                     8.199-11.386 0-6.627-5.373-12-12-12z"/>
        </svg>
        View on GitHub
    </a>
</div>
""", unsafe_allow_html=True)
    
    # Main content area
    if 'realtime_data' in st.session_state:
        # Show real-time data first
        data = st.session_state['realtime_data']
        symbol = st.session_state['symbol']
        symbol_name = st.session_state['symbol_name']
        
        visualizer = OilDataVisualizer()
        
        # Data overview
        st.markdown('## real-time data overview')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Current Price",
                value=f"${data['Close'].iloc[-1]:.2f}",
                delta=f"{data['Close'].pct_change().iloc[-1]*100:.2f}%"
            )
        
        with col2:
            st.metric(
                label="Total Return",
                value=f"{((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100:.2f}%"
            )
        
        with col3:
            volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
            st.metric(
                label="Annualized Volatility",
                value=f"{volatility:.2f}%"
            )
        
        with col4:
            st.metric(
                label="Data Points",
                value=f"{len(data):,}"
            )
        
        # Charts
        st.markdown('## price analysis')
        
        # Price chart
        price_fig = visualizer.create_price_chart(data, symbol_name)
        st.plotly_chart(price_fig, use_container_width=True)
        
        # Returns analysis
        st.markdown('## returns analysis')
        returns_fig = visualizer.create_returns_chart(data, symbol_name)
        st.plotly_chart(returns_fig, use_container_width=True)
        
        # Technical indicators
        st.markdown('## technical indicators')
        tech_fig = visualizer.create_technical_indicators(data, symbol_name)
        st.plotly_chart(tech_fig, use_container_width=True)
        
        # Data download button
        import base64
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'''
            <a href="data:file/csv;base64,{b64}" download="raw_data.csv"
               style="
                   display: inline-block;
                   background-color: #e26d5c;
                   color: #fbfcf8;
                   font-size: 1.1rem;
                   padding: 0.5rem 1.5rem;
                   border-radius: 0.5rem;
                   text-decoration: none;
                   text-align: center;
                   margin-top: 1rem;
                   box-shadow: 0 2px 4px rgba(0,0,0,0.05);
               ">
               download raw data
            </a>
        '''
        st.markdown(href, unsafe_allow_html=True)
        
        # Show processed data if available
        if 'processed_data' in st.session_state:
            st.markdown('---')
            st.markdown('## processed data analysis')
            
            # Show processed data from pipeline
            processed_data = st.session_state['processed_data']
            features = processed_data['features']
            wti_raw = processed_data['wti_raw']
            brent_raw = processed_data['brent_raw']
            usd_raw = processed_data['usd_raw']
            
            # Data overview
            st.markdown('### processed data overview')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Features Created",
                    value=f"{features.shape[1]:,}"
                )
            
            with col2:
                st.metric(
                    label="Data Points",
                    value=f"{features.shape[0]:,}"
                )
            
            with col3:
                st.metric(
                    label="Date Range",
                    value=f"{features.index.min().strftime('%Y-%m-%d')} to {features.index.max().strftime('%Y-%m-%d')}"
                )
            
            with col4:
                st.metric(
                    label="Data Sources",
                    value="WTI, Brent, USD"
                )
            
            # Feature analysis
            st.markdown('### engineered features analysis')
            feature_fig = visualizer.create_feature_analysis(features)
            if feature_fig is not None:
                st.plotly_chart(feature_fig, use_container_width=True)
            
            # Raw data charts
            st.markdown('### raw data analysis')
            
            # WTI analysis
            if wti_raw is not None:
                st.markdown('#### WTI crude oil')
                wti_data = wti_raw.reset_index()
                price_fig = visualizer.create_price_chart(wti_data, 'WTI')
                st.plotly_chart(price_fig, use_container_width=True)
            
            # Spread analysis
            if wti_raw is not None and brent_raw is not None:
                st.markdown('#### WTI-brent spread analysis')
                wti_data = wti_raw.reset_index()
                brent_data = brent_raw.reset_index()
                spread_fig = visualizer.create_spread_analysis(wti_data, brent_data)
                if spread_fig is not None:
                    st.plotly_chart(spread_fig, use_container_width=True)
            
            # Download button for engineered features
            import base64
            csv = features.to_csv(index=True)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'''
                <a href="data:file/csv;base64,{b64}" download="engineered_features.csv"
                   style="
                       display: inline-block;
                       background-color: #e26d5c;
                       color: #fbfcf8;
                       font-size: 1.1rem;
                       padding: 0.5rem 1.5rem;
                       border-radius: 0.5rem;
                       text-decoration: none;
                       text-align: center;
                       margin-top: 1rem;
                       box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                   ">
                   download engineered features
                </a>
            '''
            st.markdown(href, unsafe_allow_html=True)

            # --- EDA SECTION ---
            st.header("exploratory data analysis")
            eda_option = st.selectbox(
                "Choose EDA Visualization",
                [
                    "Correlation Matrix",
                    "Scatter Plot",
                    "ADF Test",
                    "Rolling Mean & STD",
                    "ACF & PACF",
                    "Returns Heatmap"
                ]
            )
            if eda_option == "Correlation Matrix":
                fig = plot_correlation_matrix(features)
                st.pyplot(fig)
            elif eda_option == "Scatter Plot":
                x = st.selectbox("X Variable", features.columns)
                y = st.selectbox("Y Variable", features.columns)
                fig = plot_scatter(features, x, y)
                st.pyplot(fig)
            elif eda_option == "ADF Test":
                col = st.selectbox("Column for ADF Test", features.columns)
                st.write("ADF Test Results:")
                adf_test(features[col])
            elif eda_option == "Rolling Mean & STD":
                col = st.selectbox("Column for Rolling Stats", features.columns)
                fig = plot_rolling_stats(features[col])
                st.pyplot(fig)
            elif eda_option == "ACF & PACF":
                col = st.selectbox("Column for ACF/PACF", features.columns)
                fig = plot_acf_pacf(features[col])
                st.pyplot(fig)
            elif eda_option == "Returns Heatmap":
                fig = plot_returns_heatmap(features)
                st.pyplot(fig)
            # --- END EDA SECTION ---
            
            # --- ARIMA FORECASTING SECTION ---
            st.header("ARIMA forecasting")
            st.markdown("### WTI log returns forecasting")
            
            # Get WTI data for forecasting
            if wti_raw is not None:
                # Prepare log returns
                forecaster = ARIMAForecaster()
                log_returns = forecaster.prepare_data(wti_raw, 'Close')
                
                # Display log returns info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Return", f"{log_returns.mean():.4f}")
                with col2:
                    st.metric("Volatility", f"{log_returns.std():.4f}")
                with col3:
                    st.metric("Data Points", f"{len(log_returns):,}")
                
                # Stationarity test
                st.markdown("#### stationarity test")
                stationarity_result = forecaster.check_stationarity(log_returns, "WTI Log Returns")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ADF Statistic", f"{stationarity_result['adf_statistic']:.4f}")
                with col2:
                    st.metric("P-Value", f"{stationarity_result['p_value']:.4f}")
                with col3:
                    if stationarity_result['is_stationary']:
                        st.metric("Status", f"stationary")
                    else:
                        st.metric("Status", f"non-stationary")
                
                # Model parameters
                st.markdown("#### model configuration")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    auto_select = st.checkbox("Auto-select ARIMA order", value=True)
                
                with col2:
                    if not auto_select:
                        p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1)
                        d = st.number_input("Differencing (d)", min_value=0, max_value=2, value=0)
                        q = st.number_input("MA order (q)", min_value=0, max_value=5, value=1)
                        order = (p, d, q)
                    else:
                        order = None
                
                with col3:
                    forecast_steps = st.number_input("Forecast Steps", min_value=5, max_value=100, value=30)
                
                # Fit model button
                if st.button("fit ARIMA model", type="primary"):
                    with st.spinner("Fitting ARIMA model..."):
                        try:
                            # Fit the model
                            model_info = forecaster.fit_arima(log_returns, order=order, auto_select=auto_select)
                            
                            # Display model summary with auto-selected order
                            st.markdown("#### model summary")
                            
                            # Prominently display the selected ARIMA order
                            order = model_info['order']
                            p, d, q = order
                            
                            st.markdown(f"""
                            **Selected ARIMA Order: ({p}, {d}, {q})**
                            
                            - **p = {p}**: Autoregressive order (number of lagged observations)
                            - **d = {d}**: Differencing order (number of times series was differenced)
                            - **q = {q}**: Moving average order (number of lagged forecast errors)
                            """)
                            
                            if auto_select:
                                st.success(f"(✓) **Auto-selected using AIC optimization** - This order minimizes the Akaike Information Criterion")
                            else:
                                st.info(f"ⓘ **Manually specified** - Order ({p}, {d}, {q}) was set by user")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("AIC", f"{model_info['aic']:.2f}")
                            with col2:
                                st.metric("BIC", f"{model_info['bic']:.2f}")
                            with col3:
                                st.metric("HQIC", f"{model_info['hqic']:.2f}")
                            with col4:
                                st.metric("Model Order", f"({p}, {d}, {q})")
                            
                            # Model parameters
                            st.markdown("#### Model Parameters")
                            params_df = pd.DataFrame({
                                'Parameter': model_info['params'].index,
                                'Value': model_info['params'].values,
                                'P-Value': model_info['pvalues'].values
                            })
                            
                            # Download button for model parameters
                            import base64
                            csv_params = params_df.to_csv(index=False)
                            b64_params = base64.b64encode(csv_params.encode()).decode()
                            href = f'''
                                <a href="data:file/csv;base64,{b64_params}" download="arima_model_parameters_{p}_{d}_{q}.csv"
                                   style="
                                       display: inline-block;
                                       background-color: #e26d5c;
                                       color: #fff;
                                       font-size: 1.1rem;
                                       padding: 0.5rem 1.5rem;
                                       border-radius: 0.5rem;
                                       text-decoration: none;
                                       text-align: center;
                                       margin-top: 1rem;
                                       box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                   ">
                                   download model parameters
                                </a>
                            '''
                            st.markdown(href, unsafe_allow_html=True)
                            
                            # Order selection explanation
                            if auto_select:
                                st.markdown("#### Order Selection Process")
                                st.markdown(f"""
                                **Grid Search Results:**
                                - **Search Range**: p ∈ [0, 5], d ∈ [0, 2], q ∈ [0, 5]
                                - **Criterion**: Akaike Information Criterion (AIC)
                                - **Best Order**: ({p}, {d}, {q}) with AIC = {model_info['aic']:.2f}
                                
                                **Interpretation:**
                                - **p = {p}**: The model uses {p} previous log return values
                                - **d = {d}**: The series was differenced {d} time(s) to achieve stationarity
                                - **q = {q}**: The model uses {q} previous forecast errors
                                """)
                                
                                if d == 0:
                                    st.info("ⓘ **No differencing needed** - The log returns series is already stationary")
                                elif d == 1:
                                    st.info("ⓘ **First differencing applied** - Series was differenced once to achieve stationarity")
                                else:
                                    st.info(f"ⓘ **{d}th differencing applied** - Series was differenced {d} times to achieve stationarity")
                            
                            # Generate forecast
                            forecast_results = forecaster.forecast(steps=forecast_steps)
                            
                            # Display forecast
                            st.markdown("#### 30-Step Ahead Forecast")
                            
                            # Plot forecast
                            fig = forecaster.plot_forecast(log_returns, forecast_results, "WTI Log Returns Forecast")
                            st.pyplot(fig)
                            
                            # Forecast metrics
                            st.markdown("#### Forecast Statistics")
                            forecast = forecast_results['forecast']
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean Forecast", f"{forecast.mean():.4f}")
                            with col2:
                                st.metric("Forecast Std", f"{forecast.std():.4f}")
                            with col3:
                                st.metric("Min Forecast", f"{forecast.min():.4f}")
                            with col4:
                                st.metric("Max Forecast", f"{forecast.max():.4f}")
                            
                            # Model evaluation (in-sample)
                            st.markdown("#### Model Evaluation (In-Sample)")
                            
                            # Calculate in-sample predictions
                            fitted_values = model_info['fitted_values']
                            actual_values = log_returns[fitted_values.index]
                            
                            # Calculate metrics
                            mse = mean_squared_error(actual_values, fitted_values)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(actual_values, fitted_values)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("RMSE", f"{rmse:.6f}")
                            with col2:
                                st.metric("MAE", f"{mae:.6f}")
                            with col3:
                                st.metric("R²", f"{1 - mse/actual_values.var():.4f}")
                            
                            # Naïve model comparison
                            st.markdown("#### Naïve Model Comparison")
                            
                            # Naïve model: yesterday's return = today's prediction
                            naive_predictions = log_returns.shift(1).dropna()
                            naive_actual = log_returns[naive_predictions.index]
                            
                            naive_mse = mean_squared_error(naive_actual, naive_predictions)
                            naive_rmse = np.sqrt(naive_mse)
                            naive_mae = mean_absolute_error(naive_actual, naive_predictions)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("ARIMA RMSE", f"{rmse:.6f}", delta=f"{((naive_rmse - rmse) / naive_rmse * 100):.1f}%")
                            with col2:
                                st.metric("Naïve RMSE", f"{naive_rmse:.6f}")
                            with col3:
                                st.metric("ARIMA MAE", f"{mae:.6f}", delta=f"{((naive_mae - mae) / naive_mae * 100):.1f}%")
                            with col4:
                                st.metric("Naïve MAE", f"{naive_mae:.6f}")
                            
                            # Model diagnostics
                            st.markdown("#### Model Diagnostics")
                            fig = forecaster.plot_residuals()
                            st.pyplot(fig)
                            
                            # Residuals statistics
                            residuals_stats = model_info['residuals_stats']
                            st.markdown("#### Residuals Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean", f"{residuals_stats['mean']:.6f}")
                            with col2:
                                st.metric("Std", f"{residuals_stats['std']:.6f}")
                            with col3:
                                st.metric("Skewness", f"{residuals_stats['skewness']:.4f}")
                            with col4:
                                st.metric("Kurtosis", f"{residuals_stats['kurtosis']:.4f}")
                            
                            # Residuals autocorrelation test
                            st.markdown("#### Residuals Autocorrelation Test")
                            from statsmodels.stats.diagnostic import acorr_ljungbox
                            
                            residuals = model_info['resid'].dropna()
                            lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Ljung-Box Statistic", f"{lb_test['lb_stat'].iloc[-1]:.4f}")
                            with col2:
                                st.metric("P-Value", f"{lb_test['lb_pvalue'].iloc[-1]:.4f}")
                            
                            # Interpretation
                            if lb_test['lb_pvalue'].iloc[-1] > 0.05:
                                st.success("(✓) Residuals show no significant autocorrelation (good model fit)")
                            else:
                                st.warning("⚠ Residuals show significant autocorrelation (model may need improvement)")
                            
                            # Store results in session state
                            st.session_state['arima_results'] = {
                                'forecaster': forecaster,
                                'log_returns': log_returns,
                                'model_info': model_info,
                                'forecast_results': forecast_results,
                                'evaluation': {
                                    'rmse': rmse,
                                    'mae': mae,
                                    'naive_rmse': naive_rmse,
                                    'naive_mae': naive_mae
                                }
                            }
                            
                            # Show forecast values table
                            st.markdown("#### Forecast Values")
                            forecast_df = pd.DataFrame({
                                'Date': forecast.index,
                                'Forecast': forecast.values,
                                'Lower CI': forecast_results['conf_int'].iloc[:, 0] if forecast_results['conf_int'] is not None else None,
                                'Upper CI': forecast_results['conf_int'].iloc[:, 1] if forecast_results['conf_int'] is not None else None
                            })
                            
                            # Download button for forecast values
                            import base64
                            csv_forecast = forecast_df.to_csv(index=False)
                            b64_forecast = base64.b64encode(csv_forecast.encode()).decode()
                            href = f'''
                                <a href="data:file/csv;base64,{b64_forecast}" download="arima_forecast_values_{p}_{d}_{q}.csv"
                                   style="
                                       display: inline-block;
                                       background-color: #e26d5c;
                                       color: #fff;
                                       font-size: 1.1rem;
                                       padding: 0.5rem 1.5rem;
                                       border-radius: 0.5rem;
                                       text-decoration: none;
                                       text-align: center;
                                       margin-top: 1rem;
                                       box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                   ">
                                   download forecast values
                                </a>
                            '''
                            st.markdown(href, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error fitting ARIMA model: {str(e)}")
                            st.error(f"Full error: {e}")
                    
                    # Show ACF/PACF plots for model selection
                    if st.checkbox("Show ACF/PACF plots for model selection"):
                        fig = forecaster.plot_acf_pacf(log_returns)
                        st.pyplot(fig)
            
            # --- END ARIMA FORECASTING SECTION ---
            
            # --- GARCH VOLATILITY FORECASTING SECTION ---
            st.header("GARCH volatility forecasting")
            st.markdown("### WTI volatility modeling")
            
            # Get WTI data for volatility forecasting
            if wti_raw is not None:
                # Prepare log returns for GARCH
                garch_forecaster = GARCHForecaster()
                garch_returns = garch_forecaster.prepare_data(wti_raw, 'Close')
                
                # Display returns info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Return", f"{garch_returns.mean():.4f}")
                with col2:
                    st.metric("Volatility", f"{garch_returns.std():.4f}")
                with col3:
                    st.metric("Data Points", f"{len(garch_returns):,}")
                
                # Volatility clustering test
                st.markdown("#### volatility clustering test")
                clustering_result = garch_forecaster.check_volatility_clustering(garch_returns)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Squared Returns Autocorr", f"{clustering_result['autocorrelation']:.4f}")
                with col2:
                    st.metric("Ljung-Box P-Value", f"{clustering_result['ljung_box_pvalue']:.4f}")
                with col3:
                    if clustering_result['has_clustering']:
                        st.metric("Status", "volatility clustering detected")
                    else:
                        st.metric("Status", "no clustering detected")
                
                # Model configuration
                st.markdown("#### model configuration")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    model_type = st.selectbox("GARCH Model Type", ["GARCH", "EGARCH", "GJR-GARCH"])
                
                with col2:
                    auto_select_garch = st.checkbox("Auto-select GARCH order", value=True)
                
                with col3:
                    if not auto_select_garch:
                        p_garch = st.number_input("GARCH order (p)", min_value=1, max_value=3, value=1)
                        q_garch = st.number_input("GARCH order (q)", min_value=1, max_value=3, value=1)
                        order_garch = (p_garch, q_garch)
                    else:
                        order_garch = None
                
                with col4:
                    forecast_steps_garch = st.number_input("Forecast Steps (GARCH)", min_value=5, max_value=100, value=30)
                
                # Fit GARCH model button
                if st.button("fit GARCH model", type="primary"):
                    with st.spinner("Fitting GARCH model..."):
                        try:
                            # Fit the GARCH model
                            garch_model_info = garch_forecaster.fit_garch(
                                garch_returns, 
                                model_type=model_type, 
                                p=order_garch[0] if order_garch else 1, 
                                q=order_garch[1] if order_garch else 1, 
                                auto_select=auto_select_garch
                            )
                            
                            # Display model summary
                            st.markdown("#### Model Summary")
                            
                            # Prominently display the selected GARCH order
                            order = garch_model_info['order']
                            p, q = order
                            
                            st.markdown(f"""
                            **Selected {model_type} Order: ({p}, {q})**
                            
                            - **p = {p}**: GARCH order (number of lagged conditional variances)
                            - **q = {q}**: ARCH order (number of lagged squared innovations)
                            - **Model Type**: {model_type}
                            """)
                            
                            if auto_select_garch:
                                st.success(f"(✓) **Auto-selected using AIC optimization** - This order minimizes the Akaike Information Criterion")
                            else:
                                st.info(f"ⓘ **Manually specified** - Order ({p}, {q}) was set by user")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("AIC", f"{garch_model_info['aic']:.2f}")
                            with col2:
                                st.metric("BIC", f"{garch_model_info['bic']:.2f}")
                            with col3:
                                st.metric("Log-Likelihood", f"{garch_model_info['loglikelihood']:.2f}")
                            with col4:
                                st.metric("Model Order", f"({p}, {q})")
                            
                            # Model parameters
                            st.markdown("#### Model Parameters")
                            params_df_garch = pd.DataFrame({
                                'Parameter': garch_model_info['params'].index,
                                'Value': garch_model_info['params'].values,
                                'P-Value': garch_model_info['pvalues'].values
                            })
                            
                            # Download button for GARCH model parameters
                            import base64
                            csv_params_garch = params_df_garch.to_csv(index=False)
                            b64_params_garch = base64.b64encode(csv_params_garch.encode()).decode()
                            href = f'''
                                <a href="data:file/csv;base64,{b64_params_garch}" download="garch_model_parameters_{model_type}_{p}_{q}.csv"
                                   style="
                                       display: inline-block;
                                       background-color: #e26d5c;
                                       color: #fff;
                                       font-size: 1.1rem;
                                       padding: 0.5rem 1.5rem;
                                       border-radius: 0.5rem;
                                       text-decoration: none;
                                       text-align: center;
                                       margin-top: 1rem;
                                       box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                   ">
                                   download GARCH model parameters
                                </a>
                            '''
                            st.markdown(href, unsafe_allow_html=True)
                            
                            # Generate volatility forecast
                            garch_forecast_results = garch_forecaster.forecast(steps=forecast_steps_garch)
                            
                            # Display volatility forecast
                            st.markdown("#### volatility forecast")
                            
                            # Plot volatility forecast
                            fig_garch = garch_forecaster.plot_volatility_forecast(garch_returns, garch_forecast_results, f"{model_type} Volatility Forecast")
                            st.pyplot(fig_garch)
                            
                            # Volatility forecast metrics
                            st.markdown("#### Volatility Forecast Statistics")
                            forecast_vol = garch_forecast_results['forecast']
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean Forecast Vol", f"{forecast_vol.mean():.4f}")
                            with col2:
                                st.metric("Forecast Vol Std", f"{forecast_vol.std():.4f}")
                            with col3:
                                st.metric("Min Forecast Vol", f"{forecast_vol.min():.4f}")
                            with col4:
                                st.metric("Max Forecast Vol", f"{forecast_vol.max():.4f}")
                            
                            # Historical volatility statistics
                            st.markdown("#### Historical Volatility Statistics")
                            hist_vol = garch_model_info['conditional_volatility']
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean Historical Vol", f"{hist_vol.mean():.4f}")
                            with col2:
                                st.metric("Historical Vol Std", f"{hist_vol.std():.4f}")
                            with col3:
                                st.metric("Min Historical Vol", f"{hist_vol.min():.4f}")
                            with col4:
                                st.metric("Max Historical Vol", f"{hist_vol.max():.4f}")
                            
                            # Model diagnostics
                            st.markdown("#### Model Diagnostics")
                            fig_diagnostics = garch_forecaster.plot_model_diagnostics()
                            st.pyplot(fig_diagnostics)
                            
                            # Residuals statistics
                            summary = garch_forecaster.get_model_summary()
                            residuals_stats_garch = summary['residuals_stats']
                            volatility_stats = summary['volatility_stats']
                            
                            st.markdown("#### Standardized Residuals Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean", f"{residuals_stats_garch['mean']:.6f}")
                            with col2:
                                st.metric("Std", f"{residuals_stats_garch['std']:.6f}")
                            with col3:
                                st.metric("Skewness", f"{residuals_stats_garch['skewness']:.4f}")
                            with col4:
                                st.metric("Kurtosis", f"{residuals_stats_garch['kurtosis']:.4f}")
                            
                            # Volatility clustering analysis
                            st.markdown("#### Volatility Clustering Analysis")
                            fig_clustering = garch_forecaster.plot_volatility_clustering(garch_returns)
                            st.pyplot(fig_clustering)
                            
                            # Store results in session state
                            st.session_state['garch_results'] = {
                                'forecaster': garch_forecaster,
                                'returns': garch_returns,
                                'model_info': garch_model_info,
                                'forecast_results': garch_forecast_results,
                                'clustering_result': clustering_result
                            }
                            
                            # Show volatility forecast values
                            st.markdown("#### Volatility Forecast Values")
                            forecast_vol_df = pd.DataFrame({
                                'Date': forecast_vol.index,
                                'Forecasted_Volatility': forecast_vol.values,
                                'Forecasted_Variance': garch_forecast_results['variance'].values
                            })
                            
                            # Download button for volatility forecast values
                            csv_vol_forecast = forecast_vol_df.to_csv(index=False)
                            b64_vol_forecast = base64.b64encode(csv_vol_forecast.encode()).decode()
                            href = f'''
                                <a href="data:file/csv;base64,{b64_vol_forecast}" download="garch_volatility_forecast_{model_type}_{p}_{q}.csv"
                                   style="
                                       display: inline-block;
                                       background-color: #e26d5c;
                                       color: #fff;
                                       font-size: 1.1rem;
                                       padding: 0.5rem 1.5rem;
                                       border-radius: 0.5rem;
                                       text-decoration: none;
                                       text-align: center;
                                       margin-top: 1rem;
                                       box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                   ">
                                   download volatility forecast values
                                </a>
                            '''
                            st.markdown(href, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error fitting GARCH model: {str(e)}")
                            st.error(f"Full error: {e}")
                    
                    # Show volatility clustering plots for model selection
                    if st.checkbox("Show volatility clustering analysis"):
                        fig_clustering = garch_forecaster.plot_volatility_clustering(garch_returns)
                        st.pyplot(fig_clustering)
            
            # --- END GARCH VOLATILITY FORECASTING SECTION ---
            
            # --- DYNAMIC HEDGE RATIO CALCULATION SECTION ---
            st.header("dynamic hedge ratio calculation")
            st.markdown("### rolling beta and volatility-based hedge ratios")
            
            # Get WTI and Brent data for hedge ratio calculation
            if wti_raw is not None and brent_raw is not None:
                # Prepare returns for hedge ratio calculation
                hedge_calculator = HedgeRatioCalculator()
                
                # Calculate log returns
                wti_returns = np.log(wti_raw['Close'] / wti_raw['Close'].shift(1)).dropna()
                brent_returns = np.log(brent_raw['Close'] / brent_raw['Close'].shift(1)).dropna()
                
                # Align data
                aligned_data = pd.DataFrame({
                    'wti': wti_returns,
                    'brent': brent_returns
                }).dropna()
                
                wti_aligned = aligned_data['wti'].astype(float)
                brent_aligned = aligned_data['brent'].astype(float)
                
                # Display returns info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("WTI Mean Return", f"{wti_aligned.mean():.4f}")
                with col2:
                    st.metric("Brent Mean Return", f"{brent_aligned.mean():.4f}")
                with col3:
                    correlation = float(wti_aligned.corr(brent_aligned))
                    st.metric("Correlation", f"{correlation:.4f}")
                with col4:
                    st.metric("Data Points", f"{len(wti_aligned):,}")
                
                # Hedge ratio configuration
                st.markdown("#### hedge ratio configuration")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    hedge_method = st.selectbox("Hedge Ratio Method", ["OLS", "Rolling OLS", "Volatility-based"])
                
                with col2:
                    rolling_window = st.number_input("Rolling Window", min_value=20, max_value=252, value=60)
                
                with col3:
                    asset_choice = st.selectbox("Asset to Hedge", ["WTI", "Brent"])
                
                # Determine asset and hedge based on choice
                if asset_choice == "WTI":
                    asset_returns = wti_aligned
                    hedge_returns = brent_aligned
                    asset_name = "WTI"
                    hedge_name = "Brent"
                else:
                    asset_returns = brent_aligned
                    hedge_returns = wti_aligned
                    asset_name = "Brent"
                    hedge_name = "WTI"
                
                # Calculate hedge ratios button
                if st.button("calculate hedge ratios", type="primary"):
                    with st.spinner("Calculating hedge ratios..."):
                        try:
                            # Calculate OLS hedge ratio
                            ols_hedge = hedge_calculator._ols_hedge_ratio(asset_returns, hedge_returns)
                            
                            # Display OLS hedge ratio results
                            st.markdown("#### OLS hedge ratio results")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Beta (Hedge Ratio)", f"{ols_hedge['beta']:.4f}")
                            with col2:
                                st.metric("R²", f"{ols_hedge['r2']:.4f}")
                            with col3:
                                st.metric("P-Value", f"{ols_hedge['p_value']:.4f}")
                            with col4:
                                st.metric("T-Statistic", f"{ols_hedge['t_stat']:.4f}")
                            
                            # Interpretation
                            if ols_hedge['p_value'] < 0.05:
                                st.success(f"(✓) **Statistically significant hedge ratio** - The relationship between {asset_name} and {hedge_name} is significant")
                            else:
                                st.warning(f"⚠ **Non-significant hedge ratio** - The relationship between {asset_name} and {hedge_name} may not be reliable")
                            
                            # Calculate rolling hedge ratios
                            st.markdown("#### rolling hedge ratios")
                            
                            # Rolling beta
                            rolling_beta = hedge_calculator.calculate_rolling_beta(asset_returns, hedge_returns, window=rolling_window)
                            
                            # Volatility-based hedge ratio
                            vol_hedge = hedge_calculator.calculate_volatility_based_hedge_ratio(asset_returns, hedge_returns, window=rolling_window)
                            
                            # Display rolling statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean Rolling Beta", f"{rolling_beta['beta'].mean():.4f}")
                            with col2:
                                st.metric("Beta Std", f"{rolling_beta['beta'].std():.4f}")
                            with col3:
                                st.metric("Mean Vol Hedge Ratio", f"{vol_hedge['vol_hedge_ratio'].mean():.4f}")
                            with col4:
                                st.metric("Mean Correlation", f"{vol_hedge['correlation'].mean():.4f}")
                            
                            # Plot rolling hedge ratios
                            st.markdown("#### rolling hedge ratio analysis")
                            fig_rolling = hedge_calculator.plot_rolling_hedge_ratios(asset_returns, hedge_returns, window=rolling_window)
                            st.pyplot(fig_rolling)
                            
                            # Hedge effectiveness analysis
                            st.markdown("#### hedge effectiveness analysis")
                            
                            # Use OLS hedge ratio for effectiveness calculation
                            hedge_ratio = ols_hedge['beta']
                            effectiveness = hedge_calculator.calculate_hedge_effectiveness(asset_returns, hedge_returns, hedge_ratio)
                            
                            # Display effectiveness metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Variance Reduction", f"{effectiveness['variance_reduction']:.2%}")
                            with col2:
                                st.metric("Risk Reduction", f"{effectiveness['risk_reduction']:.2%}")
                            with col3:
                                st.metric("Unhedged Volatility", f"{effectiveness['unhedged_volatility']:.4f}")
                            with col4:
                                st.metric("Hedged Volatility", f"{effectiveness['hedged_volatility']:.4f}")
                            
                            # Plot hedge effectiveness
                            fig_effectiveness = hedge_calculator.plot_hedge_effectiveness(asset_returns, hedge_returns, hedge_ratio)
                            st.pyplot(fig_effectiveness)
                            
                            # Dynamic hedge ratio strategy
                            st.markdown("#### dynamic hedge ratio strategy")
                            
                            # Calculate dynamic hedge ratios
                            dynamic_hedge_ols = rolling_beta['beta']
                            dynamic_hedge_vol = vol_hedge['vol_hedge_ratio']
                            
                            # Compare effectiveness of different strategies
                            effectiveness_ols = hedge_calculator.calculate_hedge_effectiveness(asset_returns, hedge_returns, dynamic_hedge_ols)
                            effectiveness_vol = hedge_calculator.calculate_hedge_effectiveness(asset_returns, hedge_returns, dynamic_hedge_vol)
                            
                            # Display strategy comparison
                            st.markdown("**Strategy Comparison:**")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown("**Static OLS Strategy**")
                                st.metric("Variance Reduction", f"{effectiveness['variance_reduction']:.2%}")
                                st.metric("Risk Reduction", f"{effectiveness['risk_reduction']:.2%}")
                            
                            with col2:
                                st.markdown("**Dynamic OLS Strategy**")
                                st.metric("Variance Reduction", f"{effectiveness_ols['variance_reduction']:.2%}")
                                st.metric("Risk Reduction", f"{effectiveness_ols['risk_reduction']:.2%}")
                            
                            with col3:
                                st.markdown("**Volatility-based Strategy**")
                                st.metric("Variance Reduction", f"{effectiveness_vol['variance_reduction']:.2%}")
                                st.metric("Risk Reduction", f"{effectiveness_vol['risk_reduction']:.2%}")
                            
                            # Determine best strategy
                            strategies = {
                                'Static OLS': effectiveness['variance_reduction'],
                                'Dynamic OLS': effectiveness_ols['variance_reduction'],
                                'Volatility-based': effectiveness_vol['variance_reduction']
                            }
                            
                            best_strategy = max(strategies.items(), key=lambda x: x[1])[0]
                            st.success(f"**Best Strategy**: {best_strategy} with {strategies[best_strategy]:.2%} variance reduction")
                            
                            # Store results in session state
                            st.session_state['hedge_results'] = {
                                'calculator': hedge_calculator,
                                'asset_returns': asset_returns,
                                'hedge_returns': hedge_returns,
                                'ols_hedge': ols_hedge,
                                'rolling_beta': rolling_beta,
                                'vol_hedge': vol_hedge,
                                'effectiveness': effectiveness,
                                'effectiveness_ols': effectiveness_ols,
                                'effectiveness_vol': effectiveness_vol,
                                'best_strategy': best_strategy
                            }
                            
                            # Download hedge ratio data
                            st.markdown("#### hedge ratio data")
                            
                            # Create comprehensive hedge ratio dataset with proper alignment
                            # Find common dates between rolling_beta and vol_hedge
                            common_dates = rolling_beta.index.intersection(vol_hedge.index)
                            
                            if len(common_dates) > 0:
                                # Align both datasets to common dates
                                rolling_beta_aligned = rolling_beta.reindex(common_dates, method='ffill')
                                vol_hedge_aligned = vol_hedge.reindex(common_dates, method='ffill')
                                
                                hedge_data = pd.DataFrame({
                                    'Date': common_dates,
                                    'Rolling_Beta': rolling_beta_aligned['beta'],
                                    'Rolling_R2': rolling_beta_aligned['r2'],
                                    'Rolling_P_Value': rolling_beta_aligned['p_value'],
                                    'Volatility_Ratio': vol_hedge_aligned['volatility_ratio'],
                                    'Rolling_Correlation': vol_hedge_aligned['correlation'],
                                    'Volatility_Hedge_Ratio': vol_hedge_aligned['vol_hedge_ratio'],
                                    'Asset_Volatility': vol_hedge_aligned['asset_volatility'],
                                    'Hedge_Volatility': vol_hedge_aligned['hedge_volatility']
                                })
                            else:
                                # If no common dates, create separate datasets
                                st.warning("⚠ Rolling beta and volatility hedge ratios have no common dates - creating separate datasets")
                                
                                # Create rolling beta dataset
                                rolling_beta_data = pd.DataFrame({
                                    'Date': rolling_beta.index,
                                    'Rolling_Beta': rolling_beta['beta'],
                                    'Rolling_R2': rolling_beta['r2'],
                                    'Rolling_P_Value': rolling_beta['p_value']
                                })
                                
                                # Create volatility hedge dataset
                                vol_hedge_data = pd.DataFrame({
                                    'Date': vol_hedge.index,
                                    'Volatility_Ratio': vol_hedge['volatility_ratio'],
                                    'Rolling_Correlation': vol_hedge['correlation'],
                                    'Volatility_Hedge_Ratio': vol_hedge['vol_hedge_ratio'],
                                    'Asset_Volatility': vol_hedge['asset_volatility'],
                                    'Hedge_Volatility': vol_hedge['hedge_volatility']
                                })
                                
                                # Use the longer dataset as the main one
                                if len(rolling_beta_data) >= len(vol_hedge_data):
                                    hedge_data = rolling_beta_data
                                else:
                                    hedge_data = vol_hedge_data
                            
                            # Download button for hedge ratio data
                            import base64
                            csv_hedge = hedge_data.to_csv(index=False)
                            b64_hedge = base64.b64encode(csv_hedge.encode()).decode()
                            href = f'''
                                <a href="data:file/csv;base64,{b64_hedge}" download="hedge_ratios_{asset_name}_vs_{hedge_name}.csv"
                                   style="
                                       display: inline-block;
                                       background-color: #e26d5c;
                                       color: #fff;
                                       font-size: 1.1rem;
                                       padding: 0.5rem 1.5rem;
                                       border-radius: 0.5rem;
                                       text-decoration: none;
                                       text-align: center;
                                       margin-top: 1rem;
                                       box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                   ">
                                   download hedge ratio data
                                </a>
                            '''
                            st.markdown(href, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error calculating hedge ratios: {str(e)}")
                            st.error(f"Full error: {e}")
                    
                    # Show hedge ratio analysis plots
                    if st.checkbox("Show hedge ratio analysis"):
                        try:
                            # Calculate basic hedge ratios for visualization
                            ols_hedge = hedge_calculator._ols_hedge_ratio(asset_returns, hedge_returns)
                            rolling_beta = hedge_calculator.calculate_rolling_beta(asset_returns, hedge_returns, window=rolling_window)
                            vol_hedge = hedge_calculator.calculate_volatility_based_hedge_ratio(asset_returns, hedge_returns, window=rolling_window)
                            
                            # Plot rolling hedge ratios
                            fig_rolling = hedge_calculator.plot_rolling_hedge_ratios(asset_returns, hedge_returns, window=rolling_window)
                            st.pyplot(fig_rolling)
                            
                            # Plot hedge effectiveness
                            hedge_ratio = ols_hedge['beta']
                            fig_effectiveness = hedge_calculator.plot_hedge_effectiveness(asset_returns, hedge_returns, hedge_ratio)
                            st.pyplot(fig_effectiveness)
                            
                        except Exception as e:
                            st.error(f"Error generating hedge ratio plots: {str(e)}")
            
            # --- END DYNAMIC HEDGE RATIO CALCULATION SECTION ---
            
            # --- BACKTEST DYNAMIC HEDGING SECTION ---
            st.header("backtest dynamic hedging")
            st.markdown("### realistic trading simulation with transaction costs")
            
            # Check if hedge results are available
            if 'hedge_results' in st.session_state:
                hedge_results = st.session_state['hedge_results']
                
                # Get data from hedge results
                asset_returns = hedge_results['asset_returns']
                hedge_returns = hedge_results['hedge_returns']
                rolling_beta = hedge_results['rolling_beta']
                vol_hedge = hedge_results['vol_hedge']
                
                # Get original price data for cost calculation
                if asset_choice == "WTI":
                    asset_prices = wti_raw['Close']
                    hedge_prices = brent_raw['Close']
                else:
                    asset_prices = brent_raw['Close']
                    hedge_prices = wti_raw['Close']
                
                # Backtest configuration
                st.markdown("#### backtest configuration")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    initial_capital = st.number_input("Initial Capital ($)", min_value=100000, max_value=10000000, value=1000000, step=100000)
                
                with col2:
                    transaction_cost = st.number_input("Transaction Cost (%)", min_value=0.01, max_value=1.0, value=0.1, step=0.01) / 100
                
                with col3:
                    slippage = st.number_input("Slippage (%)", min_value=0.01, max_value=1.0, value=0.05, step=0.01) / 100
                
                with col4:
                    target_volatility = st.number_input("Target Volatility (%)", min_value=5.0, max_value=50.0, value=15.0, step=1.0) / 100
                
                # Rebalancing configuration
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    rebalance_frequency = st.selectbox("Rebalance Frequency", ["daily", "weekly", "monthly"])
                
                with col2:
                    strategy_type = st.selectbox("Strategy Type", ["Rolling OLS", "Volatility-based", "Both"])
                
                with col3:
                    include_benchmark = st.checkbox("Include Benchmark Comparison", value=True)
                
                # Run backtest button
                if st.button("run backtest", type="primary"):
                    with st.spinner("Running backtest..."):
                        try:
                            # Initialize backtesting engine
                            backtest = DynamicHedgingBacktest(
                                initial_capital=initial_capital,
                                transaction_cost=transaction_cost,
                                slippage=slippage
                            )
                            
                            # Run backtests based on strategy type
                            strategies_run = []
                            
                            # Show data alignment info
                            st.info(f"**Data Alignment**: Asset returns: {len(asset_returns)}, Hedge returns: {len(hedge_returns)}, Rolling beta: {len(rolling_beta)}")
                            
                            # Check data alignment
                            common_dates_ols = asset_returns.index.intersection(rolling_beta['beta'].index)
                            common_dates_vol = asset_returns.index.intersection(vol_hedge['vol_hedge_ratio'].index)
                            
                            st.info(f"**Common Dates**: OLS strategy: {len(common_dates_ols)}, Volatility strategy: {len(common_dates_vol)}")
                            
                            if strategy_type in ["Rolling OLS", "Both"]:
                                # Run Rolling OLS strategy
                                ols_results = backtest.run_backtest(
                                    asset_returns=asset_returns,
                                    hedge_returns=hedge_returns,
                                    hedge_ratios=rolling_beta['beta'],
                                    asset_prices=asset_prices,
                                    hedge_prices=hedge_prices,
                                    strategy_name="Rolling OLS Hedge",
                                    rebalance_frequency=rebalance_frequency,
                                    target_volatility=target_volatility
                                )
                                strategies_run.append("Rolling OLS Hedge")
                            
                            if strategy_type in ["Volatility-based", "Both"]:
                                # Run Volatility-based strategy
                                vol_results = backtest.run_backtest(
                                    asset_returns=asset_returns,
                                    hedge_returns=hedge_returns,
                                    hedge_ratios=vol_hedge['vol_hedge_ratio'],
                                    asset_prices=asset_prices,
                                    hedge_prices=hedge_prices,
                                    strategy_name="Volatility-based Hedge",
                                    rebalance_frequency=rebalance_frequency,
                                    target_volatility=target_volatility
                                )
                                strategies_run.append("Volatility-based Hedge")
                            
                            # Benchmark returns (unhedged asset)
                            benchmark_returns = asset_returns if include_benchmark else None
                            
                            # Display backtest results
                            st.markdown("#### backtest results")
                            
                            # Performance metrics for each strategy
                            for strategy_name in strategies_run:
                                st.markdown(f"**{strategy_name} Performance:**")
                                
                                # Calculate performance metrics
                                metrics = backtest.calculate_performance_metrics(
                                    backtest.results[strategy_name]['net_returns'],
                                    benchmark_returns
                                )
                                
                                # Display key metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Return", f"{metrics['total_return']:.2%}")
                                    st.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
                                with col2:
                                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                                    st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                                with col3:
                                    st.metric("Volatility", f"{metrics['volatility']:.2%}")
                                    st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
                                with col4:
                                    st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.2f}")
                                    st.metric("VaR (95%)", f"{metrics['var_95']:.2%}")
                                
                                # Risk metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("CVaR (95%)", f"{metrics['cvar_95']:.2%}")
                                with col2:
                                    st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
                                with col3:
                                    st.metric("Avg Win", f"{metrics['avg_win']:.2%}")
                                with col4:
                                    st.metric("Avg Loss", f"{metrics['avg_loss']:.2%}")
                                
                                # Benchmark comparison if available
                                if include_benchmark and not pd.isna(metrics['alpha']):
                                    st.markdown("**Benchmark Comparison:**")
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Alpha", f"{metrics['alpha']:.2%}")
                                    with col2:
                                        st.metric("Beta", f"{metrics['beta']:.2f}")
                                    with col3:
                                        st.metric("Information Ratio", f"{metrics['information_ratio']:.2f}")
                                    with col4:
                                        st.metric("Tracking Error", f"{metrics['tracking_error']:.2%}")
                                
                                # Transaction cost analysis
                                results = backtest.results[strategy_name]
                                total_costs = results['total_costs'].sum()
                                cost_impact = total_costs / initial_capital
                                
                                st.markdown("**Transaction Cost Analysis:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Costs", f"${total_costs:,.0f}")
                                with col2:
                                    st.metric("Cost Impact", f"{cost_impact:.2%}")
                                with col3:
                                    st.metric("Avg Daily Cost", f"${results['total_costs'].mean():.0f}")
                            
                            # Strategy comparison if multiple strategies
                            if len(strategies_run) > 1:
                                st.markdown("#### strategy comparison")
                                
                                # Compare strategies
                                fig_comparison, comparison_df = backtest.compare_strategies(strategies_run, benchmark_returns)
                                st.pyplot(fig_comparison)
                                
                                # Display comparison table
                                st.markdown("**Performance Comparison Table:**")
                                st.table(comparison_df.round(4))
                            
                            # Detailed backtest plots
                            st.markdown("#### detailed backtest analysis")
                            
                            # Plot for each strategy
                            for strategy_name in strategies_run:
                                st.markdown(f"**{strategy_name} Analysis:**")
                                fig_backtest = backtest.plot_backtest_results(strategy_name, benchmark_returns)
                                st.pyplot(fig_backtest)
                            
                            # Generate comprehensive report
                            st.markdown("#### backtest report")
                            
                            for strategy_name in strategies_run:
                                report = backtest.generate_backtest_report(strategy_name, benchmark_returns)
                                
                                st.markdown(f"**{strategy_name} Report:**")
                                st.markdown(f"""
                                - **Backtest Period**: {report['backtest_period']}
                                - **Initial Capital**: ${report['initial_capital']:,.0f}
                                - **Final Portfolio Value**: ${report['final_portfolio_value']:,.0f}
                                - **Total Transaction Costs**: ${report['total_transaction_costs']:,.0f}
                                - **Cost Impact**: {report['cost_impact']:.2%}
                                """)
                            
                            # Store backtest results in session state
                            st.session_state['backtest_results'] = {
                                'backtest': backtest,
                                'strategies_run': strategies_run,
                                'benchmark_returns': benchmark_returns,
                                'configuration': {
                                    'initial_capital': initial_capital,
                                    'transaction_cost': transaction_cost,
                                    'slippage': slippage,
                                    'target_volatility': target_volatility,
                                    'rebalance_frequency': rebalance_frequency
                                }
                            }
                            
                            # Download backtest results
                            st.markdown("#### download backtest results")
                            
                            # Create comprehensive backtest dataset
                            all_results = []
                            for strategy_name in strategies_run:
                                results = backtest.results[strategy_name]
                                strategy_data = pd.DataFrame({
                                    'Date': results['dates'],
                                    'Strategy': strategy_name,
                                    'Portfolio_Value': results['portfolio_value'],
                                    'Net_Returns': results['net_returns'],
                                    'Gross_Returns': results['gross_returns'],
                                    'Transaction_Costs': results['total_costs'],
                                    'Position_Size': results['position_sizes'],
                                    'Hedge_Position': results['hedge_positions'],
                                    'Hedge_Ratio': results['hedge_ratios'],
                                    'Cumulative_Returns': results['cumulative_returns']
                                })
                                all_results.append(strategy_data)
                            
                            if all_results:
                                combined_results = pd.concat(all_results, ignore_index=True)
                                
                                # Download button for backtest results
                                import base64
                                csv_backtest = combined_results.to_csv(index=False)
                                b64_backtest = base64.b64encode(csv_backtest.encode()).decode()
                                href = f'''
                                    <a href="data:file/csv;base64,{b64_backtest}" download="backtest_results_{asset_name}_vs_{hedge_name}.csv"
                                       style="
                                           display: inline-block;
                                           background-color: #e26d5c;
                                           color: #fff;
                                           font-size: 1.1rem;
                                           padding: 0.5rem 1.5rem;
                                           border-radius: 0.5rem;
                                           text-decoration: none;
                                           text-align: center;
                                           margin-top: 1rem;
                                           box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                       ">
                                       download backtest results
                                    </a>
                                '''
                                st.markdown(href, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error running backtest: {str(e)}")
                            st.error(f"Full error: {e}")
            
            else:
                st.warning("⚠ **No hedge ratio results available** - Please run the Dynamic Hedge Ratio Calculation first to generate hedge ratios for backtesting.")
            
            # --- END BACKTEST DYNAMIC HEDGING SECTION ---
            
            # --- EVALUATE & COMPARE METHODS SECTION ---
            st.header("evaluate & compare methods")
            st.markdown("### method evaluation and comparison")
            
            # Check if we have results from different methods
            methods_available = []
            
            if 'arima_results' in st.session_state:
                methods_available.append('ARIMA Forecasting')
            
            if 'garch_results' in st.session_state:
                methods_available.append('GARCH Volatility')
            
            if 'hedge_results' in st.session_state:
                methods_available.append('Dynamic Hedge Ratios')
            
            if 'backtest_results' in st.session_state:
                methods_available.append('Backtesting Results')
            
            if not methods_available:
                st.warning("⚠ **No method results available** - Please run ARIMA forecasting, GARCH volatility, hedge ratio calculation, and backtesting first to enable method comparison.")
            else:
                st.success(f"(✓) **Methods Available**: {', '.join(methods_available)}")
                
                # Method comparison configuration
                st.markdown("#### comparison configuration")
                col1, col2 = st.columns(2)
                
                with col1:
                    comparison_metric = st.selectbox(
                        "Primary Comparison Metric",
                        ["Forecast Accuracy", "Risk Reduction", "Returns Performance", "Model Fit Quality"]
                    )
                
                with col2:
                    include_benchmarks = st.checkbox("Include Benchmark Comparisons", value=True)
                
                # Run comprehensive evaluation
                if st.button("run comprehensive evaluation", type="primary"):
                    with st.spinner("Running comprehensive method evaluation..."):
                        try:
                            # Initialize evaluation results
                            evaluation_results = {}
                            
                            # 1. ARIMA FORECASTING EVALUATION
                            if 'arima_results' in st.session_state:
                                st.markdown("#### ARIMA forecasting evaluation")
                                
                                arima_results = st.session_state['arima_results']
                                log_returns = arima_results['log_returns']
                                model_info = arima_results['model_info']
                                forecast_results = arima_results['forecast_results']
                                evaluation = arima_results['evaluation']
                                
                                # ARIMA Performance Metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Forecast RMSE", f"{evaluation['rmse']:.6f}")
                                    st.metric("Model AIC", f"{model_info['aic']:.2f}")
                                with col2:
                                    st.metric("Forecast MAE", f"{evaluation['mae']:.6f}")
                                    st.metric("Model BIC", f"{model_info['bic']:.2f}")
                                with col3:
                                    st.metric("Naive RMSE", f"{evaluation['naive_rmse']:.6f}")
                                    st.metric("Improvement vs Naive", f"{((evaluation['naive_rmse'] - evaluation['rmse']) / evaluation['naive_rmse'] * 100):.1f}%")
                                with col4:
                                    st.metric("Data Points", f"{len(log_returns):,}")
                                    st.metric("Forecast Steps", f"{len(forecast_results['forecast']):,}")
                                
                                # ARIMA Model Quality Assessment
                                st.markdown("**Model Quality Assessment:**")
                                
                                # Residuals analysis
                                residuals = model_info['resid'].dropna()
                                residuals_stats = model_info['residuals_stats']
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    if abs(residuals_stats['mean']) < 0.001:
                                        st.success("(✓) Residuals Mean: Good")
                                    else:
                                        st.warning("⚠ Residuals Mean: Needs attention")
                                    st.metric("Mean", f"{residuals_stats['mean']:.6f}")
                                
                                with col2:
                                    if residuals_stats['std'] < 0.02:
                                        st.success("(✓) Residuals Std: Good")
                                    else:
                                        st.warning("⚠ Residuals Std: High variability")
                                    st.metric("Std", f"{residuals_stats['std']:.6f}")
                                
                                with col3:
                                    if abs(residuals_stats['skewness']) < 1:
                                        st.success("(✓) Skewness: Normal")
                                    else:
                                        st.warning("⚠ Skewness: Non-normal")
                                    st.metric("Skewness", f"{residuals_stats['skewness']:.4f}")
                                
                                with col4:
                                    if residuals_stats['kurtosis'] < 5:
                                        st.success("(✓) Kurtosis: Normal")
                                    else:
                                        st.warning("⚠ Kurtosis: Heavy tails")
                                    st.metric("Kurtosis", f"{residuals_stats['kurtosis']:.4f}")
                                
                                evaluation_results['ARIMA'] = {
                                    'forecast_accuracy': 1 - (evaluation['rmse'] / evaluation['naive_rmse']),
                                    'model_fit': model_info['aic'],
                                    'residuals_quality': abs(residuals_stats['mean']) + residuals_stats['std'],
                                    'forecast_stability': forecast_results['forecast'].std()
                                }
                            
                            # 2. GARCH VOLATILITY EVALUATION
                            if 'garch_results' in st.session_state:
                                st.markdown("#### GARCH volatility evaluation")
                                
                                garch_results = st.session_state['garch_results']
                                returns = garch_results['returns']
                                model_info = garch_results['model_info']
                                forecast_results = garch_results['forecast_results']
                                clustering_result = garch_results['clustering_result']
                                
                                # GARCH Performance Metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Model AIC", f"{model_info['aic']:.2f}")
                                    st.metric("Log-Likelihood", f"{model_info['loglikelihood']:.2f}")
                                with col2:
                                    st.metric("Volatility Clustering", f"{clustering_result['autocorrelation']:.4f}")
                                    st.metric("Clustering P-Value", f"{clustering_result['ljung_box_pvalue']:.4f}")
                                with col3:
                                    st.metric("Forecast Vol Mean", f"{forecast_results['forecast'].mean():.4f}")
                                    st.metric("Forecast Vol Std", f"{forecast_results['forecast'].std():.4f}")
                                with col4:
                                    st.metric("Historical Vol Mean", f"{model_info['conditional_volatility'].mean():.4f}")
                                    st.metric("Data Points", f"{len(returns):,}")
                                
                                # GARCH Model Quality Assessment
                                st.markdown("**Volatility Model Assessment:**")
                                
                                # Volatility clustering detection
                                if clustering_result['has_clustering']:
                                    st.success("(✓) **Volatility Clustering Detected** - GARCH model is appropriate")
                                else:
                                    st.warning("⚠ **No Volatility Clustering** - GARCH model may not be optimal")
                                
                                # Forecast vs Historical volatility comparison
                                vol_forecast_mean = forecast_results['forecast'].mean()
                                vol_historical_mean = model_info['conditional_volatility'].mean()
                                vol_consistency = abs(vol_forecast_mean - vol_historical_mean) / vol_historical_mean
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if vol_consistency < 0.2:
                                        st.success("(✓) Forecast Consistency: Good")
                                    else:
                                        st.warning("⚠ Forecast Consistency: High deviation")
                                    st.metric("Consistency", f"{vol_consistency:.2%}")
                                
                                with col2:
                                    if model_info['aic'] < -1000:
                                        st.success("(✓) Model Fit: Excellent")
                                    elif model_info['aic'] < -500:
                                        st.success("(✓) Model Fit: Good")
                                    else:
                                        st.warning("⚠ Model Fit: Needs improvement")
                                    st.metric("AIC", f"{model_info['aic']:.2f}")
                                
                                with col3:
                                    if clustering_result['ljung_box_pvalue'] < 0.05:
                                        st.success("(✓) Clustering Significance: High")
                                    else:
                                        st.warning("⚠ Clustering Significance: Low")
                                    st.metric("P-Value", f"{clustering_result['ljung_box_pvalue']:.4f}")
                                
                                evaluation_results['GARCH'] = {
                                    'volatility_forecast_accuracy': 1 - vol_consistency,
                                    'model_fit': model_info['aic'],
                                    'clustering_detection': clustering_result['autocorrelation'],
                                    'forecast_stability': forecast_results['forecast'].std()
                                }
                            
                            # 3. DYNAMIC HEDGE RATIO EVALUATION
                            if 'hedge_results' in st.session_state:
                                st.markdown("#### Dynamic hedge ratio evaluation")
                                
                                hedge_results = st.session_state['hedge_results']
                                effectiveness = hedge_results['effectiveness']
                                effectiveness_ols = hedge_results['effectiveness_ols']
                                effectiveness_vol = hedge_results['effectiveness_vol']
                                best_strategy = hedge_results['best_strategy']
                                
                                # Hedge Ratio Performance Metrics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Best Strategy", best_strategy)
                                    st.metric("Static OLS Variance Reduction", f"{effectiveness['variance_reduction']:.2%}")
                                with col2:
                                    st.metric("Dynamic OLS Variance Reduction", f"{effectiveness_ols['variance_reduction']:.2%}")
                                    st.metric("Volatility-based Variance Reduction", f"{effectiveness_vol['variance_reduction']:.2%}")
                                with col3:
                                    st.metric("Best Risk Reduction", f"{max([effectiveness['risk_reduction'], effectiveness_ols['risk_reduction'], effectiveness_vol['risk_reduction']]):.2%}")
                                    st.metric("Hedged Volatility", f"{effectiveness['hedged_volatility']:.4f}")
                                with col4:
                                    st.metric("Unhedged Volatility", f"{effectiveness['unhedged_volatility']:.4f}")
                                    st.metric("Hedge Effectiveness", f"{effectiveness['variance_reduction']:.2%}")
                                
                                # Strategy Comparison
                                st.markdown("**Strategy Comparison:**")
                                
                                strategies_comparison = pd.DataFrame({
                                    'Strategy': ['Static OLS', 'Dynamic OLS', 'Volatility-based'],
                                    'Variance Reduction': [
                                        effectiveness['variance_reduction'],
                                        effectiveness_ols['variance_reduction'],
                                        effectiveness_vol['variance_reduction']
                                    ],
                                    'Risk Reduction': [
                                        effectiveness['risk_reduction'],
                                        effectiveness_ols['risk_reduction'],
                                        effectiveness_vol['risk_reduction']
                                    ],
                                    'Hedged Volatility': [
                                        effectiveness['hedged_volatility'],
                                        effectiveness_ols['hedged_volatility'],
                                        effectiveness_vol['hedged_volatility']
                                    ]
                                })
                                
                                st.table(strategies_comparison.round(4))
                                
                                # Best strategy analysis
                                st.success(f"**Best Strategy**: {best_strategy}")
                                
                                if best_strategy == "Dynamic OLS":
                                    st.info("**Insight**: Dynamic hedge ratios provide better risk management than static approaches")
                                elif best_strategy == "Volatility-based":
                                    st.info("**Insight**: Volatility-based hedging adapts well to changing market conditions")
                                else:
                                    st.info("**Insight**: Static OLS provides stable but less adaptive hedging")
                                
                                evaluation_results['Hedge_Ratios'] = {
                                    'best_variance_reduction': max([effectiveness['variance_reduction'], effectiveness_ols['variance_reduction'], effectiveness_vol['variance_reduction']]),
                                    'best_risk_reduction': max([effectiveness['risk_reduction'], effectiveness_ols['risk_reduction'], effectiveness_vol['risk_reduction']]),
                                    'strategy_adaptability': effectiveness_ols['variance_reduction'] - effectiveness['variance_reduction'],
                                    'volatility_effectiveness': effectiveness_vol['variance_reduction']
                                }
                            
                            # 4. BACKTESTING EVALUATION
                            if 'backtest_results' in st.session_state:
                                st.markdown("#### Backtesting evaluation")
                                
                                backtest_results = st.session_state['backtest_results']
                                backtest = backtest_results['backtest']
                                strategies_run = backtest_results['strategies_run']
                                benchmark_returns = backtest_results['benchmark_returns']
                                configuration = backtest_results['configuration']
                                
                                # Backtesting Performance Summary
                                st.markdown("**Backtesting Performance Summary:**")
                                
                                backtest_summary = []
                                for strategy_name in strategies_run:
                                    results = backtest.results[strategy_name]
                                    metrics = backtest.calculate_performance_metrics(results['net_returns'], benchmark_returns)
                                    
                                    backtest_summary.append({
                                        'Strategy': strategy_name,
                                        'Total Return': metrics['total_return'],
                                        'Annualized Return': metrics['annualized_return'],
                                        'Sharpe Ratio': metrics['sharpe_ratio'],
                                        'Max Drawdown': metrics['max_drawdown'],
                                        'Volatility': metrics['volatility'],
                                        'Win Rate': metrics['win_rate'],
                                        'Calmar Ratio': metrics['calmar_ratio'],
                                        'Total Costs': results['total_costs'].sum(),
                                        'Cost Impact': results['total_costs'].sum() / configuration['initial_capital']
                                    })
                                
                                backtest_df = pd.DataFrame(backtest_summary)
                                st.table(backtest_df.round(4))
                                
                                # Best performing strategy
                                best_backtest_strategy = backtest_df.loc[backtest_df['Sharpe Ratio'].idxmax(), 'Strategy']
                                best_sharpe = backtest_df['Sharpe Ratio'].max()
                                
                                st.success(f"**Best Backtesting Strategy**: {best_backtest_strategy} (Sharpe: {best_sharpe:.2f})")
                                
                                # Transaction cost analysis
                                st.markdown("**Transaction Cost Analysis:**")
                                avg_cost_impact = backtest_df['Cost Impact'].mean()
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if avg_cost_impact < 0.01:
                                        st.success("(✓) Cost Impact: Low")
                                    elif avg_cost_impact < 0.05:
                                        st.warning("⚠ Cost Impact: Moderate")
                                    else:
                                        st.error("⚠ Cost Impact: High")
                                    st.metric("Avg Cost Impact", f"{avg_cost_impact:.2%}")
                                
                                with col2:
                                    st.metric("Total Strategies", len(strategies_run))
                                    st.metric("Initial Capital", f"${configuration['initial_capital']:,.0f}")
                                
                                with col3:
                                    st.metric("Best Sharpe Ratio", f"{best_sharpe:.2f}")
                                    st.metric("Best Win Rate", f"{backtest_df['Win Rate'].max():.1%}")
                                
                                evaluation_results['Backtesting'] = {
                                    'best_sharpe_ratio': best_sharpe,
                                    'avg_cost_impact': avg_cost_impact,
                                    'best_total_return': backtest_df['Total Return'].max(),
                                    'strategy_count': len(strategies_run)
                                }
                            
                            # 5. COMPREHENSIVE METHOD COMPARISON
                            if len(evaluation_results) > 1:
                                st.markdown("#### comprehensive method comparison")
                                
                                # Create comparison matrix
                                comparison_metrics = {}
                                
                                for method, results in evaluation_results.items():
                                    comparison_metrics[method] = {
                                        'Forecast Accuracy': results.get('forecast_accuracy', results.get('volatility_forecast_accuracy', 0)),
                                        'Risk Reduction': results.get('best_variance_reduction', results.get('best_risk_reduction', 0)),
                                        'Model Quality': abs(results.get('model_fit', 0)) if results.get('model_fit') else 0,
                                        'Performance': results.get('best_sharpe_ratio', results.get('best_total_return', 0))
                                    }
                                
                                comparison_df = pd.DataFrame(comparison_metrics).T
                                st.table(comparison_df.round(4))
                                
                                # Method ranking
                                st.markdown("**Method Rankings:**")
                                
                                # Rank by different criteria
                                rankings = {}
                                
                                # Forecast Accuracy Ranking
                                if 'Forecast Accuracy' in comparison_df.columns:
                                    forecast_ranking = comparison_df['Forecast Accuracy'].sort_values(ascending=False)
                                    rankings['Forecast Accuracy'] = forecast_ranking
                                
                                # Risk Reduction Ranking
                                if 'Risk Reduction' in comparison_df.columns:
                                    risk_ranking = comparison_df['Risk Reduction'].sort_values(ascending=False)
                                    rankings['Risk Reduction'] = risk_ranking
                                
                                # Model Quality Ranking
                                if 'Model Quality' in comparison_df.columns:
                                    quality_ranking = comparison_df['Model Quality'].sort_values(ascending=False)
                                    rankings['Model Quality'] = quality_ranking
                                
                                # Performance Ranking
                                if 'Performance' in comparison_df.columns:
                                    perf_ranking = comparison_df['Performance'].sort_values(ascending=False)
                                    rankings['Performance'] = perf_ranking
                                
                                # Display rankings
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if 'Forecast Accuracy' in rankings:
                                        st.markdown("**Forecast Accuracy Ranking:**")
                                        for i, (method, score) in enumerate(rankings['Forecast Accuracy'].items(), 1):
                                            st.write(f"{i}. {method}: {score:.4f}")
                                    
                                    if 'Risk Reduction' in rankings:
                                        st.markdown("**Risk Reduction Ranking:**")
                                        for i, (method, score) in enumerate(rankings['Risk Reduction'].items(), 1):
                                            st.write(f"{i}. {method}: {score:.2%}")
                                
                                with col2:
                                    if 'Model Quality' in rankings:
                                        st.markdown("**Model Quality Ranking:**")
                                        for i, (method, score) in enumerate(rankings['Model Quality'].items(), 1):
                                            st.write(f"{i}. {method}: {score:.2f}")
                                    
                                    if 'Performance' in rankings:
                                        st.markdown("**Performance Ranking:**")
                                        for i, (method, score) in enumerate(rankings['Performance'].items(), 1):
                                            st.write(f"{i}. {method}: {score:.4f}")
                                
                                # Overall recommendations
                                st.markdown("#### overall recommendations")
                                
                                # Find best overall method
                                if len(rankings) > 0:
                                    # Calculate average rank for each method
                                    method_ranks = {}
                                    for method in comparison_df.index:
                                        ranks = []
                                        for ranking_name, ranking in rankings.items():
                                            if method in ranking.index:
                                                ranks.append(ranking.index.get_loc(method) + 1)
                                        if ranks:
                                            method_ranks[method] = np.mean(ranks)
                                    
                                    # Best overall method
                                    best_overall = min(method_ranks.items(), key=lambda x: x[1])[0]
                                    
                                    st.success(f"**Best Overall Method**: {best_overall}")
                                    
                                    # Method-specific recommendations
                                    st.markdown("**Method-Specific Recommendations:**")
                                    
                                    if 'ARIMA' in evaluation_results:
                                        st.markdown("**ARIMA Forecasting**: Best for short-term price predictions and trend analysis")
                                    
                                    if 'GARCH' in evaluation_results:
                                        st.markdown("**GARCH Volatility**: Best for volatility forecasting and risk modeling")
                                    
                                    if 'Hedge_Ratios' in evaluation_results:
                                        st.markdown("**Dynamic Hedge Ratios**: Best for risk management and portfolio protection")
                                    
                                    if 'Backtesting' in evaluation_results:
                                        st.markdown("**Backtesting**: Best for strategy validation and performance assessment")
                                    
                                    # Integration recommendations
                                    st.markdown("**Integration Strategy:**")
                                    st.markdown("""
                                    **Recommended Approach**:
                                    1. Use **ARIMA** for price/return forecasting
                                    2. Use **GARCH** for volatility forecasting  
                                    3. Use **Dynamic Hedge Ratios** for risk management
                                    4. Use **Backtesting** for strategy validation
                                    5. Combine all methods for comprehensive risk management
                                    """)
                                
                                # Download comprehensive evaluation results
                                st.markdown("#### download evaluation results")
                                
                                # Create comprehensive evaluation dataset
                                eval_data = []
                                for method, results in evaluation_results.items():
                                    method_data = {'Method': method}
                                    method_data.update(results)
                                    eval_data.append(method_data)
                                
                                eval_df = pd.DataFrame(eval_data)
                                
                                # Download button for evaluation results
                                import base64
                                csv_eval = eval_df.to_csv(index=False)
                                b64_eval = base64.b64encode(csv_eval.encode()).decode()
                                href = f'''
                                    <a href="data:file/csv;base64,{b64_eval}" download="method_evaluation_results.csv"
                                       style="
                                           display: inline-block;
                                           background-color: #e26d5c;
                                           color: #fff;
                                           font-size: 1.1rem;
                                           padding: 0.5rem 1.5rem;
                                           border-radius: 0.5rem;
                                           text-decoration: none;
                                           text-align: center;
                                           margin-top: 1rem;
                                           box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                       ">
                                       download evaluation results
                                    </a>
                                '''
                                st.markdown(href, unsafe_allow_html=True)
                            
                            # Store evaluation results in session state
                            st.session_state['evaluation_results'] = evaluation_results
                            
                        except Exception as e:
                            st.error(f"Error running comprehensive evaluation: {str(e)}")
                            st.error(f"Full error: {e}")
                
                # Show method comparison plots if available
                if st.checkbox("Show method comparison visualizations"):
                    if 'evaluation_results' in st.session_state and len(st.session_state['evaluation_results']) > 1:
                        try:
                            evaluation_results = st.session_state['evaluation_results']
                            
                            # Create comparison visualization
                            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                            
                            # Prepare data for plotting
                            methods = list(evaluation_results.keys())
                            
                            # Plot 1: Forecast Accuracy Comparison
                            forecast_accuracies = []
                            for method in methods:
                                results = evaluation_results[method]
                                if 'forecast_accuracy' in results:
                                    forecast_accuracies.append(results['forecast_accuracy'])
                                elif 'volatility_forecast_accuracy' in results:
                                    forecast_accuracies.append(results['volatility_forecast_accuracy'])
                                else:
                                    forecast_accuracies.append(0)
                            
                            if any(acc > 0 for acc in forecast_accuracies):
                                axes[0, 0].bar(methods, forecast_accuracies, color=['#e26d5c', '#e74c3c', '#3498db', '#f39c12'][:len(methods)])
                                axes[0, 0].set_title('Forecast Accuracy Comparison', fontsize=14, fontweight='bold')
                                axes[0, 0].set_ylabel('Accuracy')
                                axes[0, 0].grid(True, alpha=0.3)
                            
                            # Plot 2: Risk Reduction Comparison
                            risk_reductions = []
                            for method in methods:
                                results = evaluation_results[method]
                                if 'best_variance_reduction' in results:
                                    risk_reductions.append(results['best_variance_reduction'])
                                elif 'best_risk_reduction' in results:
                                    risk_reductions.append(results['best_risk_reduction'])
                                else:
                                    risk_reductions.append(0)
                            
                            if any(rr > 0 for rr in risk_reductions):
                                axes[0, 1].bar(methods, risk_reductions, color=['#e26d5c', '#e74c3c', '#3498db', '#f39c12'][:len(methods)])
                                axes[0, 1].set_title('Risk Reduction Comparison', fontsize=14, fontweight='bold')
                                axes[0, 1].set_ylabel('Risk Reduction')
                                axes[0, 1].grid(True, alpha=0.3)
                            
                            # Plot 3: Model Quality Comparison
                            model_qualities = []
                            for method in methods:
                                results = evaluation_results[method]
                                if 'model_fit' in results:
                                    model_qualities.append(abs(results['model_fit']))
                                else:
                                    model_qualities.append(0)
                            
                            if any(mq > 0 for mq in model_qualities):
                                axes[1, 0].bar(methods, model_qualities, color=['#e26d5c', '#e74c3c', '#3498db', '#f39c12'][:len(methods)])
                                axes[1, 0].set_title('Model Quality Comparison', fontsize=14, fontweight='bold')
                                axes[1, 0].set_ylabel('Model Quality (|AIC|)')
                                axes[1, 0].grid(True, alpha=0.3)
                            
                            # Plot 4: Performance Comparison
                            performances = []
                            for method in methods:
                                results = evaluation_results[method]
                                if 'best_sharpe_ratio' in results:
                                    performances.append(results['best_sharpe_ratio'])
                                elif 'best_total_return' in results:
                                    performances.append(results['best_total_return'])
                                else:
                                    performances.append(0)
                            
                            if any(perf > 0 for perf in performances):
                                axes[1, 1].bar(methods, performances, color=['#e26d5c', '#e74c3c', '#3498db', '#f39c12'][:len(methods)])
                                axes[1, 1].set_title('Performance Comparison', fontsize=14, fontweight='bold')
                                axes[1, 1].set_ylabel('Performance')
                                axes[1, 1].grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                        except Exception as e:
                            st.error(f"Error generating comparison visualizations: {str(e)}")
                    else:
                        st.warning("⚠ **No evaluation results available** - Run comprehensive evaluation first")
            
            # --- END EVALUATE & COMPARE METHODS SECTION ---
            
            # --- ROBUSTNESS CHECKS SECTION ---
            st.header("robustness checks")
            st.markdown("### model validation and stability analysis")
            
            # Check if we have results from different methods
            methods_available = []
            
            if 'arima_results' in st.session_state:
                methods_available.append('ARIMA Forecasting')
            
            if 'garch_results' in st.session_state:
                methods_available.append('GARCH Volatility')
            
            if 'hedge_results' in st.session_state:
                methods_available.append('Dynamic Hedge Ratios')
            
            if 'backtest_results' in st.session_state:
                methods_available.append('Backtesting Results')
            
            if not methods_available:
                st.warning("⚠ **No method results available** - Please run ARIMA forecasting, GARCH volatility, hedge ratio calculation, and backtesting first to enable robustness checks.")
            else:
                st.success(f"(✓) **Methods Available for Robustness Checks**: {', '.join(methods_available)}")
                
                # Robustness checks configuration
                st.markdown("#### robustness checks configuration")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    check_type = st.selectbox(
                        "Robustness Check Type",
                        ["Walk-Forward Validation", "Parameter Sensitivity", "Regime Analysis", "Stress Testing", "Cross-Validation Stability", "Outlier Robustness", "Comprehensive Analysis"]
                    )
                
                with col2:
                    target_method = st.selectbox(
                        "Target Method",
                        methods_available
                    )
                
                with col3:
                    include_plots = st.checkbox("Include Visualization Plots", value=True)
                
                # Run robustness checks
                if st.button("run robustness checks", type="primary"):
                    with st.spinner("Running robustness checks..."):
                        try:
                            # Initialize robustness checker
                            robustness_checker = RobustnessChecker()
                            all_results = {}
                            
                            # Get data based on target method
                            if target_method == 'ARIMA Forecasting' and 'arima_results' in st.session_state:
                                data = st.session_state['arima_results']['log_returns']
                                
                                # Define ARIMA model function for walk-forward validation
                                def arima_model_func(train_data, forecast_steps):
                                    from statsmodels.tsa.arima.model import ARIMA
                                    try:
                                        # Auto-select order
                                        model = ARIMA(train_data, order=(5, 0, 5))
                                        fitted_model = model.fit()
                                        forecast = fitted_model.forecast(steps=forecast_steps)
                                        return forecast.values
                                    except:
                                        return np.zeros(forecast_steps)
                                
                                # Define ARIMA performance function for sensitivity analysis
                                def arima_performance_func(data, params):
                                    from statsmodels.tsa.arima.model import ARIMA
                                    try:
                                        model = ARIMA(data, order=(params.get('p', 1), params.get('d', 0), params.get('q', 1)))
                                        fitted_model = model.fit()
                                        return fitted_model.aic
                                    except:
                                        return 1000
                            
                            elif target_method == 'GARCH Volatility' and 'garch_results' in st.session_state:
                                data = st.session_state['garch_results']['returns']
                                
                                # Define GARCH model function
                                def garch_model_func(train_data, forecast_steps):
                                    try:
                                        from arch.univariate import GARCH
                                        model = GARCH(train_data, vol='GARCH', p=1, q=1)
                                        fitted_model = model.fit(disp='off')
                                        forecast = fitted_model.forecast(horizon=forecast_steps)
                                        return np.sqrt(forecast.variance.values[-1, :])
                                    except:
                                        return np.ones(forecast_steps) * train_data.std()
                                
                                def garch_performance_func(data, params):
                                    try:
                                        from arch.univariate import GARCH
                                        model = GARCH(data, vol='GARCH', p=params.get('p', 1), q=params.get('q', 1))
                                        fitted_model = model.fit(disp='off')
                                        return fitted_model.aic
                                    except:
                                        return 1000
                            
                            elif target_method == 'Dynamic Hedge Ratios' and 'hedge_results' in st.session_state:
                                data = st.session_state['hedge_results']['asset_returns']
                                
                                def hedge_model_func(train_data, forecast_steps):
                                    # Simple rolling beta calculation
                                    if len(train_data) >= 60:
                                        rolling_beta = train_data.rolling(window=60).mean()
                                        return np.full(forecast_steps, rolling_beta.iloc[-1])
                                    else:
                                        return np.zeros(forecast_steps)
                                
                                def hedge_performance_func(data, params):
                                    window = params.get('window', 60)
                                    if len(data) >= window:
                                        rolling_vol = data.rolling(window=window).std()
                                        return rolling_vol.mean()
                                    else:
                                        return data.std()
                            
                            else:
                                st.error(f"No data available for {target_method}")
                                return
                            
                            # Run selected robustness check
                            if check_type == "Walk-Forward Validation":
                                st.markdown("#### walk-forward validation")
                                
                                # Configuration
                                col1, col2 = st.columns(2)
                                with col1:
                                    window_size = st.number_input("Training Window Size", min_value=60, max_value=500, value=252)
                                with col2:
                                    step_size = st.number_input("Step Size", min_value=10, max_value=100, value=63)
                                
                                # Run walk-forward validation
                                wf_results = robustness_checker.walk_forward_validation(
                                    data, arima_model_func if target_method == 'ARIMA Forecasting' else 
                                    garch_model_func if target_method == 'GARCH Volatility' else hedge_model_func,
                                    window_size=window_size, step_size=step_size
                                )
                                
                                all_results['walk_forward'] = wf_results
                                
                                # Display results
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("RMSE", f"{wf_results['rmse']:.6f}")
                                with col2:
                                    st.metric("MAE", f"{wf_results['mae']:.6f}")
                                with col3:
                                    st.metric("Directional Accuracy", f"{wf_results['directional_accuracy']:.2%}")
                                with col4:
                                    st.metric("Validation Periods", f"{len(wf_results['predictions']):,}")
                                
                                # Plot results
                                if include_plots:
                                    fig_wf = robustness_checker.plot_walk_forward_results(wf_results)
                                    st.pyplot(fig_wf)
                            
                            elif check_type == "Parameter Sensitivity":
                                st.markdown("#### parameter sensitivity analysis")
                                
                                # Define parameter ranges based on method
                                if target_method == 'ARIMA Forecasting':
                                    base_params = {'p': 1, 'd': 0, 'q': 1}
                                    param_ranges = {
                                        'p': range(0, 6),
                                        'd': range(0, 3),
                                        'q': range(0, 6)
                                    }
                                elif target_method == 'GARCH Volatility':
                                    base_params = {'p': 1, 'q': 1}
                                    param_ranges = {
                                        'p': range(1, 4),
                                        'q': range(1, 4)
                                    }
                                else:  # Hedge Ratios
                                    base_params = {'window': 60}
                                    param_ranges = {
                                        'window': range(20, 121, 20)
                                    }
                                
                                # Run sensitivity analysis
                                sensitivity_results = robustness_checker.parameter_sensitivity_analysis(
                                    data, base_params, param_ranges,
                                    arima_performance_func if target_method == 'ARIMA Forecasting' else
                                    garch_performance_func if target_method == 'GARCH Volatility' else hedge_performance_func
                                )
                                
                                all_results['sensitivity'] = sensitivity_results
                                
                                # Display results
                                st.markdown("**Parameter Sensitivity Results:**")
                                for param_name, results in sensitivity_results.items():
                                    st.markdown(f"**{param_name}:**")
                                    st.table(results.round(4))
                                
                                # Plot results
                                if include_plots:
                                    fig_sens = robustness_checker.plot_sensitivity_analysis(sensitivity_results)
                                    st.pyplot(fig_sens)
                            
                            elif check_type == "Regime Analysis":
                                st.markdown("#### regime analysis")
                                
                                # Run regime analysis
                                regime_results = robustness_checker.regime_analysis(data)
                                all_results['regime_analysis'] = regime_results
                                
                                # Display results
                                st.markdown("**Market Regime Analysis:**")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Low Volatility Periods", f"{regime_results['regime_stats'].get('Low Volatility', {}).get('count', 0):,}")
                                with col2:
                                    st.metric("Medium Volatility Periods", f"{regime_results['regime_stats'].get('Medium Volatility', {}).get('count', 0):,}")
                                with col3:
                                    st.metric("High Volatility Periods", f"{regime_results['regime_stats'].get('High Volatility', {}).get('count', 0):,}")
                                
                                # Regime statistics table
                                regime_stats_df = pd.DataFrame(regime_results['regime_stats']).T
                                st.markdown("**Regime Statistics:**")
                                st.table(regime_stats_df.round(4))
                                
                                # Plot results
                                if include_plots:
                                    fig_regime = robustness_checker.plot_regime_analysis(regime_results)
                                    st.pyplot(fig_regime)
                            
                            elif check_type == "Stress Testing":
                                st.markdown("#### stress testing")
                                
                                # Define stress scenarios
                                stress_scenarios = {
                                    'Volatility Shock (2x)': {'volatility_shock': 2.0},
                                    'Volatility Shock (3x)': {'volatility_shock': 3.0},
                                    'Trend Shock': {'trend_shock': -0.5},
                                    'Jump Shock': {'jump_shock': -0.1, 'jump_probability': 0.02},
                                    'Combined Stress': {'volatility_shock': 2.0, 'trend_shock': -0.3, 'jump_shock': -0.05}
                                }
                                
                                # Run stress testing
                                stress_results = robustness_checker.stress_testing(data, stress_scenarios)
                                all_results['stress_testing'] = stress_results
                                
                                # Display results
                                st.markdown("**Stress Test Results:**")
                                
                                stress_summary = []
                                for scenario_name, results in stress_results.items():
                                    stress_summary.append({
                                        'Scenario': scenario_name,
                                        'Max Drawdown': f"{results['max_drawdown']:.2%}",
                                        'VaR (95%)': f"{results['var_95']:.2%}",
                                        'CVaR (95%)': f"{results['cvar_95']:.2%}",
                                        'Volatility': f"{results['volatility']:.4f}"
                                    })
                                
                                stress_df = pd.DataFrame(stress_summary)
                                st.table(stress_df)
                                
                                # Plot results
                                if include_plots:
                                    fig_stress = robustness_checker.plot_stress_test_results(stress_results)
                                    st.pyplot(fig_stress)
                            
                            elif check_type == "Cross-Validation Stability":
                                st.markdown("#### cross-validation stability")
                                
                                # Configuration
                                cv_splits = st.number_input("CV Splits", min_value=3, max_value=10, value=5)
                                
                                # Run cross-validation
                                cv_results = robustness_checker.cross_validation_stability(
                                    data, 
                                    arima_model_func if target_method == 'ARIMA Forecasting' else 
                                    garch_model_func if target_method == 'GARCH Volatility' else hedge_model_func,
                                    cv_splits=cv_splits
                                )
                                
                                all_results['cv_stability'] = cv_results
                                
                                # Display results
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("CV Mean Score", f"{cv_results['cv_mean']:.6f}")
                                with col2:
                                    st.metric("CV Std", f"{cv_results['cv_std']:.6f}")
                                with col3:
                                    st.metric("CV Coefficient of Variation", f"{cv_results['cv_cv']:.4f}")
                                with col4:
                                    st.metric("Stability Score", f"{cv_results['stability_score']:.4f}")
                                
                                # CV scores table
                                cv_scores_df = pd.DataFrame({
                                    'Fold': range(1, len(cv_results['cv_scores']) + 1),
                                    'Score': cv_results['cv_scores']
                                })
                                st.markdown("**Cross-Validation Scores:**")
                                st.table(cv_scores_df.round(6))
                            
                            elif check_type == "Outlier Robustness":
                                st.markdown("#### outlier robustness")
                                
                                # Run outlier robustness test
                                outlier_results = robustness_checker.outlier_robustness(data)
                                all_results['outlier_robustness'] = outlier_results
                                
                                # Display results
                                st.markdown("**Outlier Robustness Results:**")
                                
                                for fraction, results in outlier_results.items():
                                    st.markdown(f"**{fraction.replace('_', ' ').title()}:**")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Original Statistics:**")
                                        orig_stats = pd.DataFrame([results['original_stats']])
                                        st.table(orig_stats.round(4))
                                    
                                    with col2:
                                        st.markdown("**Contaminated Statistics:**")
                                        cont_stats = pd.DataFrame([results['contaminated_stats']])
                                        st.table(cont_stats.round(4))
                                    
                                    st.markdown("**Robustness Metrics (Lower is better):**")
                                    robustness_df = pd.DataFrame([results['robustness_metrics']])
                                    st.table(robustness_df.round(4))
                            
                            elif check_type == "Comprehensive Analysis":
                                st.markdown("#### comprehensive robustness analysis")
                                
                                # Run all robustness checks
                                st.info("Running comprehensive robustness analysis... This may take a few minutes.")
                                
                                # Walk-forward validation
                                wf_results = robustness_checker.walk_forward_validation(
                                    data, arima_model_func if target_method == 'ARIMA Forecasting' else 
                                    garch_model_func if target_method == 'GARCH Volatility' else hedge_model_func
                                )
                                all_results['walk_forward'] = wf_results
                                
                                # Cross-validation stability
                                cv_results = robustness_checker.cross_validation_stability(
                                    data, arima_model_func if target_method == 'ARIMA Forecasting' else 
                                    garch_model_func if target_method == 'GARCH Volatility' else hedge_model_func
                                )
                                all_results['cv_stability'] = cv_results
                                
                                # Regime analysis
                                regime_results = robustness_checker.regime_analysis(data)
                                all_results['regime_analysis'] = regime_results
                                
                                # Stress testing
                                stress_scenarios = {
                                    'Volatility Shock': {'volatility_shock': 2.0},
                                    'Trend Shock': {'trend_shock': -0.3},
                                    'Jump Shock': {'jump_shock': -0.1, 'jump_probability': 0.01}
                                }
                                stress_results = robustness_checker.stress_testing(data, stress_scenarios)
                                all_results['stress_testing'] = stress_results
                                
                                # Generate comprehensive report
                                report = robustness_checker.generate_robustness_report(all_results)
                                
                                # Display comprehensive results
                                st.markdown("#### comprehensive robustness report")
                                
                                # Overall robustness score
                                if 'overall_robustness_score' in report['summary']:
                                    overall_score = report['summary']['overall_robustness_score']
                                    st.markdown(f"**Overall Robustness Score: {overall_score:.4f}**")
                                    
                                    if overall_score >= 0.8:
                                        st.success(" **Excellent Robustness** - Model shows high stability and reliability")
                                    elif overall_score >= 0.6:
                                        st.warning("⚠ **Good Robustness** - Model shows acceptable stability with some areas for improvement")
                                    else:
                                        st.error("**X Poor Robustness** - Model shows significant stability issues")
                                
                                # Summary metrics
                                st.markdown("**Summary Metrics:**")
                                summary_data = []
                                
                                if 'walk_forward' in report['summary']:
                                    wf_summary = report['summary']['walk_forward']
                                    summary_data.append({
                                        'Check': 'Walk-Forward Validation',
                                        'RMSE': f"{wf_summary['rmse']:.6f}",
                                        'MAE': f"{wf_summary['mae']:.6f}",
                                        'Directional Accuracy': f"{wf_summary['directional_accuracy']:.2%}"
                                    })
                                
                                if 'cv_stability' in report['summary']:
                                    cv_summary = report['summary']['cv_stability']
                                    summary_data.append({
                                        'Check': 'Cross-Validation Stability',
                                        'Stability Score': f"{cv_summary['stability_score']:.4f}",
                                        'CV Coefficient': f"{cv_summary['cv_cv']:.4f}",
                                        'Directional Accuracy': '-'
                                    })
                                
                                if 'stress_testing' in report['summary']:
                                    stress_summary = report['summary']['stress_testing']
                                    summary_data.append({
                                        'Check': 'Stress Testing',
                                        'Worst Case Drawdown': f"{stress_summary['worst_case_drawdown']:.2%}",
                                        'Avg VaR (95%)': f"{stress_summary['avg_var_95']:.2%}",
                                        'Directional Accuracy': '-'
                                    })
                                
                                if summary_data:
                                    summary_df = pd.DataFrame(summary_data)
                                    st.table(summary_df)
                                
                                # Recommendations
                                if report['recommendations']:
                                    st.markdown("**Recommendations:**")
                                    for i, recommendation in enumerate(report['recommendations'], 1):
                                        st.markdown(f"{i}. {recommendation}")
                                
                                # Store results in session state
                                st.session_state['robustness_results'] = {
                                    'checker': robustness_checker,
                                    'all_results': all_results,
                                    'report': report,
                                    'target_method': target_method
                                }
                                
                                # Download robustness results
                                st.markdown("#### download robustness results")
                                
                                # Create comprehensive robustness dataset
                                robustness_data = []
                                
                                # Walk-forward results
                                if 'walk_forward' in all_results:
                                    wf_data = all_results['walk_forward']
                                    for i in range(len(wf_data['predictions'])):
                                        robustness_data.append({
                                            'Date': wf_data['dates'][i] if i < len(wf_data['dates']) else None,
                                            'Check_Type': 'Walk_Forward',
                                            'Actual': wf_data['actuals'][i],
                                            'Predicted': wf_data['predictions'][i],
                                            'Residual': wf_data['actuals'][i] - wf_data['predictions'][i]
                                        })
                                
                                # Cross-validation results
                                if 'cv_stability' in all_results:
                                    cv_data = all_results['cv_stability']
                                    for i, score in enumerate(cv_data['cv_scores']):
                                        robustness_data.append({
                                            'Date': None,
                                            'Check_Type': f'CV_Fold_{i+1}',
                                            'Actual': None,
                                            'Predicted': None,
                                            'Residual': score
                                        })
                                
                                if robustness_data:
                                    robustness_df = pd.DataFrame(robustness_data)
                                    
                                    # Download button for robustness results
                                    import base64
                                    csv_robustness = robustness_df.to_csv(index=False)
                                    b64_robustness = base64.b64encode(csv_robustness.encode()).decode()
                                    href = f'''
                                        <a href="data:file/csv;base64,{b64_robustness}" download="robustness_checks_{target_method.replace(" ", "_")}.csv"
                                           style="
                                               display: inline-block;
                                               background-color: #e26d5c;
                                               color: #fff;
                                               font-size: 1.1rem;
                                               padding: 0.5rem 1.5rem;
                                               border-radius: 0.5rem;
                                               text-decoration: none;
                                               text-align: center;
                                               margin-top: 1rem;
                                               box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                           ">
                                           download robustness results
                                        </a>
                                    '''
                                    st.markdown(href, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error running robustness checks: {str(e)}")
                            st.error(f"Full error: {e}")
                    
                    # Show robustness analysis plots if available
                    if st.checkbox("Show robustness analysis visualizations"):
                        if 'robustness_results' in st.session_state:
                            try:
                                robustness_results = st.session_state['robustness_results']
                                all_results = robustness_results['all_results']
                                
                                # Show available plots
                                available_plots = []
                                if 'walk_forward' in all_results:
                                    available_plots.append('Walk-Forward Results')
                                if 'sensitivity' in all_results:
                                    available_plots.append('Parameter Sensitivity')
                                if 'regime_analysis' in all_results:
                                    available_plots.append('Regime Analysis')
                                if 'stress_testing' in all_results:
                                    available_plots.append('Stress Test Results')
                                
                                if available_plots:
                                    plot_choice = st.selectbox("Select Plot to Display", available_plots)
                                    
                                    if plot_choice == 'Walk-Forward Results':
                                        fig = robustness_results['checker'].plot_walk_forward_results(all_results['walk_forward'])
                                        st.pyplot(fig)
                                    elif plot_choice == 'Parameter Sensitivity':
                                        fig = robustness_results['checker'].plot_sensitivity_analysis(all_results['sensitivity'])
                                        st.pyplot(fig)
                                    elif plot_choice == 'Regime Analysis':
                                        fig = robustness_results['checker'].plot_regime_analysis(all_results['regime_analysis'])
                                        st.pyplot(fig)
                                    elif plot_choice == 'Stress Test Results':
                                        fig = robustness_results['checker'].plot_stress_test_results(all_results['stress_testing'])
                                        st.pyplot(fig)
                                else:
                                    st.warning("⚠ **No robustness plots available** - Run robustness checks first")
                                    
                            except Exception as e:
                                st.error(f"Error generating robustness plots: {str(e)}")
                        else:
                            st.warning("⚠ **No robustness results available** - Run robustness checks first")
            
            # --- END ROBUSTNESS CHECKS SECTION ---
            
            # --- REPORTING SECTION ---
            st.header("reporting")
            st.markdown("### analysis reports and documentation")
            
            # Check if we have results from different methods
            methods_available = []
            
            if 'arima_results' in st.session_state:
                methods_available.append('ARIMA Forecasting')
            
            if 'garch_results' in st.session_state:
                methods_available.append('GARCH Volatility')
            
            if 'hedge_results' in st.session_state:
                methods_available.append('Dynamic Hedge Ratios')
            
            if 'backtest_results' in st.session_state:
                methods_available.append('Backtesting Results')
            
            if 'evaluation_results' in st.session_state:
                methods_available.append('Method Evaluation')
            
            if 'robustness_results' in st.session_state:
                methods_available.append('Robustness Checks')
            
            if not methods_available:
                st.warning("⚠ **No analysis results available** - Please run ARIMA forecasting, GARCH volatility, hedge ratio calculation, backtesting, evaluation, and robustness checks first to enable comprehensive reporting.")
            else:
                st.success(f"(✓) **Methods Available for Reporting**: {', '.join(methods_available)}")
                
                # Reporting configuration
                st.markdown("#### reporting configuration")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    report_type = st.selectbox(
                        "Report Type",
                        ["Executive Summary", "Detailed Technical Report", "Comprehensive Analysis Report", "Custom Report"]
                    )
                
                with col2:
                    include_visualizations = st.checkbox("Include Visualizations", value=True)
                
                with col3:
                    export_format = st.selectbox(
                        "Export Format",
                        ["CSV", "JSON", "PDF Summary"]
                    )
                
                # Generate report button
                if st.button("generate comprehensive report", type="primary"):
                    with st.spinner("Generating comprehensive report..."):
                        try:
                            # Initialize reporter
                            reporter = OilHedgingReporter()
                            
                            # Collect all available results
                            all_results = {}
                            
                            if 'arima_results' in st.session_state:
                                all_results['arima_results'] = st.session_state['arima_results']
                            
                            if 'garch_results' in st.session_state:
                                all_results['garch_results'] = st.session_state['garch_results']
                            
                            if 'hedge_results' in st.session_state:
                                all_results['hedge_results'] = st.session_state['hedge_results']
                            
                            if 'backtest_results' in st.session_state:
                                all_results['backtest_results'] = st.session_state['backtest_results']
                            
                            if 'evaluation_results' in st.session_state:
                                all_results['evaluation_results'] = st.session_state['evaluation_results']
                            
                            if 'robustness_results' in st.session_state:
                                all_results['robustness_results'] = st.session_state['robustness_results']
                            
                            # Generate report based on type
                            if report_type == "Executive Summary":
                                report = reporter.generate_executive_summary(all_results)
                                
                                # Display executive summary
                                st.markdown("#### executive summary")
                                
                                # Report metadata
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Report Date", report.get('report_date', 'N/A'))
                                with col2:
                                    st.metric("Analysis Period", report.get('analysis_period', 'N/A'))
                                with col3:
                                    risk_level = report.get('risk_assessment', {}).get('risk_level', 'N/A')
                                    st.metric("Risk Level", risk_level)
                                
                                # Key findings
                                st.markdown("**Key Findings:**")
                                key_findings = report.get('key_findings', [])
                                for i, finding in enumerate(key_findings, 1):
                                    st.markdown(f"{i}. {finding}")
                                
                                # Recommendations
                                st.markdown("**Recommendations:**")
                                recommendations = report.get('recommendations', [])
                                for i, recommendation in enumerate(recommendations, 1):
                                    st.markdown(f"{i}. {recommendation}")
                                
                                # Risk assessment
                                st.markdown("**Risk Assessment:**")
                                risk_assessment = report.get('risk_assessment', {})
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Risk Level:**")
                                    if risk_assessment['risk_level'] == 'High':
                                        st.error(f"{risk_assessment['risk_level']}")
                                    elif risk_assessment['risk_level'] == 'Medium':
                                        st.warning(f"{risk_assessment['risk_level']}")
                                    else:
                                        st.success(f"{risk_assessment['risk_level']}")
                                
                                with col2:
                                    st.markdown("**Key Risks:**")
                                    for risk in risk_assessment['key_risks']:
                                        st.markdown(f"• {risk}")
                                
                                # Performance summary
                                st.markdown("**Performance Summary:**")
                                performance_summary = report.get('performance_summary', {})
                                
                                if 'forecast_accuracy' in performance_summary:
                                    forecast_acc = performance_summary['forecast_accuracy']
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Forecast RMSE", f"{forecast_acc['rmse']:.6f}")
                                    with col2:
                                        st.metric("Forecast MAE", f"{forecast_acc['mae']:.6f}")
                                    with col3:
                                        st.metric("Improvement vs Naive", f"{forecast_acc['improvement_vs_naive']:.1f}%")
                                
                                if 'hedging_effectiveness' in performance_summary:
                                    hedge_eff = performance_summary['hedging_effectiveness']
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Best Strategy", hedge_eff['best_strategy'])
                                    with col2:
                                        st.metric("Variance Reduction", f"{hedge_eff['variance_reduction']:.2%}")
                                    with col3:
                                        st.metric("Risk Reduction", f"{hedge_eff['risk_reduction']:.2%}")
                                
                                if 'backtesting' in performance_summary:
                                    backtest_perf = performance_summary['backtesting']
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Best Strategy", backtest_perf['best_strategy'])
                                    with col2:
                                        st.metric("Best Sharpe Ratio", f"{backtest_perf['best_sharpe']:.2f}")
                                    with col3:
                                        st.metric("Total Strategies", backtest_perf['total_strategies'])
                            
                            elif report_type == "Detailed Technical Report":
                                report = reporter.generate_detailed_report(all_results)
                                
                                # Display detailed report
                                st.markdown("#### detailed technical report")
                                
                                # Executive summary section
                                st.markdown("**Executive Summary:**")
                                exec_summary = report.get('executive_summary', {})
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Report Date", exec_summary.get('report_date', 'N/A'))
                                with col2:
                                    st.metric("Analysis Period", exec_summary.get('analysis_period', 'N/A'))
                                with col3:
                                    risk_level = exec_summary.get('risk_assessment', {}).get('risk_level', 'N/A')
                                    st.metric("Risk Level", risk_level)
                                
                                # Methodology
                                st.markdown("**Methodology:**")
                                methodology = report.get('methodology', {})
                                
                                st.markdown("**Data Sources:**")
                                for source, description in methodology['data_sources'].items():
                                    st.markdown(f"• **{source.replace('_', ' ').title()}**: {description}")
                                
                                st.markdown("**Methods Used:**")
                                for method, description in methodology['methods_used'].items():
                                    st.markdown(f"• **{method.replace('_', ' ').title()}**: {description}")
                                
                                # Detailed results by method
                                st.markdown("**Detailed Results by Method:**")
                                
                                detailed_results = report.get('detailed_results', {})
                                for method_name, method_results in detailed_results.items():
                                    st.markdown(f"**{method_name.replace('_', ' ').title()}:**")
                                    
                                    if isinstance(method_results, dict):
                                        # Display key metrics
                                        if 'model_specification' in method_results:
                                            spec = method_results['model_specification']
                                            col1, col2, col3, col4 = st.columns(4)
                                            with col1:
                                                if 'order' in spec:
                                                    st.metric("Model Order", str(spec['order']))
                                            with col2:
                                                if 'aic' in spec:
                                                    st.metric("AIC", f"{spec['aic']:.2f}")
                                            with col3:
                                                if 'bic' in spec:
                                                    st.metric("BIC", f"{spec['bic']:.2f}")
                                            with col4:
                                                if 'log_likelihood' in spec:
                                                    st.metric("Log-Likelihood", f"{spec['log_likelihood']:.2f}")
                                        
                                        if 'forecast_performance' in method_results:
                                            perf = method_results['forecast_performance']
                                            col1, col2, col3 = st.columns(3)
                                            with col1:
                                                if 'rmse' in perf:
                                                    st.metric("RMSE", f"{perf['rmse']:.6f}")
                                            with col2:
                                                if 'mae' in perf:
                                                    st.metric("MAE", f"{perf['mae']:.6f}")
                                            with col3:
                                                if 'improvement_vs_naive' in perf:
                                                    st.metric("Improvement vs Naive", f"{perf['improvement_vs_naive']:.1f}%")
                                        
                                        if 'static_ols' in method_results:
                                            static = method_results['static_ols']
                                            col1, col2, col3, col4 = st.columns(4)
                                            with col1:
                                                st.metric("Variance Reduction", f"{static['variance_reduction']:.2%}")
                                            with col2:
                                                st.metric("Risk Reduction", f"{static['risk_reduction']:.2%}")
                                            with col3:
                                                st.metric("Hedged Volatility", f"{static['hedged_volatility']:.4f}")
                                            with col4:
                                                st.metric("Unhedged Volatility", f"{static['unhedged_volatility']:.4f}")
                                        
                                        if 'strategy_results' in method_results:
                                            strategies = method_results['strategy_results']
                                            st.markdown("**Strategy Performance:**")
                                            
                                            strategy_data = []
                                            for strategy, results in strategies.items():
                                                strategy_data.append({
                                                    'Strategy': strategy,
                                                    'Total Return': f"{results['total_return']:.2%}",
                                                    'Sharpe Ratio': f"{results['sharpe_ratio']:.2f}",
                                                    'Max Drawdown': f"{results['max_drawdown']:.2%}",
                                                    'Volatility': f"{results['volatility']:.2%}",
                                                    'Win Rate': f"{results['win_rate']:.1%}"
                                                })
                                            
                                            strategy_df = pd.DataFrame(strategy_data)
                                            st.table(strategy_df)
                            
                            elif report_type == "Comprehensive Analysis Report":
                                # Generate both executive summary and detailed report
                                exec_summary = reporter.generate_executive_summary(all_results)
                                detailed_report = reporter.generate_detailed_report(all_results)
                                
                                # Display comprehensive report
                                st.markdown("#### comprehensive analysis report")
                                
                                # Executive summary
                                st.markdown("**Executive Summary:**")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Report Date", exec_summary.get('report_date', 'N/A'))
                                with col2:
                                    st.metric("Analysis Period", exec_summary.get('analysis_period', 'N/A'))
                                with col3:
                                    risk_level = exec_summary.get('risk_assessment', {}).get('risk_level', 'N/A')
                                    st.metric("Risk Level", risk_level)
                                
                                # Key findings and recommendations
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Key Findings:**")
                                    key_findings = exec_summary.get('key_findings', [])
                                    for i, finding in enumerate(key_findings, 1):
                                        st.markdown(f"{i}. {finding}")
                                
                                with col2:
                                    st.markdown("**Recommendations:**")
                                    recommendations = exec_summary.get('recommendations', [])
                                    for i, recommendation in enumerate(recommendations, 1):
                                        st.markdown(f"{i}. {recommendation}")
                                
                                # Method comparison
                                if 'evaluation_results' in all_results:
                                    st.markdown("**Method Comparison:**")
                                    evaluation_results = all_results['evaluation_results']
                                    
                                    comparison_data = []
                                    for method, results in evaluation_results.items():
                                        comparison_data.append({
                                            'Method': method,
                                            'Forecast Accuracy': results.get('forecast_accuracy', results.get('volatility_forecast_accuracy', 0)),
                                            'Risk Reduction': results.get('best_variance_reduction', results.get('best_risk_reduction', 0)),
                                            'Model Quality': abs(results.get('model_fit', 0)) if results.get('model_fit') else 0,
                                            'Performance': results.get('best_sharpe_ratio', results.get('best_total_return', 0))
                                        })
                                    
                                    comparison_df = pd.DataFrame(comparison_data)
                                    st.table(comparison_df.round(4))
                                
                                # Risk assessment
                                st.markdown("**Risk Assessment:**")
                                risk_assessment = exec_summary.get('risk_assessment', {})
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Risk Level:**")
                                    if risk_assessment['risk_level'] == 'High':
                                        st.error(f"{risk_assessment['risk_level']}")
                                    elif risk_assessment['risk_level'] == 'Medium':
                                        st.warning(f"{risk_assessment['risk_level']}")
                                    else:
                                        st.success(f"{risk_assessment['risk_level']}")
                                
                                with col2:
                                    st.markdown("**Key Risks:**")
                                    for risk in risk_assessment['key_risks']:
                                        st.markdown(f"• {risk}")
                                
                                # Performance summary
                                st.markdown("**Performance Summary:**")
                                performance_summary = exec_summary.get('performance_summary', {})
                                
                                if performance_summary:
                                    # Create performance metrics display
                                    perf_data = []
                                    for category, metrics in performance_summary.items():
                                        if isinstance(metrics, dict):
                                            for metric, value in metrics.items():
                                                if isinstance(value, (int, float)):
                                                    perf_data.append({
                                                        'Category': category.replace('_', ' ').title(),
                                                        'Metric': metric.replace('_', ' ').title(),
                                                        'Value': value
                                                    })
                                    
                                    if perf_data:
                                        perf_df = pd.DataFrame(perf_data)
                                        st.table(perf_df.round(4))
                            
                            # Generate visualizations if requested
                            if include_visualizations:
                                st.markdown("#### report visualizations")
                                
                                try:
                                    figs = reporter.create_report_visualizations(all_results)
                                    
                                    if 'performance_comparison' in figs:
                                        st.markdown("**Performance Comparison:**")
                                        st.pyplot(figs['performance_comparison'])
                                    
                                    if 'risk_assessment' in figs:
                                        st.markdown("**Risk Assessment:**")
                                        st.pyplot(figs['risk_assessment'])
                                    
                                    if 'method_effectiveness' in figs:
                                        st.markdown("**Method Effectiveness:**")
                                        st.pyplot(figs['method_effectiveness'])
                                    
                                except Exception as e:
                                    st.error(f"Error generating visualizations: {str(e)}")
                            
                            # Export report
                            st.markdown("#### export report")
                            
                            if export_format == "CSV":
                                if report_type == "Executive Summary":
                                    report = reporter.generate_executive_summary(all_results)
                                elif report_type == "Detailed Technical Report":
                                    report = reporter.generate_detailed_report(all_results)
                                else:
                                    report = reporter.generate_detailed_report(all_results)
                                
                                export_result = reporter.export_report_to_csv(report, f"oil_hedging_{report_type.lower().replace(' ', '_')}")
                                st.success(f"(✓) {export_result}")
                                
                                # Download button for CSV files
                                st.markdown("**Download CSV Reports:**")
                                st.markdown("""
                                The following CSV files have been generated:
                                - Executive Summary
                                - Detailed Results by Method
                                - Performance Metrics
                                - Risk Assessment
                                """)
                            
                            elif export_format == "JSON":
                                # Create JSON export
                                import json
                                
                                if report_type == "Executive Summary":
                                    report = reporter.generate_executive_summary(all_results)
                                else:
                                    report = reporter.generate_detailed_report(all_results)
                                
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                json_filename = f"oil_hedging_report_{report_type.lower().replace(' ', '_')}_{timestamp}.json"
                                
                                # Convert report to JSON-serializable format
                                json_report = _convert_to_json_serializable(report)
                                
                                # Download button for JSON
                                import base64
                                json_str = json.dumps(json_report, indent=2, default=str)
                                b64_json = base64.b64encode(json_str.encode()).decode()
                                href = f'''
                                    <a href="data:application/json;base64,{b64_json}" download="{json_filename}"
                                       style="
                                           display: inline-block;
                                           background-color: #e26d5c;
                                           color: #fff;
                                           font-size: 1.1rem;
                                           padding: 0.5rem 1.5rem;
                                           border-radius: 0.5rem;
                                           text-decoration: none;
                                           text-align: center;
                                           margin-top: 1rem;
                                           box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                       ">
                                       download JSON report
                                    </a>
                                '''
                                st.markdown(href, unsafe_allow_html=True)
                            
                            elif export_format == "PDF Summary":
                                st.info("📄 **PDF Summary Export** - This feature would require additional PDF generation libraries. For now, use CSV or JSON export options.")
                            
                            # Store report in session state
                            st.session_state['generated_report'] = {
                                'reporter': reporter,
                                'report_type': report_type,
                                'all_results': all_results,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            
                        except Exception as e:
                            st.error(f"Error generating report: {str(e)}")
                            st.error(f"Full error: {e}")
                    
                    # Show report visualizations if available
                    if st.checkbox("Show additional report visualizations"):
                        if 'generated_report' in st.session_state:
                            try:
                                generated_report = st.session_state['generated_report']
                                reporter = generated_report['reporter']
                                all_results = generated_report['all_results']
                                
                                # Show available visualizations
                                available_plots = []
                                if 'evaluation_results' in all_results:
                                    available_plots.extend(['Performance Comparison', 'Method Effectiveness'])
                                if 'hedge_results' in all_results and 'backtest_results' in all_results:
                                    available_plots.append('Risk Assessment')
                                
                                if available_plots:
                                    plot_choice = st.selectbox("Select Visualization", available_plots)
                                    
                                    if plot_choice == 'Performance Comparison':
                                        fig = reporter._create_performance_comparison_chart(all_results['evaluation_results'])
                                        st.pyplot(fig)
                                    elif plot_choice == 'Method Effectiveness':
                                        fig = reporter._create_method_effectiveness_chart(all_results['evaluation_results'])
                                        st.pyplot(fig)
                                    elif plot_choice == 'Risk Assessment':
                                        fig = reporter._create_risk_assessment_chart(all_results['hedge_results'], all_results['backtest_results'])
                                        st.pyplot(fig)
                                else:
                                    st.warning("⚠ **No additional visualizations available** - Run comprehensive analysis first")
                                    
                            except Exception as e:
                                st.error(f"Error generating additional visualizations: {str(e)}")
                        else:
                            st.warning("⚠ **No report available** - Generate a report first")
            
            # --- END REPORTING SECTION ---
        
        else:
            # Welcome message
            st.markdown("""
            ## welcome to the oil futures hedging dashboard!
            
            This dashboard provides comprehensive oil futures analysis with:
            
            - **Real-time data visualization** - Price charts, returns analysis, and technical indicators
            - **Processed data analysis** - Engineered features, spread analysis, and advanced analytics
            - **Forecasting models** - ARIMA and GARCH forecasting for price and volatility
            - **Hedge ratio analysis** - Dynamic hedge ratio calculations and effectiveness testing
            - **Backtesting** - Strategy performance evaluation and risk assessment
            - **Robustness checks** - Model sensitivity and stability analysis
            - **Comprehensive reporting** - Executive summaries and detailed technical reports
            
            ### getting started:
            1. Select your symbol and date range from the sidebar
            2. Click "Load Data & Analyze" to fetch both real-time and processed data
            3. Explore the integrated analysis results
            4. Run forecasting, backtesting, and other advanced analyses
            """)
            
            # Show available symbols
            st.markdown('### available symbols:')
            symbol_options = {
                'WTI Crude Oil': 'CL=F',
                'Brent Crude Oil': 'BZ=F',
                'USD Index': 'DX-Y.NYB'
            }
            
            for name, ticker in symbol_options.items():
                st.write(f"- **{name}**: `{ticker}`")

if __name__ == "__main__":
    main()