# Oil Futures Data
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import time

def fetch_wti_futures():
    """
    I'm using yfinance as a free resource for WTI futures.
    """
    try:
        # WTI Crude Oil futures symbol on yfinance
        # CL=F is the front month WTI futures contract
        wti = yf.Ticker("CL=F")
        
        # Get historical data for the last 5 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        
        # Fetch daily data
        data = wti.history(start=start_date, end=end_date)
        
        # Clean and format the data
        data = data.reset_index()
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock_Splits']
        
        # Add contract info
        data['Symbol'] = 'WTI_Front_Month'
        data['Contract_Type'] = 'Futures'
        
        # Save to data/raw/
        data.to_csv('data/raw/wti_futures.csv', index=False)
        
        print(f"Successfully fetched WTI futures data: {len(data)} records")
        return data
        
    except Exception as e:
        print(f"Error fetching WTI futures data: {e}")
        return None

def fetch_wti_futures_multiple_contracts():
    """
    Fetch multiple WTI futures contracts for term structure analysis.
    Using yfinance symbols for different contract months.
    """
    contracts = {
        'CL=F': 'Front_Month',
        'CL1!': 'Next_Month', 
        'CL2!': 'Second_Month',
        'CL3!': 'Third_Month'
    }
    
    all_data = {}
    
    for symbol, contract_name in contracts.items():
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2y")  # 2 years of data
            
            if not data.empty:
                data = data.reset_index()
                data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock_Splits']
                data['Contract'] = contract_name
                data['Symbol'] = symbol
                
                all_data[contract_name] = data
                
                # Save individual contract data
                data.to_csv(f'data/raw/wti_{contract_name.lower()}.csv', index=False)
                
                print(f"Fetched {contract_name}: {len(data)} records")
                
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            print(f"Error fetching {contract_name}: {e}")
    
    return all_data

def fetch_brent_futures():
    """Fetch Brent crude oil futures data using yfinance."""
    try:
        # Brent Crude Oil futures symbol
        brent = yf.Ticker("BZ=F")
        
        # Get historical data for the last 5 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        
        data = brent.history(start=start_date, end=end_date)
        data = data.reset_index()
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock_Splits']
        data['Symbol'] = 'Brent_Front_Month'
        data['Contract_Type'] = 'Futures'
        
        data.to_csv('data/raw/brent_futures.csv', index=False)
        print(f"Successfully fetched Brent futures data: {len(data)} records")
        return data
        
    except Exception as e:
        print(f"Error fetching Brent futures data: {e}")
        return None

def fetch_oil_options():
    """Get options data with implied volatility (placeholder for now)."""
    # Note: Options data typically requires paid subscriptions
    # For free alternatives, you might use yfinance for basic options data
    print("Options data fetching - requires paid subscription for comprehensive data")
    return None

# Macroeconomic Data
def fetch_interest_rates():
    """Get Treasury yields, Fed rates from FRED API."""
    # Note: Requires FRED API key
    print("Interest rates data - requires FRED API key")
    return None

def fetch_usd_index():
    """Get USD index data using yfinance."""
    try:
        usd = yf.Ticker("DX-Y.NYB")
        data = usd.history(period="5y")
        data = data.reset_index()
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock_Splits']
        data['Symbol'] = 'USD_Index'
        
        data.to_csv('data/raw/usd_index.csv', index=False)
        print(f"Successfully fetched USD index data: {len(data)} records")
        return data
        
    except Exception as e:
        print(f"Error fetching USD index data: {e}")
        return None

def fetch_inflation_data():
    """Get CPI, PPI from FRED (placeholder)."""
    # Note: Requires FRED API key
    print("Inflation data - requires FRED API key")
    return None

# Energy Market Data
def fetch_oil_inventories():
    """Get EIA inventory reports (placeholder)."""
    # Note: EIA API requires registration
    print("Oil inventories data - requires EIA API registration")
    return None

def fetch_production_data():
    """Get OPEC, US production data (placeholder)."""
    print("Production data - requires specialized data sources")
    return None

# Technical Data
def fetch_term_structure():
    """Calculate futures term structure from multiple contracts."""
    # This will use data from fetch_wti_futures_multiple_contracts()
    print("Term structure calculation - requires multiple contract data")
    return None

def fetch_roll_yields():
    """Calculate roll yields between contracts."""
    print("Roll yields calculation - requires multiple contract data")
    return None

# Main Collection Function
def collect_all_data():
    """Orchestrates all data collection and saves to data/raw/ and data/processed/."""
    print("Starting data collection...")
    
    # Fetch all available data
    wti_data = fetch_wti_futures()
    brent_data = fetch_brent_futures()
    usd_data = fetch_usd_index()
    
    print("Data collection completed!")
    return {
        'wti': wti_data,
        'brent': brent_data,
        'usd_index': usd_data
    } 