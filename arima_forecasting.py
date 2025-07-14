import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ARIMAForecaster:
    """
    ARIMA forecasting for WTI crude oil log returns.
    Handles stationarity, model selection, fitting, and evaluation.
    """
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.forecast_results = None
        self.model_info = {}
        
    def prepare_data(self, price_data, target_column='Close'):
        """
        Prepare log returns from price data.
        
        Args:
            price_data (pd.DataFrame): Price data with datetime index
            target_column (str): Column name for price data
            
        Returns:
            pd.Series: Log returns series
        """
        if target_column not in price_data.columns:
            raise ValueError(f"Column '{target_column}' not found in data")
            
        # Calculate log returns
        log_returns = np.log(price_data[target_column] / price_data[target_column].shift(1))
        
        # Remove first row (NaN from shift)
        log_returns = log_returns.dropna()
        
        return log_returns
    
    def check_stationarity(self, series, title="Time Series"):
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        
        Args:
            series (pd.Series): Time series to test
            title (str): Title for the test
            
        Returns:
            dict: Test results
        """
        result = adfuller(series.dropna())
        
        # Handle different versions of statsmodels
        try:
            critical_values = result[4] if len(result) > 4 else {}
        except (IndexError, TypeError):
            critical_values = {}
        
        test_results = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': critical_values,
            'is_stationary': result[1] < 0.05,
            'title': title
        }
        
        return test_results
    
    def plot_acf_pacf(self, series, lags=40, figsize=(12, 4)):
        """
        Plot ACF and PACF for model order selection.
        
        Args:
            series (pd.Series): Time series
            lags (int): Number of lags to plot
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: Figure with ACF and PACF plots
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # ACF plot
        plot_acf(series.dropna(), lags=lags, ax=ax1, alpha=0.05)
        ax1.set_title('Autocorrelation Function (ACF)')
        ax1.grid(True, alpha=0.3)
        
        # PACF plot
        plot_pacf(series.dropna(), lags=lags, ax=ax2, alpha=0.05)
        ax2.set_title('Partial Autocorrelation Function (PACF)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def auto_select_order(self, series, max_p=5, max_d=2, max_q=5, seasonal=False):
        """
        Automatically select ARIMA order using AIC/BIC.
        
        Args:
            series (pd.Series): Time series
            max_p (int): Maximum AR order
            max_d (int): Maximum differencing order
            max_q (int): Maximum MA order
            seasonal (bool): Whether to consider seasonal components
            
        Returns:
            tuple: Best (p, d, q) order
        """
        best_aic = np.inf
        best_order = None
        
        # Grid search for best order
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            
                    except:
                        continue
        
        return best_order
    
    def fit_arima(self, series, order=None, auto_select=True, max_p=5, max_d=2, max_q=5):
        """
        Fit ARIMA model to the series.
        
        Args:
            series (pd.Series): Time series
            order (tuple): ARIMA order (p, d, q)
            auto_select (bool): Whether to auto-select order
            max_p, max_d, max_q (int): Maximum orders for auto-selection
            
        Returns:
            dict: Model information and fitted model
        """
        if auto_select and order is None:
            order = self.auto_select_order(series, max_p, max_d, max_q)
            print(f"Auto-selected ARIMA order: {order}")
        
        if order is None:
            raise ValueError("Must provide order or set auto_select=True")
        
        # Fit the model
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        
        # Store model info
        self.model_info = {
            'order': order,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'hqic': fitted_model.hqic,
            'params': fitted_model.params,
            'pvalues': fitted_model.pvalues,
            'resid': fitted_model.resid,
            'fitted_values': fitted_model.fittedvalues
        }
        # Add residuals_stats
        self.model_info['residuals_stats'] = {
            'mean': self.model_info['resid'].mean(),
            'std': self.model_info['resid'].std(),
            'skewness': self.model_info['resid'].skew(),
            'kurtosis': self.model_info['resid'].kurtosis()
        }
        self.fitted_model = fitted_model
        return self.model_info
    
    def forecast(self, steps=30, alpha=0.05):
        """
        Generate forecasts using fitted ARIMA model.
        
        Args:
            steps (int): Number of steps to forecast
            alpha (float): Confidence level for intervals
            
        Returns:
            dict: Forecast results
        """
        if self.fitted_model is None:
            raise ValueError("Must fit model before forecasting")
        
        # Generate forecast
        forecast_result = self.fitted_model.forecast(steps=steps, alpha=alpha)
        
        # Extract components
        if hasattr(forecast_result, 'predicted_mean'):
            forecast_mean = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()
        else:
            # Handle different forecast result formats
            forecast_mean = forecast_result
            conf_int = None
        
        self.forecast_results = {
            'forecast': forecast_mean,
            'conf_int': conf_int,
            'steps': steps,
            'alpha': alpha
        }
        
        return self.forecast_results
    
    def evaluate_forecast(self, actual, forecast):
        """
        Evaluate forecast accuracy.
        
        Args:
            actual (pd.Series): Actual values
            forecast (pd.Series): Forecasted values
            
        Returns:
            dict: Evaluation metrics
        """
        # Align series
        common_index = actual.index.intersection(forecast.index)
        actual_aligned = actual.loc[common_index]
        forecast_aligned = forecast.loc[common_index]
        
        # Calculate metrics
        mse = mean_squared_error(actual_aligned, forecast_aligned)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_aligned, forecast_aligned)
        mape = np.mean(np.abs((actual_aligned - forecast_aligned) / actual_aligned)) * 100
        
        # Directional accuracy
        actual_direction = np.sign(actual_aligned.diff().dropna())
        forecast_direction = np.sign(forecast_aligned.diff().dropna())
        directional_accuracy = np.mean(actual_direction == forecast_direction) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'actual': actual_aligned,
            'forecast': forecast_aligned
        }
    
    def plot_forecast(self, series, forecast_results, title="ARIMA Forecast"):
        """
        Plot the original series and forecast.
        
        Args:
            series (pd.Series): Original time series
            forecast_results (dict): Forecast results
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Forecast plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot original series
        ax.plot(series.index, series.values, label='Actual', color='#e26d5c', linewidth=2)
        
        # Plot forecast
        forecast = forecast_results['forecast']
        ax.plot(forecast.index, forecast.values, label='Forecast', color='#dc3545', linewidth=2, linestyle='--')
        
        # Plot confidence intervals if available
        if forecast_results['conf_int'] is not None:
            conf_int = forecast_results['conf_int']
            ax.fill_between(forecast.index, 
                          conf_int.iloc[:, 0], 
                          conf_int.iloc[:, 1], 
                          alpha=0.3, color='#dc3545', label=f'{int((1-forecast_results["alpha"])*100)}% CI')
        
        ax.set_title(title, fontsize=14, fontweight='bold', color='#e26d5c')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Log Returns', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(self):
        """
        Plot model residuals for diagnostics.
        
        Returns:
            matplotlib.figure.Figure: Residuals plot
        """
        if self.fitted_model is None:
            raise ValueError("Must fit model before plotting residuals")
        
        residuals = self.model_info['resid']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Residuals time series
        axes[0, 0].plot(residuals.index, residuals.values, color='#e26d5c')
        axes[0, 0].set_title('Residuals Time Series')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[0, 1].hist(residuals.values, bins=30, color='#e26d5c', alpha=0.7)
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals.values, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # ACF of residuals
        plot_acf(residuals.values, lags=20, ax=axes[1, 1], alpha=0.05)
        axes[1, 1].set_title('ACF of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_model_summary(self):
        """
        Get comprehensive model summary.
        
        Returns:
            dict: Model summary information
        """
        if self.fitted_model is None:
            raise ValueError("Must fit model before getting summary")
        
        summary = {
            'order': self.model_info['order'],
            'aic': self.model_info['aic'],
            'bic': self.model_info['bic'],
            'hqic': self.model_info['hqic'],
            'params': self.model_info['params'],
            'pvalues': self.model_info['pvalues'],
            'residuals_stats': {
                'mean': self.model_info['resid'].mean(),
                'std': self.model_info['resid'].std(),
                'skewness': self.model_info['resid'].skew(),
                'kurtosis': self.model_info['resid'].kurtosis()
            }
        }
        
        return summary 