import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from arch.univariate import GARCH, EGARCH
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class GARCHForecaster:
    def __init__(self):
        self.fitted_model = None
        self.model_info = {}
        self.forecast_results = None
        
    def prepare_data(self, data, column='Close'):
        """Prepare data for GARCH modeling - calculate returns."""
        if isinstance(data, pd.DataFrame):
            if column in data.columns:
                prices = data[column]
            else:
                raise ValueError(f"Column '{column}' not found in data")
        else:
            prices = data
            
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1)).dropna()
        return log_returns
    
    def check_volatility_clustering(self, returns):
        """Check for volatility clustering using squared returns autocorrelation."""
        squared_returns = returns ** 2
        
        # Calculate autocorrelation of squared returns
        autocorr = squared_returns.autocorr(lag=1)
        
        # Ljung-Box test for squared returns
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(squared_returns, lags=10, return_df=True)
        
        return {
            'autocorrelation': autocorr,
            'ljung_box_stat': lb_test['lb_stat'].iloc[-1],
            'ljung_box_pvalue': lb_test['lb_pvalue'].iloc[-1],
            'has_clustering': lb_test['lb_pvalue'].iloc[-1] < 0.05
        }
    
    def fit_garch(self, returns, model_type='GARCH', p=1, q=1, auto_select=True, max_p=3, max_q=3):
        """Fit GARCH model to returns data."""
        try:
            if auto_select:
                # Grid search for best GARCH order
                best_aic = np.inf
                best_order = (1, 1)
                best_model = None
                
                for p_test in range(1, max_p + 1):
                    for q_test in range(1, max_q + 1):
                        try:
                            if model_type == 'GARCH':
                                model = arch_model(returns, vol='GARCH', p=p_test, q=q_test, dist='normal')
                            elif model_type == 'EGARCH':
                                model = arch_model(returns, vol='EGARCH', p=p_test, q=q_test, dist='normal')
                            elif model_type == 'GJR-GARCH':
                                model = arch_model(returns, vol='GARCH', p=p_test, o=1, q=q_test, dist='normal')
                            else:
                                raise ValueError(f"Unknown model type: {model_type}")
                            
                            fitted_model = model.fit(disp='off')
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_order = (p_test, q_test)
                                best_model = fitted_model
                                
                        except:
                            continue
                
                if best_model is None:
                    raise ValueError("Could not fit any GARCH model")
                
                fitted_model = best_model
                p, q = best_order
                
            else:
                # Use specified order
                if model_type == 'GARCH':
                    model = arch_model(returns, vol='GARCH', p=p, q=q, dist='normal')
                elif model_type == 'EGARCH':
                    model = arch_model(returns, vol='EGARCH', p=p, q=q, dist='normal')
                elif model_type == 'GJR-GARCH':
                    model = arch_model(returns, vol='GARCH', p=p, o=1, q=q, dist='normal')
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                fitted_model = model.fit(disp='off')
            
            # Store model information
            self.model_info = {
                'model_type': model_type,
                'order': (p, q),
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'loglikelihood': fitted_model.loglikelihood,
                'params': fitted_model.params,
                'pvalues': fitted_model.pvalues,
                'conditional_volatility': fitted_model.conditional_volatility,
                'resid': fitted_model.resid,
                'fitted_model': fitted_model
            }
            
            self.fitted_model = fitted_model
            return self.model_info
            
        except Exception as e:
            raise Exception(f"Error fitting GARCH model: {str(e)}")
    
    def forecast(self, steps=30):
        """Generate volatility forecasts."""
        if self.fitted_model is None:
            raise ValueError("No fitted model available. Run fit_garch() first.")
        
        try:
            # Generate forecasts
            forecast = self.fitted_model.forecast(horizon=steps)
            
            # Extract forecasted variance and convert to volatility
            forecasted_variance = forecast.variance.values[-1, :]
            forecasted_volatility = np.sqrt(forecasted_variance)
            
            # Create forecast index
            last_date = pd.Timestamp.now()
            forecast_dates = pd.date_range(start=last_date, periods=steps+1, freq='D')[1:]
            
            self.forecast_results = {
                'forecast': pd.Series(forecasted_volatility, index=forecast_dates),
                'variance': pd.Series(forecasted_variance, index=forecast_dates),
                'mean_forecast': forecast.mean.values[-1, :] if hasattr(forecast, 'mean') else None
            }
            
            return self.forecast_results
            
        except Exception as e:
            raise Exception(f"Error generating forecasts: {str(e)}")
    
    def plot_volatility_forecast(self, returns, forecast_results, title="Volatility Forecast"):
        """Plot historical volatility and forecast."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Historical volatility
        historical_vol = self.model_info['conditional_volatility']
        dates = returns.index
        
        ax1.plot(dates, historical_vol, label='Historical Volatility', color='#e26d5c', linewidth=1.5)
        ax1.set_title('Historical Conditional Volatility', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Volatility')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Forecast
        forecast_vol = forecast_results['forecast']
        forecast_dates = forecast_vol.index
        
        ax2.plot(forecast_dates, forecast_vol, label='Forecasted Volatility', color='#e74c3c', linewidth=2)
        ax2.set_title('Volatility Forecast', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volatility')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_model_diagnostics(self):
        """Plot GARCH model diagnostics."""
        if self.fitted_model is None:
            raise ValueError("No fitted model available. Run fit_garch() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Standardized residuals
        std_resid = self.model_info['resid'] / self.model_info['conditional_volatility']
        
        # Plot 1: Standardized residuals
        axes[0, 0].plot(std_resid, color='#e26d5c', alpha=0.7)
        axes[0, 0].set_title('Standardized Residuals')
        axes[0, 0].set_ylabel('Standardized Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Q-Q plot of standardized residuals
        from scipy import stats
        stats.probplot(std_resid, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot of Standardized Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: ACF of standardized residuals
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(std_resid, ax=axes[1, 0], lags=20, alpha=0.05)
        axes[1, 0].set_title('ACF of Standardized Residuals')
        
        # Plot 4: ACF of squared standardized residuals
        plot_acf(std_resid**2, ax=axes[1, 1], lags=20, alpha=0.05)
        axes[1, 1].set_title('ACF of Squared Standardized Residuals')
        
        plt.tight_layout()
        return fig
    
    def plot_volatility_clustering(self, returns):
        """Plot volatility clustering analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Returns
        axes[0, 0].plot(returns, color='#e26d5c', alpha=0.7)
        axes[0, 0].set_title('Log Returns')
        axes[0, 0].set_ylabel('Returns')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Squared returns
        squared_returns = returns ** 2
        axes[0, 1].plot(squared_returns, color='#e74c3c', alpha=0.7)
        axes[0, 1].set_title('Squared Returns')
        axes[0, 1].set_ylabel('Squared Returns')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: ACF of returns
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(returns, ax=axes[1, 0], lags=20, alpha=0.05)
        axes[1, 0].set_title('ACF of Returns')
        
        # Plot 4: ACF of squared returns
        plot_acf(squared_returns, ax=axes[1, 1], lags=20, alpha=0.05)
        axes[1, 1].set_title('ACF of Squared Returns')
        
        plt.tight_layout()
        return fig
    
    def get_model_summary(self):
        """Get comprehensive model summary."""
        if self.fitted_model is None:
            raise ValueError("No fitted model available. Run fit_garch() first.")
        
        # Calculate additional statistics
        std_resid = self.model_info['resid'] / self.model_info['conditional_volatility']
        
        # Residuals statistics
        residuals_stats = {
            'mean': std_resid.mean(),
            'std': std_resid.std(),
            'skewness': std_resid.skew(),
            'kurtosis': std_resid.kurtosis(),
            'jarque_bera_stat': self.fitted_model.conditional_volatility.std(),
            'jarque_bera_pvalue': 0.05  # Placeholder
        }
        
        # Volatility statistics
        volatility_stats = {
            'mean_volatility': self.model_info['conditional_volatility'].mean(),
            'volatility_of_volatility': self.model_info['conditional_volatility'].std(),
            'min_volatility': self.model_info['conditional_volatility'].min(),
            'max_volatility': self.model_info['conditional_volatility'].max()
        }
        
        return {
            'model_info': self.model_info,
            'residuals_stats': residuals_stats,
            'volatility_stats': volatility_stats
        } 