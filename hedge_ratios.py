import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

class HedgeRatioCalculator:
    def __init__(self):
        self.hedge_ratios = {}
        self.rolling_betas = None
        self.volatility_ratios = None
        
    def calculate_rolling_beta(self, asset_returns, hedge_returns, window=60, min_periods=30):
        """Calculate rolling beta (hedge ratio) using rolling regression."""
        if len(asset_returns) != len(hedge_returns):
            raise ValueError("Asset and hedge returns must have the same length")
        
        # Align data
        aligned_data = pd.DataFrame({
            'asset': asset_returns,
            'hedge': hedge_returns
        }).dropna()
        
        asset = aligned_data['asset']
        hedge = aligned_data['hedge']
        
        # Calculate rolling beta
        rolling_betas = []
        rolling_r2 = []
        rolling_pvalues = []
        
        for i in range(window, len(asset)):
            asset_window = asset.iloc[i-window:i]
            hedge_window = hedge.iloc[i-window:i]
            
            # Linear regression
            try:
                X = np.array(hedge_window.values).reshape(-1, 1)
                y = np.array(asset_window.values)
                
                # Check for valid data
                if len(X) < 2 or len(y) < 2:
                    rolling_betas.append(np.nan)
                    rolling_r2.append(np.nan)
                    rolling_pvalues.append(np.nan)
                    continue
                
                reg = LinearRegression()
                reg.fit(X, y)
                
                beta = reg.coef_[0]
                r2 = r2_score(y, reg.predict(X))
                
                # Calculate p-value for beta
                y_pred = reg.predict(X)
                residuals = y - y_pred
                mse = np.sum(residuals**2) / (len(y) - 2)
                var_beta = mse / np.sum((hedge_window - hedge_window.mean())**2)
                se_beta = np.sqrt(var_beta)
                t_stat = beta / se_beta
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 2))
                
                rolling_betas.append(beta)
                rolling_r2.append(r2)
                rolling_pvalues.append(p_value)
                
            except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
                rolling_betas.append(np.nan)
                rolling_r2.append(np.nan)
                rolling_pvalues.append(np.nan)
        
        # Create results DataFrame
        if len(rolling_betas) > 0 and len(asset.index) >= window:
            try:
                # Ensure we have the right number of indices
                if len(rolling_betas) == len(asset.index) - window:
                    results = pd.DataFrame({
                        'beta': rolling_betas,
                        'r2': rolling_r2,
                        'p_value': rolling_pvalues
                    }, index=asset.index[window:])
                else:
                    # Fallback if lengths don't match
                    results = pd.DataFrame({
                        'beta': rolling_betas,
                        'r2': rolling_r2,
                        'p_value': rolling_pvalues
                    })
            except (ValueError, IndexError):
                # Fallback if index alignment fails
                results = pd.DataFrame({
                    'beta': rolling_betas,
                    'r2': rolling_r2,
                    'p_value': rolling_pvalues
                })
        else:
            # Handle case where no valid results
            results = pd.DataFrame(data=[], columns=pd.Index(['beta', 'r2', 'p_value']))
        
        self.rolling_betas = results
        return results
    
    def calculate_volatility_based_hedge_ratio(self, asset_returns, hedge_returns, window=60):
        """Calculate volatility-based hedge ratio using rolling volatility ratios."""
        if len(asset_returns) != len(hedge_returns):
            raise ValueError("Asset and hedge returns must have the same length")
        
        # Align data
        aligned_data = pd.DataFrame({
            'asset': asset_returns,
            'hedge': hedge_returns
        }).dropna()
        
        asset = aligned_data['asset']
        hedge = aligned_data['hedge']
        
        # Calculate rolling volatilities
        asset_vol = asset.rolling(window=window, min_periods=window//2).std()
        hedge_vol = hedge.rolling(window=window, min_periods=window//2).std()
        
        # Calculate volatility-based hedge ratio
        vol_ratio = asset_vol / hedge_vol
        
        # Calculate correlation-based adjustment
        rolling_corr = asset.rolling(window=window, min_periods=window//2).corr(hedge)
        
        # Final volatility-based hedge ratio
        vol_hedge_ratio = vol_ratio * rolling_corr
        
        results = pd.DataFrame({
            'volatility_ratio': vol_ratio,
            'correlation': rolling_corr,
            'vol_hedge_ratio': vol_hedge_ratio,
            'asset_volatility': asset_vol,
            'hedge_volatility': hedge_vol
        })
        
        self.volatility_ratios = results
        return results
    
    def calculate_optimal_hedge_ratio(self, asset_returns, hedge_returns, method='ols', window=60):
        """Calculate optimal hedge ratio using different methods."""
        if method == 'ols':
            return self._ols_hedge_ratio(asset_returns, hedge_returns)
        elif method == 'rolling_ols':
            return self.calculate_rolling_beta(asset_returns, hedge_returns, window)
        elif method == 'volatility':
            return self.calculate_volatility_based_hedge_ratio(asset_returns, hedge_returns, window)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _ols_hedge_ratio(self, asset_returns, hedge_returns):
        """Calculate OLS hedge ratio for entire period."""
        # Align data
        aligned_data = pd.DataFrame({
            'asset': asset_returns,
            'hedge': hedge_returns
        }).dropna()
        
        X = np.array(aligned_data['hedge'].values).reshape(-1, 1)
        y = np.array(aligned_data['asset'].values)
        
        # OLS regression
        reg = LinearRegression()
        reg.fit(X, y)
        
        beta = reg.coef_[0]
        r2 = r2_score(y, reg.predict(X))
        
        # Calculate statistics
        y_pred = reg.predict(X)
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (len(y) - 2)
        var_beta = mse / np.sum((aligned_data['hedge'] - aligned_data['hedge'].mean())**2)
        se_beta = np.sqrt(var_beta)
        t_stat = beta / se_beta
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 2))
        
        return {
            'beta': beta,
            'r2': r2,
            'p_value': p_value,
            'se_beta': se_beta,
            't_stat': t_stat
        }
    
    def calculate_hedge_effectiveness(self, asset_returns, hedge_returns, hedge_ratio):
        """Calculate hedge effectiveness using different metrics."""
        # Align data
        aligned_data = pd.DataFrame({
            'asset': asset_returns,
            'hedge': hedge_returns
        }).dropna()
        
        asset = aligned_data['asset']
        hedge = aligned_data['hedge']
        
        # Calculate hedged portfolio returns
        if isinstance(hedge_ratio, (int, float)):
            # Constant hedge ratio
            hedged_returns = asset - hedge_ratio * hedge
        else:
            # Dynamic hedge ratio
            hedge_ratio_aligned = hedge_ratio.reindex(asset.index, method='ffill')
            hedged_returns = asset - hedge_ratio_aligned * hedge
        
        # Calculate hedge effectiveness metrics
        try:
            unhedged_var = asset.var()
            hedged_var = hedged_returns.var()
            
            # Variance reduction
            variance_reduction = (unhedged_var - hedged_var) / unhedged_var if unhedged_var != 0 else 0.0
            
            # Risk reduction
            unhedged_vol = asset.std()
            hedged_vol = hedged_returns.std()
            risk_reduction = (unhedged_vol - hedged_vol) / unhedged_vol if unhedged_vol != 0 else 0.0
        except (ZeroDivisionError, ValueError):
            variance_reduction = 0.0
            risk_reduction = 0.0
            unhedged_vol = 0.0
            hedged_vol = 0.0
        
        # Maximum drawdown
        try:
            # Ensure we're working with pandas Series
            if not isinstance(asset, pd.Series):
                asset = pd.Series(asset)
            if not isinstance(hedged_returns, pd.Series):
                hedged_returns = pd.Series(hedged_returns)
                
            unhedged_cum = (1 + asset).cumprod()
            hedged_cum = (1 + pd.Series(hedged_returns)).cumprod()
            
            # Calculate drawdown safely
            unhedged_dd = (unhedged_cum / unhedged_cum.expanding().max() - 1).min()
            hedged_dd = (hedged_cum / hedged_cum.expanding().max() - 1).min()
        except (ZeroDivisionError, ValueError, AttributeError):
            unhedged_dd = 0.0
            hedged_dd = 0.0
        
        return {
            'variance_reduction': variance_reduction,
            'risk_reduction': risk_reduction,
            'unhedged_volatility': unhedged_vol,
            'hedged_volatility': hedged_vol,
            'unhedged_max_drawdown': unhedged_dd,
            'hedged_max_drawdown': hedged_dd,
            'hedged_returns': hedged_returns
        }
    
    def plot_rolling_hedge_ratios(self, asset_returns, hedge_returns, window=60):
        """Plot rolling hedge ratios and related metrics."""
        # Calculate rolling beta
        rolling_beta = self.calculate_rolling_beta(asset_returns, hedge_returns, window)
        
        # Calculate volatility-based hedge ratio
        vol_hedge = self.calculate_volatility_based_hedge_ratio(asset_returns, hedge_returns, window)
        
        # Find common dates for alignment
        common_dates = rolling_beta.index.intersection(vol_hedge.index)
        
        if len(common_dates) == 0:
            # If no common dates, use the shorter one
            if len(rolling_beta) <= len(vol_hedge):
                common_dates = rolling_beta.index
            else:
                common_dates = vol_hedge.index
        
        # Align data to common dates
        rolling_beta_aligned = rolling_beta.reindex(common_dates, method='ffill')
        vol_hedge_aligned = vol_hedge.reindex(common_dates, method='ffill')
        
        # Create subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Plot 1: Rolling Beta
        axes[0, 0].plot(rolling_beta_aligned.index, rolling_beta_aligned['beta'], color='#e26d5c', linewidth=2)
        axes[0, 0].set_title('Rolling Beta (OLS Hedge Ratio)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Beta')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Rolling R²
        axes[0, 1].plot(rolling_beta_aligned.index, rolling_beta_aligned['r2'], color='#e74c3c', linewidth=2)
        axes[0, 1].set_title('Rolling R²', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Volatility-based Hedge Ratio
        axes[1, 0].plot(vol_hedge_aligned.index, vol_hedge_aligned['vol_hedge_ratio'], color='#3498db', linewidth=2)
        axes[1, 0].set_title('Volatility-based Hedge Ratio', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Hedge Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Rolling Correlation
        axes[1, 1].plot(vol_hedge_aligned.index, vol_hedge_aligned['correlation'], color='#f39c12', linewidth=2)
        axes[1, 1].set_title('Rolling Correlation', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Volatility Comparison
        axes[2, 0].plot(vol_hedge_aligned.index, vol_hedge_aligned['asset_volatility'], label='Asset Volatility', color='#e26d5c', linewidth=2)
        axes[2, 0].plot(vol_hedge_aligned.index, vol_hedge_aligned['hedge_volatility'], label='Hedge Volatility', color='#e74c3c', linewidth=2)
        axes[2, 0].set_title('Rolling Volatilities', fontsize=14, fontweight='bold')
        axes[2, 0].set_ylabel('Volatility')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Hedge Ratio Comparison
        axes[2, 1].plot(rolling_beta_aligned.index, rolling_beta_aligned['beta'], label='OLS Beta', color='#e26d5c', linewidth=2)
        axes[2, 1].plot(vol_hedge_aligned.index, vol_hedge_aligned['vol_hedge_ratio'], label='Volatility-based', color='#3498db', linewidth=2)
        axes[2, 1].set_title('Hedge Ratio Comparison', fontsize=14, fontweight='bold')
        axes[2, 1].set_ylabel('Hedge Ratio')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_hedge_effectiveness(self, asset_returns, hedge_returns, hedge_ratio):
        """Plot hedge effectiveness analysis."""
        # Calculate hedge effectiveness
        effectiveness = self.calculate_hedge_effectiveness(asset_returns, hedge_returns, hedge_ratio)
        
        # Get aligned data from effectiveness calculation
        hedged_returns = effectiveness['hedged_returns']
        
        # Align all data to the same index (hedged returns index)
        aligned_asset = asset_returns.reindex(hedged_returns.index, method='ffill')
        aligned_hedge = hedge_returns.reindex(hedged_returns.index, method='ffill')
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Cumulative Returns Comparison
        asset_cum = (1 + aligned_asset).cumprod()
        hedge_cum = (1 + aligned_hedge).cumprod()
        hedged_cum = (1 + hedged_returns).cumprod()
        
        axes[0, 0].plot(asset_cum.index, asset_cum, label='Asset', color='#e26d5c', linewidth=2)
        axes[0, 0].plot(hedge_cum.index, hedge_cum, label='Hedge', color='#e74c3c', linewidth=2)
        axes[0, 0].plot(hedged_cum.index, hedged_cum, label='Hedged Portfolio', color='#3498db', linewidth=2)
        axes[0, 0].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Rolling Volatility Comparison
        asset_vol = aligned_asset.rolling(window=60).std() * np.sqrt(252)
        hedge_vol = aligned_hedge.rolling(window=60).std() * np.sqrt(252)
        hedged_vol = hedged_returns.rolling(window=60).std() * np.sqrt(252)
        
        axes[0, 1].plot(asset_vol.index, asset_vol, label='Asset', color='#e26d5c', linewidth=2)
        axes[0, 1].plot(hedge_vol.index, hedge_vol, label='Hedge', color='#e74c3c', linewidth=2)
        axes[0, 1].plot(hedged_vol.index, hedged_vol, label='Hedged Portfolio', color='#3498db', linewidth=2)
        axes[0, 1].set_title('Rolling Volatility (Annualized)', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Volatility')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Returns Distribution
        axes[1, 0].hist(asset_returns, bins=50, alpha=0.7, label='Asset', color='#e26d5c')
        axes[1, 0].hist(effectiveness['hedged_returns'], bins=50, alpha=0.7, label='Hedged Portfolio', color='#3498db')
        axes[1, 0].set_title('Returns Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Returns')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Hedge Effectiveness Metrics
        metrics = ['Variance\nReduction', 'Risk\nReduction']
        values = [effectiveness['variance_reduction'], effectiveness['risk_reduction']]
        colors = ['#e26d5c', '#e74c3c']
        
        bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
        axes[1, 1].set_title('Hedge Effectiveness Metrics', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Reduction (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def get_hedge_summary(self, asset_returns, hedge_returns, hedge_ratio):
        """Get comprehensive hedge ratio summary."""
        # Calculate different hedge ratios
        ols_hedge = self._ols_hedge_ratio(asset_returns, hedge_returns)
        rolling_beta = self.calculate_rolling_beta(asset_returns, hedge_returns)
        vol_hedge = self.calculate_volatility_based_hedge_ratio(asset_returns, hedge_returns)
        effectiveness = self.calculate_hedge_effectiveness(asset_returns, hedge_returns, hedge_ratio)
        
        return {
            'ols_hedge': ols_hedge,
            'rolling_beta': rolling_beta,
            'volatility_hedge': vol_hedge,
            'effectiveness': effectiveness
        } 