import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class RobustnessChecker:
    def __init__(self):
        self.results = {}
        
    def walk_forward_validation(self, data, model_func, window_size=252, step_size=63):
        """
        Perform walk-forward validation for time series models.
        
        Args:
            data: Time series data
            model_func: Function that takes training data and returns predictions
            window_size: Size of training window (default: 1 year)
            step_size: Step size for moving window (default: 3 months)
            
        Returns:
            Dictionary with validation results
        """
        predictions = []
        actuals = []
        dates = []
        
        for i in range(window_size, len(data) - step_size, step_size):
            # Training data
            train_data = data.iloc[i-window_size:i]
            # Test data
            test_data = data.iloc[i:i+step_size]
            
            # Get predictions
            pred = model_func(train_data, len(test_data))
            
            # Store results
            predictions.extend(pred)
            actuals.extend(test_data.values)
            dates.extend(test_data.index)
            
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        
        # Directional accuracy
        actual_direction = np.diff(actuals) > 0
        pred_direction = np.diff(predictions) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction)
        
        return {
            'predictions': predictions,
            'actuals': actuals,
            'dates': dates,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy
        }
    
    def parameter_sensitivity_analysis(self, data, base_params, param_ranges, model_func):
        """
        Analyze sensitivity of model parameters.
        
        Args:
            data: Input data
            base_params: Base parameter dictionary
            param_ranges: Dictionary of parameter ranges to test
            model_func: Function that takes data and parameters, returns metric
            
        Returns:
            Dictionary with sensitivity results
        """
        sensitivity_results = {}
        
        for param_name, param_range in param_ranges.items():
            param_results = []
            
            for param_value in param_range:
                # Create parameter set
                test_params = base_params.copy()
                test_params[param_name] = param_value
                
                # Get model performance
                try:
                    metric = model_func(data, test_params)
                    param_results.append({
                        'param_value': param_value,
                        'metric': metric
                    })
                except Exception as e:
                    print(f"Error with {param_name}={param_value}: {e}")
                    continue
            
            sensitivity_results[param_name] = pd.DataFrame(param_results)
        
        return sensitivity_results
    
    def regime_analysis(self, data, n_regimes=3):
        """
        Analyze model performance across different market regimes.
        
        Args:
            data: Time series data
            n_regimes: Number of regimes to identify
            
        Returns:
            Dictionary with regime analysis results
        """
        # Calculate rolling volatility to identify regimes
        rolling_vol = data.rolling(window=60).std()
        
        # Use quantiles to define regimes
        vol_quantiles = rolling_vol.quantile([0.33, 0.67])
        
        # Define regimes
        low_vol_threshold = vol_quantiles.iloc[0]
        high_vol_threshold = vol_quantiles.iloc[1]
        
        regimes = pd.Series(index=data.index, dtype=str)
        regimes[rolling_vol <= low_vol_threshold] = 'Low Volatility'
        regimes[(rolling_vol > low_vol_threshold) & (rolling_vol <= high_vol_threshold)] = 'Medium Volatility'
        regimes[rolling_vol > high_vol_threshold] = 'High Volatility'
        
        # Calculate regime statistics
        regime_stats = {}
        for regime in ['Low Volatility', 'Medium Volatility', 'High Volatility']:
            regime_data = data[regimes == regime]
            if len(regime_data) > 0:
                regime_stats[regime] = {
                    'count': len(regime_data),
                    'mean': regime_data.mean(),
                    'std': regime_data.std(),
                    'skewness': regime_data.skew(),
                    'kurtosis': regime_data.kurtosis()
                }
        
        return {
            'regimes': regimes,
            'regime_stats': regime_stats,
            'rolling_volatility': rolling_vol
        }
    
    def stress_testing(self, data, stress_scenarios):
        """
        Perform stress testing under various market scenarios.
        
        Args:
            data: Historical data
            stress_scenarios: Dictionary of stress scenarios
            
        Returns:
            Dictionary with stress test results
        """
        stress_results = {}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            # Apply stress scenario
            stressed_data = self._apply_stress_scenario(data, scenario_params)
            
            # Calculate performance under stress
            stress_results[scenario_name] = {
                'stressed_data': stressed_data,
                'max_drawdown': self._calculate_max_drawdown(stressed_data),
                'var_95': np.percentile(stressed_data, 5),
                'cvar_95': stressed_data[stressed_data <= np.percentile(stressed_data, 5)].mean(),
                'volatility': stressed_data.std()
            }
        
        return stress_results
    
    def _apply_stress_scenario(self, data, scenario_params):
        """Apply stress scenario to data."""
        stressed_data = data.copy()
        
        if 'volatility_shock' in scenario_params:
            # Increase volatility
            vol_multiplier = scenario_params['volatility_shock']
            stressed_data = stressed_data * vol_multiplier
        
        if 'trend_shock' in scenario_params:
            # Add trend component
            trend = np.linspace(0, scenario_params['trend_shock'], len(data))
            stressed_data = stressed_data + trend
        
        if 'jump_shock' in scenario_params:
            # Add jump component
            jump_size = scenario_params['jump_shock']
            jump_prob = scenario_params.get('jump_probability', 0.01)
            
            jumps = np.random.choice([0, jump_size], size=len(data), p=[1-jump_prob, jump_prob])
            stressed_data = stressed_data + jumps
        
        return stressed_data
    
    def _calculate_max_drawdown(self, data):
        """Calculate maximum drawdown."""
        cumulative = (1 + data).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def cross_validation_stability(self, data, model_func, cv_splits=5):
        """
        Assess model stability using cross-validation.
        
        Args:
            data: Input data
            model_func: Model function
            cv_splits: Number of CV splits
            
        Returns:
            Dictionary with CV stability results
        """
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        
        cv_scores = []
        cv_predictions = []
        
        for train_idx, test_idx in tscv.split(data):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            
            # Fit model and predict
            try:
                predictions = model_func(train_data, len(test_data))
                
                # Calculate score
                score = mean_squared_error(test_data, predictions)
                cv_scores.append(score)
                cv_predictions.append(predictions)
                
            except Exception as e:
                print(f"CV fold error: {e}")
                continue
        
        # Calculate stability metrics
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        cv_cv = cv_std / cv_mean  # Coefficient of variation
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'cv_cv': cv_cv,
            'cv_predictions': cv_predictions,
            'stability_score': 1 / (1 + cv_cv)  # Higher is more stable
        }
    
    def outlier_robustness(self, data, outlier_fractions=[0.01, 0.05, 0.10]):
        """
        Test model robustness to outliers.
        
        Args:
            data: Input data
            outlier_fractions: List of outlier fractions to test
            
        Returns:
            Dictionary with outlier robustness results
        """
        robustness_results = {}
        
        for fraction in outlier_fractions:
            # Add outliers
            contaminated_data = self._add_outliers(data, fraction)
            
            # Calculate statistics
            original_stats = {
                'mean': data.mean(),
                'std': data.std(),
                'skewness': data.skew(),
                'kurtosis': data.kurtosis()
            }
            
            contaminated_stats = {
                'mean': contaminated_data.mean(),
                'std': contaminated_data.std(),
                'skewness': contaminated_data.skew(),
                'kurtosis': contaminated_data.kurtosis()
            }
            
            # Calculate robustness metrics
            robustness_metrics = {}
            for stat in ['mean', 'std', 'skewness', 'kurtosis']:
                original_val = original_stats[stat]
                contaminated_val = contaminated_stats[stat]
                
                if original_val != 0:
                    robustness_metrics[stat] = abs(contaminated_val - original_val) / abs(original_val)
                else:
                    robustness_metrics[stat] = abs(contaminated_val - original_val)
            
            robustness_results[f'outlier_fraction_{fraction}'] = {
                'contaminated_data': contaminated_data,
                'original_stats': original_stats,
                'contaminated_stats': contaminated_stats,
                'robustness_metrics': robustness_metrics
            }
        
        return robustness_results
    
    def _add_outliers(self, data, fraction):
        """Add outliers to data."""
        contaminated_data = data.copy()
        n_outliers = int(len(data) * fraction)
        
        # Randomly select positions for outliers
        outlier_positions = np.random.choice(len(data), n_outliers, replace=False)
        
        # Add large outliers
        outlier_values = np.random.normal(0, 5 * data.std(), n_outliers)
        contaminated_data.iloc[outlier_positions] = outlier_values
        
        return contaminated_data
    
    def plot_walk_forward_results(self, walk_forward_results):
        """Plot walk-forward validation results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot predictions vs actuals
        axes[0, 0].plot(walk_forward_results['dates'], walk_forward_results['actuals'], 
                       label='Actual', alpha=0.7)
        axes[0, 0].plot(walk_forward_results['dates'], walk_forward_results['predictions'], 
                       label='Predicted', alpha=0.7)
        axes[0, 0].set_title('Walk-Forward Predictions vs Actuals')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot residuals
        residuals = np.array(walk_forward_results['actuals']) - np.array(walk_forward_results['predictions'])
        axes[0, 1].plot(walk_forward_results['dates'], residuals)
        axes[0, 1].set_title('Residuals Over Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_sensitivity_analysis(self, sensitivity_results):
        """Plot parameter sensitivity analysis results."""
        n_params = len(sensitivity_results)
        fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
        
        if n_params == 1:
            axes = [axes]
        
        for i, (param_name, results) in enumerate(sensitivity_results.items()):
            axes[i].plot(results['param_value'], results['metric'], 'o-')
            axes[i].set_xlabel(param_name)
            axes[i].set_ylabel('Performance Metric')
            axes[i].set_title(f'Sensitivity: {param_name}')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_regime_analysis(self, regime_results):
        """Plot regime analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot regimes over time
        regime_colors = {'Low Volatility': 'green', 'Medium Volatility': 'orange', 'High Volatility': 'red'}
        for regime in regime_results['regimes'].unique():
            regime_data = regime_results['regimes'] == regime
            axes[0, 0].scatter(regime_results['regimes'].index[regime_data], 
                             regime_results['rolling_volatility'][regime_data], 
                             c=regime_colors[regime], alpha=0.6, label=regime)
        axes[0, 0].set_title('Market Regimes Over Time')
        axes[0, 0].set_ylabel('Rolling Volatility')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Regime distribution
        regime_counts = regime_results['regimes'].value_counts()
        axes[0, 1].pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%')
        axes[0, 1].set_title('Regime Distribution')
        
        # Regime statistics
        regime_stats_df = pd.DataFrame(regime_results['regime_stats']).T
        regime_stats_df[['mean', 'std']].plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Regime Statistics')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Volatility distribution by regime
        regime_data = []
        regime_labels = []
        for regime in ['Low Volatility', 'Medium Volatility', 'High Volatility']:
            regime_mask = regime_results['regimes'] == regime
            if regime_mask.any():
                regime_data.append(regime_results['rolling_volatility'][regime_mask])
                regime_labels.append(regime)
        
        axes[1, 1].boxplot(regime_data, labels=regime_labels)
        axes[1, 1].set_title('Volatility Distribution by Regime')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_stress_test_results(self, stress_results):
        """Plot stress test results."""
        n_scenarios = len(stress_results)
        fig, axes = plt.subplots(2, n_scenarios, figsize=(5*n_scenarios, 10))
        
        if n_scenarios == 1:
            axes = axes.reshape(2, 1)
        
        for i, (scenario_name, results) in enumerate(stress_results.items()):
            # Plot stressed data
            axes[0, i].plot(results['stressed_data'], alpha=0.7)
            axes[0, i].set_title(f'Stressed Data: {scenario_name}')
            axes[0, i].grid(True, alpha=0.3)
            
            # Plot performance metrics
            metrics = ['max_drawdown', 'var_95', 'cvar_95', 'volatility']
            metric_values = [results[metric] for metric in metrics]
            axes[1, i].bar(metrics, metric_values)
            axes[1, i].set_title(f'Performance Metrics: {scenario_name}')
            axes[1, i].tick_params(axis='x', rotation=45)
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_robustness_report(self, all_results):
        """Generate comprehensive robustness report."""
        report = {
            'summary': {},
            'recommendations': []
        }
        
        # Walk-forward validation summary
        if 'walk_forward' in all_results:
            wf_results = all_results['walk_forward']
            report['summary']['walk_forward'] = {
                'rmse': wf_results['rmse'],
                'mae': wf_results['mae'],
                'directional_accuracy': wf_results['directional_accuracy']
            }
            
            if wf_results['directional_accuracy'] < 0.5:
                report['recommendations'].append("Low directional accuracy in walk-forward validation - consider model improvements")
        
        # Cross-validation stability
        if 'cv_stability' in all_results:
            cv_results = all_results['cv_stability']
            report['summary']['cv_stability'] = {
                'stability_score': cv_results['stability_score'],
                'cv_cv': cv_results['cv_cv']
            }
            
            if cv_results['stability_score'] < 0.7:
                report['recommendations'].append("Low model stability - consider regularization or simpler model")
        
        # Parameter sensitivity
        if 'sensitivity' in all_results:
            sensitivity_results = all_results['sensitivity']
            high_sensitivity_params = []
            
            for param_name, results in sensitivity_results.items():
                metric_range = results['metric'].max() - results['metric'].min()
                if metric_range > 0.1:  # Threshold for high sensitivity
                    high_sensitivity_params.append(param_name)
            
            report['summary']['sensitivity'] = {
                'high_sensitivity_params': high_sensitivity_params
            }
            
            if high_sensitivity_params:
                report['recommendations'].append(f"High sensitivity to parameters: {', '.join(high_sensitivity_params)} - consider parameter optimization")
        
        # Regime analysis
        if 'regime_analysis' in all_results:
            regime_results = all_results['regime_analysis']
            regime_stats = regime_results['regime_stats']
            
            # Check for regime-dependent performance
            regime_performance = {}
            for regime, stats in regime_stats.items():
                regime_performance[regime] = stats['mean']
            
            report['summary']['regime_analysis'] = {
                'regime_performance': regime_performance
            }
            
            # Check for significant performance differences across regimes
            performance_values = list(regime_performance.values())
            if max(performance_values) - min(performance_values) > 0.1:
                report['recommendations'].append("Significant performance variation across regimes - consider regime-specific models")
        
        # Stress testing
        if 'stress_testing' in all_results:
            stress_results = all_results['stress_testing']
            
            # Check for extreme losses under stress
            max_losses = []
            for scenario_name, results in stress_results.items():
                max_losses.append(results['max_drawdown'])
            
            report['summary']['stress_testing'] = {
                'worst_case_drawdown': min(max_losses),
                'avg_var_95': np.mean([results['var_95'] for results in stress_results.values()])
            }
            
            if min(max_losses) < -0.3:  # 30% drawdown threshold
                report['recommendations'].append("High potential losses under stress - consider risk management improvements")
        
        # Overall robustness score
        robustness_scores = []
        if 'walk_forward' in report['summary']:
            robustness_scores.append(min(1.0, 1.0 - report['summary']['walk_forward']['rmse']))
        if 'cv_stability' in report['summary']:
            robustness_scores.append(report['summary']['cv_stability']['stability_score'])
        
        if robustness_scores:
            report['summary']['overall_robustness_score'] = np.mean(robustness_scores)
            
            if report['summary']['overall_robustness_score'] < 0.6:
                report['recommendations'].append("Overall low robustness - consider comprehensive model review")
        
        return report 