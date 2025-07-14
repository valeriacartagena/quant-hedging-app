import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OilHedgingReporter:
    def __init__(self):
        self.report_data = {}
        self.report_sections = []
        
    def generate_executive_summary(self, results_dict):
        """
        Generate an executive summary of all analysis results.
        
        Args:
            results_dict: Dictionary containing results from all methods
        """
        summary = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'analysis_period': self._get_analysis_period(results_dict),
            'key_findings': [],
            'recommendations': [],
            'risk_assessment': {},
            'performance_summary': {}
        }
        
        # Extract key findings from each method
        if 'arima_results' in results_dict:
            arima_results = results_dict['arima_results']
            evaluation = arima_results.get('evaluation', {})
            
            if evaluation.get('rmse', 0) < evaluation.get('naive_rmse', float('inf')):
                summary['key_findings'].append(
                    f"ARIMA forecasting shows {((evaluation.get('naive_rmse', 1) - evaluation.get('rmse', 0)) / evaluation.get('naive_rmse', 1) * 100):.1f}% improvement over naïve model"
                )
            
            summary['performance_summary']['forecast_accuracy'] = {
                'rmse': evaluation.get('rmse', 0),
                'mae': evaluation.get('mae', 0),
                'improvement_vs_naive': ((evaluation.get('naive_rmse', 1) - evaluation.get('rmse', 0)) / evaluation.get('naive_rmse', 1) * 100) if evaluation.get('naive_rmse', 0) > 0 else 0
            }
        
        if 'garch_results' in results_dict:
            garch_results = results_dict['garch_results']
            clustering_result = garch_results.get('clustering_result', {})
            
            if clustering_result.get('has_clustering', False):
                summary['key_findings'].append(
                    f"Strong volatility clustering detected (autocorrelation: {clustering_result.get('autocorrelation', 0):.3f})"
                )
            
            summary['performance_summary']['volatility_modeling'] = {
                'clustering_detected': clustering_result.get('has_clustering', False),
                'autocorrelation': clustering_result.get('autocorrelation', 0),
                'model_aic': garch_results.get('model_info', {}).get('aic', 0)
            }
        
        if 'hedge_results' in results_dict:
            hedge_results = results_dict['hedge_results']
            effectiveness = hedge_results.get('effectiveness', {})
            best_strategy = hedge_results.get('best_strategy', 'Unknown')
            
            summary['key_findings'].append(
                f"Best hedging strategy: {best_strategy} with {effectiveness.get('variance_reduction', 0):.1%} variance reduction"
            )
            
            summary['performance_summary']['hedging_effectiveness'] = {
                'best_strategy': best_strategy,
                'variance_reduction': effectiveness.get('variance_reduction', 0),
                'risk_reduction': effectiveness.get('risk_reduction', 0),
                'hedged_volatility': effectiveness.get('hedged_volatility', 0)
            }
        
        if 'backtest_results' in results_dict:
            backtest_results = results_dict['backtest_results']
            strategies_run = backtest_results.get('strategies_run', [])
            
            if strategies_run:
                best_sharpe = 0
                best_strategy = None
                
                for strategy in strategies_run:
                    results = backtest_results['backtest'].results[strategy]
                    metrics = backtest_results['backtest'].calculate_performance_metrics(
                        results['net_returns'], backtest_results.get('benchmark_returns')
                    )
                    
                    if metrics.get('sharpe_ratio', 0) > best_sharpe:
                        best_sharpe = metrics.get('sharpe_ratio', 0)
                        best_strategy = strategy
                
                if best_strategy:
                    summary['key_findings'].append(
                        f"Best backtesting strategy: {best_strategy} with Sharpe ratio of {best_sharpe:.2f}"
                    )
                
                summary['performance_summary']['backtesting'] = {
                    'best_strategy': best_strategy,
                    'best_sharpe': best_sharpe,
                    'total_strategies': len(strategies_run)
                }
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations(results_dict)
        
        # Risk assessment
        summary['risk_assessment'] = self._assess_overall_risk(results_dict)
        
        return summary
    
    def _get_analysis_period(self, results_dict):
        """Extract analysis period from results."""
        if 'arima_results' in results_dict:
            log_returns = results_dict['arima_results'].get('log_returns')
            if log_returns is not None and len(log_returns) > 0:
                return f"{log_returns.index[0].strftime('%Y-%m-%d')} to {log_returns.index[-1].strftime('%Y-%m-%d')}"
        
        return "Period not specified"
    
    def _generate_recommendations(self, results_dict):
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        if 'arima_results' in results_dict:
            evaluation = results_dict['arima_results'].get('evaluation', {})
            improvement = ((evaluation.get('naive_rmse', 1) - evaluation.get('rmse', 0)) / evaluation.get('naive_rmse', 1) * 100) if evaluation.get('naive_rmse', 0) > 0 else 0
            
            if improvement > 10:
                recommendations.append("ARIMA model shows significant forecasting improvement - consider using for short-term price predictions")
            else:
                recommendations.append("ARIMA model improvement is limited - consider alternative forecasting methods or feature engineering")
        
        if 'garch_results' in results_dict:
            clustering_result = results_dict['garch_results'].get('clustering_result', {})
            
            if clustering_result.get('has_clustering', False):
                recommendations.append("Volatility clustering detected - GARCH models are appropriate for risk modeling")
            else:
                recommendations.append("No significant volatility clustering - consider simpler volatility models")
        
        if 'hedge_results' in results_dict:
            hedge_results = results_dict['hedge_results']
            best_strategy = hedge_results.get('best_strategy', '')
            
            if 'Dynamic' in best_strategy:
                recommendations.append("Dynamic hedge ratios outperform static approaches - implement rolling rebalancing")
            elif 'Volatility' in best_strategy:
                recommendations.append("Volatility-based hedging shows best performance - adapt to changing market conditions")
            else:
                recommendations.append("Static hedging provides stable risk reduction - suitable for conservative strategies")
        
        if 'backtest_results' in results_dict:
            backtest_results = results_dict['backtest_results']
            configuration = backtest_results.get('configuration', {})
            cost_impact = configuration.get('transaction_cost', 0)
            
            if cost_impact > 0.02:
                recommendations.append("High transaction costs detected - consider reducing rebalancing frequency")
            else:
                recommendations.append("Transaction costs are reasonable - current rebalancing strategy is cost-effective")
        
        return recommendations
    
    def _assess_overall_risk(self, results_dict):
        """Assess overall risk profile based on all results."""
        risk_assessment = {
            'risk_level': 'Medium',
            'key_risks': [],
            'risk_metrics': {}
        }
        
        # Volatility risk
        if 'garch_results' in results_dict:
            forecast_results = results_dict['garch_results'].get('forecast_results', {})
            forecast_vol = forecast_results.get('forecast', pd.Series())
            
            if len(forecast_vol) > 0:
                avg_vol = forecast_vol.mean()
                if avg_vol > 0.03:
                    risk_assessment['key_risks'].append("High forecasted volatility")
                    risk_assessment['risk_level'] = 'High'
                elif avg_vol < 0.01:
                    risk_assessment['key_risks'].append("Low forecasted volatility")
                    risk_assessment['risk_level'] = 'Low'
                
                risk_assessment['risk_metrics']['forecast_volatility'] = avg_vol
        
        # Hedging effectiveness risk
        if 'hedge_results' in results_dict:
            effectiveness = results_dict['hedge_results'].get('effectiveness', {})
            variance_reduction = effectiveness.get('variance_reduction', 0)
            
            if variance_reduction < 0.3:
                risk_assessment['key_risks'].append("Limited hedging effectiveness")
                if risk_assessment['risk_level'] != 'High':
                    risk_assessment['risk_level'] = 'Medium-High'
            
            risk_assessment['risk_metrics']['hedging_effectiveness'] = variance_reduction
        
        # Backtesting risk
        if 'backtest_results' in results_dict:
            backtest_results = results_dict['backtest_results']
            strategies_run = backtest_results.get('strategies_run', [])
            
            worst_drawdown = 0
            for strategy in strategies_run:
                results = backtest_results['backtest'].results[strategy]
                metrics = backtest_results['backtest'].calculate_performance_metrics(
                    results['net_returns'], backtest_results.get('benchmark_returns')
                )
                drawdown = abs(metrics.get('max_drawdown', 0))
                if drawdown > worst_drawdown:
                    worst_drawdown = drawdown
            
            if worst_drawdown > 0.2:
                risk_assessment['key_risks'].append("High maximum drawdown in backtesting")
                risk_assessment['risk_level'] = 'High'
            
            risk_assessment['risk_metrics']['max_drawdown'] = worst_drawdown
        
        return risk_assessment
    
    def generate_detailed_report(self, results_dict):
        """
        Generate a detailed technical report with all analysis results.
        """
        report = {
            'executive_summary': self.generate_executive_summary(results_dict),
            'methodology': self._generate_methodology_section(),
            'detailed_results': {},
            'appendix': {}
        }
        
        # ARIMA Results
        if 'arima_results' in results_dict:
            report['detailed_results']['arima_forecasting'] = self._format_arima_results(
                results_dict['arima_results']
            )
        
        # GARCH Results
        if 'garch_results' in results_dict:
            report['detailed_results']['garch_volatility'] = self._format_garch_results(
                results_dict['garch_results']
            )
        
        # Hedge Ratio Results
        if 'hedge_results' in results_dict:
            report['detailed_results']['hedge_ratios'] = self._format_hedge_results(
                results_dict['hedge_results']
            )
        
        # Backtesting Results
        if 'backtest_results' in results_dict:
            report['detailed_results']['backtesting'] = self._format_backtest_results(
                results_dict['backtest_results']
            )
        
        # Evaluation Results
        if 'evaluation_results' in results_dict:
            report['detailed_results']['evaluation'] = self._format_evaluation_results(
                results_dict['evaluation_results']
            )
        
        # Robustness Results
        if 'robustness_results' in results_dict:
            report['detailed_results']['robustness'] = self._format_robustness_results(
                results_dict['robustness_results']
            )
        
        return report
    
    def _generate_methodology_section(self):
        """Generate methodology section for the report."""
        return {
            'data_sources': {
                'wti_futures': 'WTI Crude Oil futures data from Yahoo Finance',
                'brent_futures': 'Brent Crude Oil futures data from Yahoo Finance',
                'usd_index': 'US Dollar Index data from Yahoo Finance'
            },
            'methods_used': {
                'arima_forecasting': 'Autoregressive Integrated Moving Average models for time series forecasting',
                'garch_volatility': 'Generalized Autoregressive Conditional Heteroskedasticity models for volatility modeling',
                'hedge_ratios': 'Dynamic hedge ratio calculation using rolling regression and volatility-based methods',
                'backtesting': 'Realistic trading simulation with transaction costs and slippage',
                'robustness_checks': 'Walk-forward validation, parameter sensitivity, and stress testing'
            },
            'key_metrics': {
                'forecast_accuracy': 'RMSE, MAE, and directional accuracy',
                'risk_metrics': 'Sharpe ratio, maximum drawdown, VaR, CVaR',
                'hedging_effectiveness': 'Variance reduction and risk reduction',
                'model_quality': 'AIC, BIC, and residual diagnostics'
            }
        }
    
    def _format_arima_results(self, arima_results):
        """Format ARIMA results for detailed report."""
        model_info = arima_results.get('model_info', {})
        evaluation = arima_results.get('evaluation', {})
        forecast_results = arima_results.get('forecast_results', {})
        
        return {
            'model_specification': {
                'order': model_info.get('order', (0, 0, 0)),
                'aic': model_info.get('aic', 0),
                'bic': model_info.get('bic', 0),
                'hqic': model_info.get('hqic', 0)
            },
            'forecast_performance': {
                'rmse': evaluation.get('rmse', 0),
                'mae': evaluation.get('mae', 0),
                'naive_rmse': evaluation.get('naive_rmse', 0),
                'improvement_vs_naive': ((evaluation.get('naive_rmse', 1) - evaluation.get('rmse', 0)) / evaluation.get('naive_rmse', 1) * 100) if evaluation.get('naive_rmse', 0) > 0 else 0
            },
            'forecast_statistics': {
                'forecast_mean': forecast_results.get('forecast', pd.Series()).mean() if len(forecast_results.get('forecast', pd.Series())) > 0 else 0,
                'forecast_std': forecast_results.get('forecast', pd.Series()).std() if len(forecast_results.get('forecast', pd.Series())) > 0 else 0,
                'forecast_steps': len(forecast_results.get('forecast', pd.Series()))
            },
            'residuals_analysis': model_info.get('residuals_stats', {})
        }
    
    def _format_garch_results(self, garch_results):
        """Format GARCH results for detailed report."""
        model_info = garch_results.get('model_info', {})
        clustering_result = garch_results.get('clustering_result', {})
        forecast_results = garch_results.get('forecast_results', {})
        
        return {
            'model_specification': {
                'model_type': model_info.get('model_type', 'GARCH'),
                'order': model_info.get('order', (1, 1)),
                'aic': model_info.get('aic', 0),
                'bic': model_info.get('bic', 0),
                'log_likelihood': model_info.get('loglikelihood', 0)
            },
            'volatility_clustering': {
                'autocorrelation': clustering_result.get('autocorrelation', 0),
                'ljung_box_pvalue': clustering_result.get('ljung_box_pvalue', 0),
                'has_clustering': clustering_result.get('has_clustering', False)
            },
            'forecast_statistics': {
                'forecast_mean': forecast_results.get('forecast', pd.Series()).mean() if len(forecast_results.get('forecast', pd.Series())) > 0 else 0,
                'forecast_std': forecast_results.get('forecast', pd.Series()).std() if len(forecast_results.get('forecast', pd.Series())) > 0 else 0,
                'historical_mean': model_info.get('conditional_volatility', pd.Series()).mean() if len(model_info.get('conditional_volatility', pd.Series())) > 0 else 0
            }
        }
    
    def _format_hedge_results(self, hedge_results):
        """Format hedge ratio results for detailed report."""
        effectiveness = hedge_results.get('effectiveness', {})
        effectiveness_ols = hedge_results.get('effectiveness_ols', {})
        effectiveness_vol = hedge_results.get('effectiveness_vol', {})
        best_strategy = hedge_results.get('best_strategy', 'Unknown')
        
        return {
            'static_ols': {
                'variance_reduction': effectiveness.get('variance_reduction', 0),
                'risk_reduction': effectiveness.get('risk_reduction', 0),
                'hedged_volatility': effectiveness.get('hedged_volatility', 0),
                'unhedged_volatility': effectiveness.get('unhedged_volatility', 0)
            },
            'dynamic_ols': {
                'variance_reduction': effectiveness_ols.get('variance_reduction', 0),
                'risk_reduction': effectiveness_ols.get('risk_reduction', 0),
                'hedged_volatility': effectiveness_ols.get('hedged_volatility', 0)
            },
            'volatility_based': {
                'variance_reduction': effectiveness_vol.get('variance_reduction', 0),
                'risk_reduction': effectiveness_vol.get('risk_reduction', 0),
                'hedged_volatility': effectiveness_vol.get('hedged_volatility', 0)
            },
            'best_strategy': best_strategy,
            'strategy_comparison': {
                'static_vs_dynamic': effectiveness_ols.get('variance_reduction', 0) - effectiveness.get('variance_reduction', 0),
                'static_vs_volatility': effectiveness_vol.get('variance_reduction', 0) - effectiveness.get('variance_reduction', 0)
            }
        }
    
    def _format_backtest_results(self, backtest_results):
        """Format backtesting results for detailed report."""
        backtest = backtest_results.get('backtest')
        strategies_run = backtest_results.get('strategies_run', [])
        configuration = backtest_results.get('configuration', {})
        
        strategy_results = {}
        for strategy in strategies_run:
            if backtest and strategy in backtest.results:
                results = backtest.results[strategy]
                metrics = backtest.calculate_performance_metrics(
                    results['net_returns'], backtest_results.get('benchmark_returns')
                )
                
                strategy_results[strategy] = {
                    'total_return': metrics.get('total_return', 0),
                    'annualized_return': metrics.get('annualized_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'volatility': metrics.get('volatility', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'calmar_ratio': metrics.get('calmar_ratio', 0),
                    'var_95': metrics.get('var_95', 0),
                    'cvar_95': metrics.get('cvar_95', 0),
                    'total_costs': results['total_costs'].sum() if 'total_costs' in results else 0
                }
        
        return {
            'configuration': configuration,
            'strategy_results': strategy_results,
            'best_strategy': max(strategy_results.items(), key=lambda x: x[1].get('sharpe_ratio', 0))[0] if strategy_results else None,
            'cost_analysis': {
                'total_costs': sum(result.get('total_costs', 0) for result in strategy_results.values()),
                'avg_cost_impact': sum(result.get('total_costs', 0) for result in strategy_results.values()) / configuration.get('initial_capital', 1) if configuration.get('initial_capital', 0) > 0 else 0
            }
        }
    
    def _format_evaluation_results(self, evaluation_results):
        """Format evaluation results for detailed report."""
        return {
            'methods_evaluated': list(evaluation_results.keys()),
            'comparison_metrics': evaluation_results,
            'method_rankings': self._rank_methods(evaluation_results)
        }
    
    def _format_robustness_results(self, robustness_results):
        """Format robustness results for detailed report."""
        all_results = robustness_results.get('all_results', {})
        report = robustness_results.get('report', {})
        
        return {
            'checks_performed': list(all_results.keys()),
            'overall_robustness_score': report.get('summary', {}).get('overall_robustness_score', 0),
            'recommendations': report.get('recommendations', []),
            'detailed_results': all_results
        }
    
    def _rank_methods(self, evaluation_results):
        """Rank methods based on different criteria."""
        rankings = {}
        
        # Extract metrics for ranking
        metrics_data = {}
        for method, results in evaluation_results.items():
            metrics_data[method] = {
                'forecast_accuracy': results.get('forecast_accuracy', results.get('volatility_forecast_accuracy', 0)),
                'risk_reduction': results.get('best_variance_reduction', results.get('best_risk_reduction', 0)),
                'model_quality': abs(results.get('model_fit', 0)) if results.get('model_fit') else 0,
                'performance': results.get('best_sharpe_ratio', results.get('best_total_return', 0))
            }
        
        # Create rankings
        for metric in ['forecast_accuracy', 'risk_reduction', 'model_quality', 'performance']:
            if any(data.get(metric, 0) > 0 for data in metrics_data.values()):
                sorted_methods = sorted(metrics_data.items(), key=lambda x: x[1].get(metric, 0), reverse=True)
                rankings[metric] = [method for method, _ in sorted_methods]
        
        return rankings
    
    def create_report_visualizations(self, results_dict):
        """
        Create comprehensive visualizations for the report.
        """
        figs = {}
        
        # Performance comparison chart
        if 'evaluation_results' in results_dict:
            figs['performance_comparison'] = self._create_performance_comparison_chart(
                results_dict['evaluation_results']
            )
        
        # Risk assessment chart
        if 'hedge_results' in results_dict and 'backtest_results' in results_dict:
            figs['risk_assessment'] = self._create_risk_assessment_chart(
                results_dict['hedge_results'], results_dict['backtest_results']
            )
        
        # Method effectiveness chart
        if 'evaluation_results' in results_dict:
            figs['method_effectiveness'] = self._create_method_effectiveness_chart(
                results_dict['evaluation_results']
            )
        
        return figs
    
    def _create_performance_comparison_chart(self, evaluation_results):
        """Create performance comparison visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        methods = list(evaluation_results.keys())
        
        # Extract metrics
        forecast_accuracies = []
        risk_reductions = []
        model_qualities = []
        performances = []
        
        for method in methods:
            results = evaluation_results[method]
            forecast_accuracies.append(results.get('forecast_accuracy', results.get('volatility_forecast_accuracy', 0)))
            risk_reductions.append(results.get('best_variance_reduction', results.get('best_risk_reduction', 0)))
            model_qualities.append(abs(results.get('model_fit', 0)) if results.get('model_fit') else 0)
            performances.append(results.get('best_sharpe_ratio', results.get('best_total_return', 0)))
        
        # Plot 1: Forecast Accuracy
        if any(acc > 0 for acc in forecast_accuracies):
            axes[0, 0].bar(methods, forecast_accuracies, color=['#e26d5c', '#e74c3c', '#3498db', '#f39c12'][:len(methods)])
            axes[0, 0].set_title('Forecast Accuracy', fontsize=14, fontweight='bold')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Risk Reduction
        if any(rr > 0 for rr in risk_reductions):
            axes[0, 1].bar(methods, risk_reductions, color=['#e26d5c', '#e74c3c', '#3498db', '#f39c12'][:len(methods)])
            axes[0, 1].set_title('Risk Reduction', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('Risk Reduction')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Model Quality
        if any(mq > 0 for mq in model_qualities):
            axes[1, 0].bar(methods, model_qualities, color=['#e26d5c', '#e74c3c', '#3498db', '#f39c12'][:len(methods)])
            axes[1, 0].set_title('Model Quality (|AIC|)', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('Model Quality')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Performance
        if any(perf > 0 for perf in performances):
            axes[1, 1].bar(methods, performances, color=['#e26d5c', '#e74c3c', '#3498db', '#f39c12'][:len(methods)])
            axes[1, 1].set_title('Performance', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('Performance')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_risk_assessment_chart(self, hedge_results, backtest_results):
        """Create risk assessment visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Hedge effectiveness
        effectiveness = hedge_results.get('effectiveness', {})
        effectiveness_ols = hedge_results.get('effectiveness_ols', {})
        effectiveness_vol = hedge_results.get('effectiveness_vol', {})
        
        strategies = ['Static OLS', 'Dynamic OLS', 'Volatility-based']
        variance_reductions = [
            effectiveness.get('variance_reduction', 0),
            effectiveness_ols.get('variance_reduction', 0),
            effectiveness_vol.get('variance_reduction', 0)
        ]
        
        axes[0, 0].bar(strategies, variance_reductions, color=['#e26d5c', '#3498db', '#e74c3c'])
        axes[0, 0].set_title('Hedging Effectiveness', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Variance Reduction')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Backtesting drawdowns
        if backtest_results.get('backtest') and backtest_results.get('strategies_run'):
            strategies_run = backtest_results['strategies_run']
            drawdowns = []
            
            for strategy in strategies_run:
                results = backtest_results['backtest'].results[strategy]
                metrics = backtest_results['backtest'].calculate_performance_metrics(
                    results['net_returns'], backtest_results.get('benchmark_returns')
                )
                drawdowns.append(abs(metrics.get('max_drawdown', 0)))
            
            axes[0, 1].bar(strategies_run, drawdowns, color=['#e26d5c', '#3498db', '#e74c3c'][:len(strategies_run)])
            axes[0, 1].set_title('Maximum Drawdown', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('Drawdown')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Sharpe ratios
        if backtest_results.get('backtest') and backtest_results.get('strategies_run'):
            sharpe_ratios = []
            
            for strategy in strategies_run:
                results = backtest_results['backtest'].results[strategy]
                metrics = backtest_results['backtest'].calculate_performance_metrics(
                    results['net_returns'], backtest_results.get('benchmark_returns')
                )
                sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
            
            axes[1, 0].bar(strategies_run, sharpe_ratios, color=['#e26d5c', '#3498db', '#e74c3c'][:len(strategies_run)])
            axes[1, 0].set_title('Sharpe Ratios', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Risk-return scatter
        if backtest_results.get('backtest') and backtest_results.get('strategies_run'):
            volatilities = []
            returns = []
            
            for strategy in strategies_run:
                results = backtest_results['backtest'].results[strategy]
                metrics = backtest_results['backtest'].calculate_performance_metrics(
                    results['net_returns'], backtest_results.get('benchmark_returns')
                )
                volatilities.append(metrics.get('volatility', 0))
                returns.append(metrics.get('annualized_return', 0))
            
            axes[1, 1].scatter(volatilities, returns, s=100, c=['#e26d5c', '#3498db', '#e74c3c'][:len(strategies_run)])
            for i, strategy in enumerate(strategies_run):
                axes[1, 1].annotate(strategy, (volatilities[i], returns[i]), xytext=(5, 5), textcoords='offset points')
            axes[1, 1].set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Volatility')
            axes[1, 1].set_ylabel('Annualized Return')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _create_method_effectiveness_chart(self, evaluation_results):
        """Create method effectiveness visualization."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = list(evaluation_results.keys())
        
        # Create radar chart data
        categories = ['Forecast Accuracy', 'Risk Reduction', 'Model Quality', 'Performance']
        data = []
        
        for method in methods:
            results = evaluation_results[method]
            method_data = [
                results.get('forecast_accuracy', results.get('volatility_forecast_accuracy', 0)),
                results.get('best_variance_reduction', results.get('best_risk_reduction', 0)),
                min(abs(results.get('model_fit', 0)) / 1000, 1) if results.get('model_fit') else 0,  # Normalize AIC
                min(results.get('best_sharpe_ratio', results.get('best_total_return', 0)) / 2, 1)  # Normalize performance
            ]
            data.append(method_data)
        
        # Plot radar chart
        angles = list(np.linspace(0, 2 * np.pi, len(categories), endpoint=False))
        angles.append(angles[0])  # Complete the circle
        
        colors = ['#e26d5c', '#e74c3c', '#3498db', '#f39c12'][:len(methods)]
        
        for i, method in enumerate(methods):
            values = data[i] + data[i][:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Method Effectiveness Comparison', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def export_report_to_csv(self, report, filename_prefix="oil_hedging_report"):
        """
        Export report data to CSV files.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Handle both executive summary and detailed report formats
        if 'executive_summary' in report:
            # Detailed report format
            exec_summary = report['executive_summary']
            exec_summary_df = pd.DataFrame([{
                'Report_Date': exec_summary.get('report_date', 'N/A'),
                'Analysis_Period': exec_summary.get('analysis_period', 'N/A'),
                'Risk_Level': exec_summary.get('risk_assessment', {}).get('risk_level', 'N/A'),
                'Key_Findings_Count': len(exec_summary.get('key_findings', [])),
                'Recommendations_Count': len(exec_summary.get('recommendations', []))
            }])
            exec_summary_df.to_csv(f"{filename_prefix}_executive_summary_{timestamp}.csv", index=False)
            
            # Detailed results
            if 'detailed_results' in report:
                for method, results in report['detailed_results'].items():
                    if isinstance(results, dict):
                        # Flatten nested dictionaries
                        flattened_results = self._flatten_dict(results)
                        results_df = pd.DataFrame([flattened_results])
                        results_df.to_csv(f"{filename_prefix}_{method}_{timestamp}.csv", index=False)
        else:
            # Executive summary format (direct return from generate_executive_summary)
            exec_summary_df = pd.DataFrame([{
                'Report_Date': report.get('report_date', 'N/A'),
                'Analysis_Period': report.get('analysis_period', 'N/A'),
                'Risk_Level': report.get('risk_assessment', {}).get('risk_level', 'N/A'),
                'Key_Findings_Count': len(report.get('key_findings', [])),
                'Recommendations_Count': len(report.get('recommendations', []))
            }])
            exec_summary_df.to_csv(f"{filename_prefix}_executive_summary_{timestamp}.csv", index=False)
        
        return f"Reports exported with timestamp: {timestamp}"
    
    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items) 