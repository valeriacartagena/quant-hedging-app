import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

class DynamicHedgingBacktest:
    def __init__(self, initial_capital=1000000, transaction_cost=0.001, slippage=0.0005):
        """
        Initialize backtesting engine for dynamic hedging strategies.
        
        Parameters:
        - initial_capital: Starting capital in USD
        - transaction_cost: Transaction cost as percentage (e.g., 0.001 = 0.1%)
        - slippage: Slippage cost as percentage (e.g., 0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.results = {}
        
    def calculate_position_sizes(self, asset_returns, hedge_returns, hedge_ratios, 
                               rebalance_frequency='daily', target_volatility=0.15):
        """
        Calculate position sizes for dynamic hedging strategy.
        
        Parameters:
        - asset_returns: Returns of the asset to hedge
        - hedge_returns: Returns of the hedging instrument
        - hedge_ratios: Dynamic hedge ratios (Series)
        - rebalance_frequency: How often to rebalance ('daily', 'weekly', 'monthly')
        - target_volatility: Target portfolio volatility
        """
        # Data should already be aligned from run_backtest, but ensure consistency
        common_index = asset_returns.index.intersection(hedge_ratios.index)
        
        aligned_data = pd.DataFrame({
            'asset_returns': asset_returns.loc[common_index],
            'hedge_returns': hedge_returns.loc[common_index],
            'hedge_ratios': hedge_ratios.loc[common_index]
        }).dropna()
        
        # Calculate rolling volatility for position sizing
        rolling_vol = aligned_data['asset_returns'].rolling(window=60).std() * np.sqrt(252)
        
        # Position sizing based on target volatility
        position_sizes = target_volatility / (rolling_vol * np.sqrt(252))
        position_sizes = position_sizes.fillna(1.0)  # Default to 1.0 if no volatility data
        
        # Apply rebalancing frequency
        if rebalance_frequency == 'daily':
            rebalance_dates = aligned_data.index
        elif rebalance_frequency == 'weekly':
            rebalance_dates = aligned_data.resample('W').last().index
        elif rebalance_frequency == 'monthly':
            rebalance_dates = aligned_data.resample('M').last().index
        else:
            rebalance_dates = aligned_data.index
        
        # Forward fill position sizes between rebalancing dates
        position_sizes_rebalanced = position_sizes.reindex(aligned_data.index)
        for date in rebalance_dates:
            if date in position_sizes_rebalanced.index:
                position_sizes_rebalanced.loc[date:] = position_sizes.loc[date]
        
        return position_sizes_rebalanced
    
    def calculate_transaction_costs(self, position_changes, prices):
        """
        Calculate transaction costs for position changes.
        
        Parameters:
        - position_changes: Changes in position sizes
        - prices: Asset prices for cost calculation
        """
        # Calculate notional value of trades
        trade_notional = abs(position_changes) * prices
        
        # Transaction costs
        transaction_costs = trade_notional * self.transaction_cost
        
        # Slippage costs (assume proportional to trade size)
        slippage_costs = trade_notional * self.slippage
        
        total_costs = transaction_costs + slippage_costs
        
        return total_costs
    
    def run_backtest(self, asset_returns, hedge_returns, hedge_ratios, 
                    asset_prices, hedge_prices, strategy_name="Dynamic Hedge",
                    rebalance_frequency='daily', target_volatility=0.15):
        """
        Run backtest for dynamic hedging strategy.
        
        Parameters:
        - asset_returns: Returns of the asset to hedge
        - hedge_returns: Returns of the hedging instrument
        - hedge_ratios: Dynamic hedge ratios
        - asset_prices: Asset prices for cost calculation
        - hedge_prices: Hedge instrument prices for cost calculation
        - strategy_name: Name of the strategy
        - rebalance_frequency: Rebalancing frequency
        - target_volatility: Target portfolio volatility
        """
        # First, align the hedge ratios with the returns data
        # The hedge ratios might start later due to rolling window calculation
        common_index = asset_returns.index.intersection(hedge_ratios.index)
        
        if len(common_index) == 0:
            raise ValueError("No common dates between returns and hedge ratios")
        
        # Align all data to the common index
        aligned_data = pd.DataFrame({
            'asset_returns': asset_returns.loc[common_index],
            'hedge_returns': hedge_returns.loc[common_index],
            'hedge_ratios': hedge_ratios.loc[common_index],
            'asset_prices': asset_prices.loc[common_index],
            'hedge_prices': hedge_prices.loc[common_index]
        }).dropna()
        
        # Calculate position sizes
        position_sizes = self.calculate_position_sizes(
            aligned_data['asset_returns'], 
            aligned_data['hedge_returns'], 
            aligned_data['hedge_ratios'],
            rebalance_frequency,
            target_volatility
        )
        
        # Calculate hedge positions
        hedge_positions = position_sizes * aligned_data['hedge_ratios']
        
        # Calculate position changes for transaction costs
        asset_position_changes = position_sizes.diff().fillna(position_sizes.iloc[0])
        hedge_position_changes = hedge_positions.diff().fillna(hedge_positions.iloc[0])
        
        # Calculate transaction costs
        asset_costs = self.calculate_transaction_costs(asset_position_changes, aligned_data['asset_prices'])
        hedge_costs = self.calculate_transaction_costs(hedge_position_changes, aligned_data['hedge_prices'])
        total_costs = asset_costs + hedge_costs
        
        # Calculate portfolio returns
        asset_contribution = position_sizes * aligned_data['asset_returns']
        hedge_contribution = -hedge_positions * aligned_data['hedge_returns']  # Negative for hedging
        gross_returns = asset_contribution + hedge_contribution
        
        # Net returns after costs
        net_returns = gross_returns - total_costs
        
        # Calculate cumulative returns
        cumulative_returns = (1 + net_returns).cumprod()
        
        # Calculate portfolio value
        portfolio_value = self.initial_capital * cumulative_returns
        
        # Store results
        results = {
            'strategy_name': strategy_name,
            'dates': aligned_data.index,
            'asset_returns': aligned_data['asset_returns'],
            'hedge_returns': aligned_data['hedge_returns'],
            'net_returns': net_returns,
            'gross_returns': gross_returns,
            'total_costs': total_costs,
            'portfolio_value': portfolio_value,
            'position_sizes': position_sizes,
            'hedge_positions': hedge_positions,
            'hedge_ratios': aligned_data['hedge_ratios'],
            'cumulative_returns': cumulative_returns
        }
        
        self.results[strategy_name] = results
        return results
    
    def calculate_performance_metrics(self, returns, benchmark_returns=None):
        """
        Calculate performance metrics.
        
        Parameters:
        - returns: Strategy returns
        - benchmark_returns: Benchmark returns for comparison
        """
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR and CVaR (95% confidence)
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Average win/loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Benchmark comparison
        if benchmark_returns is not None:
            # Align returns
            aligned_data = pd.DataFrame({
                'strategy': returns,
                'benchmark': benchmark_returns
            }).dropna()
            
            strategy_returns = aligned_data['strategy']
            benchmark_returns_aligned = aligned_data['benchmark']
            
            # Alpha and Beta
            covariance = np.cov(strategy_returns, benchmark_returns_aligned)[0, 1]
            benchmark_variance = benchmark_returns_aligned.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            benchmark_annualized = (1 + benchmark_returns_aligned).prod() ** (252 / len(benchmark_returns_aligned)) - 1
            alpha = annualized_return - (beta * benchmark_annualized)
            
            # Information ratio
            excess_returns = strategy_returns - benchmark_returns_aligned
            information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
            
            # Tracking error
            tracking_error = excess_returns.std() * np.sqrt(252)
        else:
            alpha = beta = information_ratio = tracking_error = np.nan
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error
        }
    
    def plot_backtest_results(self, strategy_name, benchmark_returns=None):
        """
        Plot comprehensive backtest results.
        
        Parameters:
        - strategy_name: Name of the strategy to plot
        - benchmark_returns: Optional benchmark returns for comparison
        """
        if strategy_name not in self.results:
            raise ValueError(f"Strategy {strategy_name} not found in results")
        
        results = self.results[strategy_name]
        
        # Align benchmark returns with strategy dates if provided
        aligned_benchmark = None
        if benchmark_returns is not None:
            # Find common dates between strategy and benchmark
            common_dates = results['dates'].intersection(benchmark_returns.index)
            if len(common_dates) > 0:
                aligned_benchmark = benchmark_returns.loc[common_dates]
                # Reindex strategy results to match benchmark dates
                strategy_dates = results['dates'].intersection(common_dates)
                strategy_results = {key: value.loc[strategy_dates] if hasattr(value, 'loc') else value 
                                  for key, value in results.items()}
            else:
                st.warning("No common dates between strategy and benchmark - skipping benchmark comparison")
                aligned_benchmark = None
                strategy_results = results
        else:
            strategy_results = results
        
        # Create subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Plot 1: Portfolio Value
        axes[0, 0].plot(strategy_results['dates'], strategy_results['portfolio_value'], 
                       color='#e26d5c', linewidth=2, label='Hedged Portfolio')
        if aligned_benchmark is not None:
            benchmark_cumulative = (1 + aligned_benchmark).cumprod() * self.initial_capital
            axes[0, 0].plot(strategy_results['dates'], benchmark_cumulative, 
                           color='#e74c3c', linewidth=2, label='Benchmark', alpha=0.7)
        axes[0, 0].set_title('Portfolio Value', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Returns
        axes[0, 1].plot(strategy_results['dates'], strategy_results['cumulative_returns'], 
                       color='#3498db', linewidth=2, label='Hedged Portfolio')
        if aligned_benchmark is not None:
            benchmark_cumulative_returns = (1 + aligned_benchmark).cumprod()
            axes[0, 1].plot(strategy_results['dates'], benchmark_cumulative_returns, 
                           color='#f39c12', linewidth=2, label='Benchmark', alpha=0.7)
        axes[0, 1].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Cumulative Return')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Rolling Volatility
        rolling_vol = strategy_results['net_returns'].rolling(window=60).std() * np.sqrt(252)
        axes[1, 0].plot(strategy_results['dates'], rolling_vol, color='#9b59b6', linewidth=2)
        axes[1, 0].set_title('Rolling Volatility (60-day)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Volatility')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Hedge Ratios
        axes[1, 1].plot(strategy_results['dates'], strategy_results['hedge_ratios'], color='#e67e22', linewidth=2)
        axes[1, 1].set_title('Dynamic Hedge Ratios', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Hedge Ratio')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Transaction Costs
        axes[2, 0].plot(strategy_results['dates'], strategy_results['total_costs'].cumsum(), 
                       color='#e74c3c', linewidth=2)
        axes[2, 0].set_title('Cumulative Transaction Costs', fontsize=14, fontweight='bold')
        axes[2, 0].set_ylabel('Cumulative Costs ($)')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot 6: Returns Distribution
        axes[2, 1].hist(strategy_results['net_returns'], bins=50, alpha=0.7, color='#e26d5c', density=True)
        axes[2, 1].set_title('Returns Distribution', fontsize=14, fontweight='bold')
        axes[2, 1].set_xlabel('Returns')
        axes[2, 1].set_ylabel('Density')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def compare_strategies(self, strategy_names, benchmark_returns=None):
        """
        Compare multiple strategies.
        
        Parameters:
        - strategy_names: List of strategy names to compare
        - benchmark_returns: Optional benchmark returns
        """
        comparison_data = {}
        
        for strategy_name in strategy_names:
            if strategy_name in self.results:
                metrics = self.calculate_performance_metrics(
                    self.results[strategy_name]['net_returns'], 
                    benchmark_returns
                )
                comparison_data[strategy_name] = metrics
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data).T
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Annualized Returns
        strategies = list(comparison_data.keys())
        returns = [comparison_data[s]['annualized_return'] for s in strategies]
        colors = ['#e26d5c', '#e74c3c', '#3498db', '#f39c12']
        
        bars1 = axes[0, 0].bar(strategies, returns, color=colors[:len(strategies)], alpha=0.7)
        axes[0, 0].set_title('Annualized Returns', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, returns):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{value:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Sharpe Ratios
        sharpe_ratios = [comparison_data[s]['sharpe_ratio'] for s in strategies]
        bars2 = axes[0, 1].bar(strategies, sharpe_ratios, color=colors[:len(strategies)], alpha=0.7)
        axes[0, 1].set_title('Sharpe Ratios', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, sharpe_ratios):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Maximum Drawdowns
        max_drawdowns = [comparison_data[s]['max_drawdown'] for s in strategies]
        bars3 = axes[1, 0].bar(strategies, max_drawdowns, color=colors[:len(strategies)], alpha=0.7)
        axes[1, 0].set_title('Maximum Drawdowns', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars3, max_drawdowns):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height - 0.01,
                           f'{value:.2%}', ha='center', va='top', fontweight='bold')
        
        # Plot 4: Win Rates
        win_rates = [comparison_data[s]['win_rate'] for s in strategies]
        bars4 = axes[1, 1].bar(strategies, win_rates, color=colors[:len(strategies)], alpha=0.7)
        axes[1, 1].set_title('Win Rates', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Win Rate (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars4, win_rates):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig, comparison_df
    
    def generate_backtest_report(self, strategy_name, benchmark_returns=None):
        """
        Generate comprehensive backtest report.
        
        Parameters:
        - strategy_name: Name of the strategy
        - benchmark_returns: Optional benchmark returns
        """
        if strategy_name not in self.results:
            raise ValueError(f"Strategy {strategy_name} not found in results")
        
        results = self.results[strategy_name]
        metrics = self.calculate_performance_metrics(results['net_returns'], benchmark_returns)
        
        report = {
            'strategy_name': strategy_name,
            'backtest_period': f"{results['dates'][0].strftime('%Y-%m-%d')} to {results['dates'][-1].strftime('%Y-%m-%d')}",
            'initial_capital': self.initial_capital,
            'final_portfolio_value': results['portfolio_value'].iloc[-1],
            'total_return': metrics['total_return'],
            'annualized_return': metrics['annualized_return'],
            'volatility': metrics['volatility'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'calmar_ratio': metrics['calmar_ratio'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'var_95': metrics['var_95'],
            'cvar_95': metrics['cvar_95'],
            'total_transaction_costs': results['total_costs'].sum(),
            'avg_daily_costs': results['total_costs'].mean(),
            'cost_impact': results['total_costs'].sum() / self.initial_capital
        }
        
        if benchmark_returns is not None:
            report.update({
                'alpha': metrics['alpha'],
                'beta': metrics['beta'],
                'information_ratio': metrics['information_ratio'],
                'tracking_error': metrics['tracking_error']
            })
        
        return report 