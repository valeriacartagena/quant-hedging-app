import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.colors as mcolors

# Project color palette
PRIMARY = '#e26d5c'
TEXT = '#000000'
BG = '#fbfcf8'

# 1. Correlation Matrix

def plot_correlation_matrix(df, columns=None, figsize=(10,8), max_features=20):
    """Plot a correlation matrix heatmap for selected columns and return the figure. Limits to top N features by variance if too many columns. Uses blue (0) to green (1) colormap."""
    if columns is not None:
        subset = df[columns].dropna()
    else:
        subset = df.dropna()
    # Limit to top N features by variance for readability
    if subset.shape[1] > max_features:
        top_cols = subset.var().sort_values(ascending=False).head(max_features).index
        subset = subset[top_cols]
    corr_matrix = subset.corr()
    # Custom blue-to-green colormap (blue for 0, green for 1)
    white_red = mcolors.LinearSegmentedColormap.from_list("white_red", ["#ffe8c2", "#e26d5c"], N=256)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap=white_red,
        annot_kws={"size": 8, "color": '#000'},
        cbar_kws={"shrink": .8},
        ax=ax,
        square=True,
        vmin=0, vmax=1
    )
    ax.set_title("Correlation Matrix", color=PRIMARY, fontsize=20, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig

# 2. Scatter Plot

def plot_scatter(df, x, y, alpha=0.5, figsize=(7,5)):
    """Scatter plot for two variables, returns the figure."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df[x], df[y], alpha=alpha, color=PRIMARY)
    ax.set_xlabel(x, color=TEXT)
    ax.set_ylabel(y, color=TEXT)
    ax.set_title(f"{x} vs. {y}", color=PRIMARY)
    plt.tight_layout()
    return fig

# 3. Augmented Dickey-Fuller (ADF) Test

def adf_test(series):
    """Run Augmented Dickey-Fuller test and print results."""
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    
    # Handle different versions of statsmodels
    try:
        if len(result) > 4 and isinstance(result[4], dict):
            for key, value in result[4].items():
                print(f"Critical Value {key}: {value:.4f}")
    except (IndexError, TypeError):
        print("Critical values not available")
    
    if result[1] < 0.05:
        print("\nInterpretation: Likely stationary (reject H0)")
    else:
        print("\nInterpretation: Likely non-stationary (fail to reject H0)")

# 4. Rolling Mean & STD

def plot_rolling_stats(series, window=30, figsize=(10,5)):
    """Plot rolling mean and std for a series, returns the figure."""
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(series, label='Returns', alpha=0.5, color=PRIMARY)
    ax.plot(roll_mean, label=f'Rolling Mean ({window}-day)', color='red')
    ax.plot(roll_std, label=f'Rolling STD ({window}-day)', color='green')
    ax.set_title("Rolling Mean & Standard Deviation", color=PRIMARY)
    ax.legend()
    plt.tight_layout()
    return fig

# 5. ACF & PACF Plots

def plot_acf_pacf(series, lags=30, figsize=(12,5)):
    """Plot ACF and PACF side by side, returns the figure."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_acf(series.dropna(), lags=lags, ax=axes[0], color=PRIMARY)
    axes[0].set_title('Autocorrelation (ACF)', color=PRIMARY)
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], color=PRIMARY)
    axes[1].set_title('Partial Autocorrelation (PACF)', color=PRIMARY)
    plt.tight_layout()
    return fig

def plot_returns_heatmap(df, returns_col='Returns'):
    """Plot a heatmap of mean returns by month and day of week. Returns the figure."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError('DataFrame index must be a DatetimeIndex')
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    heatmap_data = df.pivot_table(
        values=returns_col,
        index='Month',
        columns='DayOfWeek',
        aggfunc='mean'
    )
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, ax=ax)
    ax.set_title('Returns by Month vs DayOfWeek', color=PRIMARY)
    ax.set_xlabel('Day of Week', color=TEXT)
    ax.set_ylabel('Month', color=TEXT)
    plt.tight_layout()
    return fig 