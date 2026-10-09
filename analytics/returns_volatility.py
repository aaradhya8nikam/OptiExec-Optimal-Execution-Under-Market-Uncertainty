"""
Returns and Volatility Analytics for Market Microstructure.
Implements classical Close-to-Close, Parkinson (High-Low), and Garman-Klass (OHLC) volatility estimators.
"""

from typing import Union
import numpy as np
import pandas as pd


def calculate_returns(prices: Union[pd.Series, np.ndarray], method: str = "log") -> np.ndarray:
    """
    Computes simple or logarithmic returns from a price series.
    """
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 2:
        return np.array([])
    if method == "log":
        return np.log(arr[1:] / arr[:-1])
    elif method == "simple":
        return (arr[1:] - arr[:-1]) / arr[:-1]
    else:
        raise ValueError(f"Unknown return method: {method}. Choose 'log' or 'simple'.")


def calculate_historical_volatility(
    prices: Union[pd.Series, np.ndarray],
    intervals_per_day: int = 75,
    trading_days_per_year: int = 252
) -> float:
    """
    Standard sample standard deviation of log returns annualized:
    sigma_annual = std(r) * sqrt(trading_days_per_year * intervals_per_day)
    """
    returns = calculate_returns(prices, method="log")
    if len(returns) < 2:
        return 0.0
    bar_vol = np.std(returns, ddof=1)
    annualized_vol = bar_vol * np.sqrt(trading_days_per_year * intervals_per_day)
    return float(annualized_vol)


def calculate_parkinson_volatility(
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    intervals_per_day: int = 75,
    trading_days_per_year: int = 252
) -> float:
    """
    Parkinson Volatility estimator based on High-Low range (Parkinson, 1980).
    More efficient than close-to-close by utilizing intraday price extremes.
    sigma_P = sqrt( 1 / (4 * ln(2) * N) * sum( (ln(H_i / L_i))^2 ) )
    """
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    if len(h) != len(l) or len(h) == 0:
        return 0.0
    
    # Avoid zero or negative lows
    ratio = np.maximum(h, 1e-6) / np.maximum(l, 1e-6)
    log_hl = np.log(ratio)
    factor = 1.0 / (4.0 * np.log(2.0))
    bar_variance = factor * np.mean(log_hl ** 2)
    bar_vol = np.sqrt(bar_variance)
    annualized_vol = bar_vol * np.sqrt(trading_days_per_year * intervals_per_day)
    return float(annualized_vol)


def calculate_garman_klass_volatility(
    open_p: Union[pd.Series, np.ndarray],
    high: Union[pd.Series, np.ndarray],
    low: Union[pd.Series, np.ndarray],
    close: Union[pd.Series, np.ndarray],
    intervals_per_day: int = 75,
    trading_days_per_year: int = 252
) -> float:
    """
    Garman-Klass Volatility estimator (Garman & Klass, 1980).
    Extends Parkinson by incorporating Open and Close prices for maximum efficiency.
    sigma_GK^2 = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
    """
    o = np.asarray(open_p, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    
    if not (len(o) == len(h) == len(l) == len(c)) or len(o) == 0:
        return 0.0
    
    log_hl = np.log(np.maximum(h, 1e-6) / np.maximum(l, 1e-6))
    log_co = np.log(np.maximum(c, 1e-6) / np.maximum(o, 1e-6))
    
    gk_terms = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    bar_variance = np.maximum(np.mean(gk_terms), 0.0)
    bar_vol = np.sqrt(bar_variance)
    annualized_vol = bar_vol * np.sqrt(trading_days_per_year * intervals_per_day)
    return float(annualized_vol)


def calculate_rolling_volatility(
    prices: Union[pd.Series, np.ndarray],
    window: int = 20,
    intervals_per_day: int = 75,
    trading_days_per_year: int = 252
) -> pd.Series:
    """
    Calculates rolling annualized historical volatility series over a rolling window.
    """
    s = pd.Series(prices)
    returns = np.log(s / s.shift(1))
    rolling_std = returns.rolling(window=window).std()
    annual_factor = np.sqrt(trading_days_per_year * intervals_per_day)
    return rolling_std * annual_factor


def scale_volatility_to_interval(
    annual_vol: float,
    interval_duration_min: float,
    trading_minutes_per_day: float = 375.0,  # 9:15 to 15:30 in NSE/BSE
    trading_days_per_year: int = 252
) -> float:
    """
    Scales annualized volatility sigma_annual to interval volatility sigma_tau:
    sigma_tau = sigma_annual * sqrt(interval_duration_min / (trading_days_per_year * trading_minutes_per_day))
    """
    total_minutes_per_year = trading_days_per_year * trading_minutes_per_day
    dt = interval_duration_min / total_minutes_per_year
    return float(annual_vol * np.sqrt(dt))
