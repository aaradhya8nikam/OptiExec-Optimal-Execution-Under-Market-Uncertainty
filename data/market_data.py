"""
Market Data module: Ingestion, simulation, and intraday volume modeling.
"""

from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_u_shape_volume_profile(num_intervals: int, u_intensity: float = 1.8) -> np.ndarray:
    """
    Generates a normalized intraday U-shape volume distribution.
    Institutional trading volume typically follows a U-curve: elevated at market open,
    quiescent during midday, and surging prior to the closing cross.

    Parameters:
    -----------
    num_intervals : int
        Number of discrete intervals (e.g. 12 intervals for 1 hr with 5-min bins).
    u_intensity : float
        Curvature factor; higher values mean sharper open/close spikes.

    Returns:
    --------
    np.ndarray
        Array of shape (num_intervals,) normalized such that sum(weights) = 1.0.
    """
    if num_intervals <= 0:
        raise ValueError("num_intervals must be >= 1")
    if num_intervals == 1:
        return np.array([1.0])

    t = np.linspace(-1.0, 1.0, num_intervals)
    # Quadratic U-shape + base liquidity floor
    weights = 1.0 + u_intensity * (t ** 2)
    # Give slight asymmetry (closing rush slightly heavier than open)
    weights += 0.15 * t
    weights = np.maximum(weights, 0.05)
    return weights / np.sum(weights)


def generate_synthetic_ohlcv(
    symbol: str = "RELIANCE.NS",
    start_price: float = 1000.0,
    annual_vol: float = 0.25,
    adv: float = 5_000_000.0,
    days: int = 30,
    intervals_per_day: int = 75,  # 375 mins / 5 min bins = 75 bins/day
    seed: Optional[int] = 42
) -> pd.DataFrame:
    """
    Generates a realistic intraday OHLCV time-series dataframe with:
    - Geometric Brownian Motion price dynamics
    - Realistic U-shape intraday volume patterns
    - High/Low wick simulation
    - Bid-Ask spread simulation
    """
    if seed is not None:
        np.random.seed(seed)

    total_bars = days * intervals_per_day
    dt_day = 1.0 / (252.0 * intervals_per_day)
    sigma_bar = annual_vol * np.sqrt(dt_day)

    base_time = datetime(2026, 9, 1, 9, 15)
    timestamps = []
    
    # Generate timestamps skipping overnight / weekend gaps
    cur_date = base_time
    for d in range(days):
        day_start = cur_date + timedelta(days=d)
        # Skip weekends
        while day_start.weekday() >= 5:
            day_start += timedelta(days=1)
        for i in range(intervals_per_day):
            timestamps.append(day_start + timedelta(minutes=i * 5))

    # Log returns
    log_returns = np.random.normal(0.0, sigma_bar, total_bars)
    close_prices = start_price * np.exp(np.cumsum(log_returns))
    
    # Open price is previous close with small gap noise
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = start_price
    open_prices += np.random.normal(0.0, 0.2 * sigma_bar * close_prices)

    # High and Low
    wick_noise_high = np.abs(np.random.normal(0.0, 0.6 * sigma_bar * close_prices))
    wick_noise_low = np.abs(np.random.normal(0.0, 0.6 * sigma_bar * close_prices))
    high_prices = np.maximum(open_prices, close_prices) + wick_noise_high
    low_prices = np.minimum(open_prices, close_prices) - wick_noise_low
    low_prices = np.maximum(low_prices, 0.01)

    # Volume: U-shape intraday modulated by daily volume variation
    u_profile = generate_u_shape_volume_profile(intervals_per_day)
    daily_factors = np.random.lognormal(mean=0.0, sigma=0.2, size=days)
    
    volumes = []
    for d in range(days):
        day_vol = adv * daily_factors[d]
        bar_vol = day_vol * u_profile * np.random.lognormal(mean=0.0, sigma=0.1, size=intervals_per_day)
        volumes.extend(bar_vol)

    volumes = np.array(volumes, dtype=float)

    # Bid-Ask spread: 1 to 5 bps with spread widening during low volume periods
    spread_bps = 2.5 * (np.mean(volumes) / np.maximum(volumes, 1.0)) ** 0.3
    spread_bps = np.clip(spread_bps, 1.0, 15.0)

    df = pd.DataFrame({
        "timestamp": timestamps[:total_bars],
        "open": np.round(open_prices, 2),
        "high": np.round(high_prices, 2),
        "low": np.round(low_prices, 2),
        "close": np.round(close_prices, 2),
        "volume": np.round(volumes, 0),
        "spread_bps": np.round(spread_bps, 2)
    })
    
    df["vwap"] = ((df["high"] + df["low"] + df["close"]) / 3.0 * df["volume"]).cumsum() / df["volume"].cumsum()
    return df


class MarketDataLoader:
    """Manages market data ingestion, caching, and parametric estimation."""
    
    PRESET_ASSETS: Dict[str, Dict[str, float]] = {
        "RELIANCE.NS": {"start_price": 1350.0, "annual_vol": 0.22, "adv": 7_500_000.0, "spread_bps": 2.0},
        "INFY.NS": {"start_price": 1820.0, "annual_vol": 0.26, "adv": 5_200_000.0, "spread_bps": 2.5},
        "TCS.NS": {"start_price": 4100.0, "annual_vol": 0.19, "adv": 2_800_000.0, "spread_bps": 2.2},
        "HDFCBANK.NS": {"start_price": 1650.0, "annual_vol": 0.21, "adv": 12_000_000.0, "spread_bps": 1.8},
        "AAPL": {"start_price": 225.0, "annual_vol": 0.24, "adv": 48_000_000.0, "spread_bps": 1.0},
        "NVDA": {"start_price": 120.0, "annual_vol": 0.45, "adv": 65_000_000.0, "spread_bps": 1.2},
        "MSFT": {"start_price": 420.0, "annual_vol": 0.20, "adv": 22_000_000.0, "spread_bps": 1.1},
    }

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}

    def get_market_data(
        self,
        symbol: str = "RELIANCE.NS",
        days: int = 30,
        intervals_per_day: int = 75,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Retrieves or synthesizes OHLCV dataset for the symbol."""
        cache_key = f"{symbol}_{days}_{intervals_per_day}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()

        params = self.PRESET_ASSETS.get(
            symbol,
            {"start_price": 1000.0, "annual_vol": 0.25, "adv": 5_000_000.0, "spread_bps": 2.5}
        )

        df = generate_synthetic_ohlcv(
            symbol=symbol,
            start_price=params["start_price"],
            annual_vol=params["annual_vol"],
            adv=params["adv"],
            days=days,
            intervals_per_day=intervals_per_day
        )
        self._cache[cache_key] = df
        return df.copy()

    @staticmethod
    def load_from_csv(file_path: str) -> pd.DataFrame:
        """Loads and validates a user-provided OHLCV CSV file."""
        df = pd.read_csv(file_path)
        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV file is missing required columns: {missing}")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        if "spread_bps" not in df.columns:
            df["spread_bps"] = 2.5
        return df
