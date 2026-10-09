"""
Liquidity and Volume Analytics module.
Computes ADV, empirical volume curves, participation rate, and market tightness metrics.
"""

from typing import Dict, Any, Union
import numpy as np
import pandas as pd


def calculate_adv(df: pd.DataFrame) -> float:
    """
    Computes Average Daily Volume (ADV) in shares from OHLCV dataframe.
    """
    if "volume" not in df.columns:
        raise ValueError("DataFrame must contain 'volume' column")
    
    if "timestamp" in df.columns:
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["timestamp"]).dt.date
        daily_vols = df_copy.groupby("date")["volume"].sum()
        return float(daily_vols.mean()) if len(daily_vols) > 0 else float(df["volume"].sum())
    
    return float(df["volume"].mean() * 75)


def estimate_intraday_volume_profile(df: pd.DataFrame, num_target_intervals: int = 12) -> np.ndarray:
    """
    Extracts or aggregates historical intraday volume profile into N target intervals.
    Returns normalized volume fractions [w_1, w_2, ... w_N] summing to 1.0.
    """
    if "volume" not in df.columns:
        from data.market_data import generate_u_shape_volume_profile
        return generate_u_shape_volume_profile(num_target_intervals)

    # Group by time-of-day interval if timestamp exists
    if "timestamp" in df.columns:
        df_copy = df.copy()
        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])
        df_copy["time_bin"] = df_copy["timestamp"].dt.hour * 60 + df_copy["timestamp"].dt.minute
        
        # Bin into target intervals
        time_bins = df_copy["time_bin"].unique()
        time_bins.sort()
        
        if len(time_bins) >= num_target_intervals:
            # Chunk the time bins into num_target_intervals buckets
            chunks = np.array_split(time_bins, num_target_intervals)
            profile = []
            for chunk in chunks:
                vol_in_chunk = df_copy[df_copy["time_bin"].isin(chunk)]["volume"].sum()
                profile.append(vol_in_chunk)
            profile_arr = np.array(profile, dtype=float)
            total = profile_arr.sum()
            if total > 0:
                return profile_arr / total

    # Fallback to parametric U-curve
    from data.market_data import generate_u_shape_volume_profile
    return generate_u_shape_volume_profile(num_target_intervals)


def calculate_participation_rate(
    order_quantity: float,
    execution_horizon_min: float,
    adv: float,
    trading_minutes_per_day: float = 375.0
) -> float:
    """
    Calculates the institutional participation rate (% of market volume consumed):
    Participation Rate = Order Quantity / Expected Market Volume in Horizon
    """
    expected_market_volume = adv * (execution_horizon_min / trading_minutes_per_day)
    if expected_market_volume <= 0:
        return 1.0
    return float(order_quantity / expected_market_volume)


def calculate_order_to_adv_ratio(order_quantity: float, adv: float) -> float:
    """
    Percentage of Average Daily Volume that the order represents.
    """
    if adv <= 0:
        return 1.0
    return float(order_quantity / adv)


def calculate_liquidity_summary(
    df: pd.DataFrame,
    order_quantity: float,
    execution_horizon_min: float
) -> Dict[str, Any]:
    """
    Produces a comprehensive liquidity diagnostic dictionary.
    """
    adv = calculate_adv(df)
    order_to_adv = calculate_order_to_adv_ratio(order_quantity, adv)
    part_rate = calculate_participation_rate(order_quantity, execution_horizon_min, adv)
    
    avg_spread = float(df["spread_bps"].mean()) if "spread_bps" in df.columns else 2.5
    
    # Categorize liquidity regime
    if part_rate < 0.05 and order_to_adv < 0.02:
        regime = "High Liquidity (Low Impact Risk)"
    elif part_rate < 0.15 and order_to_adv < 0.08:
        regime = "Moderate Liquidity (Standard Institutional Execution)"
    else:
        regime = "Constrained Liquidity (Severe Market Impact Warning)"

    return {
        "adv": adv,
        "order_to_adv_pct": order_to_adv * 100.0,
        "participation_rate_pct": part_rate * 100.0,
        "avg_spread_bps": avg_spread,
        "liquidity_regime": regime
    }
