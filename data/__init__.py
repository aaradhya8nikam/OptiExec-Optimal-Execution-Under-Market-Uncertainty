"""
Market Data module for Optimal Execution Engine.
Handles OHLCV ingestion, intraday volume profile generation, and synthetic market feeds.
"""

from .market_data import MarketDataLoader, generate_u_shape_volume_profile, generate_synthetic_ohlcv
