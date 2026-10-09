"""
Utility script to generate sample CSV datasets in data/sample_datasets/
"""

import os
from .market_data import MarketDataLoader


def generate_all_sample_datasets(output_dir: str = "data/sample_datasets"):
    """Generates and saves sample CSV datasets for standard equities."""
    os.makedirs(output_dir, exist_ok=True)
    loader = MarketDataLoader()
    
    for symbol in MarketDataLoader.PRESET_ASSETS.keys():
        clean_name = symbol.replace(".", "_")
        file_path = os.path.join(output_dir, f"{clean_name}_30d_5min.csv")
        df = loader.get_market_data(symbol=symbol, days=30, use_cache=False)
        df.to_csv(file_path, index=False)
        print(f"Generated sample dataset: {file_path} ({len(df)} rows)")


if __name__ == "__main__":
    generate_all_sample_datasets()
