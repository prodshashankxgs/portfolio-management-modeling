"""
Data Collection Module for Portfolio Management PPI Modeling
Phase One: Data Collection Strategy Implementation
"""

import os
import time
import requests
import polars as pl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import json
from pathlib import Path

class DataCollector:
    """
    Modular data collection class for PPI Portfolio Management modeling.
    Implements caching, rate limiting, and API management for FRED and BLS data.
    """
    
    def __init__(self, fred_api_key: Optional[str] = None, cache_dir: str = "data_cache"):
        """
        Initialize DataCollector with API keys and cache directory.
        
        Args:
            fred_api_key: FRED API key (can be set via environment variable)
            cache_dir: Directory for caching downloaded data
        """
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Rate limiting parameters
        self.last_api_call = 0
        self.min_call_interval = 0.1  # 100ms between calls
        
        # BLS API base URL
        self.bls_base_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        
        # FRED API base URL
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        
    def _rate_limit(self):
        """Implement rate limiting between API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.min_call_interval:
            time.sleep(self.min_call_interval - time_since_last_call)
        self.last_api_call = time.time()
    
    def _get_cache_path(self, series_id: str, source: str) -> Path:
        """Generate cache file path for a given series."""
        return self.cache_dir / f"{source}_{series_id}.json"
    
    def _load_from_cache(self, series_id: str, source: str, max_age_days: int = 1) -> Optional[Dict]:
        """Load data from cache if it exists and is recent enough."""
        cache_path = self._get_cache_path(series_id, source)
        
        if not cache_path.exists():
            return None
            
        # Check if cache is too old
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age.days > max_age_days:
            return None
            
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    def _save_to_cache(self, data: Dict, series_id: str, source: str):
        """Save data to cache."""
        cache_path = self._get_cache_path(series_id, source)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save to cache: {e}")
    
    def get_fred_data(self, series_id: str, start_date: str = "2000-01-01", 
                     end_date: Optional[str] = None, use_cache: bool = True) -> pl.DataFrame:
        """
        Fetch data from FRED API with caching and rate limiting.
        
        Args:
            series_id: FRED series identifier
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)
            use_cache: Whether to use cached data if available
            
        Returns:
            Polars DataFrame with date and value columns
        """
        if not self.fred_api_key:
            raise ValueError("FRED API key not provided. Set FRED_API_KEY environment variable.")
        
        # Check cache first
        if use_cache:
            cached_data = self._load_from_cache(series_id, "fred")
            if cached_data:
                return pl.DataFrame(cached_data)
        
        # Prepare API request
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }
        
        # Make API call with rate limiting
        self._rate_limit()
        
        try:
            response = requests.get(self.fred_base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'observations' not in data:
                raise ValueError(f"No observations found for series {series_id}")
            
            # Convert to DataFrame
            observations = data['observations']
            df_data = []
            
            for obs in observations:
                if obs['value'] != '.':  # FRED uses '.' for missing values
                    df_data.append({
                        'date': obs['date'],
                        'value': float(obs['value']),
                        'series_id': series_id
                    })
            
            df = pl.DataFrame(df_data)
            df = df.with_columns(pl.col('date').str.to_date())
            
            # Cache the result
            if use_cache:
                self._save_to_cache(df.to_dicts(), series_id, "fred")
            
            return df
            
        except requests.RequestException as e:
            raise RuntimeError(f"Error fetching FRED data for {series_id}: {e}")
    
    def get_target_variable(self) -> pl.DataFrame:
        """
        Fetch the primary target variable: PPI for Portfolio Management Services.
        
        Returns:
            Polars DataFrame with PPI data
        """
        print("Fetching target variable: PPI for Portfolio Management Services")
        return self.get_fred_data('PCU5239252392', start_date='2000-01-01')
    
    def get_market_returns_data(self) -> Dict[str, pl.DataFrame]:
        """
        Fetch market returns data from FRED.
        
        Returns:
            Dictionary of DataFrames for different market indices
        """
        market_series = {
            'SP500': 'SP500',           # S&P 500
            'NASDAQCOM': 'NASDAQCOM',   # NASDAQ Composite
            'RUT': 'RUT',               # Russell 2000 (if available)
            'VIXCLS': 'VIXCLS'          # VIX Volatility Index
        }
        
        market_data = {}
        
        for name, series_id in market_series.items():
            try:
                print(f"Fetching {name} data...")
                df = self.get_fred_data(series_id, start_date='2000-01-01')
                market_data[name] = df
            except Exception as e:
                print(f"Warning: Could not fetch {name} ({series_id}): {e}")
                
        return market_data
    
    def get_bond_market_data(self) -> Dict[str, pl.DataFrame]:
        """
        Fetch bond market data from FRED.
        
        Returns:
            Dictionary of DataFrames for bond market indicators
        """
        bond_series = {
            'DGS10': 'DGS10',           # 10-Year Treasury Constant Maturity Rate
            'DGS2': 'DGS2',             # 2-Year Treasury Constant Maturity Rate
            'DGS3MO': 'DGS3MO',         # 3-Month Treasury Constant Maturity Rate
            'BAMLC0A0CM': 'BAMLC0A0CM', # ICE BofA US Corporate Bond Index Option-Adjusted Spread
        }
        
        bond_data = {}
        
        for name, series_id in bond_series.items():
            try:
                print(f"Fetching {name} data...")
                df = self.get_fred_data(series_id, start_date='2000-01-01')
                bond_data[name] = df
            except Exception as e:
                print(f"Warning: Could not fetch {name} ({series_id}): {e}")
                
        return bond_data
    
    def get_macro_indicators(self) -> Dict[str, pl.DataFrame]:
        """
        Fetch macroeconomic indicators from FRED.
        
        Returns:
            Dictionary of DataFrames for macro indicators
        """
        macro_series = {
            'DTWEXBGS': 'DTWEXBGS',     # Trade Weighted U.S. Dollar Index: Broad, Goods and Services
            'DCOILWTICO': 'DCOILWTICO', # Crude Oil Prices: West Texas Intermediate
            'GOLDAMGBD228NLBM': 'GOLDAMGBD228NLBM', # Gold Fixing Price
            'UNRATE': 'UNRATE',         # Unemployment Rate
            'CPIAUCSL': 'CPIAUCSL',     # Consumer Price Index for All Urban Consumers
        }
        
        macro_data = {}
        
        for name, series_id in macro_series.items():
            try:
                print(f"Fetching {name} data...")
                df = self.get_fred_data(series_id, start_date='2000-01-01')
                macro_data[name] = df
            except Exception as e:
                print(f"Warning: Could not fetch {name} ({series_id}): {e}")
                
        return macro_data
    
    def collect_all_data(self) -> Dict[str, Union[pl.DataFrame, Dict[str, pl.DataFrame]]]:
        """
        Collect all data sources for the modeling project.
        
        Returns:
            Dictionary containing all collected data
        """
        print("Starting comprehensive data collection for Phase One...")
        
        all_data = {}
        
        # Target variable
        all_data['target'] = self.get_target_variable()
        
        # Market data
        all_data['market_returns'] = self.get_market_returns_data()
        
        # Bond market data
        all_data['bond_market'] = self.get_bond_market_data()
        
        # Macro indicators
        all_data['macro_indicators'] = self.get_macro_indicators()
        
        print("Data collection complete!")
        return all_data
    
    def save_collected_data(self, data: Dict, filename: str = "collected_data.json"):
        """
        Save all collected data to a file.
        
        Args:
            data: Dictionary of collected data
            filename: Output filename
        """
        output_path = self.cache_dir / filename
        
        # Convert Polars DataFrames to dictionaries for JSON serialization
        serializable_data = {}
        
        for key, value in data.items():
            if isinstance(value, pl.DataFrame):
                serializable_data[key] = value.to_dicts()
            elif isinstance(value, dict):
                serializable_data[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, pl.DataFrame):
                        serializable_data[key][subkey] = subvalue.to_dicts()
                    else:
                        serializable_data[key][subkey] = subvalue
            else:
                serializable_data[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)
        
        print(f"Data saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Collect all data
    all_data = collector.collect_all_data()
    
    # Save to file
    collector.save_collected_data(all_data)
    
    # Print summary
    print("\nData Collection Summary:")
    print(f"Target variable shape: {all_data['target'].shape}")
    
    for category, data_dict in all_data.items():
        if category != 'target' and isinstance(data_dict, dict):
            print(f"\n{category.title()}:")
            for name, df in data_dict.items():
                if isinstance(df, pl.DataFrame):
                    print(f"  {name}: {df.shape}")
