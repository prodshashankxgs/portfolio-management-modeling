"""
Phase Two: Exploratory Data Analysis Module (Pure Polars)
Portfolio Management PPI Modeling Project
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class ExploratoryAnalyzer:
    """
    Pure Polars implementation for exploratory data analysis of PPI modeling.
    """
    
    def __init__(self, data_dir: str = "data_cache/parquet"):
        self.data_dir = Path(data_dir)
        self.data = {}
        self.target_df = None
        self.predictors_df = None
        self.analysis_df = None
        
    def load_data(self) -> bool:
        """Load all Parquet data files."""
        if not self.data_dir.exists():
            print(f"⚠️ Data directory {self.data_dir} not found.")
            return False
        
        for file in self.data_dir.glob("*.parquet"):
            try:
                df = pl.read_parquet(file)
                key = file.stem
                self.data[key] = df
                print(f"✓ Loaded {key}: {df.shape}")
            except Exception as e:
                print(f"⚠️ Error loading {file}: {e}")
        
        # Extract target data
        self.target_df = self.data.get('target_ppidf01')
        return len(self.data) > 0
    
    def prepare_target_variable(self) -> Optional[pl.DataFrame]:
        """Prepare target variable with MoM% changes."""
        if self.target_df is None:
            return None
            
        target_processed = self.target_df.sort('date').with_columns([
            (pl.col('value').pct_change() * 100).alias('mom_pct'),
            pl.col('value').log().alias('log_value'),
            (pl.col('value') / pl.col('value').shift(12) - 1).alias('yoy_change'),
            pl.col('date').dt.year().alias('year'),
            pl.col('date').dt.month().alias('month'),
            pl.col('date').dt.quarter().alias('quarter')
        ])
        
        return target_processed
    
    def prepare_predictors(self) -> Optional[pl.DataFrame]:
        """Prepare and combine predictor variables."""
        predictor_dfs = []
        
        for key, df in self.data.items():
            if key.startswith('target_'):
                continue
                
            try:
                processed_df = df.sort('date').with_columns([
                    pl.col('value').alias(f'{key}_level'),
                    (pl.col('value').pct_change() * 100).alias(f'{key}_mom_pct'),
                    pl.col('value').rolling_mean(window_size=3).alias(f'{key}_ma3'),
                    pl.col('value').rolling_std(window_size=3).alias(f'{key}_vol3')
                ]).select(['date', f'{key}_level', f'{key}_mom_pct', f'{key}_ma3', f'{key}_vol3'])
                
                predictor_dfs.append(processed_df)
                
            except Exception as e:
                print(f"⚠️ Error processing {key}: {e}")
        
        if not predictor_dfs:
            return None
        
        # Join all predictors
        combined_df = predictor_dfs[0]
        for df in predictor_dfs[1:]:
            combined_df = combined_df.join(df, on='date', how='outer')
        
        return combined_df.sort('date')
    
    def calculate_correlations(self, target_processed: pl.DataFrame, predictors_df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Calculate correlations between target and predictors."""
        # Join target with predictors
        analysis_df = target_processed.select(['date', 'mom_pct']).join(
            predictors_df, on='date', how='inner'
        ).drop_nulls()
        
        self.analysis_df = analysis_df
        
        # Calculate correlations
        target_col = 'mom_pct'
        predictor_cols = [col for col in analysis_df.columns if col not in ['date', target_col]]
        
        correlations = []
        for col in predictor_cols:
            try:
                corr = analysis_df.select(pl.corr(target_col, col))['correlation'][0]
                if corr is not None and not np.isnan(corr):
                    correlations.append((col, corr))
            except:
                continue
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return pl.DataFrame({
            'variable': [item[0] for item in correlations],
            'correlation': [item[1] for item in correlations]
        })
    
    def lead_lag_analysis(self, target_col: str, predictor_col: str, max_lags: int = 6) -> Dict[int, float]:
        """Calculate lead-lag correlations."""
        if self.analysis_df is None:
            return {}
            
        correlations = {}
        df = self.analysis_df
        
        for lag in range(-max_lags, max_lags + 1):
            try:
                if lag == 0:
                    corr = df.select(pl.corr(target_col, predictor_col))['correlation'][0]
                elif lag > 0:
                    corr = df.select(pl.corr(target_col, pl.col(predictor_col).shift(lag)))['correlation'][0]
                else:
                    corr = df.select(pl.corr(pl.col(target_col).shift(-lag), predictor_col))['correlation'][0]
                
                if corr is not None and not np.isnan(corr):
                    correlations[lag] = corr
            except:
                continue
        
        return correlations
    
    def run_full_analysis(self) -> Dict:
        """Run complete exploratory analysis."""
        print("Starting Phase Two: Exploratory Data Analysis")
        print("=" * 50)
        
        # Load data
        if not self.load_data():
            return {"error": "Failed to load data"}
        
        # Prepare target
        target_processed = self.prepare_target_variable()
        if target_processed is None:
            return {"error": "Failed to prepare target variable"}
        
        # Prepare predictors
        predictors_df = self.prepare_predictors()
        if predictors_df is None:
            return {"error": "Failed to prepare predictors"}
        
        # Calculate correlations
        corr_df = self.calculate_correlations(target_processed, predictors_df)
        if corr_df is None:
            return {"error": "Failed to calculate correlations"}
        
        # Lead-lag analysis for top predictors
        top_predictors = corr_df.head(5)['variable'].to_list()
        lead_lag_results = {}
        
        for predictor in top_predictors:
            if predictor in self.analysis_df.columns:
                correlations = self.lead_lag_analysis('mom_pct', predictor)
                lead_lag_results[predictor] = correlations
        
        # Summary statistics
        mom_stats = target_processed.select([
            pl.col('mom_pct').drop_nulls().count().alias('count'),
            pl.col('mom_pct').drop_nulls().mean().alias('mean'),
            pl.col('mom_pct').drop_nulls().std().alias('std'),
            pl.col('mom_pct').drop_nulls().min().alias('min'),
            pl.col('mom_pct').drop_nulls().max().alias('max')
        ])
        
        return {
            "target_stats": mom_stats,
            "correlations": corr_df,
            "lead_lag_results": lead_lag_results,
            "analysis_df": self.analysis_df,
            "data_shape": self.analysis_df.shape if self.analysis_df is not None else None
        }

if __name__ == "__main__":
    # Run analysis
    analyzer = ExploratoryAnalyzer()
    results = analyzer.run_full_analysis()
    
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(f"\n✓ Analysis complete!")
        print(f"Data shape: {results['data_shape']}")
        print(f"Top correlations:")
        print(results['correlations'].head(10))
