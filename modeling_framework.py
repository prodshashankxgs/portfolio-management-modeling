"""
Phase Three: Modeling & Validation Framework (Pure Polars)
Portfolio Management PPI Modeling Project

Implements model hierarchy:
- Baseline AR models
- Linear models (Ridge/Lasso) with lagged features
- Tree-based models (Random Forest, XGBoost)
- Time series cross-validation framework
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesModeler:
    """
    Pure Polars implementation for time series modeling and validation.
    """
    
    def __init__(self, data_dir: str = "data_cache/parquet"):
        self.data_dir = Path(data_dir)
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def load_analysis_data(self) -> Optional[pl.DataFrame]:
        """Load preprocessed analysis data from exploratory phase."""
        try:
            # Try to load from exploratory analysis results
            from exploratory_analysis import ExploratoryAnalyzer
            
            analyzer = ExploratoryAnalyzer(str(self.data_dir))
            results = analyzer.run_full_analysis()
            
            if "error" in results:
                print(f"Error loading analysis data: {results['error']}")
                return None
                
            return results["analysis_df"]
            
        except Exception as e:
            print(f"Error loading analysis data: {e}")
            return None
    
    def create_lagged_features(self, df: pl.DataFrame, target_col: str, 
                             predictor_cols: List[str], max_lags: int = 6) -> pl.DataFrame:
        """Create lagged features for time series modeling."""
        
        # Start with original data
        feature_df = df.select(['date', target_col] + predictor_cols)
        
        # Add lagged versions of predictors
        for col in predictor_cols:
            for lag in range(1, max_lags + 1):
                feature_df = feature_df.with_columns([
                    pl.col(col).shift(lag).alias(f'{col}_lag{lag}')
                ])
        
        # Add lagged target for AR components
        for lag in range(1, 4):  # AR(3) components
            feature_df = feature_df.with_columns([
                pl.col(target_col).shift(lag).alias(f'{target_col}_lag{lag}')
            ])
        
        # Add rolling statistics
        for col in predictor_cols:
            feature_df = feature_df.with_columns([
                pl.col(col).rolling_mean(window_size=3).alias(f'{col}_ma3'),
                pl.col(col).rolling_std(window_size=3).alias(f'{col}_vol3'),
                pl.col(col).rolling_mean(window_size=6).alias(f'{col}_ma6')
            ])
        
        # Add momentum indicators
        for col in predictor_cols:
            feature_df = feature_df.with_columns([
                (pl.col(col) - pl.col(col).shift(3)).alias(f'{col}_mom3'),
                (pl.col(col) / pl.col(col).shift(6) - 1).alias(f'{col}_mom6')
            ])
        
        return feature_df.drop_nulls()
    
    def time_series_split(self, df: pl.DataFrame, train_size: float = 0.7, 
                         validation_size: float = 0.15) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split data for time series validation (no shuffling)."""
        
        n_total = df.height
        n_train = int(n_total * train_size)
        n_val = int(n_total * validation_size)
        
        train_df = df.head(n_train)
        val_df = df.slice(n_train, n_val)
        test_df = df.tail(n_total - n_train - n_val)
        
        return train_df, val_df, test_df
    
    def expanding_window_cv(self, df: pl.DataFrame, target_col: str, 
                           feature_cols: List[str], min_train_size: int = 60) -> List[Tuple[pl.DataFrame, pl.DataFrame]]:
        """Create expanding window cross-validation splits."""
        
        splits = []
        n_total = df.height
        
        # Start with minimum training size, expand by 12 months each time
        for end_train in range(min_train_size, n_total - 12, 12):
            train_df = df.head(end_train)
            test_df = df.slice(end_train, 12)  # Next 12 months for testing
            
            if test_df.height > 0:
                splits.append((train_df, test_df))
        
        return splits
    
    def fit_baseline_ar_model(self, train_df: pl.DataFrame, target_col: str) -> Dict[str, Any]:
        """Fit simple autoregressive baseline model."""
        
        # Prepare AR features (lags 1-3)
        ar_features = []
        for lag in range(1, 4):
            ar_features.append(f'{target_col}_lag{lag}')
        
        # Filter to available features
        available_features = [col for col in ar_features if col in train_df.columns]
        
        if not available_features:
            return {"error": "No AR features available"}
        
        # Prepare training data
        train_clean = train_df.select(['date', target_col] + available_features).drop_nulls()
        
        if train_clean.height < 10:
            return {"error": "Insufficient training data"}
        
        X_train = train_clean.select(available_features).to_numpy()
        y_train = train_clean[target_col].to_numpy()
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Calculate in-sample metrics
        y_pred = model.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)
        
        return {
            "model": model,
            "features": available_features,
            "rmse": rmse,
            "mae": mae,
            "n_train": len(y_train)
        }
    
    def fit_ridge_model(self, train_df: pl.DataFrame, target_col: str, 
                       feature_cols: List[str], alpha: float = 1.0) -> Dict[str, Any]:
        """Fit Ridge regression with lagged features."""
        
        # Prepare training data
        all_cols = [target_col] + feature_cols
        train_clean = train_df.select(['date'] + all_cols).drop_nulls()
        
        if train_clean.height < 20:
            return {"error": "Insufficient training data"}
        
        X_train = train_clean.select(feature_cols).to_numpy()
        y_train = train_clean[target_col].to_numpy()
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Fit Ridge model
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_train_scaled)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)
        
        return {
            "model": model,
            "scaler": scaler,
            "features": feature_cols,
            "rmse": rmse,
            "mae": mae,
            "n_train": len(y_train)
        }
    
    def fit_random_forest(self, train_df: pl.DataFrame, target_col: str, 
                         feature_cols: List[str], n_estimators: int = 100) -> Dict[str, Any]:
        """Fit Random Forest model."""
        
        # Prepare training data
        all_cols = [target_col] + feature_cols
        train_clean = train_df.select(['date'] + all_cols).drop_nulls()
        
        if train_clean.height < 30:
            return {"error": "Insufficient training data"}
        
        X_train = train_clean.select(feature_cols).to_numpy()
        y_train = train_clean[target_col].to_numpy()
        
        # Fit Random Forest
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = model.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)
        
        # Feature importance
        importance = dict(zip(feature_cols, model.feature_importances_))
        
        return {
            "model": model,
            "features": feature_cols,
            "rmse": rmse,
            "mae": mae,
            "n_train": len(y_train),
            "feature_importance": importance
        }
    
    def predict_model(self, model_result: Dict[str, Any], test_df: pl.DataFrame, 
                     target_col: str) -> Optional[np.ndarray]:
        """Make predictions using fitted model."""
        
        if "error" in model_result:
            return None
        
        model = model_result["model"]
        features = model_result["features"]
        
        # Prepare test data
        test_clean = test_df.select(['date', target_col] + features).drop_nulls()
        
        if test_clean.height == 0:
            return None
        
        X_test = test_clean.select(features).to_numpy()
        
        # Apply scaling if available
        if "scaler" in model_result:
            X_test = model_result["scaler"].transform(X_test)
        
        return model.predict(X_test)
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Directional accuracy
        direction_true = np.sign(y_true)
        direction_pred = np.sign(y_pred)
        directional_accuracy = np.mean(direction_true == direction_pred)
        
        return {
            "rmse": rmse,
            "mae": mae,
            "directional_accuracy": directional_accuracy,
            "n_predictions": len(y_true)
        }
    
    def run_model_comparison(self, df: pl.DataFrame, target_col: str = 'mom_pct') -> Dict[str, Any]:
        """Run comprehensive model comparison."""
        
        print("Starting Phase Three: Modeling & Validation")
        print("=" * 50)
        
        # Get feature columns (exclude date and target)
        feature_cols = [col for col in df.columns if col not in ['date', target_col]]
        
        print(f"Target: {target_col}")
        print(f"Features available: {len(feature_cols)}")
        print(f"Data shape: {df.shape}")
        
        # Create lagged features
        print("\nCreating lagged features...")
        feature_df = self.create_lagged_features(df, target_col, feature_cols[:10], max_lags=3)  # Limit for performance
        
        print(f"Enhanced feature set: {feature_df.shape}")
        
        # Split data
        train_df, val_df, test_df = self.time_series_split(feature_df)
        
        print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
        
        # Get all feature columns after lagging
        all_features = [col for col in feature_df.columns if col not in ['date', target_col]]
        
        # Model results
        results = {}
        
        # 1. Baseline AR Model
        print("\n1. Fitting Baseline AR Model...")
        ar_result = self.fit_baseline_ar_model(train_df, target_col)
        if "error" not in ar_result:
            ar_pred = self.predict_model(ar_result, test_df, target_col)
            if ar_pred is not None:
                y_test = test_df.select(target_col).drop_nulls().to_numpy()
                if len(y_test) == len(ar_pred):
                    ar_metrics = self.evaluate_model(y_test, ar_pred)
                    results["AR_Baseline"] = {**ar_result, **ar_metrics}
                    print(f"   AR Model RMSE: {ar_metrics['rmse']:.4f}")
        
        # 2. Ridge Regression
        print("\n2. Fitting Ridge Regression...")
        ridge_result = self.fit_ridge_model(train_df, target_col, all_features[:20])  # Limit features
        if "error" not in ridge_result:
            ridge_pred = self.predict_model(ridge_result, test_df, target_col)
            if ridge_pred is not None:
                y_test = test_df.select(target_col).drop_nulls().to_numpy()
                if len(y_test) == len(ridge_pred):
                    ridge_metrics = self.evaluate_model(y_test, ridge_pred)
                    results["Ridge"] = {**ridge_result, **ridge_metrics}
                    print(f"   Ridge RMSE: {ridge_metrics['rmse']:.4f}")
        
        # 3. Random Forest
        print("\n3. Fitting Random Forest...")
        rf_result = self.fit_random_forest(train_df, target_col, all_features[:15])  # Limit features
        if "error" not in rf_result:
            rf_pred = self.predict_model(rf_result, test_df, target_col)
            if rf_pred is not None:
                y_test = test_df.select(target_col).drop_nulls().to_numpy()
                if len(y_test) == len(rf_pred):
                    rf_metrics = self.evaluate_model(y_test, rf_pred)
                    results["Random_Forest"] = {**rf_result, **rf_metrics}
                    print(f"   Random Forest RMSE: {rf_metrics['rmse']:.4f}")
        
        return {
            "results": results,
            "data_splits": {
                "train_shape": train_df.shape,
                "val_shape": val_df.shape,
                "test_shape": test_df.shape
            },
            "feature_count": len(all_features)
        }
    
    def print_model_summary(self, comparison_results: Dict[str, Any]):
        """Print comprehensive model comparison summary."""
        
        print("\nMODEL COMPARISON SUMMARY")
        print("=" * 60)
        
        results = comparison_results["results"]
        
        if not results:
            print("No successful model results to display.")
            return
        
        # Create summary table
        print(f"{'Model':<15} {'RMSE':<8} {'MAE':<8} {'Dir.Acc':<8} {'Features':<10}")
        print("-" * 60)
        
        for model_name, model_result in results.items():
            rmse = model_result.get('rmse', 0)
            mae = model_result.get('mae', 0)
            dir_acc = model_result.get('directional_accuracy', 0)
            n_features = len(model_result.get('features', []))
            
            print(f"{model_name:<15} {rmse:<8.4f} {mae:<8.4f} {dir_acc:<8.3f} {n_features:<10}")
        
        # Best model
        best_model = min(results.items(), key=lambda x: x[1].get('rmse', float('inf')))
        print(f"\n🏆 Best Model: {best_model[0]} (RMSE: {best_model[1]['rmse']:.4f})")
        
        # Feature importance for tree models
        for model_name, model_result in results.items():
            if 'feature_importance' in model_result:
                print(f"\n📊 {model_name} - Top 10 Features:")
                importance = model_result['feature_importance']
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for i, (feature, imp) in enumerate(sorted_features[:10]):
                    print(f"   {i+1:2d}. {feature:<25} {imp:.4f}")


if __name__ == "__main__":
    # Run modeling framework
    modeler = TimeSeriesModeler()
    
    # Load data
    analysis_df = modeler.load_analysis_data()
    
    if analysis_df is not None:
        # Run model comparison
        results = modeler.run_model_comparison(analysis_df)
        
        # Print summary
        modeler.print_model_summary(results)
    else:
        print("Failed to load analysis data. Please run Phase Two first.")
