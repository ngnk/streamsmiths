"""
Complete Time Series Analysis Script for videos_log_v2
Run this on YOUR LOCAL MACHINE where network access to Neon is available

Usage:
    python neon_timeseries_analysis.py

Requirements:
    pip install polars pandas numpy statsmodels scipy scikit-learn matplotlib python-dotenv pyarrow connectorx
"""

import os
import pandas as pd
import polars as pl
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from dotenv import load_dotenv


# ============================================================================
# NEON DATABASE CONNECTION
# ============================================================================

def fetch_videos_from_neon(table_name: str = "videos_log_v2") -> pl.DataFrame:
    """Fetch videos_log_v2 table from Neon database"""
    load_dotenv()
    
    db_uri = os.getenv("NEON_DATABASE_URL")
    if not db_uri:
        # If no .env file, use the direct connection string
        db_uri = "postgresql://neondb_owner:npg_c38OpMvawKVg@ep-withered-sky-aeqx6en8-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    
    # Clean the connection string
    db_uri = db_uri.strip()
    if db_uri.startswith("psql "):
        db_uri = db_uri[5:].strip()
    db_uri = db_uri.strip("'\"")
    
    print(f"[FETCH] Connecting to Neon database...")
    print(f"[FETCH] Fetching table: {table_name}")
    
    try:
        # Read from database using Polars
        query = f"SELECT * FROM {table_name} ORDER BY ingestion_timestamp"
        df = pl.read_database_uri(query=query, uri=db_uri)
        
        print(f"[FETCH] ✓ Fetched {len(df)} rows from {table_name}")
        return df
    except Exception as e:
        print(f"[FETCH] ✗ Error: {e}")
        print("\nTrying pandas + SQLAlchemy as backup...")
        return fetch_with_pandas(table_name, db_uri)


def fetch_with_pandas(table_name: str, db_uri: str) -> pl.DataFrame:
    """Backup method using pandas + SQLAlchemy"""
    from sqlalchemy import create_engine
    
    engine = create_engine(db_uri)
    query = f"SELECT * FROM {table_name} ORDER BY ingestion_timestamp"
    
    df_pandas = pd.read_sql(query, engine)
    print(f"[FETCH] ✓ Fetched {len(df_pandas)} rows using pandas")
    
    # Convert to Polars
    return pl.from_pandas(df_pandas)


# ============================================================================
# TIME SERIES DATA PREPARATION
# ============================================================================

def prepare_timeseries_data(df: pl.DataFrame, aggregation: str = "hourly") -> pd.DataFrame:
    """
    Convert videos_log_v2 data into aggregated time series.
    
    Args:
        df: Polars DataFrame from Neon
        aggregation: "hourly" or "daily"
        
    Returns:
        Pandas DataFrame with aggregated metrics
    """
    print(f"\n[PREPARE] Original data: {len(df)} video records")
    print(f"[PREPARE] Aggregation: {aggregation}")
    
    # Convert to pandas for easier time series manipulation
    df_pd = df.to_pandas()
    
    # Parse timestamps
    df_pd['ingestion_timestamp'] = pd.to_datetime(df_pd['ingestion_timestamp'])
    
    # Create time bins
    if aggregation == "hourly":
        df_pd['time_bin'] = df_pd['ingestion_timestamp'].dt.floor('H')
    else:  # daily
        df_pd['time_bin'] = df_pd['ingestion_timestamp'].dt.date
    
    # Calculate engagement_rate if not present or recalculate
    df_pd['engagement_rate'] = ((df_pd['like_count'] + df_pd['comment_count']) / 
                                 df_pd['view_count'].replace(0, 1)) * 100
    
    # Aggregate
    agg_metrics = df_pd.groupby('time_bin').agg({
        'view_count': 'sum',
        'like_count': 'sum',
        'comment_count': 'sum',
        'engagement_rate': 'mean',
        'views_per_day': 'mean',
        'video_id': 'count'
    }).reset_index()
    
    agg_metrics.columns = ['time_bin', 'total_views', 'total_likes', 'total_comments',
                           'avg_engagement_rate', 'avg_views_per_day', 'num_videos']
    
    agg_metrics = agg_metrics.sort_values('time_bin').reset_index(drop=True)
    
    print(f"[PREPARE] Aggregated to: {len(agg_metrics)} time points")
    print(f"[PREPARE] Time range: {agg_metrics['time_bin'].min()} to {agg_metrics['time_bin'].max()}")
    
    return agg_metrics


# ============================================================================
# DYNAMIC LINEAR MODEL (DLM)
# ============================================================================

class DLMForwardBackward:
    """Dynamic Linear Model with Forward Filtering and Backward Smoothing"""
    
    def __init__(self, obs_variance=1.0, state_variance=1.0):
        self.obs_variance = obs_variance
        self.state_variance = state_variance
        self.filtered_means = None
        self.filtered_covs = None
        self.smoothed_means = None
        self.smoothed_covs = None
        
    def forward_filter(self, y):
        """Kalman Filter - Forward Pass"""
        n = len(y)
        filtered_means = np.zeros(n)
        filtered_covs = np.zeros(n)
        predicted_means = np.zeros(n)
        predicted_covs = np.zeros(n)
        
        m0 = y[0]
        C0 = self.state_variance * 10
        
        for t in range(n):
            if t == 0:
                predicted_means[t] = m0
                predicted_covs[t] = C0
            else:
                predicted_means[t] = filtered_means[t-1]
                predicted_covs[t] = filtered_covs[t-1] + self.state_variance
            
            forecast_error = y[t] - predicted_means[t]
            forecast_variance = predicted_covs[t] + self.obs_variance
            kalman_gain = predicted_covs[t] / forecast_variance
            
            filtered_means[t] = predicted_means[t] + kalman_gain * forecast_error
            filtered_covs[t] = (1 - kalman_gain) * predicted_covs[t]
        
        self.filtered_means = filtered_means
        self.filtered_covs = filtered_covs
        self.predicted_means = predicted_means
        self.predicted_covs = predicted_covs
        
        return filtered_means
    
    def backward_smooth(self, y):
        """Rauch-Tung-Striebel Smoother - Backward Pass"""
        if self.filtered_means is None:
            self.forward_filter(y)
        
        n = len(y)
        smoothed_means = np.zeros(n)
        smoothed_covs = np.zeros(n)
        
        smoothed_means[-1] = self.filtered_means[-1]
        smoothed_covs[-1] = self.filtered_covs[-1]
        
        for t in range(n-2, -1, -1):
            J_t = self.filtered_covs[t] / (self.filtered_covs[t] + self.state_variance)
            
            smoothed_means[t] = self.filtered_means[t] + J_t * (
                smoothed_means[t+1] - self.filtered_means[t]
            )
            smoothed_covs[t] = self.filtered_covs[t] + J_t**2 * (
                smoothed_covs[t+1] - self.filtered_covs[t] - self.state_variance
            )
        
        self.smoothed_means = smoothed_means
        self.smoothed_covs = smoothed_covs
        
        return smoothed_means
    
    def fit_predict(self, y_train, y_test=None):
        """Fit model and make predictions"""
        self.forward_filter(y_train)
        self.backward_smooth(y_train)
        
        if y_test is not None:
            n_test = len(y_test)
            predictions = np.zeros(n_test)
            predictions[0] = self.filtered_means[-1]
            
            for i in range(1, n_test):
                predictions[i] = predictions[i-1]
            
            return self.smoothed_means, predictions
        
        return self.smoothed_means, None


# ============================================================================
# MODEL FITTING FUNCTIONS
# ============================================================================

def fit_arima_model(y_train, y_test, order=(1, 1, 1)):
    """Fit ARIMA model"""
    try:
        model = ARIMA(y_train, order=order)
        model_fit = model.fit()
        
        fitted = model_fit.fittedvalues
        forecast = model_fit.forecast(steps=len(y_test))
        
        return {
            'model': model_fit,
            'fitted': fitted,
            'forecast': forecast,
            'aic': model_fit.aic,
            'bic': model_fit.bic,
            'params': order
        }
    except Exception as e:
        print(f"ARIMA fitting error: {e}")
        return None


def fit_ar1_model(y_train, y_test):
    """Fit AR(1) model"""
    try:
        model = AutoReg(y_train, lags=1)
        model_fit = model.fit()
        
        fitted = model_fit.fittedvalues
        forecast = model_fit.forecast(steps=len(y_test))
        
        return {
            'model': model_fit,
            'fitted': fitted,
            'forecast': forecast,
            'aic': model_fit.aic,
            'bic': model_fit.bic,
            'params': 1
        }
    except Exception as e:
        print(f"AR(1) fitting error: {e}")
        return None


def fit_dlm_model(y_train, y_test):
    """Fit DLM with Forward-Backward Smoother"""
    try:
        obs_var = np.var(y_train) * 0.1
        state_var = np.var(np.diff(y_train)) * 0.5
        
        dlm = DLMForwardBackward(obs_variance=obs_var, state_variance=state_var)
        fitted, forecast = dlm.fit_predict(y_train, y_test)
        
        n = len(y_train)
        k = 2
        rss = np.sum((y_train - fitted)**2)
        
        aic = n * np.log(rss / n) + 2 * k
        bic = n * np.log(rss / n) + k * np.log(n)
        
        return {
            'model': dlm,
            'fitted': fitted,
            'forecast': forecast,
            'aic': aic,
            'bic': bic,
            'params': {'obs_var': obs_var, 'state_var': state_var}
        }
    except Exception as e:
        print(f"DLM fitting error: {e}")
        return None


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'MSE': mse
    }


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_model_comparison(data, target_col='total_views', train_size=0.7):
    """Run all models and compare performance"""
    
    if len(data) < 10:
        raise ValueError(
            f"\n{'='*60}\n"
            f"INSUFFICIENT DATA: Only {len(data)} time points available!\n"
            f"{'='*60}\n"
            "Time series analysis requires at least 10 observations (preferably 20+).\n"
        )
    
    y = data[target_col].values
    dates = data['time_bin'].values
    
    split_idx = int(len(y) * train_size)
    split_idx = max(5, min(split_idx, len(y) - 5))
    
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    dates_train = dates[:split_idx]
    dates_test = dates[split_idx:]
    
    print(f"\nAnalyzing: {target_col}")
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
    print(f"Time range: {dates[0]} to {dates[-1]}")
    print("="*60)
    
    results = {}
    
    # Fit ARIMA
    print("\nFitting ARIMA(1,1,1)...")
    arima_result = fit_arima_model(y_train, y_test, order=(1, 1, 1))
    if arima_result:
        results['ARIMA'] = arima_result
        print(f"  AIC: {arima_result['aic']:.2f}, BIC: {arima_result['bic']:.2f}")
    
    # Fit AR(1)
    print("\nFitting AR(1)...")
    ar1_result = fit_ar1_model(y_train, y_test)
    if ar1_result:
        results['AR1'] = ar1_result
        print(f"  AIC: {ar1_result['aic']:.2f}, BIC: {ar1_result['bic']:.2f}")
    
    # Fit DLM
    print("\nFitting DLM (Forward-Backward Smoother)...")
    dlm_result = fit_dlm_model(y_train, y_test)
    if dlm_result:
        results['DLM'] = dlm_result
        print(f"  AIC: {dlm_result['aic']:.2f}, BIC: {dlm_result['bic']:.2f}")
    
    if not results:
        print("\nNo models were successfully fitted.")
        return None
    
    # Calculate metrics
    print("\n" + "="*60)
    print("TRAINING SET PERFORMANCE:")
    print("="*60)
    
    train_metrics = {}
    for model_name, result in results.items():
        fitted = result['fitted']
        min_len = min(len(y_train), len(fitted))
        if min_len > 0:
            metrics = calculate_metrics(y_train[-min_len:], fitted[-min_len:])
            train_metrics[model_name] = metrics
            
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    print("\n" + "="*60)
    print("TEST SET PERFORMANCE:")
    print("="*60)
    
    test_metrics = {}
    for model_name, result in results.items():
        forecast = result['forecast']
        min_len = min(len(y_test), len(forecast))
        if min_len > 0:
            metrics = calculate_metrics(y_test[:min_len], forecast[:min_len])
            test_metrics[model_name] = metrics
            
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    
    # Model comparison summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY:")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'AIC': [r['aic'] for r in results.values()],
        'BIC': [r['bic'] for r in results.values()],
        'Train_RMSE': [train_metrics.get(m, {}).get('RMSE', np.nan) for m in results.keys()],
        'Test_RMSE': [test_metrics.get(m, {}).get('RMSE', np.nan) for m in results.keys()],
        'Train_R2': [train_metrics.get(m, {}).get('R2', np.nan) for m in results.keys()],
        'Test_R2': [test_metrics.get(m, {}).get('R2', np.nan) for m in results.keys()]
    })
    
    print(comparison_df.to_string(index=False))
    
    # Save results
    output_dir = Path('./results')
    output_dir.mkdir(exist_ok=True)
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    print(f"\n✓ Results saved to {output_dir}/model_comparison.csv")
    
    return {
        'results': results,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'comparison_df': comparison_df,
        'data': {
            'y_train': y_train,
            'y_test': y_test,
            'dates_train': dates_train,
            'dates_test': dates_test,
            'target_col': target_col
        }
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*70)
    print("TIME SERIES FORECASTING - videos_log_v2")
    print("="*70)
    
    # Fetch data from Neon
    df_videos = fetch_videos_from_neon("videos_log_v2")
    
    # Try hourly aggregation first
    hourly_data = prepare_timeseries_data(df_videos, aggregation="hourly")
    
    # If not enough hourly data, try daily
    if len(hourly_data) < 10:
        print("\n⚠️  Not enough hourly data points. Switching to daily aggregation...")
        daily_data = prepare_timeseries_data(df_videos, aggregation="daily")
        time_series_data = daily_data
    else:
        time_series_data = hourly_data
    
    # Run forecasting models
    print("\n" + "="*70)
    print("RUNNING FORECASTING MODELS")
    print("="*70)
    
    # Analyze total views
    print("\n📊 TARGET: Total Views")
    results_views = run_model_comparison(time_series_data, target_col='total_views')
    
    # Analyze total likes (optional)
    print("\n\n📊 TARGET: Total Likes")
    results_likes = run_model_comparison(time_series_data, target_col='total_likes')
    
    # Analyze engagement rate (optional)
    print("\n\n📊 TARGET: Average Engagement Rate")
    results_engagement = run_model_comparison(time_series_data, target_col='avg_engagement_rate')
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print("\nResults saved to: ./results/model_comparison.csv")


if __name__ == "__main__":
    main()