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
    """Fetch videos_log_v2 table from Neon database or fall back to local parquet"""
    load_dotenv()
    
    db_uri = os.getenv("NEON_DATABASE_URL")
    
    # Fallback to hardcoded connection string if no .env file
    if not db_uri:
        print("[FETCH] No .env file found, using hardcoded connection string...")
        db_uri = "postgresql://neondb_owner:npg_c38OpMvawKVg@ep-withered-sky-aeqx6en8-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    
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
        print(f"[FETCH] ✗ Polars connection error: {str(e)[:100]}")
        print("[FETCH] Trying pandas + SQLAlchemy as backup...")
        
        try:
            return fetch_with_pandas(table_name, db_uri)
        except Exception as e2:
            print(f"[FETCH] ✗ SQLAlchemy connection error: {str(e2)[:100]}")
            print("\n" + "="*70)
            print("DATABASE CONNECTION FAILED")
            print("="*70)
            print("Possible solutions:")
            print("1. Install required packages: pip install psycopg2-binary sqlalchemy")
            print("2. Check network/firewall settings")
            print("3. Verify database is accessible")
            print("4. Use local parquet file instead (see below)")
            print("="*70)
            print("\n[FETCH] Falling back to local parquet file...")
            return fetch_from_local_parquet()


def fetch_with_pandas(table_name: str, db_uri: str) -> pl.DataFrame:
    """Backup method using pandas + SQLAlchemy"""
    from sqlalchemy import create_engine
    
    engine = create_engine(db_uri)
    query = f"SELECT * FROM {table_name} ORDER BY ingestion_timestamp"
    
    df_pandas = pd.read_sql(query, engine)
    print(f"[FETCH] ✓ Fetched {len(df_pandas)} rows using pandas")
    
    # Convert to Polars
    return pl.from_pandas(df_pandas)


def fetch_from_local_parquet() -> pl.DataFrame:
    """Fallback to local parquet file if database connection fails"""
    possible_paths = [
        "silver_data/videos.parquet",
        "data/videos.parquet",
        "videos.parquet",
        "../silver_data/videos.parquet"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"[FETCH] ✓ Found local parquet: {path}")
            df = pl.read_parquet(path)
            
            # Rename columns if needed to match database schema
            cols = set(df.columns)
            if 'ingest_timestamp' in cols and 'ingestion_timestamp' not in cols:
                df = df.rename({'ingest_timestamp': 'ingestion_timestamp'})
            
            # Add views_per_day if missing
            if 'views_per_day' not in df.columns:
                df = df.with_columns([pl.lit(0).alias('views_per_day')])
            
            print(f"[FETCH] ✓ Loaded {len(df)} rows from local parquet")
            return df
    
    raise FileNotFoundError(
        "\n" + "="*70 + "\n"
        "NO DATA SOURCE AVAILABLE\n"
        "="*70 + "\n"
        "Neither Neon database connection nor local parquet file found.\n\n"
        "Please either:\n"
        "1. Fix your .env with correct NEON_DATABASE_URL, or\n"
        "2. Place a parquet file at one of these locations:\n" +
        "\n".join(f"   - {p}" for p in possible_paths) + "\n" +
        "="*70
    )


# ============================================================================
# TIME SERIES DATA PREPARATION
# ============================================================================

def prepare_timeseries_data(df: pl.DataFrame, aggregation: str = "minutely") -> pd.DataFrame:
    """
    Convert videos_log_v2 data into aggregated time series.
    
    Args:
        df: Polars DataFrame from Neon
        aggregation: "minutely", "hourly" or "daily"
        
    Returns:
        Pandas DataFrame with aggregated metrics
    """
    print(f"\n[PREPARE] Original data: {len(df)} video records")
    print(f"[PREPARE] Aggregation: {aggregation}")
    
    # Convert to pandas for easier time series manipulation
    df_pd = df.to_pandas()
    
    # Check which timestamp to use
    timestamp_col = 'ingestion_timestamp'
    
    # Parse ingestion timestamps
    df_pd['ingestion_timestamp'] = pd.to_datetime(df_pd['ingestion_timestamp'])
    
    # Check time spread of ingestion_timestamp
    ingestion_unique = df_pd['ingestion_timestamp'].dt.floor('h').nunique()
    
    if ingestion_unique < 5:
        print(f"[PREPARE] ⚠️  ingestion_timestamp has only {ingestion_unique} unique hours")
        print(f"[PREPARE] This indicates batch loading - switching to published_at")
        
        # Use published_at instead
        if 'published_at' in df_pd.columns:
            timestamp_col = 'published_at'
            df_pd['published_at'] = pd.to_datetime(df_pd['published_at'], errors='coerce')
            df_pd = df_pd.dropna(subset=['published_at'])
            print(f"[PREPARE] Using published_at timestamps (when videos were published)")
        else:
            print(f"[PREPARE] ✗ No published_at column found!")
            print(f"[PREPARE] Cannot perform time series analysis with batch-loaded data")
            timestamp_col = 'ingestion_timestamp'
    
    # Create time bins based on chosen timestamp
    if aggregation == "minutely":
        df_pd['time_bin'] = df_pd[timestamp_col].dt.floor('T')
    elif aggregation == "hourly":
        df_pd['time_bin'] = df_pd[timestamp_col].dt.floor('h')
    else:  # daily
        df_pd['time_bin'] = df_pd[timestamp_col].dt.floor('D')
    
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
            pred_cov = self.filtered_covs[-1] + self.state_variance
            
            # Simple exponential smoothing forecast
            last_state = self.filtered_means[-1]
            predictions[0] = last_state
            
            for i in range(1, n_test):
                predictions[i] = predictions[i-1]  # Random walk forecast
            
            return self.smoothed_means, predictions
        
        return self.smoothed_means, None


# ============================================================================
# MODEL FITTING FUNCTIONS
# ============================================================================

def fit_arima_model(y_train, y_test, order=None):
    """Fit ARIMA model with automatic order selection if order=None"""
    if order is None:
        # Try multiple orders and select best AIC
        orders_to_try = [
            (0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1),
            (0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1),
            (2, 0, 0), (0, 0, 2), (2, 1, 0), (0, 1, 2),
            (2, 1, 1), (1, 1, 2), (2, 0, 1), (1, 0, 2)
        ]
        best_aic = np.inf
        best_order = (1, 1, 1)
        best_model = None
        
        for test_order in orders_to_try:
            try:
                model = ARIMA(y_train, order=test_order)
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = test_order
                    best_model = model_fit
            except:
                continue
        
        if best_model is None:
            print("  Could not fit any ARIMA model")
            return None
        
        print(f"  Best ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        order = best_order
        model_fit = best_model
    else:
        try:
            model = ARIMA(y_train, order=order)
            model_fit = model.fit()
        except Exception as e:
            print(f"  ARIMA fitting error: {e}")
            return None
    
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
        print(f"  AR(1) fitting error: {e}")
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
        print(f"  DLM fitting error: {e}")
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
# VISUALIZATION FUNCTION
# ============================================================================

def create_visualization(results_dict, output_dir='outputs'):
    """Create visualization of fitted models and forecasts"""
    if not results_dict:
        print("\n[VIZ] No results to visualize")
        return
    
    data = results_dict['data']
    results = results_dict['results']
    target_col = data['target_col']
    
    y_train = data['y_train']
    y_test = data['y_test']
    dates_train = data['dates_train']
    dates_test = data['dates_test']
    
    # Combine for full series plotting
    y_full = np.concatenate([y_train, y_test])
    dates_full = np.concatenate([dates_train, dates_test])
    
    print(f"\n[VIZ] Creating visualization for {target_col}...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot full observed series
    ax.plot(pd.to_datetime(dates_full), y_full, 
            label="Observed", color="#1f77b4", linewidth=2, alpha=0.7)
    
    # Plot each model's fitted and forecast
    colors = {'ARIMA': '#ff7f0e', 'AR1': '#2ca02c', 'DLM': '#d62728'}
    
    for model_name, result in results.items():
        color = colors.get(model_name, '#000000')
        
        # Get fitted values
        fitted = np.asarray(result['fitted'])
        if len(fitted) > 0:
            # Plot fitted line
            model_label = model_name
            if model_name == 'ARIMA' and 'params' in result:
                model_label = f"ARIMA{result['params']}"
            
            ax.plot(pd.to_datetime(dates_train[-len(fitted):]), fitted,
                   label=f"{model_label} fitted", linestyle="--", 
                   color=color, linewidth=2)
        
        # Get forecast values
        forecast = result.get('forecast')
        if forecast is not None:
            forecast = np.asarray(forecast)
            if len(forecast) > 0:
                ax.plot(pd.to_datetime(dates_test[:len(forecast)]), forecast,
                       label=f"{model_label} forecast", linestyle=":", 
                       color=color, linewidth=2)
    
    # Add vertical line at train/test split
    split_date = pd.to_datetime(dates_train[-1])
    ax.axvline(split_date, color='black', linestyle='--', 
               alpha=0.3, linewidth=1.5, label='Train/Test Split')
    
    # Formatting
    title_map = {
        'total_views': 'Total Views',
        'total_likes': 'Total Likes',
        'total_comments': 'Total Comments',
        'avg_engagement_rate': 'Average Engagement Rate',
        'avg_views_per_day': 'Average Views Per Day',
        'num_videos': 'Number of Videos'
    }
    
    title = title_map.get(target_col, target_col.replace('_', ' ').title())
    ax.set_title(f"{title} — Observed, Fitted, and Forecasts", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(title, fontsize=12)
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filename = f"{target_col}_forecast.png"
    save_path = output_path / filename
    
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[VIZ] ✓ Saved: {save_path}")
    
    plt.close()


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
    
    # Fit ARIMA with automatic order selection
    print("\nFitting ARIMA (auto order selection)...")
    arima_result = fit_arima_model(y_train, y_test, order=None)
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
    
    # Check if data has time spread in ingestion_timestamp
    df_pd_check = df_videos.to_pandas()
    df_pd_check['ingestion_timestamp'] = pd.to_datetime(df_pd_check['ingestion_timestamp'])
    ingestion_hours = df_pd_check['ingestion_timestamp'].dt.floor('h').nunique()
    
    # If ingestion timestamps are concentrated, start with daily aggregation
    # Otherwise, try minutely first
    if ingestion_hours < 5:
        print("\n[INFO] Detected batch ingestion - starting with daily aggregation")
        print("[INFO] (Will use published_at timestamps instead of ingestion_timestamp)")
        time_series_data = prepare_timeseries_data(df_videos, aggregation="daily")
    else:
        # Try minutely aggregation first
        minutely_data = prepare_timeseries_data(df_videos, aggregation="minutely")

        # If not enough minutely data, try hourly then daily
        if len(minutely_data) < 10:
            print("\n⚠️  Not enough minutely data points. Switching to hourly aggregation...")
            hourly_data = prepare_timeseries_data(df_videos, aggregation="hourly")
            if len(hourly_data) < 10:
                print("\n⚠️  Not enough hourly data points. Switching to daily aggregation...")
                daily_data = prepare_timeseries_data(df_videos, aggregation="daily")
                time_series_data = daily_data
            else:
                time_series_data = hourly_data
        else:
            time_series_data = minutely_data
    
    # Check if we have enough data
    if len(time_series_data) < 10:
        print("\n" + "="*70)
        print("❌ INSUFFICIENT DATA FOR TIME SERIES ANALYSIS")
        print("="*70)
        print(f"Found only {len(time_series_data)} time points (need at least 10)")
        print("\nPossible issues:")
        print("1. Data was loaded in a single batch (all same ingestion_timestamp)")
        print("2. Not enough videos with published_at dates")
        print("3. Published dates are too concentrated")
        print("\nSuggestions:")
        print("- Check your data source has videos published over time")
        print("- Verify published_at column exists and has valid dates")
        print("- Load more historical video data")
        print("="*70)
        return None
    
    # Run forecasting models
    print("\n" + "="*70)
    print("RUNNING FORECASTING MODELS")
    print("="*70)
    
    # Store results for visualization
    all_results = {}
    
    # Analyze total views
    print("\n📊 TARGET: Total Views")
    results_views = run_model_comparison(time_series_data, target_col='total_views')
    if results_views:
        all_results['total_views'] = results_views
    
    # Analyze total likes
    print("\n\n📊 TARGET: Total Likes")
    results_likes = run_model_comparison(time_series_data, target_col='total_likes')
    if results_likes:
        all_results['total_likes'] = results_likes
    
    # Analyze engagement rate
    print("\n\n📊 TARGET: Average Engagement Rate")
    results_engagement = run_model_comparison(time_series_data, target_col='avg_engagement_rate')
    if results_engagement:
        all_results['avg_engagement_rate'] = results_engagement
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    for target, results in all_results.items():
        create_visualization(results, output_dir='outputs')
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print("  - ./results/model_comparison.csv")
    print("  - ./outputs/*_forecast.png")
    
    return all_results


if __name__ == "__main__":
    main()