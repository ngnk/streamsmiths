"""
Time Series Forecasting: ARIMA vs DLM vs Neural Network
Using RAW NON-AGGREGATED data (individual video observations)
"""

import os
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from dotenv import load_dotenv
from sqlalchemy import create_engine
from datetime import datetime
from math import ceil

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

try:
    from pmdarima import auto_arima
except Exception:
    auto_arima = None

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def connect_to_neon(table_name: str = "videos_log_v3") -> pl.DataFrame:
    """Connect to Neon database and fetch video data."""
    load_dotenv()
    
    db_uri = os.getenv("NEON_DATABASE_URL")
    if not db_uri:
        raise RuntimeError("NEON_DATABASE_URL not set.")
    
    db_uri = db_uri.strip()
    if db_uri.startswith("psql "):
        db_uri = db_uri[5:].strip()
    db_uri = db_uri.strip("'\"")
    
    print(f"[DATABASE] Connecting to Neon...")
    print(f"[DATABASE] Fetching table: {table_name}")
    
    try:
        engine = create_engine(db_uri, pool_pre_ping=True)
        query = f"SELECT * FROM {table_name}"
        df_pandas = pd.read_sql(query, engine)
        df = pl.from_pandas(df_pandas)
        print(f"[DATABASE] ✓ Fetched {len(df):,} rows from {table_name}")
        return df
    except Exception as e:
        print(f"[DATABASE] ✗ Connection error: {e}")
        raise


# ============================================================================
# DATA PREPARATION - RAW NON-AGGREGATED DATA
# ============================================================================

def prepare_timeseries_raw(df: pl.DataFrame, target_metric: str = "view_count") -> pd.DataFrame:
    """Use RAW individual video observations - NO AGGREGATION"""
    df_pd = df.to_pandas()
    
    # Determine timestamp column
    if 'published_at' in df_pd.columns:
        df_pd['published_at'] = pd.to_datetime(df_pd['published_at'], errors='coerce')
        df_pd = df_pd.dropna(subset=['published_at'])
        timestamp_col = 'published_at'
        print('[PREPARE] Using `published_at` as the timestamp column')
    elif 'ingestion_timestamp' in df_pd.columns:
        df_pd['ingestion_timestamp'] = pd.to_datetime(df_pd['ingestion_timestamp'], errors='coerce')
        df_pd = df_pd.dropna(subset=['ingestion_timestamp'])
        timestamp_col = 'ingestion_timestamp'
        print('[PREPARE] Using `ingestion_timestamp` as the timestamp column')
    else:
        raise RuntimeError('No usable timestamp column found')

    print('[PREPARE] Using RAW NON-AGGREGATED data (individual video observations)')
    
    # For engagement rate, we need to calculate it per observation
    if target_metric == 'engagement_rate':
        df_pd['engagement_rate'] = (
            (df_pd['like_count'] + df_pd['comment_count']) / 
            df_pd['view_count'].replace(0, 1)
        ) * 100
    
    # Select timestamp and target metric
    result = df_pd[[timestamp_col, target_metric]].copy()
    result.columns = ['timestamp', target_metric]
    
    # Sort by timestamp
    result = result.sort_values('timestamp').reset_index(drop=True)
    
    # Remove any NaN values in target metric
    result = result.dropna(subset=[target_metric])
    
    print(f"[PREPARE] Time series length: {len(result)} individual observations")
    print(f"[PREPARE] Date range: {result['timestamp'].min()} to {result['timestamp'].max()}")
    
    return result


# ============================================================================
# DYNAMIC LINEAR MODEL (Kalman Filter)
# ============================================================================

class DynamicLinearModel:
    """Dynamic Linear Model using Kalman Filter."""
    
    def __init__(self, obs_variance=None, state_variance=None):
        self.obs_variance = obs_variance
        self.state_variance = state_variance
        self.filtered_means = None
        self.filtered_vars = None
        self.smoothed_means = None
        self.smoothed_vars = None
        self._trend = 0
    
    def _initialize_variances(self, y):
        if self.obs_variance is None:
            self.obs_variance = np.var(y) * 0.1
        if self.state_variance is None:
            diffs = np.diff(y)
            self.state_variance = np.var(diffs) * 0.5 if len(diffs) > 0 else 1.0
        if len(y) > 10:
            self._trend = np.mean(np.diff(y[-50:])) if len(y) > 50 else np.mean(np.diff(y))
    
    def forward_filter(self, y):
        n = len(y)
        self.filtered_means = np.zeros(n)
        self.filtered_vars = np.zeros(n)
        
        m0, C0 = y[0], self.state_variance * 10
        
        for t in range(n):
            if t == 0:
                pred_mean, pred_var = m0, C0
            else:
                pred_mean = self.filtered_means[t-1]
                pred_var = self.filtered_vars[t-1] + self.state_variance
            
            forecast_var = pred_var + self.obs_variance
            kalman_gain = pred_var / forecast_var
            
            self.filtered_means[t] = pred_mean + kalman_gain * (y[t] - pred_mean)
            self.filtered_vars[t] = (1 - kalman_gain) * pred_var
        
        return self.filtered_means
    
    def backward_smooth(self):
        n = len(self.filtered_means)
        self.smoothed_means = np.zeros(n)
        self.smoothed_vars = np.zeros(n)
        
        self.smoothed_means[-1] = self.filtered_means[-1]
        self.smoothed_vars[-1] = self.filtered_vars[-1]
        
        for t in range(n-2, -1, -1):
            J_t = self.filtered_vars[t] / (self.filtered_vars[t] + self.state_variance)
            self.smoothed_means[t] = self.filtered_means[t] + J_t * (self.smoothed_means[t+1] - self.filtered_means[t])
            self.smoothed_vars[t] = self.filtered_vars[t] + J_t**2 * (self.smoothed_vars[t+1] - self.filtered_vars[t] - self.state_variance)
        
        return self.smoothed_means
    
    def fit(self, y):
        self._initialize_variances(y)
        self.forward_filter(y)
        self.backward_smooth()
        return self
    
    def forecast(self, steps=14):
        forecasts = np.zeros(steps)
        forecast_vars = np.zeros(steps)
        
        last_mean = self.smoothed_means[-1]
        last_var = self.smoothed_vars[-1]
        
        for h in range(steps):
            forecasts[h] = last_mean + self._trend * (h + 1)
            forecast_vars[h] = last_var + (h + 1) * self.state_variance + self.obs_variance
        
        return forecasts, forecast_vars


# ============================================================================
# NEURAL NETWORK FORECASTER
# ============================================================================

class SimpleNNForecaster:
    """Simple NN forecaster that predicts the next value directly"""
    
    def __init__(self, lags=24):
        self.lags = lags
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.last_mean = 0
        self.resid_std = 0
    
    def _build_features(self, y):
        """Build lag features"""
        X, targets = [], []
        for i in range(self.lags, len(y)):
            X.append(y[i-self.lags:i])
            targets.append(y[i])
        return np.array(X), np.array(targets)
    
    def fit(self, y):
        y = np.array(y).ravel()
        if len(y) < self.lags + 10:
            return None
        
        self.last_mean = np.mean(y[-100:]) if len(y) > 100 else np.mean(y)
        
        X, targets = self._build_features(y)
        if len(X) < 10:
            return None
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(targets.reshape(-1, 1)).ravel()
        
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=1000, random_state=42,
            early_stopping=True, validation_fraction=0.15,
            alpha=0.001, learning_rate='adaptive'
        )
        self.model.fit(X_scaled, y_scaled)
        
        preds_scaled = self.model.predict(X_scaled)
        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
        self.resid_std = np.std(targets - preds)
        
        fitted = np.full(len(y), np.nan)
        fitted[self.lags:] = preds
        return fitted
    
    def forecast(self, y, steps):
        if self.model is None:
            return None, None
        
        y = np.array(y).ravel()
        forecasts = []
        window = list(y[-self.lags:])
        
        for h in range(steps):
            X_in = np.array(window[-self.lags:]).reshape(1, -1)
            X_scaled = self.scaler_X.transform(X_in)
            pred_scaled = self.model.predict(X_scaled)[0]
            pred = self.scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            
            pred = max(pred, self.last_mean * 0.5)
            pred = min(pred, self.last_mean * 2.0)
            
            window.append(pred)
            forecasts.append(pred)
        
        forecasts = np.array(forecasts)
        forecast_std = np.array([self.resid_std * np.sqrt(1 + 0.01 * h) for h in range(steps)])
        
        return forecasts, forecast_std


def fit_nn_model(y_train, y_test, steps_ahead=24, lags=24):
    """Fit NN model"""
    print("\n[NN] Fitting MLPRegressor...")
    
    try:
        lags = min(lags, len(y_train) // 4)
        model = SimpleNNForecaster(lags=lags)
        fitted = model.fit(y_train)
        
        if fitted is None:
            print("[NN] Not enough training data")
            return None
        
        forecasts, forecast_std = model.forecast(y_train, steps_ahead)
        
        if forecasts is None:
            return None
        
        valid_mask = ~np.isnan(fitted)
        train_rmse = np.sqrt(mean_squared_error(y_train[valid_mask], fitted[valid_mask])) if np.sum(valid_mask) > 0 else np.nan
        
        print(f"[NN] Train RMSE: {train_rmse:,.0f}")
        
        return {
            'model': model,
            'fitted_train': fitted,
            'forecast': forecasts,
            'forecast_std': forecast_std,
            'metrics': {'train_rmse': train_rmse}
        }
    except Exception as e:
        print(f"[NN] ✗ Fitting error: {e}")
        return None


# ============================================================================
# MODEL FITTING
# ============================================================================

def fit_arima(y_train, y_test, order=(1, 1, 1)):
    """Fit ARIMA model"""
    print(f"\n[ARIMA] Fitting ARIMA{order}...")
    
    try:
        if auto_arima is not None:
            aa = auto_arima(y_train, seasonal=False, stepwise=True, 
                           error_action='ignore', suppress_warnings=True)
            order = aa.order
            print(f"[ARIMA] auto_arima selected order={order}")
        
        model = ARIMA(y_train, order=order)
        model_fit = model.fit()
        fitted = np.array(model_fit.fittedvalues)
        
        # Pad fitted to match y_train length
        if len(fitted) < len(y_train):
            fitted_full = np.full(len(y_train), np.nan)
            fitted_full[-len(fitted):] = fitted
            fitted = fitted_full
        
        forecast_obj = model_fit.get_forecast(steps=len(y_test))
        forecast = np.array(forecast_obj.predicted_mean)
        forecast_std = np.array(forecast_obj.se_mean)
        
        train_rmse = np.sqrt(np.nanmean((y_train - fitted)**2))
        test_rmse = np.sqrt(mean_squared_error(y_test, forecast))
        
        print(f"[ARIMA] Train RMSE: {train_rmse:,.0f}")
        print(f"[ARIMA] Test RMSE: {test_rmse:,.0f}")
        
        return {
            'model': model_fit,
            'fitted': fitted,
            'forecast': forecast,
            'forecast_std': forecast_std,
            'order': order,
            'metrics': {'train_rmse': train_rmse, 'test_rmse': test_rmse, 'aic': model_fit.aic}
        }
    except Exception as e:
        print(f"[ARIMA] ✗ Fitting error: {e}")
        return None


def fit_dlm(y_train, y_test):
    """Fit Dynamic Linear Model"""
    print(f"\n[DLM] Fitting Dynamic Linear Model...")
    
    try:
        model = DynamicLinearModel()
        model.fit(y_train)
        
        fitted = np.array(model.smoothed_means)
        forecast, forecast_vars = model.forecast(steps=len(y_test))
        forecast_std = np.sqrt(forecast_vars)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, fitted))
        test_rmse = np.sqrt(mean_squared_error(y_test, forecast))
        
        print(f"[DLM] Train RMSE: {train_rmse:,.0f}")
        print(f"[DLM] Test RMSE: {test_rmse:,.0f}")
        
        return {
            'model': model,
            'fitted': fitted,
            'forecast': forecast,
            'forecast_std': forecast_std,
            'metrics': {'train_rmse': train_rmse, 'test_rmse': test_rmse}
        }
    except Exception as e:
        print(f"[DLM] ✗ Fitting error: {e}")
        return None


# ============================================================================
# VISUALIZATION
# ============================================================================

def format_large_number(x, pos=None):
    """Format large numbers for y-axis"""
    if abs(x) >= 1e9:
        return f'{x/1e9:.1f}B'
    elif abs(x) >= 1e6:
        return f'{x/1e6:.0f}M'
    elif abs(x) >= 1e3:
        return f'{x/1e3:.0f}K'
    else:
        return f'{x:.0f}'


def create_faceted_forecast_plot(
    timestamps, y_full, train_size,
    arima_result, dlm_result, nn_result,
    ext_forecast_timestamps,
    arima_ext=None, dlm_ext=None, nn_ext=None,
    target_metric=None,
    output_path='outputs/forecast_comparison_raw.png'
):
    """Create faceted visualization with LINEAR scale and visible fan plots."""
    print(f"\n[PLOT] Creating faceted visualization...")
    
    timestamps_train = timestamps[:train_size]
    timestamps_test = timestamps[train_size:]
    y_train = y_full[:train_size]
    y_test = y_full[train_size:]
    
    colors = {
        'observed': '#2E86AB',
        'observed_test': '#A23B72',
        'fitted': '#F18F01',
        'forecast': '#C73E1D',
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(18, 15))
    plt.subplots_adjust(hspace=0.25)
    
    def plot_panel(ax, model_name, result, ext_result):
        # Get data range for y-limits
        recent_y = y_full[-500:] if len(y_full) > 500 else y_full
        y_median = np.median(recent_y)
        y_iqr = np.percentile(recent_y, 75) - np.percentile(recent_y, 25)
        
        y_lower = max(0, y_median - 3 * y_iqr)
        y_upper = y_median + 3 * y_iqr
        
        if ext_result is not None:
            f_ext, f_ext_std = ext_result
            forecast_upper = np.max(f_ext + 2 * f_ext_std)
            forecast_lower = np.min(f_ext - 2 * f_ext_std)
            y_upper = max(y_upper, forecast_upper)
            y_lower = max(0, min(y_lower, forecast_lower))
        
        # Plot observed data
        ax.plot(timestamps_train, y_train, color=colors['observed'], 
                linewidth=0.8, alpha=0.6, label='Observed (Train)')
        ax.plot(timestamps_test, y_test, color=colors['observed_test'],
                linewidth=0.8, alpha=0.6, label='Observed (Test)')
        
        # Plot fitted values
        if result is not None:
            fitted = result.get('fitted')
            if fitted is None:
                fitted = result.get('fitted_train')
            if fitted is not None:
                fitted = np.array(fitted)
                valid = ~np.isnan(fitted)
                if np.any(valid) and len(fitted) == len(timestamps_train):
                    ax.plot(timestamps_train[valid], fitted[valid],
                           color=colors['fitted'], linewidth=1.5, linestyle='--',
                           alpha=0.8, label=f'{model_name} Fitted')
        
        # Plot forecast with fan plot
        if ext_result is not None:
            f_ext, f_ext_std = ext_result
            f_ext = np.array(f_ext)
            f_ext_std = np.array(f_ext_std)
            
            ax.plot(ext_forecast_timestamps, f_ext, color=colors['forecast'],
                   linewidth=2.5, label=f'{model_name} Forecast')
            
            # Fan plot
            for alpha, z, label in [(0.4, 0.67, '50%'), (0.25, 1.28, '80%'), (0.12, 1.96, '95%')]:
                lower = np.maximum(f_ext - z * f_ext_std, 0)
                upper = f_ext + z * f_ext_std
                ax.fill_between(ext_forecast_timestamps, lower, upper,
                               alpha=alpha, color=colors['forecast'], linewidth=0)
        
        # Vertical line at forecast start
        ax.axvline(timestamps[-1], color='gray', linestyle='--', alpha=0.7, linewidth=1)
        
        ax.set_ylim(y_lower, y_upper)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_number))
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel('View Count', fontsize=11)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
        ax.grid(alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Metrics box
        if result is not None and 'metrics' in result:
            m = result['metrics']
            txt = []
            if 'train_rmse' in m and not np.isnan(m.get('train_rmse', np.nan)):
                txt.append(f"Train RMSE: {format_large_number(m['train_rmse'])}")
            if 'test_rmse' in m and m.get('test_rmse') not in ['N/A', None]:
                txt.append(f"Test RMSE: {format_large_number(m['test_rmse'])}")
            if txt:
                ax.text(0.98, 0.98, '\n'.join(txt), transform=ax.transAxes,
                       va='top', ha='right', fontsize=9, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plot_panel(axes[0], 'ARIMA', arima_result, arima_ext)
    plot_panel(axes[1], 'DLM (Kalman Filter)', dlm_result, dlm_ext)
    plot_panel(axes[2], 'Neural Network', nn_result, nn_ext)
    
    axes[2].set_xlabel('Date', fontsize=12)
    
    # CI legend
    legend_elements = [
        Patch(facecolor=colors['forecast'], alpha=0.4, label='50% CI'),
        Patch(facecolor=colors['forecast'], alpha=0.25, label='80% CI'),
        Patch(facecolor=colors['forecast'], alpha=0.12, label='95% CI')
    ]
    fig.legend(handles=legend_elements, loc='upper right', 
              bbox_to_anchor=(0.99, 0.99), fontsize=10, title='Confidence Intervals')
    
    fig.suptitle(f'Time Series Forecast: {target_metric.replace("_", " ").title()} (RAW Non-Aggregated Data)',
                fontsize=16, fontweight='bold', y=1.01)
    
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[PLOT] ✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("TIME SERIES FORECASTING: ARIMA vs DLM vs Neural Network")
    print("Using RAW NON-AGGREGATED data (individual video observations)")
    print("=" * 80)
    
    # 1. Load data
    df = connect_to_neon(table_name="videos_log_v3")
    
    # 2. Prepare time series - RAW data
    target_metric = 'view_count'
    ts_data = prepare_timeseries_raw(df, target_metric=target_metric)
    
    if len(ts_data) < 30:
        print(f"\n[ERROR] Insufficient data: only {len(ts_data)} observations")
        return
    
    # 3. Train/test split
    test_size = min(100, len(ts_data) // 5)  # Larger test set for raw data
    train_size = len(ts_data) - test_size
    
    timestamps = ts_data['timestamp'].values
    y_full = ts_data[target_metric].values
    
    y_train = y_full[:train_size]
    y_test = y_full[train_size:]
    
    print(f"\n[SPLIT] Train: {train_size}, Test: {test_size}")
    print(f"[SPLIT] Data ends: {timestamps[-1]}")
    
    # 4. Fit models on train data
    arima_result = fit_arima(y_train, y_test)
    dlm_result = fit_dlm(y_train, y_test)
    nn_result = fit_nn_model(y_train, y_test, steps_ahead=len(y_test), lags=24)
    
    # 5. Extended forecasts
    last_dt = pd.to_datetime(timestamps[-1])
    target_date = pd.Timestamp('2026-01-31')
    
    # For raw data, we forecast number of observations not hours
    # Estimate observation rate
    time_range = (pd.to_datetime(timestamps[-1]) - pd.to_datetime(timestamps[0])).total_seconds() / 3600
    obs_per_hour = len(timestamps) / time_range if time_range > 0 else 1
    
    hours_to_forecast = int((target_date - last_dt).total_seconds() / 3600)
    hours_to_forecast = max(hours_to_forecast, 24 * 60)
    forecast_steps = int(hours_to_forecast * obs_per_hour)  # Estimate number of future observations
    forecast_steps = min(forecast_steps, 10000)  # Cap at reasonable number
    
    print(f"\n[FORECAST] Extending forecasts for ~{forecast_steps} observations")
    print(f"[FORECAST] From {last_dt} to {target_date}")
    print(f"[FORECAST] Estimated observation rate: {obs_per_hour:.2f} obs/hour")
    
    # Generate timestamps for forecast - spread evenly
    time_delta_seconds = (target_date - last_dt).total_seconds()
    ext_forecast_timestamps = pd.date_range(start=last_dt, end=target_date, periods=forecast_steps + 1)[1:]
    
    # Refit on full data for extended forecasts
    arima_ext = None
    try:
        if auto_arima is not None:
            aa = auto_arima(y_full, seasonal=False, stepwise=True, 
                           error_action='ignore', suppress_warnings=True)
            ar_model = ARIMA(y_full, order=aa.order).fit()
        else:
            ar_model = ARIMA(y_full, order=(1, 1, 1)).fit()
        
        fo = ar_model.get_forecast(steps=forecast_steps)
        arima_ext = (np.array(fo.predicted_mean), np.array(fo.se_mean))
        print(f"[ARIMA] Extended forecast range: {arima_ext[0].min():,.0f} to {arima_ext[0].max():,.0f}")
    except Exception as e:
        print(f"[ARIMA] ✗ Extended forecast error: {e}")
    
    dlm_ext = None
    try:
        dlm_full = DynamicLinearModel()
        dlm_full.fit(y_full)
        f_ext, f_vars = dlm_full.forecast(steps=forecast_steps)
        dlm_ext = (np.array(f_ext), np.sqrt(np.array(f_vars)))
        print(f"[DLM] Extended forecast range: {dlm_ext[0].min():,.0f} to {dlm_ext[0].max():,.0f}")
    except Exception as e:
        print(f"[DLM] ✗ Extended forecast error: {e}")
    
    nn_ext = None
    try:
        nn = SimpleNNForecaster(lags=24)
        nn.fit(y_full)
        f_ext, f_std = nn.forecast(y_full, forecast_steps)
        if f_ext is not None:
            nn_ext = (np.array(f_ext), np.array(f_std))
            print(f"[NN] Extended forecast range: {nn_ext[0].min():,.0f} to {nn_ext[0].max():,.0f}")
    except Exception as e:
        print(f"[NN] ✗ Extended forecast error: {e}")
    
    # 6. Create visualization
    if any([arima_result, dlm_result, nn_result]):
        create_faceted_forecast_plot(
            timestamps=timestamps, y_full=y_full, train_size=train_size,
            arima_result=arima_result, dlm_result=dlm_result, nn_result=nn_result,
            ext_forecast_timestamps=ext_forecast_timestamps,
            arima_ext=arima_ext, dlm_ext=dlm_ext, nn_ext=nn_ext,
            target_metric=target_metric,
            output_path='outputs/forecast_comparison_raw.png'
        )
    
    # 7. Summary
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY (RAW DATA)")
    print("=" * 80)
    
    rows = []
    if arima_result:
        rows.append({'Model': 'ARIMA', 'Train_RMSE': arima_result['metrics']['train_rmse'],
                    'Test_RMSE': arima_result['metrics']['test_rmse']})
    if dlm_result:
        rows.append({'Model': 'DLM', 'Train_RMSE': dlm_result['metrics']['train_rmse'],
                    'Test_RMSE': dlm_result['metrics']['test_rmse']})
    if nn_result:
        rows.append({'Model': 'Neural Network', 'Train_RMSE': nn_result['metrics']['train_rmse'],
                    'Test_RMSE': 'N/A'})
    
    if rows:
        comparison = pd.DataFrame(rows)
        print(comparison.to_string(index=False))
        Path('outputs').mkdir(exist_ok=True)
        comparison.to_csv('outputs/model_comparison_raw.csv', index=False)
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()