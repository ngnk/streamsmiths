"""
Time Series Forecasting: ARIMA vs DLM
Connects to Neon database, fits models, creates faceted visualizations with fan plots
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
    """
    Connect to Neon database and fetch video data.
    Uses pandas + SQLAlchemy for compatibility with Python 3.13.
    """
    load_dotenv()
    
    db_uri = os.getenv("NEON_DATABASE_URL")
    if not db_uri:
        raise RuntimeError(
            "NEON_DATABASE_URL not set. Add it to .env file.\n"
            "Format: postgresql://user:pass@host/database?sslmode=require"
        )
    
    # Clean the connection string
    db_uri = db_uri.strip()
    if db_uri.startswith("psql "):
        db_uri = db_uri[5:].strip()
    db_uri = db_uri.strip("'\"")
    
    print(f"[DATABASE] Connecting to Neon...")
    print(f"[DATABASE] Fetching table: {table_name}")
    
    try:
        # Use pandas + SQLAlchemy (more compatible with Python 3.13)
        from sqlalchemy import create_engine
        
        engine = create_engine(db_uri, pool_pre_ping=True)
        # Do not order by ingestion_timestamp here — prefer ordering/sorting after
        # we choose the correct timestamp column (published_at vs ingestion_timestamp).
        query = f"SELECT * FROM {table_name}"
        
        # Read with pandas then convert to polars
        df_pandas = pd.read_sql(query, engine)
        df = pl.from_pandas(df_pandas)
        
        print(f"[DATABASE] ✓ Fetched {len(df):,} rows from {table_name}")
        return df
    except Exception as e:
        print(f"[DATABASE] ✗ Connection error: {e}")
        raise


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_timeseries(df: pl.DataFrame, target_metric: str = "view_count") -> pd.DataFrame:
    """
    Aggregate data into hourly time series.
    
    Parameters:
    -----------
    df : pl.DataFrame
        Raw data from database
    target_metric : str
        Metric to forecast (e.g., 'view_count', 'like_count', 'engagement_rate')
    
    Returns:
    --------
    pd.DataFrame with columns: ['timestamp', target_metric]
    
    """

    df_pd = df.to_pandas()
    
    # Determine which timestamp to use. Prefer `published_at` (video publish time)
    # over `ingestion_timestamp` (when the row was added to the DB).
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
        raise RuntimeError('No usable timestamp column found (published_at or ingestion_timestamp)')

    # Create hourly bins (we aggregate hourly by default)
    df_pd['time_bin'] = df_pd[timestamp_col].dt.floor('h')

    # --- Filter to dense period if present ---
    # Many rows may be concentrated in a recent year (e.g., 2025). If one
    # year contains the majority of records, restrict the analysis to that
    # year so models are not dominated by sparse historical data.
    try:
        df_pd['__year'] = pd.to_datetime(df_pd[timestamp_col]).dt.year
        year_counts = df_pd['__year'].value_counts().sort_values(ascending=False)
        top_year = int(year_counts.index[0])
        top_count = int(year_counts.iloc[0])
        total = len(df_pd)
        frac = top_count / total if total > 0 else 0

        # Heuristic: restrict if top year contains >= 50% of rows or at least 200 rows
        if frac >= 0.5 or top_count >= 200:
            print(f"[PREPARE] Filtering to densest year: {top_year} ({top_count}/{total} rows, {frac:.0%})")
            df_pd = df_pd[df_pd['__year'] == top_year].copy()
            # Recompute time_bin after filtering
            df_pd['time_bin'] = df_pd[timestamp_col].dt.floor('h')
        else:
            print(f"[PREPARE] No dominant year found (top {top_year}: {top_count}/{total} rows). Using full history.")

        # drop helper column
        df_pd.drop(columns=['__year'], inplace=True)
    except Exception:
        # If something goes wrong, continue with full dataset
        pass
    
    # Aggregate by hour
    if target_metric == 'engagement_rate':
        # Recalculate engagement rate at aggregated level
        agg_df = df_pd.groupby('time_bin').agg({
            'view_count': 'sum',
            'like_count': 'sum',
            'comment_count': 'sum'
        }).reset_index()
        
        agg_df['engagement_rate'] = (
            (agg_df['like_count'] + agg_df['comment_count']) / 
            agg_df['view_count'].replace(0, 1)
        ) * 100
        
        result = agg_df[['time_bin', 'engagement_rate']].copy()
        result.columns = ['timestamp', target_metric]
    else:
        agg_df = df_pd.groupby('time_bin').agg({
            target_metric: 'sum'
        }).reset_index()
        agg_df.columns = ['timestamp', target_metric]
        result = agg_df
    
    result = result.sort_values('timestamp').reset_index(drop=True)
    
    print(f"[PREPARE] Time series length: {len(result)} hours")
    print(f"[PREPARE] Date range: {result['timestamp'].min()} to {result['timestamp'].max()}")
    print(f"[PREPARE] {target_metric} range: {result[target_metric].min():.2f} to {result[target_metric].max():.2f}")
    
    return result


# ============================================================================
# DYNAMIC LINEAR MODEL (Kalman Filter)
# ============================================================================

class DynamicLinearModel:
    """
    Dynamic Linear Model using Kalman Filter with forward filtering
    and backward smoothing (Rauch-Tung-Striebel smoother).
    """
    
    def __init__(self, obs_variance=None, state_variance=None):
        self.obs_variance = obs_variance
        self.state_variance = state_variance
        
        # Storage for filtering results
        self.filtered_means = None
        self.filtered_vars = None
        self.predicted_means = None
        self.predicted_vars = None
        
        # Storage for smoothing results
        self.smoothed_means = None
        self.smoothed_vars = None
    
    def _initialize_variances(self, y):
        """Initialize observation and state variances if not provided"""
        if self.obs_variance is None:
            self.obs_variance = np.var(y) * 0.1
        if self.state_variance is None:
            diffs = np.diff(y)
            self.state_variance = np.var(diffs) * 0.5
    
    def forward_filter(self, y):
        """
        Kalman Filter: Forward pass through data
        
        Returns:
        --------
        filtered_means : array
            Filtered state estimates
        """
        n = len(y)
        
        # Initialize arrays
        self.filtered_means = np.zeros(n)
        self.filtered_vars = np.zeros(n)
        self.predicted_means = np.zeros(n)
        self.predicted_vars = np.zeros(n)
        
        # Initial state
        m0 = y[0]
        C0 = self.state_variance * 10
        
        for t in range(n):
            # Prediction step
            if t == 0:
                self.predicted_means[t] = m0
                self.predicted_vars[t] = C0
            else:
                self.predicted_means[t] = self.filtered_means[t-1]
                self.predicted_vars[t] = self.filtered_vars[t-1] + self.state_variance
            
            # Update step
            forecast_error = y[t] - self.predicted_means[t]
            forecast_var = self.predicted_vars[t] + self.obs_variance
            kalman_gain = self.predicted_vars[t] / forecast_var
            
            self.filtered_means[t] = self.predicted_means[t] + kalman_gain * forecast_error
            self.filtered_vars[t] = (1 - kalman_gain) * self.predicted_vars[t]
        
        return self.filtered_means
    
    def backward_smooth(self):
        """
        Rauch-Tung-Striebel Smoother: Backward pass through data
        
        Returns:
        --------
        smoothed_means : array
            Smoothed state estimates
        """
        n = len(self.filtered_means)
        
        self.smoothed_means = np.zeros(n)
        self.smoothed_vars = np.zeros(n)
        
        # Initialize at the end
        self.smoothed_means[-1] = self.filtered_means[-1]
        self.smoothed_vars[-1] = self.filtered_vars[-1]
        
        # Backward recursion
        for t in range(n-2, -1, -1):
            J_t = self.filtered_vars[t] / (self.filtered_vars[t] + self.state_variance)
            
            self.smoothed_means[t] = (
                self.filtered_means[t] + 
                J_t * (self.smoothed_means[t+1] - self.filtered_means[t])
            )
            
            self.smoothed_vars[t] = (
                self.filtered_vars[t] + 
                J_t**2 * (self.smoothed_vars[t+1] - self.filtered_vars[t] - self.state_variance)
            )
        
        return self.smoothed_means
    
    def fit(self, y):
        """Fit the model to data"""
        self._initialize_variances(y)
        self.forward_filter(y)
        self.backward_smooth()
        return self
    
    def forecast(self, steps=14):
        """
        Generate forecasts with uncertainty estimates
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast (default: 14 for 2 weeks)
        
        Returns:
        --------
        forecasts : array
            Point forecasts
        forecast_vars : array
            Forecast variances (for fan plot)
        """
        if self.smoothed_means is None:
            raise RuntimeError("Model must be fitted before forecasting")
        
        forecasts = np.zeros(steps)
        forecast_vars = np.zeros(steps)
        
        # Start from the last smoothed state
        last_mean = self.smoothed_means[-1]
        last_var = self.smoothed_vars[-1]
        
        for h in range(steps):
            # Random walk forecast
            forecasts[h] = last_mean
            forecast_vars[h] = last_var + (h + 1) * self.state_variance + self.obs_variance
        
        return forecasts, forecast_vars


# ============================================================================
# MODEL FITTING & EVALUATION
# ============================================================================

def fit_arima(y_train, y_test, order=(1, 1, 1)):
    """
    Fit ARIMA model
    
    Returns:
    --------
    dict with keys: 'model', 'fitted', 'forecast', 'forecast_std', 'metrics'
    """
    print(f"\n[ARIMA] Fitting ARIMA{order}...")
    
    try:
        # If pmdarima is available, try auto_arima to find a better order
        if auto_arima is not None:
            print("[ARIMA] Running auto_arima to select order (this may take a while)...")
            aa = auto_arima(y_train, seasonal=False, stepwise=True, error_action='ignore', suppress_warnings=True)
            sel_order = aa.order
            print(f"[ARIMA] auto_arima selected order={sel_order}")
            model = ARIMA(y_train, order=sel_order)
        else:
            model = ARIMA(y_train, order=order)

        model_fit = model.fit()

        # Fitted values
        fitted = model_fit.fittedvalues

        # Forecast with confidence intervals for full extended horizon (user will request steps)
        forecast_obj = model_fit.get_forecast(steps=len(y_test))
        forecast = forecast_obj.predicted_mean
        forecast_std = forecast_obj.se_mean
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(
            y_train[len(y_train)-len(fitted):], fitted
        ))
        test_rmse = np.sqrt(mean_squared_error(y_test, forecast))
        
        print(f"[ARIMA] Train RMSE: {train_rmse:.2f}")
        print(f"[ARIMA] Test RMSE: {test_rmse:.2f}")
        print(f"[ARIMA] AIC: {model_fit.aic:.2f}")
        
        return {
            'model': model_fit,
            'fitted': fitted,
            'forecast': forecast,
            'forecast_std': forecast_std,
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'aic': model_fit.aic
            }
        }
    except Exception as e:
        print(f"[ARIMA] ✗ Fitting error: {e}")
        return None


def fit_dlm(y_train, y_test):
    """
    Fit Dynamic Linear Model
    
    Returns:
    --------
    dict with keys: 'model', 'fitted', 'forecast', 'forecast_std', 'metrics'
    """
    print(f"\n[DLM] Fitting Dynamic Linear Model...")
    
    try:
        model = DynamicLinearModel()
        model.fit(y_train)
        
        # Fitted values (smoothed)
        fitted = model.smoothed_means
        
        # Forecast (for test length; extended forecast will be requested separately)
        forecast, forecast_vars = model.forecast(steps=len(y_test))
        forecast_std = np.sqrt(forecast_vars)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, fitted))
        test_rmse = np.sqrt(mean_squared_error(y_test, forecast))
        
        print(f"[DLM] Train RMSE: {train_rmse:.2f}")
        print(f"[DLM] Test RMSE: {test_rmse:.2f}")
        
        return {
            'model': model,
            'fitted': fitted,
            'forecast': forecast,
            'forecast_std': forecast_std,
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse
            }
        }
    except Exception as e:
        print(f"[DLM] ✗ Fitting error: {e}")
        return None


def fit_dlm_extended(model_obj, steps):
    """
    Given a fitted DynamicLinearModel instance, produce extended forecasts and std.
    """
    forecast, forecast_vars = model_obj.forecast(steps=steps)
    return forecast, np.sqrt(forecast_vars)


def fit_nn_model(y_train, y_test, steps_ahead=24, lags=24):
    """
    Fit a simple MLP on lag features and produce recursive multi-step forecasts.
    Returns a dict with keys: 'model', 'fitted_train', 'forecast', 'forecast_std'.
    If y_test is empty, 'forecast' will be the extended forecast of length steps_ahead.
    """
    print("\n[NN] Fitting MLPRegressor with lag features...")
    try:
        # Build lag matrix
        X = []
        y = []
        for i in range(lags, len(y_train)):
            X.append(y_train[i-lags:i])
            y.append(y_train[i])
        if len(X) < 10:
            print("[NN] Not enough training examples for NN model")
            return None

        X = np.array(X)
        y = np.array(y)

        # Scale features and target
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        Xs = scaler_X.fit_transform(X)
        ys = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=0, early_stopping=True)
        model.fit(Xs, ys)

        # Compute residual std on training set (in original scale)
        preds_train_scaled = model.predict(Xs)
        preds_train = scaler_y.inverse_transform(preds_train_scaled.reshape(-1, 1)).ravel()
        resid_std = np.sqrt(np.mean((preds_train - y)**2))

        # Fitted values aligned to the end of training series (original scale)
        fitted_full = np.full(len(y_train), np.nan)
        fitted_full[lags:len(y_train)] = preds_train

        # If y_test length > 0, produce forecast of that length for evaluation
        forecasts = None
        forecasts_std = None
        if steps_ahead > 0:
            # Recursive forecasting starting from last lags of y_train
            last_window = list(y_train[-lags:]) if len(y_train) >= lags else list(y_train[-len(y_train):])
            if len(last_window) < lags:
                pad = [np.mean(y_train)] * (lags - len(last_window))
                last_window = pad + last_window

            forecasts = []
            window = last_window.copy()
            for _ in range(steps_ahead):
                x_in = np.array(window[-lags:]).reshape(1, -1)
                x_in_s = scaler_X.transform(x_in)
                f_scaled = model.predict(x_in_s)[0]
                f = scaler_y.inverse_transform(np.array([[f_scaled]])).ravel()[0]
                forecasts.append(f)
                window.append(f)

            forecasts = np.array(forecasts)
            forecasts_std = np.full(len(forecasts), resid_std)

        return {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'fitted_train': fitted_full,
            'forecast': forecasts,
            'forecast_std': forecasts_std,
            'train_resid_std': resid_std
        }
    except Exception as e:
        print(f"[NN] ✗ Fitting error: {e}")
        return None


# ============================================================================
# VISUALIZATION WITH FAN PLOTS
# ============================================================================

def create_faceted_forecast_plot(
    timestamps,
    y_full,
    train_size,
    arima_result,
    dlm_result,
    nn_result,
    ext_forecast_timestamps,
    arima_ext=None,
    dlm_ext=None,
    nn_ext=None,
    target_metric=None,
    output_path='outputs/forecast_comparison.png'
):
    """
    Create faceted visualization with:
    - Top panel: ARIMA fitted + fan plot
    - Bottom panel: DLM fitted + fan plot
    
    Fan plots show 50%, 80%, and 95% confidence intervals
    """
    print(f"\n[PLOT] Creating faceted visualization...")
    
    # Split data
    timestamps_train = timestamps[:train_size]
    timestamps_test = timestamps[train_size:]
    y_train = y_full[:train_size]
    y_test = y_full[train_size:]
    
    # Create figure with 3 rows (ARIMA, DLM, NN)
    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True)
    
    # Color scheme
    colors = {
        'observed': '#1f77b4',
        'fitted': '#ff7f0e', 
        'forecast': '#d62728',
        'ci_dark': 0.3,
        'ci_medium': 0.2,
        'ci_light': 0.1
    }
    
    # ========== ARIMA PANEL ==========
    ax1 = axes[0]
    
    if arima_result is not None:
        # Plot observed data
        ax1.plot(timestamps_train, y_train, 
                label='Observed (Train)', color=colors['observed'], 
                linewidth=2, alpha=0.7)
        ax1.plot(timestamps_test, y_test,
                label='Observed (Test)', color=colors['observed'],
                linewidth=2, alpha=0.7, linestyle='--')
        
        # Plot fitted values
        fitted = arima_result['fitted']
        fitted_times = timestamps_train[len(timestamps_train)-len(fitted):]
        ax1.plot(fitted_times, fitted,
                label='ARIMA Fitted', color=colors['fitted'],
                linewidth=2, linestyle='--')
        
        # Plot short test forecast (for comparison)
        forecast = arima_result.get('forecast')
        forecast_std = arima_result.get('forecast_std')
        if forecast is not None and len(forecast) > 0:
            ax1.plot(timestamps_test, forecast,
                label='ARIMA Forecast (test)', color=colors['forecast'],
                linewidth=2.5)

        # Plot extended forecast if provided
        if arima_ext is not None:
            f_ext, f_ext_std = arima_ext
            days = int(len(ext_forecast_timestamps) / 24)
            ax1.plot(ext_forecast_timestamps, f_ext, label=f'ARIMA Extended Forecast (+{days}d)', color=colors['forecast'], linewidth=1.8)
            for alpha, z_score in [(colors['ci_light'], 1.96), (colors['ci_medium'], 1.28), (colors['ci_dark'], 0.67)]:
                lower = f_ext - z_score * f_ext_std
                upper = f_ext + z_score * f_ext_std
                ax1.fill_between(ext_forecast_timestamps, lower, upper, alpha=alpha, color=colors['forecast'])
        
        # Fan plot (95%, 80%, 50% intervals) for test forecast
        if forecast is not None and forecast_std is not None:
            for alpha, z_score in [(colors['ci_light'], 1.96), (colors['ci_medium'], 1.28), (colors['ci_dark'], 0.67)]:
                lower = forecast - z_score * forecast_std
                upper = forecast + z_score * forecast_std
                ax1.fill_between(timestamps_test, lower, upper, alpha=alpha, color=colors['forecast'])
        
        # Add vertical line at train/test split
        ax1.axvline(timestamps_train[-1], color='black', 
                   linestyle=':', alpha=0.5, linewidth=1.5)
        
        # Format
        ax1.set_title(f'ARIMA Model - {target_metric.replace("_", " ").title()}', 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel(target_metric.replace("_", " ").title(), fontsize=12)
        ax1.legend(loc='best', framealpha=0.95, fontsize=10)
        ax1.grid(alpha=0.3, linestyle='--')
        
        # Add metrics text
        metrics = arima_result['metrics']
        metrics_text = (
            f"Train RMSE: {metrics['train_rmse']:.2f}\n"
            f"Test RMSE: {metrics['test_rmse']:.2f}\n"
            f"AIC: {metrics['aic']:.2f}"
        )
        ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5), fontsize=10)

        # --- Auto-scale or log-scale if dynamic range is huge ---
        try:
            vals = []
            vals.extend(np.asarray(y_train).ravel().tolist())
            vals.extend(np.asarray(y_test).ravel().tolist())
            if 'fitted' in arima_result and arima_result['fitted'] is not None:
                vals.extend(np.asarray(arima_result['fitted']).ravel().tolist())
            if 'forecast' in arima_result and arima_result['forecast'] is not None:
                vals.extend(np.asarray(arima_result['forecast']).ravel().tolist())
            if arima_ext is not None:
                f_ext_vals, f_ext_std_vals = arima_ext
                vals.extend(np.asarray(f_ext_vals).ravel().tolist())
                vals.extend((np.asarray(f_ext_vals) - 1.96 * np.asarray(f_ext_std_vals)).ravel().tolist())
                vals.extend((np.asarray(f_ext_vals) + 1.96 * np.asarray(f_ext_std_vals)).ravel().tolist())

            vals_arr = np.array([v for v in vals if np.isfinite(v)])
            if vals_arr.size > 0:
                vmax = float(np.nanmax(vals_arr))
                vmin_pos = float(np.nanmin(vals_arr[vals_arr > 0])) if np.any(vals_arr > 0) else float(np.nanmin(vals_arr))
                if vmin_pos > 0 and (vmax / vmin_pos) > 1000:
                    ax1.set_yscale('log')
        except Exception:
            pass
    
    # ========== DLM PANEL ==========
    ax2 = axes[1]
    
    if dlm_result is not None:
        # Plot observed data
        ax2.plot(timestamps_train, y_train,
                label='Observed (Train)', color=colors['observed'],
                linewidth=2, alpha=0.7)
        ax2.plot(timestamps_test, y_test,
                label='Observed (Test)', color=colors['observed'],
                linewidth=2, alpha=0.7, linestyle='--')
        
        # Plot fitted values
        fitted = dlm_result['fitted']
        ax2.plot(timestamps_train, fitted,
                label='DLM Fitted', color=colors['fitted'],
                linewidth=2, linestyle='--')
        
        # Plot short test forecast
        forecast = dlm_result.get('forecast')
        forecast_std = dlm_result.get('forecast_std')
        if forecast is not None and len(forecast) > 0:
            ax2.plot(timestamps_test, forecast, label='DLM Forecast (test)', color=colors['forecast'], linewidth=2.5)

        # Extended DLM forecast
        if dlm_ext is not None:
            f_ext, f_ext_std = dlm_ext
            days = int(len(ext_forecast_timestamps) / 24)
            ax2.plot(ext_forecast_timestamps, f_ext, label=f'DLM Extended Forecast (+{days}d)', color=colors['forecast'], linewidth=1.8)
            for alpha, z_score in [(colors['ci_light'], 1.96), (colors['ci_medium'], 1.28), (colors['ci_dark'], 0.67)]:
                lower = f_ext - z_score * f_ext_std
                upper = f_ext + z_score * f_ext_std
                ax2.fill_between(ext_forecast_timestamps, lower, upper, alpha=alpha, color=colors['forecast'])

        # Fan plot for test forecast
        if forecast is not None and forecast_std is not None:
            for alpha, z_score in [(colors['ci_light'], 1.96), (colors['ci_medium'], 1.28), (colors['ci_dark'], 0.67)]:
                lower = forecast - z_score * forecast_std
                upper = forecast + z_score * forecast_std
                ax2.fill_between(timestamps_test, lower, upper, alpha=alpha, color=colors['forecast'])
        
        # Add vertical line at train/test split
        ax2.axvline(timestamps_train[-1], color='black',
                   linestyle=':', alpha=0.5, linewidth=1.5)
        
        # Format
        ax2.set_title('Dynamic Linear Model (Kalman Filter)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel(target_metric.replace("_", " ").title(), fontsize=12)
        ax2.legend(loc='best', framealpha=0.95, fontsize=10)
        ax2.grid(alpha=0.3, linestyle='--')
        
        # Add metrics text
        metrics = dlm_result['metrics']
        metrics_text = (
            f"Train RMSE: {metrics['train_rmse']:.2f}\n"
            f"Test RMSE: {metrics['test_rmse']:.2f}"
        )
        ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5), fontsize=10)

        # --- Ensure DLM panel y-limits cover observed, fitted, and forecasts ---
        try:
            vals = []
            vals.extend(np.asarray(y_train).ravel().tolist())
            vals.extend(np.asarray(y_test).ravel().tolist())
            if 'fitted' in dlm_result and dlm_result['fitted'] is not None:
                vals.extend(np.asarray(dlm_result['fitted']).ravel().tolist())
            if forecast is not None:
                vals.extend(np.asarray(forecast).ravel().tolist())
            if dlm_ext is not None:
                f_ext_vals, f_ext_std_vals = dlm_ext
                vals.extend(np.asarray(f_ext_vals).ravel().tolist())
                # include CI endpoints
                vals.extend((np.asarray(f_ext_vals) - 1.96 * np.asarray(f_ext_std_vals)).ravel().tolist())
                vals.extend((np.asarray(f_ext_vals) + 1.96 * np.asarray(f_ext_std_vals)).ravel().tolist())

            vals_arr = np.array([v for v in vals if np.isfinite(v)])
            if vals_arr.size > 0:
                ymin = float(np.nanmin(vals_arr))
                ymax = float(np.nanmax(vals_arr))
                if ymax > ymin:
                    pad = max((ymax - ymin) * 0.08, 1.0)
                    ax2.set_ylim(ymin - pad, ymax + pad)
        except Exception:
            pass
        # Apply log-scaling if dynamic range is very large
        try:
            vals_all = np.array([v for v in vals if np.isfinite(v)])
            if vals_all.size > 0:
                vmax = float(np.nanmax(vals_all))
                vmin_pos = float(np.nanmin(vals_all[vals_all > 0])) if np.any(vals_all > 0) else float(np.nanmin(vals_all))
                if vmin_pos > 0 and (vmax / vmin_pos) > 1000:
                    ax2.set_yscale('log')
        except Exception:
            pass
    
    # ========== NN PANEL ==========
    ax3 = axes[2]
    if nn_result is not None:
        ax3.plot(timestamps_train, y_train, label='Observed (Train)', color=colors['observed'], linewidth=2, alpha=0.7)
        ax3.plot(timestamps_test, y_test, label='Observed (Test)', color=colors['observed'], linewidth=2, alpha=0.7, linestyle='--')

        # Plot NN fitted training values (if available)
        if isinstance(nn_result, dict) and nn_result.get('fitted_train') is not None:
            fitted_nn = np.asarray(nn_result.get('fitted_train'))
            if np.any(~np.isnan(fitted_nn)):
                ax3.plot(timestamps[:len(fitted_nn)], fitted_nn, label='NN Fitted (train)', color=colors['fitted'], linewidth=2, linestyle='--')

        # Extended NN forecast (trained on full series)
        if nn_ext is not None:
            f_ext, f_ext_std = nn_ext
            days = int(len(ext_forecast_timestamps) / 24)
            ax3.plot(ext_forecast_timestamps, f_ext, label=f'NN Extended Forecast (+{days}d)', color=colors['forecast'], linewidth=1.8)
            for alpha, z_score in [(colors['ci_light'], 1.96), (colors['ci_medium'], 1.28), (colors['ci_dark'], 0.67)]:
                lower = f_ext - z_score * f_ext_std
                upper = f_ext + z_score * f_ext_std
                ax3.fill_between(ext_forecast_timestamps, lower, upper, alpha=alpha, color=colors['forecast'])

        # Auto-log-scale NN panel if range is extreme
        try:
            vals = []
            vals.extend(np.asarray(y_train).ravel().tolist())
            vals.extend(np.asarray(y_test).ravel().tolist())
            if isinstance(nn_result, dict) and nn_result.get('fitted_train') is not None:
                vals.extend(np.asarray(nn_result.get('fitted_train')).ravel().tolist())
            if nn_ext is not None:
                f_ext_vals, f_ext_std_vals = nn_ext
                vals.extend(np.asarray(f_ext_vals).ravel().tolist())
                vals.extend((np.asarray(f_ext_vals) - 1.96 * np.asarray(f_ext_std_vals)).ravel().tolist())
                vals.extend((np.asarray(f_ext_vals) + 1.96 * np.asarray(f_ext_std_vals)).ravel().tolist())

            vals_arr = np.array([v for v in vals if np.isfinite(v)])
            if vals_arr.size > 0:
                vmax = float(np.nanmax(vals_arr))
                vmin_pos = float(np.nanmin(vals_arr[vals_arr > 0])) if np.any(vals_arr > 0) else float(np.nanmin(vals_arr))
                if vmin_pos > 0 and (vmax / vmin_pos) > 1000:
                    ax3.set_yscale('log')
        except Exception:
            pass

        ax3.set_title('Neural Network (MLP) Forecast', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel(target_metric.replace('_', ' ').title(), fontsize=12)
        ax3.legend(loc='best', framealpha=0.95, fontsize=10)
        ax3.grid(alpha=0.3, linestyle='--')

    # Add legend for confidence intervals
    legend_elements = [
        Patch(facecolor=colors['forecast'], alpha=colors['ci_light'], 
              label='95% CI'),
        Patch(facecolor=colors['forecast'], alpha=colors['ci_medium'], 
              label='80% CI'),
        Patch(facecolor=colors['forecast'], alpha=colors['ci_dark'], 
              label='50% CI')
    ]
    fig.legend(handles=legend_elements, loc='upper right', 
              bbox_to_anchor=(0.98, 0.98), fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[PLOT] ✓ Saved: {output_path}")
    
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("TIME SERIES FORECASTING: ARIMA vs DLM")
    print("=" * 80)
    
    # 1. Connect to database
    df = connect_to_neon(table_name="videos_log_v3")
    
    # 2. Prepare time series
    # You can change target_metric to: 'view_count', 'like_count', 'engagement_rate'
    target_metric = 'view_count'
    ts_data = prepare_timeseries(df, target_metric=target_metric)
    
    if len(ts_data) < 30:
        print(f"\n[ERROR] Insufficient data: only {len(ts_data)} time points")
        print("[ERROR] Need at least 30 hours of data for reliable forecasting")
        return
    
    # 3. Split into train/test (last 14 hours = 2 weeks of hourly forecasts)
    test_size = min(14, len(ts_data) // 5)  # At least 20% for test
    train_size = len(ts_data) - test_size
    
    timestamps = ts_data['timestamp'].values
    y_full = ts_data[target_metric].values
    
    y_train = y_full[:train_size]
    y_test = y_full[train_size:]
    
    print(f"\n[SPLIT] Train size: {train_size} hours")
    print(f"[SPLIT] Test size: {test_size} hours")
    print(f"[SPLIT] Train period: {timestamps[0]} to {timestamps[train_size-1]}")
    print(f"[SPLIT] Test period: {timestamps[train_size]} to {timestamps[-1]}")
    
    # 4. Fit ARIMA model (on train for evaluation)
    arima_result = fit_arima(y_train, y_test, order=(1, 1, 1))

    # 5. Fit DLM model (on train for evaluation)
    dlm_result = fit_dlm(y_train, y_test)

    # 6. Fit NN model (on train for evaluation)
    nn_eval = fit_nn_model(y_train, y_test, steps_ahead=len(y_test), lags=24)

    # 7. Prepare extended forecasts for a shorter horizon (default: 30 days)
    default_forecast_days = 30
    hours_to_forecast = int(default_forecast_days * 24)

    last_dt = pd.to_datetime(timestamps[-1])
    ext_forecast_timestamps = [last_dt + pd.Timedelta(hours=i+1) for i in range(hours_to_forecast)]

    print(f"\n[FORECAST] Extending forecasts for {hours_to_forecast} hours (~{default_forecast_days} days)")

    # Re-fit ARIMA on full series to produce final forecasts
    arima_ext = None
    try:
        if auto_arima is not None:
            aa_full = auto_arima(y_full, seasonal=False, stepwise=True, error_action='ignore', suppress_warnings=True)
            sel_order_full = aa_full.order
            print(f"[ARIMA] auto_arima (full) selected order={sel_order_full}")
            ar_model = ARIMA(y_full, order=sel_order_full).fit()
        else:
            ar_model = ARIMA(y_full, order=(1, 1, 1)).fit()

        fo = ar_model.get_forecast(steps=hours_to_forecast)
        arima_ext = (np.array(fo.predicted_mean), np.array(fo.se_mean))
    except Exception as e:
        print(f"[ARIMA] ✗ Extended forecast error: {e}")
        arima_ext = None

    # Re-fit DLM on full series
    dlm_ext = None
    try:
        dlm_full = DynamicLinearModel()
        dlm_full.fit(y_full)
        f_ext, f_ext_std = fit_dlm_extended(dlm_full, hours_to_forecast)
        dlm_ext = (np.array(f_ext), np.array(f_ext_std))
    except Exception as e:
        print(f"[DLM] ✗ Extended forecast error: {e}")
        dlm_ext = None

    # Fit NN on full series for extended forecast
    nn_ext = None
    try:
        nn_f = fit_nn_model(y_full, np.array([]), steps_ahead=hours_to_forecast, lags=24)
        if nn_f is not None and nn_f.get('forecast') is not None:
            nn_ext = (np.array(nn_f['forecast']), np.array(nn_f['forecast_std']))
    except Exception as e:
        print(f"[NN] ✗ Extended forecast error: {e}")
        nn_ext = None
    
    # 6. Create visualization
    if arima_result is not None or dlm_result is not None or nn_eval is not None:
        create_faceted_forecast_plot(
            timestamps=timestamps,
            y_full=y_full,
            train_size=train_size,
            arima_result=arima_result,
            dlm_result=dlm_result,
            nn_result=nn_eval,
            ext_forecast_timestamps=ext_forecast_timestamps,
            arima_ext=arima_ext,
            dlm_ext=dlm_ext,
            nn_ext=nn_ext,
            target_metric=target_metric,
            output_path='outputs/forecast_comparison.png'
        )
    
    # 7. Print comparison summary
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    if arima_result and dlm_result:
        comparison = pd.DataFrame({
            'Model': ['ARIMA(1,1,1)', 'DLM'],
            'Train_RMSE': [
                arima_result['metrics']['train_rmse'],
                dlm_result['metrics']['train_rmse']
            ],
            'Test_RMSE': [
                arima_result['metrics']['test_rmse'],
                dlm_result['metrics']['test_rmse']
            ]
        })
        
        print(comparison.to_string(index=False))
        
        # Save comparison
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        comparison.to_csv(output_dir / 'model_comparison.csv', index=False)
        print(f"\n[SAVE] ✓ Comparison saved to outputs/model_comparison.csv")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nOutputs:")
    print("  • outputs/forecast_comparison.png - Faceted visualization with fan plots")
    print("  • outputs/model_comparison.csv - Performance metrics")
    print("=" * 80)


if __name__ == "__main__":
    main()