import streamlit as st
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
from typing import Optional, Dict, Tuple

try:
    from pmdarima import auto_arima
except Exception:
    auto_arima = None

# Page config
st.set_page_config(
    page_title="STREAMWATCH - YouTube Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF0000, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .intelligence-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .model-card-arima {
        background: linear-gradient(135deg, #ff7e5f 0%, #feb47b 100%);
    }
    .model-card-dlm {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .model-card-nn {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .model-card-regression {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .ensemble-card {
        background: linear-gradient(135deg, #FA8BFF 0%, #2BD2FF 50%, #2BFF88 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .channel-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .milestone-badge {
        background: #FFD700;
        color: #000;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .viral-badge {
        background: #FF4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .grade-badge {
        background-color: #e62117; 
        color: white; 
        display: inline-block; 
        padding: 5px 15px; 
        border-radius: 5px; 
        font-weight: bold; 
        font-size: 1.5rem; 
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TIME SERIES MODEL CLASSES
# ============================================================================

class DynamicLinearModel:
    """Dynamic Linear Model using Kalman Filter with trend support"""
    
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
            self._trend = np.mean(np.diff(y[-20:])) if len(y) > 20 else np.mean(np.diff(y))
        else:
            self._trend = 0
    
    def forward_filter(self, y):
        n = len(y)
        self.filtered_means = np.zeros(n)
        self.filtered_vars = np.zeros(n)
        
        m0 = y[0]
        C0 = self.state_variance * 10
        
        for t in range(n):
            if t == 0:
                pred_mean = m0
                pred_var = C0
            else:
                pred_mean = self.filtered_means[t-1]
                pred_var = self.filtered_vars[t-1] + self.state_variance
            
            forecast_error = y[t] - pred_mean
            forecast_var = pred_var + self.obs_variance
            kalman_gain = pred_var / forecast_var
            
            self.filtered_means[t] = pred_mean + kalman_gain * forecast_error
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
            self.smoothed_means[t] = (self.filtered_means[t] + 
                                     J_t * (self.smoothed_means[t+1] - self.filtered_means[t]))
            self.smoothed_vars[t] = (self.filtered_vars[t] + 
                                    J_t**2 * (self.smoothed_vars[t+1] - self.filtered_vars[t] - self.state_variance))
        
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


class SimpleNNForecaster:
    """Simple NN forecaster with lag features"""
    
    def __init__(self, lags=24):
        self.lags = lags
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.last_mean = 0
        self.resid_std = 0
    
    def _build_features(self, y):
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


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

@st.cache_resource
def get_db_connection():
    """Connect to Neon database with auto-recovery"""
    db_uri = os.getenv("NEON_DATABASE_URL")
    if not db_uri:
        st.error("⚠️ NEON_DATABASE_URL not found. Please set it in your environment.")
        st.stop()
    
    return create_engine(db_uri, pool_pre_ping=True, pool_recycle=1800)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def load_channels():
    """Load latest channel data from V3"""
    engine = get_db_connection()
    query = """
    SELECT DISTINCT ON (channel_id) *
    FROM channels_log_v3
    ORDER BY channel_id, ingestion_timestamp DESC
    """
    return pl.from_pandas(pd.read_sql(query, engine))


@st.cache_data(ttl=3600)
def load_videos():
    """Load latest video data from V3"""
    engine = get_db_connection()
    query = """
    SELECT DISTINCT ON (video_id) *
    FROM videos_log_v3
    ORDER BY video_id, ingestion_timestamp DESC
    """
    return pl.from_pandas(pd.read_sql(query, engine))


@st.cache_data(ttl=3600)
def load_video_history(video_id: str, days: int = 30):
    """Load time-series data for a specific video"""
    engine = get_db_connection()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    query = f"""
    SELECT *
    FROM videos_log_v3
    WHERE video_id = '{video_id}'
    AND ingestion_timestamp >= '{cutoff}'
    ORDER BY ingestion_timestamp ASC
    """
    return pl.from_pandas(pd.read_sql(query, engine))


@st.cache_data(ttl=3600)
def load_channel_history(channel_id: str, days: int = 30):
    """Load time-series data for a specific channel"""
    engine = get_db_connection()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    query = f"""
    SELECT *
    FROM channels_log_v3
    WHERE channel_id = '{channel_id}'
    AND ingestion_timestamp >= '{cutoff}'
    ORDER BY ingestion_timestamp ASC
    """
    return pl.from_pandas(pd.read_sql(query, engine))


@st.cache_data(ttl=3600)
def load_aggregated_timeseries():
    """Load ALL aggregated time series for Platform Overview"""
    engine = get_db_connection()
    query = """
    SELECT 
        date_trunc('hour', COALESCE(published_at::timestamp, ingestion_timestamp::timestamp)) as time_bin,
        SUM(view_count) as total_views,
        SUM(like_count) as total_likes,
        SUM(comment_count) as total_comments,
        COUNT(*) as video_count
    FROM videos_log_v3
    GROUP BY date_trunc('hour', COALESCE(published_at::timestamp, ingestion_timestamp::timestamp))
    ORDER BY time_bin ASC
    """
    return pd.read_sql(query, engine)


@st.cache_data(ttl=3600)
def load_channel_timeseries(channel_id: str):
    """Load time series for a specific channel"""
    engine = get_db_connection()
    query = """
    SELECT 
        ingestion_timestamp,
        subscriber_count,
        view_count,
        video_count
    FROM channels_log_v3
    WHERE channel_id = %s
    ORDER BY ingestion_timestamp ASC
    """
    return pd.read_sql(query, engine, params=(channel_id,))


@st.cache_data(ttl=3600, show_spinner=True)
def get_trending_data(days: int, region: str) -> pd.DataFrame:
    """Load trending videos from the v3 tables"""
    engine = get_db_connection()

    query = """
        SELECT
            t.video_id,
            t.channel_id,
            COALESCE(c.channel_title, t.channel_title, t.channel_id) AS channel_display,
            t.video_title,
            t.view_count,
            t.like_count,
            t.comment_count,
            t.duration_seconds,
            t.engagement_rate,
            t.views_per_day,
            t.days_since_publish,
            t.ingestion_timestamp,
            t.trending_region
        FROM trending_videos_log_v3 AS t
        LEFT JOIN channels_log_v3 AS c
          ON t.channel_id = c.channel_id
        WHERE t.ingestion_timestamp::timestamptz >= NOW() - INTERVAL %s
          AND t.trending_region = %s
    """

    df = pd.read_sql(query, engine, params=(f"{days} days", region))
    
    if df.empty:
        return df
    
    df["ingestion_timestamp"] = pd.to_datetime(df["ingestion_timestamp"])
    df = df.sort_values("ingestion_timestamp").reset_index(drop=True)

    # Add velocity features with proper error handling
    df["view_delta"] = df.groupby("video_id")["view_count"].diff()
    time_diff = df.groupby("video_id")["ingestion_timestamp"].diff()
    df["time_delta_hours"] = time_diff.dt.total_seconds() / 3600.0
    
    # Calculate velocity safely
    df["view_velocity_per_hour"] = df["view_delta"] / df["time_delta_hours"]
    
    # Replace invalid values (inf, -inf, nan) with None
    df["view_velocity_per_hour"] = df["view_velocity_per_hour"].replace([np.inf, -np.inf], np.nan)

    return df


# ============================================================================
# MODEL FITTING FUNCTIONS
# ============================================================================

def fit_arima_model(y_train, forecast_steps):
    """Fit ARIMA model with auto order selection"""
    try:
        if auto_arima is not None:
            aa = auto_arima(y_train, seasonal=False, stepwise=True, 
                           error_action='ignore', suppress_warnings=True)
            order = aa.order
        else:
            order = (1, 1, 1)
        
        model = ARIMA(y_train, order=order)
        model_fit = model.fit()
        
        fitted = np.array(model_fit.fittedvalues)
        forecast_obj = model_fit.get_forecast(steps=forecast_steps)
        forecast = np.array(forecast_obj.predicted_mean)
        forecast_std = np.array(forecast_obj.se_mean)
        
        if len(fitted) < len(y_train):
            fitted_full = np.full(len(y_train), np.nan)
            fitted_full[-len(fitted):] = fitted
            fitted = fitted_full
        
        train_rmse = np.sqrt(np.nanmean((y_train - fitted)**2))
        
        return {
            'fitted': fitted,
            'forecast': forecast,
            'forecast_std': forecast_std,
            'order': order,
            'metrics': {'train_rmse': train_rmse, 'aic': model_fit.aic}
        }
    except Exception as e:
        print(f"ARIMA fitting failed: {e}")
        return None


def fit_dlm_model(y_train, forecast_steps):
    """Fit DLM (Kalman Filter) model"""
    try:
        dlm = DynamicLinearModel()
        dlm.fit(y_train)
        
        fitted = np.array(dlm.smoothed_means)
        forecast, forecast_vars = dlm.forecast(steps=forecast_steps)
        forecast_std = np.sqrt(forecast_vars)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, fitted))
        
        return {
            'fitted': fitted,
            'forecast': np.array(forecast),
            'forecast_std': np.array(forecast_std),
            'metrics': {'train_rmse': train_rmse}
        }
    except Exception as e:
        print(f"DLM fitting failed: {e}")
        return None


def fit_nn_model(y_train, forecast_steps):
    """Fit Neural Network model"""
    try:
        lags = min(24, len(y_train)//4)
        nn = SimpleNNForecaster(lags=lags)
        fitted = nn.fit(y_train)
        
        if fitted is not None:
            forecast, forecast_std = nn.forecast(y_train, forecast_steps)
            
            valid_mask = ~np.isnan(fitted)
            if np.sum(valid_mask) > 0:
                train_rmse = np.sqrt(mean_squared_error(
                    y_train[valid_mask], fitted[valid_mask]
                ))
            else:
                train_rmse = np.nan
            
            return {
                'fitted': fitted,
                'forecast': forecast,
                'forecast_std': forecast_std,
                'lags': lags,
                'metrics': {'train_rmse': train_rmse}
            }
        else:
            return None
    except Exception as e:
        print(f"NN fitting failed: {e}")
        return None


def fit_regression_model(df: pd.DataFrame, forecast_days: int = 14):
    """Fit linear regression model for short-term prediction"""
    try:
        max_ts = df["ingestion_timestamp"].max()
        cutoff = max_ts - pd.Timedelta(days=2)

        train_df = df[df["ingestion_timestamp"] < cutoff].copy()
        test_df = df[df["ingestion_timestamp"] >= cutoff].copy()

        feature_cols = ["views_per_day", "days_since_publish", "engagement_rate", "duration_seconds"]

        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df["view_count"]

        X_test = test_df[feature_cols].fillna(0)
        y_test = test_df["view_count"]

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Train predictions
        train_pred = model.predict(X_train)
        
        # Test predictions
        test_df = test_df.copy()
        test_df["predicted_view_count"] = model.predict(X_test)

        # Calculate confidence intervals
        train_errors = y_train - train_pred
        mse = np.mean(train_errors**2)
        std_error = np.sqrt(mse)
        
        # Metrics
        test_errors = y_test - test_df["predicted_view_count"]
        mae = np.mean(np.abs(test_errors))
        rmse = np.sqrt(np.mean(test_errors**2))
        r2 = 1 - np.sum(test_errors**2) / np.sum((y_test - y_test.mean())**2)

        # Forecast future
        last_row = df.iloc[-1]
        forecast_dates = pd.date_range(
            start=max_ts + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        
        forecast_features = pd.DataFrame({
            'views_per_day': [last_row['views_per_day']] * forecast_days,
            'days_since_publish': [last_row['days_since_publish'] + i for i in range(1, forecast_days+1)],
            'engagement_rate': [last_row['engagement_rate']] * forecast_days,
            'duration_seconds': [last_row['duration_seconds']] * forecast_days
        })
        
        forecast = model.predict(forecast_features)
        forecast_std = np.array([std_error * np.sqrt(1 + i*0.1) for i in range(forecast_days)])

        return {
            'model': model,
            'train_size': len(train_df),
            'test_size': len(test_df),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'test_df': test_df,
            'forecast': forecast,
            'forecast_dates': forecast_dates,
            'forecast_std': forecast_std,
            'std_error': std_error
        }
    except Exception as e:
        print(f"Regression fitting failed: {e}")
        return None


# ============================================================================
# COMPUTE ALL FORECASTS - PLATFORM LEVEL
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="Computing platform forecasts...")
def compute_platform_forecasts():
    """Pre-compute all forecast models for platform-level data"""
    ts_data = load_aggregated_timeseries()
    
    if len(ts_data) < 30:
        return None
    
    timestamps = pd.to_datetime(ts_data['time_bin']).values
    y_full = ts_data['total_views'].values
    
    # Calculate forecast steps to 2026-01-31
    last_dt = pd.to_datetime(timestamps[-1])
    target_date = pd.Timestamp('2026-01-31')
    hours_to_forecast = int((target_date - last_dt).total_seconds() / 3600)
    hours_to_forecast = max(hours_to_forecast, 24 * 60)
    forecast_steps = hours_to_forecast
    
    forecast_timestamps = pd.date_range(start=last_dt, periods=forecast_steps + 1, freq='h')[1:]
    
    # Fit all models
    results = {}
    
    # ARIMA
    try:
        results['arima'] = fit_arima_model(y_full, forecast_steps)
    except Exception as e:
        print(f"Platform ARIMA failed: {e}")
        results['arima'] = None
    
    # DLM
    try:
        results['dlm'] = fit_dlm_model(y_full, forecast_steps)
    except Exception as e:
        print(f"Platform DLM failed: {e}")
        results['dlm'] = None
    
    # Neural Network
    try:
        results['nn'] = fit_nn_model(y_full, forecast_steps)
    except Exception as e:
        print(f"Platform NN failed: {e}")
        results['nn'] = None
    
    # Compute ensemble
    ensemble_forecast = None
    ensemble_std = None
    if any(r is not None for r in results.values()):
        valid_forecasts = []
        valid_stds = []
        for r in results.values():
            if r is not None:
                valid_forecasts.append(r['forecast'])
                valid_stds.append(r['forecast_std'])
        
        if len(valid_forecasts) > 0:
            ensemble_forecast = np.mean(valid_forecasts, axis=0)
            ensemble_std = np.sqrt(np.mean([s**2 for s in valid_stds], axis=0))
            
            results['ensemble'] = {
                'forecast': ensemble_forecast,
                'forecast_std': ensemble_std,
                'metrics': {'train_rmse': np.mean([r['metrics']['train_rmse'] for r in results.values() if r is not None])}
            }
    
    return {
        'timestamps': timestamps,
        'y_full': y_full,
        'forecast_timestamps': forecast_timestamps,
        'results': results,
        'forecast_steps': forecast_steps,
        'last_dt': last_dt,
        'target_date': target_date
    }


@st.cache_data(ttl=3600, show_spinner="Computing channel forecasts...")
def compute_channel_forecasts(channel_id: str, forecast_days: int = 60):
    """Compute forecasts for a specific channel"""
    ts_data = load_channel_timeseries(channel_id)
    
    if len(ts_data) < 10:
        return None
    
    ts_data['ingestion_timestamp'] = pd.to_datetime(ts_data['ingestion_timestamp'])
    ts_data = ts_data.sort_values('ingestion_timestamp')
    
    # Use 8-hour aggregation
    ts_data = ts_data.set_index('ingestion_timestamp')
    ts_data = ts_data.resample('8H').last().dropna()
    
    timestamps = ts_data.index.values
    y_full = ts_data['view_count'].values
    
    forecast_steps = forecast_days * 3  # 3 samples per day (8-hour intervals)
    last_dt = pd.to_datetime(timestamps[-1])
    forecast_timestamps = pd.date_range(start=last_dt, periods=forecast_steps + 1, freq='8H')[1:]
    
    # Fit models
    results = {}
    
    try:
        results['arima'] = fit_arima_model(y_full, forecast_steps)
    except:
        results['arima'] = None
    
    try:
        results['dlm'] = fit_dlm_model(y_full, forecast_steps)
    except:
        results['dlm'] = None
    
    try:
        results['nn'] = fit_nn_model(y_full, forecast_steps)
    except:
        results['nn'] = None
    
    # Ensemble
    if any(r is not None for r in results.values()):
        valid_forecasts = [r['forecast'] for r in results.values() if r is not None]
        valid_stds = [r['forecast_std'] for r in results.values() if r is not None]
        
        if len(valid_forecasts) > 0:
            results['ensemble'] = {
                'forecast': np.mean(valid_forecasts, axis=0),
                'forecast_std': np.sqrt(np.mean([s**2 for s in valid_stds], axis=0)),
                'metrics': {'train_rmse': np.mean([r['metrics']['train_rmse'] for r in results.values() if r is not None])}
            }
    
    return {
        'timestamps': timestamps,
        'y_full': y_full,
        'forecast_timestamps': forecast_timestamps,
        'results': results,
        'forecast_steps': forecast_steps,
        'last_dt': last_dt
    }


# ============================================================================
# MILESTONE ACHIEVEMENT PROBABILITY
# ============================================================================

def calculate_milestone_probability(
    current_views: float,
    milestone: float,
    forecast_mean: np.ndarray,
    forecast_std: np.ndarray,
    forecast_dates: pd.DatetimeIndex
) -> Dict:
    """Calculate probability of reaching milestone by date"""
    
    if current_views >= milestone:
        return {
            'achieved': True,
            'date': None,
            'probability': 1.0,
            'days_to_milestone': 0
        }
    
    views_needed = milestone - current_views
    
    # Find when forecast crosses milestone
    cross_idx = None
    for i, views in enumerate(forecast_mean):
        if views >= milestone:
            cross_idx = i
            break
    
    if cross_idx is None:
        # Milestone not reached in forecast window
        return {
            'achieved': False,
            'date': None,
            'probability': 0.0,
            'days_to_milestone': None
        }
    
    # Calculate probability using z-score
    z_score = (milestone - forecast_mean[cross_idx]) / forecast_std[cross_idx]
    probability = 1 - stats.norm.cdf(z_score)
    
    estimated_date = forecast_dates[cross_idx]
    days_to_milestone = (estimated_date - forecast_dates[0]).days
    
    return {
        'achieved': False,
        'date': estimated_date,
        'probability': probability,
        'days_to_milestone': days_to_milestone
    }


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_forecast_plot(timestamps, y_full, results, forecast_timestamps, metric_name='Views', title=None):
    """Create interactive Plotly forecast visualization"""
    
    fig = go.Figure()
    
    timestamps_dt = pd.to_datetime(timestamps)
    forecast_timestamps_dt = pd.to_datetime(forecast_timestamps)
    
    # Calculate reasonable y-axis range
    y_min = np.min(y_full) * 0.95
    y_max = np.max(y_full) * 1.05
    
    # Check forecast values to adjust range
    for result in results.values():
        if result is not None and 'forecast' in result:
            forecast_max = np.max(result['forecast'])
            if forecast_max > y_max:
                y_max = forecast_max * 1.05
    
    # Observed data
    fig.add_trace(go.Scatter(
        x=timestamps_dt,
        y=y_full,
        mode='lines',
        name='Observed Data',
        line=dict(color='#2E86AB', width=2.5),
        hovertemplate=f'<b>Observed</b><br>Date: %{{x|%Y-%m-%d %H:%M}}<br>{metric_name}: %{{y:,.0f}}<extra></extra>'
    ))
    
    model_colors = {
        'arima': '#FF6B35',
        'dlm': '#8B5CF6',
        'nn': '#10B981',
        'regression': '#F5576C',
        'ensemble': '#FA8BFF'
    }
    model_names = {
        'arima': 'ARIMA',
        'dlm': 'DLM (Kalman)',
        'nn': 'Neural Network',
        'regression': 'Linear Regression',
        'ensemble': 'Ensemble (Avg)'
    }
    
    # Plot fitted values
    for model_key in results.keys():
        result = results.get(model_key)
        if result is None:
            continue
        
        fitted = result.get('fitted')
        if fitted is not None:
            color = model_colors.get(model_key, '#666')
            name = model_names.get(model_key, model_key)
            
            fitted = np.array(fitted)
            valid = ~np.isnan(fitted)
            if np.any(valid):
                fig.add_trace(go.Scatter(
                    x=timestamps_dt[valid],
                    y=fitted[valid],
                    mode='lines',
                    name=f'{name} Fitted',
                    line=dict(color=color, width=1.5, dash='dot'),
                    opacity=0.7,
                    hovertemplate=f'<b>{name} Fitted</b><br>Date: %{{x|%Y-%m-%d %H:%M}}<br>{metric_name}: %{{y:,.0f}}<extra></extra>'
                ))
    
    # Plot forecasts with confidence intervals
    for model_key in results.keys():
        result = results.get(model_key)
        if result is None:
            continue
        
        forecast = result['forecast']
        forecast_std = result['forecast_std']
        color = model_colors.get(model_key, '#666')
        name = model_names.get(model_key, model_key)
        
        # Confidence intervals
        upper = forecast + 1.96 * forecast_std
        lower = forecast - 1.96 * forecast_std
        lower = np.maximum(lower, y_min)
        
        fig.add_trace(go.Scatter(
            x=forecast_timestamps_dt,
            y=upper,
            mode='lines',
            name=f'{name} Upper CI',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_timestamps_dt,
            y=lower,
            mode='lines',
            name=f'{name} Lower CI',
            line=dict(width=0),
            fillcolor=f'rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.2)',
            fill='tonexty',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Main forecast line
        fig.add_trace(go.Scatter(
            x=forecast_timestamps_dt,
            y=forecast,
            mode='lines+markers',
            name=f'{name} Forecast',
            line=dict(color=color, width=2.5),
            marker=dict(size=4),
            hovertemplate=f'<b>{name} Forecast</b><br>Date: %{{x|%Y-%m-%d %H:%M}}<br>{metric_name}: %{{y:,.0f}}<extra></extra>'
        ))
    
    plot_title = title if title else f'{metric_name} Forecast'
    
    # Calculate good default x-axis range (last 30 days of data + 14 days forecast)
    last_data_date = timestamps_dt[-1]
    default_start = last_data_date - pd.Timedelta(days=30)
    default_end = forecast_timestamps_dt[min(len(forecast_timestamps_dt)-1, 14*24)]  # 14 days into forecast
    
    fig.update_layout(
        title=plot_title,
        xaxis_title='Date',
        yaxis_title=metric_name,
        hovermode='x unified',
        height=600,
        yaxis=dict(
            range=[y_min, y_max],
            fixedrange=False
        ),
        xaxis=dict(
            range=[default_start, default_end],
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=30, label="30d", step="day", stepmode="backward"),
                    dict(count=90, label="90d", step="day", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    return fig


def get_trending_leaderboard_figure(df: pd.DataFrame):
    """Create trending leaderboard visualization"""
    latest_ts = df["ingestion_timestamp"].max()
    df_latest = df[df["ingestion_timestamp"] == latest_ts].copy()

    channel_stats = (
        df_latest.groupby("channel_display", as_index=False)
        .agg(
            total_views=("view_count", "sum"),
            avg_engagement=("engagement_rate", "mean"),
            videos=("video_id", "nunique"),
        )
        .sort_values("total_views", ascending=False)
        .head(10)
    )

    fig = px.bar(
        channel_stats,
        x="total_views",
        y="channel_display",
        orientation="h",
        title="Creator Leaderboard (Trending)",
        labels={"total_views": "Total Views", "channel_display": "Channel"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig, channel_stats


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_number(num):
    """Format numbers with B for billions, M for millions, K for thousands"""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{int(num)}"


def calculate_grade(subs, views):
    """Assign a Social Blade Style Letter Grade"""
    score = 0
    
    if subs > 50_000_000: score += 50
    elif subs > 10_000_000: score += 40
    elif subs > 1_000_000: score += 30
    elif subs > 100_000: score += 20
    else: score += 10

    if views > 10_000_000_000: score += 50
    elif views > 1_000_000_000: score += 40
    elif views > 100_000_000: score += 30
    elif views > 10_000_000: score += 20
    else: score += 10
    
    if score >= 90: return "A++"
    elif score >= 80: return "A+"
    elif score >= 70: return "A"
    elif score >= 60: return "B+"
    elif score >= 50: return "B"
    else: return "C"


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["Home", "Channel Leaderboard", "Video Explorer", "Milestone Tracker", "Intelligence"]
)

# Initialize session state
if 'selected_channel_id' not in st.session_state:
    st.session_state.selected_channel_id = None
if 'selected_video_id' not in st.session_state:
    st.session_state.selected_video_id = None

# Load data
channels_df = load_channels()
videos_df = load_videos()

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "Home":
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <p style='font-size: 4.2rem; font-weight: bold; background: linear-gradient(90deg, #FF0000, #FF6B6B); 
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;'>
                STREAMWATCH
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    with col1:
        st.metric("Channels", f"{len(channels_df)}")
    
    with col2:
        st.metric("Videos", format_number(len(videos_df)))
    
    with col3:
        try:
            billionaires = videos_df.filter(pl.col("is_billionaires_watch") == True).height
        except:
            billionaires = 0
        st.metric("1B+ Club", f"{billionaires}")
    
    with col4:
        try:
            approaching = videos_df.filter(pl.col("is_approaching_milestone") == True).height
        except:
            approaching = 0
        st.metric("Milestones", f"{approaching}")
    
    with col5:
        total_data_points = channels_df.height + videos_df.height
        st.metric("Data Points", format_number(total_data_points))
    
    with col6:
        try:
            top_channel_views = channels_df.sort("view_count", descending=True)["view_count"][0]
            st.metric("Top Channel", format_number(top_channel_views))
        except:
            st.metric("Top Channel", "N/A")
    
    with col7:
        try:
            total_platform_views = videos_df["view_count"].sum()
            st.metric("Platform Views", format_number(total_platform_views))
        except:
            st.metric("Platform Views", "N/A")
    
    st.markdown("---")
    
    # Platform highlights with thumbnails
    st.markdown("### Platform Highlights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Channels by Subscribers")
        top_channels = channels_df.sort("subscriber_count", descending=True).head(5)
        for idx, row in enumerate(top_channels.iter_rows(named=True)):
            col_a, col_b = st.columns([1, 4])
            with col_a:
                thumbnail = row.get('thumbnail_url')
                if thumbnail and str(thumbnail) not in ['0', '', 'None', None]:
                    try:
                        st.image(thumbnail, width=50)
                    except:
                        st.markdown("📺")
                else:
                    st.markdown("📺")
            with col_b:
                st.markdown(f"**{idx+1}. {row['channel_title']}**")
                st.caption(f"{format_number(row['subscriber_count'])} subscribers")
    
    with col2:
        st.markdown("#### Top Videos by Views")
        top_videos = videos_df.sort("view_count", descending=True).head(5)
        for idx, row in enumerate(top_videos.iter_rows(named=True)):
            col_a, col_b = st.columns([1, 4])
            with col_a:
                thumbnail = row.get('thumbnail_url')
                if thumbnail and str(thumbnail) not in ['0', '', 'None', None]:
                    try:
                        st.image(thumbnail, width=80)
                    except:
                        st.markdown("🎬")
                else:
                    st.markdown("🎬")
            with col_b:
                st.markdown(f"**{idx+1}. {row['video_title'][:45]}...**")
                st.caption(f"{format_number(row['view_count'])} views")

# ============================================================================
# CHANNEL LEADERBOARD PAGE
# ============================================================================

elif page == "Channel Leaderboard":
    st.title("Channel Leaderboard")
    
    # Check if viewing a specific channel
    if 'selected_channel_id' in st.session_state and st.session_state.selected_channel_id:
        # CHANNEL DETAIL VIEW
        channel_id = st.session_state.selected_channel_id
        try:
            channel_data = channels_df.filter(pl.col("channel_id") == channel_id).to_dicts()[0]
        except IndexError:
            st.error("Channel not found.")
            st.stop()
        
        # Back button
        if st.button("← Back to Leaderboard"):
            st.session_state.selected_channel_id = None
            st.rerun()
        
        # Channel header
        col1, col2 = st.columns([1, 2])
        
        with col1:
            thumbnail = channel_data.get('thumbnail_url')
            if thumbnail and str(thumbnail) not in ['0', '', 'None', None]:
                try:
                    st.image(thumbnail, width=250)
                except:
                    st.markdown("### 📺")
            else:
                st.markdown("### 📺")
        
        with col2:
            st.markdown(f"# {channel_data['channel_title']}")
            st.markdown(f"`{channel_data['custom_url']}`")
            
            # Grade badge
            grade = calculate_grade(channel_data['subscriber_count'], channel_data['view_count'])
            st.markdown(f'<span class="grade-badge">Grade: {grade}</span>', unsafe_allow_html=True)
            
            st.caption(channel_data['description'][:300] + "..." if channel_data['description'] else "")
        
        st.markdown("---")
        
        # Current stats
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Subscribers", f"{channel_data['subscriber_count']:,}")
        col2.metric("Total Views", format_number(channel_data['view_count']))
        col3.metric("Videos", f"{channel_data['video_count']:,}")
        col4.metric("Joined", channel_data['published_at'][:10] if channel_data.get('published_at') else "N/A")
        col5.metric("Country", channel_data.get('country', 'N/A'))
        
        st.markdown("---")
        
        # Historical data
        st.subheader("📈 Channel Growth History")
        
        history = load_channel_history(channel_id, days=30)
        
        if len(history) >= 2:
            df_history = history.to_pandas()
            df_history['ingestion_timestamp'] = pd.to_datetime(df_history['ingestion_timestamp'])
            df_history = df_history.sort_values('ingestion_timestamp')
            
            # Create dual-axis chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_history['ingestion_timestamp'],
                y=df_history['subscriber_count'],
                mode='lines+markers',
                name='Subscribers',
                line=dict(color='#FF6B35', width=3),
                yaxis='y'
            ))
            
            fig.add_trace(go.Scatter(
                x=df_history['ingestion_timestamp'],
                y=df_history['view_count'],
                mode='lines+markers',
                name='Total Views',
                line=dict(color='#667eea', width=3),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="Subscriber & View Count History (30 Days)",
                xaxis_title="Date",
                yaxis=dict(title="Subscribers", side="left"),
                yaxis2=dict(title="Total Views", side="right", overlaying="y"),
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Growth metrics
            first_snapshot = df_history.iloc[0]
            last_snapshot = df_history.iloc[-1]
            days_span = (last_snapshot['ingestion_timestamp'] - first_snapshot['ingestion_timestamp']).days
            
            if days_span > 0:
                sub_growth = last_snapshot['subscriber_count'] - first_snapshot['subscriber_count']
                view_growth = last_snapshot['view_count'] - first_snapshot['view_count']
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Subs Gained (30d)", f"+{sub_growth:,}")
                col2.metric("Views Gained (30d)", f"+{view_growth:,}")
                col3.metric("Avg Daily Sub Growth", f"{sub_growth/days_span:,.0f}")
                col4.metric("Avg Daily View Growth", f"{view_growth/days_span:,.0f}")
        else:
            st.info("Not enough historical data for growth analysis (need at least 2 snapshots)")
        
        st.markdown("---")
        
        # Top videos from this channel
        st.subheader(f"Top Videos from {channel_data['channel_title']}")
        
        channel_videos = videos_df.filter(pl.col("channel_id") == channel_id).sort("view_count", descending=True).head(10)
        
        for idx, vid_row in enumerate(channel_videos.iter_rows(named=True)):
            col1, col2 = st.columns([1, 4])
            
            with col1:
                thumbnail = vid_row.get('thumbnail_url')
                if thumbnail and str(thumbnail) not in ['0', '', 'None', None]:
                    try:
                        st.image(thumbnail, width=180)
                    except:
                        st.markdown("🎬")
                else:
                    st.markdown("🎬")
            
            with col2:
                if st.button(f"{vid_row['video_title']}", key=f"vid_{vid_row['video_id']}"):
                    st.session_state.selected_video_id = vid_row['video_id']
                    st.rerun()
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Views", f"{vid_row['view_count']:,}")
                col_b.metric("Engagement", f"{vid_row['engagement_rate']:.2f}%")
                col_c.metric("Views/Day", f"{vid_row['views_per_day']:,.0f}")
            
            st.markdown("---")
    
    else:
        # LEADERBOARD VIEW (default)
        sort_by = st.selectbox(
            "Sort by:",
            ["Subscribers", "Total Views", "Video Count"]
        )
        
        sort_column_map = {
            "Subscribers": "subscriber_count",
            "Total Views": "view_count",
            "Video Count": "video_count"
        }
        
        sorted_channels = channels_df.sort(sort_column_map[sort_by], descending=True)
        
        for idx, row in enumerate(sorted_channels.iter_rows(named=True)):
            col1, col2, col3 = st.columns([1, 5, 2])
            
            with col1:
                st.markdown(f"### #{idx+1}")
                thumbnail = row.get('thumbnail_url')
                if thumbnail and str(thumbnail) not in ['0', '', 'None', None]:
                    try:
                        st.image(thumbnail, width=100)
                    except:
                        st.markdown("📺")
                else:
                    st.markdown("📺")
            
            with col2:
                if st.button(f"{row['channel_title']}", key=f"ch_{row['channel_id']}"):
                    st.session_state.selected_channel_id = row['channel_id']
                    st.rerun()
                
                st.markdown(f"`{row['custom_url']}`")
                st.caption(row['description'][:200] + "..." if row['description'] else "")
            
            with col3:
                st.metric("Subscribers", f"{row['subscriber_count']:,}")
                st.metric("Total Views", format_number(row['view_count']))
                st.metric("Videos", f"{row['video_count']:,}")
            
            st.markdown("---")

# ============================================================================
# VIDEO EXPLORER PAGE
# ============================================================================

elif page == "Video Explorer":
    st.title("Video Explorer")
    
    # Check if viewing a specific video
    if 'selected_video_id' in st.session_state and st.session_state.selected_video_id:
        # VIDEO DETAIL VIEW
        video_id = st.session_state.selected_video_id
        try:
            video_data = videos_df.filter(pl.col("video_id") == video_id).to_dicts()[0]
        except IndexError:
            st.error("Video not found.")
            st.stop()
        
        # Back button
        if st.button("← Back to Explorer"):
            st.session_state.selected_video_id = None
            st.rerun()
        
        # Video header
        col1, col2 = st.columns([1, 2])
        
        with col1:
            thumbnail = video_data.get('thumbnail_url')
            if thumbnail and str(thumbnail) not in ['0', '', 'None', None]:
                try:
                    st.image(thumbnail, width=350)
                except:
                    st.markdown("### 🎬")
            else:
                st.markdown("### 🎬")
        
        with col2:
            st.markdown(f"# {video_data['video_title']}")
            c_title = video_data.get('channel_title', 'Unknown Channel')
            st.markdown(f"## {c_title}")
            st.markdown(f"`{video_data['custom_url']}`")
            
            # Status badges
            if video_data.get('is_billionaires_watch'):
                st.markdown('<span class="milestone-badge">Billionaires Club (1B+)</span>', unsafe_allow_html=True)
            if video_data.get('is_approaching_milestone'):
                st.markdown(f'<span class="viral-badge">Approaching {video_data["next_milestone"]:,}</span>', unsafe_allow_html=True)
            if video_data.get('is_highly_viral'):
                st.markdown('<span class="viral-badge">Highly Viral</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Current stats
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Views", f"{video_data['view_count']:,}")
        col2.metric("Likes", f"{video_data['like_count']:,}")
        col3.metric("Comments", f"{video_data['comment_count']:,}")
        col4.metric("Engagement", f"{video_data['engagement_rate']:.2f}%")
        col5.metric("Views/Day", f"{video_data['views_per_day']:,.0f}")
        
        st.markdown("---")
        
        # Milestone tracking
        st.subheader(f"Milestone Progress: {video_data['milestone_tier']}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Views", f"{video_data['view_count']:,}")
        col2.metric("Next Milestone", f"{video_data['next_milestone']:,}")
        col3.metric("Views Needed", f"{video_data['views_to_next_milestone']:,}")
        
        # Progress bar
        progress_pct = video_data['milestone_progress_pct'] / 100
        st.progress(progress_pct)
        st.caption(f"{video_data['milestone_progress_pct']:.1f}% progress toward {video_data['next_milestone']:,} views")
        
        # Simple milestone projection
        views_per_day = video_data.get('views_per_day', 0)
        views_needed = video_data.get('views_to_next_milestone', 0)
        
        col1, col2 = st.columns(2)
        with col1:
            if views_per_day > 0:
                days_to_milestone = int(views_needed / views_per_day)
                st.metric(
                    "Days to Next Milestone",
                    f"{days_to_milestone:,} days",
                    help="Linear projection based on current velocity"
                )
            else:
                st.metric("Days to Next Milestone", "N/A (low velocity)")
        
        with col2:
            st.metric(
                "Daily Velocity",
                f"{views_per_day:,.0f} views/day",
                help="Average views gained per day"
            )
        
        st.markdown("---")
        
        # Historical view count
        st.subheader("View Count History")
        
        history = load_video_history(video_id, days=30)
        
        if len(history) >= 2:
            df_history = history.to_pandas()
            df_history['ingestion_timestamp'] = pd.to_datetime(df_history['ingestion_timestamp'])
            df_history = df_history.sort_values('ingestion_timestamp')
            
            # Calculate growth
            first_snapshot = df_history.iloc[0]
            last_snapshot = df_history.iloc[-1]
            days_span = (last_snapshot['ingestion_timestamp'] - first_snapshot['ingestion_timestamp']).days
            
            if days_span > 0:
                view_growth = last_snapshot['view_count'] - first_snapshot['view_count']
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Views Gained (30 days)", f"+{view_growth:,}")
                col2.metric("Avg Daily Growth", f"{view_growth/days_span:,.0f} views/day")
                col3.metric("Growth Rate", f"{(view_growth/first_snapshot['view_count']*100):.2f}%")
            
            # Chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_history['ingestion_timestamp'],
                y=df_history['view_count'],
                mode='lines+markers',
                name='Actual Views',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(102,126,234,0.1)'
            ))
            
            # Add milestone line
            fig.add_hline(
                y=video_data['next_milestone'],
                line_dash="dash",
                line_color="gold",
                annotation_text=f"Next Milestone: {video_data['next_milestone']:,}",
                annotation_position="right"
            )
            
            fig.update_layout(
                title="View Count History (30 Days)",
                xaxis_title="Date",
                yaxis_title="View Count",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough historical data (need at least 2 snapshots)")
        
        st.markdown("---")
        
        # Engagement analysis
        st.subheader("Engagement Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Engagement Ratios:**")
            st.metric("Like-to-View", f"{video_data['like_view_ratio']:.4f}", help="Audience positivity")
            st.metric("Comment-to-View", f"{video_data['comment_view_ratio']:.4f}", help="Engagement level")
            st.metric("Like-to-Comment", f"{video_data['like_comment_ratio']:.2f}", help="Positive sentiment")
        
        with col2:
            st.markdown("**Video Details:**")
            st.metric("Video Age", f"{video_data['days_since_publish']} days")
            st.metric("Duration", f"{video_data['duration_seconds']//60} min {video_data['duration_seconds']%60} sec")
            
            category_names = {
                "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music",
                "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events",
                "20": "Gaming", "22": "People & Blogs", "23": "Comedy",
                "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style",
                "27": "Education", "28": "Science & Technology"
            }
            if video_data.get('category_id'):
                category = category_names.get(str(video_data['category_id']), f"Category {video_data['category_id']}")
                st.metric("Category", category)
    
    else:
        # VIDEO EXPLORER VIEW (default)
        col1, col2 = st.columns(2)
        
        with col1:
            filter_type = st.selectbox(
                "Filter by:",
                ["All Videos", "Billionaires Watch (1B+)", "Approaching Milestone", "Highly Viral"]
            )
        
        with col2:
            milestone_tier = st.selectbox(
                "Milestone Tier:",
                ["All", "1B+", "500M-1B", "250M-500M", "100M-250M", "50M-100M", "25M-50M", "10M-25M"]
            )
        
        # Apply filters
        filtered_videos = videos_df
        
        if filter_type == "Billionaires Watch (1B+)":
            filtered_videos = filtered_videos.filter(pl.col("is_billionaires_watch") == True)
        elif filter_type == "Approaching Milestone":
            filtered_videos = filtered_videos.filter(pl.col("is_approaching_milestone") == True)
        elif filter_type == "Highly Viral":
            filtered_videos = filtered_videos.filter(pl.col("is_highly_viral") == True)
        
        if milestone_tier != "All":
            filtered_videos = filtered_videos.filter(pl.col("milestone_tier") == milestone_tier)
        
        filtered_videos = filtered_videos.sort("view_count", descending=True)
        
        st.write(f"**{len(filtered_videos)} videos found**")
        
        # Display videos
        for row in filtered_videos.head(20).iter_rows(named=True):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                thumbnail = row.get('thumbnail_url')
                if thumbnail and str(thumbnail) not in ['0', '', 'None', None]:
                    try:
                        st.image(thumbnail, width=200)
                    except:
                        st.markdown("🎬")
                else:
                    st.markdown("🎬")
            
            with col2:
                if st.button(f"{row['video_title']}", key=f"explore_{row['video_id']}"):
                    st.session_state.selected_video_id = row['video_id']
                    st.rerun()
                
                c_title = row.get('channel_title', 'Unknown Channel')
                st.markdown(f"**{c_title}** • `{row['custom_url']}`")
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Views", f"{row['view_count']:,}")
                col_b.metric("Engagement", f"{row['engagement_rate']:.2f}%")
                col_c.metric("Views/Day", f"{row['views_per_day']:,.0f}")
                
                badges = []
                if row.get('is_billionaires_watch'):
                    badges.append("Billionaires Club")
                if row.get('is_approaching_milestone'):
                    badges.append(f"{row['next_milestone']:,} approaching")
                if row.get('is_highly_viral'):
                    badges.append("Highly Viral")
                
                if badges:
                    st.caption(" • ".join(badges))
            
            st.markdown("---")

# ============================================================================
# MILESTONE TRACKER PAGE
# ============================================================================

elif page == "Milestone Tracker":
    st.title("Milestone Tracker")
    st.markdown("### Videos Approaching Major Milestones")
    st.caption("Within 5% of reaching their next milestone tier")
    
    approaching_videos = videos_df.filter(pl.col("is_approaching_milestone") == True).sort("milestone_progress_pct", descending=True)
    
    if len(approaching_videos) == 0:
        st.info("No videos currently approaching milestones (within 5% threshold)")
    else:
        for row in approaching_videos.iter_rows(named=True):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                thumbnail = row.get('thumbnail_url')
                if thumbnail and str(thumbnail) not in ['0', '', 'None', None]:
                    try:
                        st.image(thumbnail, width=180)
                    except:
                        st.markdown("🎬")
                else:
                    st.markdown("🎬")
            
            with col2:
                st.markdown(f"### {row['video_title']}")
                c_title = row.get('channel_title', 'Unknown Channel')
                st.markdown(f"**{c_title}**")
                
                progress = row['milestone_progress_pct'] / 100
                st.progress(progress)
                st.caption(f"Progress: {row['milestone_progress_pct']:.1f}% to {row['next_milestone']:,} views")
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Current Views", f"{row['view_count']:,}")
                col_b.metric("Views Needed", f"{row['views_to_next_milestone']:,}")
                
                if row['days_to_milestone']:
                    col_c.metric("Est. Days", f"{int(row['days_to_milestone'])}")
                else:
                    col_c.metric("Est. Days", "N/A")
            
            st.markdown("---")

# ============================================================================
# 🧠 INTELLIGENCE PAGE - THE MASTERPIECE
# ============================================================================

elif page == "Intelligence":
    st.title("Intelligence - AI-Powered Analytics & Forecasting")
    
    # Intelligence Mode Selector
    intelligence_mode = st.radio(
        "Select Intelligence Mode:",
        ["Platform Overview", "Channel Intelligence", "Video Intelligence", "Model Performance"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # ========================================================================
    # MODE 1: PLATFORM OVERVIEW
    # ========================================================================
    
    if intelligence_mode == "Platform Overview":
        st.subheader("Platform-Wide Forecasting")
        st.caption("Aggregate predictions across all tracked videos using advanced time series models")
        
        forecast_data = compute_platform_forecasts()
        
        if forecast_data is None:
            st.error("Insufficient data for forecasting. Need at least 30 hourly observations.")
            st.stop()
        
        timestamps = forecast_data['timestamps']
        y_full = forecast_data['y_full']
        forecast_timestamps = forecast_data['forecast_timestamps']
        results = forecast_data['results']
        forecast_steps = forecast_data['forecast_steps']
        last_dt = forecast_data['last_dt']
        target_date = forecast_data['target_date']
        
        st.success(f"✅ Loaded {len(y_full):,} hourly observations from {pd.to_datetime(timestamps[0]).strftime('%Y-%m-%d')} to {last_dt.strftime('%Y-%m-%d')}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Data Points", f"{len(y_full):,}")
        col2.metric("Forecast Steps", f"{forecast_steps:,} hours ({forecast_steps//24} days)")
        col3.metric("Forecast To", target_date.strftime("%Y-%m-%d"))
        
        st.markdown("---")
        
        if any(results.values()):
            st.subheader("Interactive Forecast Visualization")
            st.caption("🖱️ **Tip**: Use range selector or drag slider to zoom • Hover for details")
            
            fig = create_forecast_plot(
                timestamps=timestamps,
                y_full=y_full,
                results=results,
                forecast_timestamps=forecast_timestamps,
                metric_name='Total Platform Views',
                title='Platform-Wide View Count Forecast'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Model performance table
            st.subheader("Model Performance Metrics")
            
            # Build comparison data
            comparison_data = []
            for model_name in ['arima', 'dlm', 'nn', 'ensemble']:
                result = results.get(model_name)
                if result:
                    model_display = {
                        'arima': 'ARIMA',
                        'dlm': 'DLM (Kalman)',
                        'nn': 'Neural Network',
                        'ensemble': 'Ensemble'
                    }[model_name]
                    
                    rmse = result['metrics']['train_rmse']
                    final_forecast = result['forecast'][-1]
                    
                    row_data = {
                        'Model': model_display,
                        'RMSE': f"{rmse:,.0f}",
                        'Final Forecast': format_number(final_forecast)
                    }
                    
                    # Add model-specific info
                    if model_name == 'arima' and 'order' in result:
                        row_data['Details'] = f"Order {result['order']}, AIC: {result['metrics']['aic']:,.0f}"
                    elif model_name == 'dlm':
                        row_data['Details'] = "Adaptive State-Space"
                    elif model_name == 'nn':
                        row_data['Details'] = "MLP (64-32-1)"
                    elif model_name == 'ensemble':
                        row_data['Details'] = "Average of All Models"
                    
                    comparison_data.append(row_data)
            
            # Display as DataFrame
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)
            
            with st.expander("📖 About These Models"):
                st.markdown("""
                ### Model Descriptions
                
                **ARIMA (AutoRegressive Integrated Moving Average)**
                - Captures linear trends and autocorrelation
                - Auto-selected order using `pmdarima`
                - Best for data with clear patterns
                
                **DLM (Dynamic Linear Model)**
                - Kalman filtering for state-space estimation
                - Adapts to changing trends in real-time
                - Smooth probabilistic forecasts
                
                **Neural Network (MLP)**
                - Learns non-linear patterns from lag features
                - 64-32 hidden layer architecture
                - Captures complex dependencies
                
                **Ensemble**
                - Averages all model forecasts
                - Reduces individual model bias
                - Most robust prediction
                
                ### Interpreting Results
                - **RMSE**: Lower = better fit
                - **Confidence Intervals**: Wider = higher uncertainty
                - **Trend**: Extended historical pattern
                """)
        
        else:
            st.error("All models failed to fit. Check data quality.")
    
    # ========================================================================
    # MODE 2: CHANNEL INTELLIGENCE
    # ========================================================================
    
    elif intelligence_mode == "Channel Intelligence":
        st.subheader("Channel-Specific Forecasting")
        st.caption("Predict subscriber growth and view trends for individual channels")
        
        # Channel selector
        channel_options = channels_df.sort("subscriber_count", descending=True)
        channel_list = [(row['channel_title'], row['channel_id']) for row in channel_options.iter_rows(named=True)]
        
        selected_channel_name = st.selectbox(
            "Select a channel:",
            options=[name for name, _ in channel_list],
            format_func=lambda x: x
        )
        
        selected_channel_id = next(cid for name, cid in channel_list if name == selected_channel_name)
        
        # Forecast horizon
        forecast_days = st.slider("Forecast horizon (days):", 14, 120, 60)
        
        if st.button("🚀 Generate Channel Forecast"):
            with st.spinner("Computing channel forecasts..."):
                channel_forecast = compute_channel_forecasts(selected_channel_id, forecast_days)
            
            if channel_forecast is None:
                st.error("Insufficient historical data for this channel (need 10+ observations)")
            else:
                timestamps = channel_forecast['timestamps']
                y_full = channel_forecast['y_full']
                forecast_timestamps = channel_forecast['forecast_timestamps']
                results = channel_forecast['results']
                
                st.success(f"✅ Loaded {len(y_full)} observations")
                
                # Get channel info
                channel_info = channels_df.filter(pl.col("channel_id") == selected_channel_id).to_dicts()[0]
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Subs", format_number(channel_info['subscriber_count']))
                col2.metric("Total Views", format_number(channel_info['view_count']))
                col3.metric("Videos", f"{channel_info['video_count']:,}")
                col4.metric("Grade", calculate_grade(channel_info['subscriber_count'], channel_info['view_count']))
                
                st.markdown("---")
                
                # Plot
                fig = create_forecast_plot(
                    timestamps=timestamps,
                    y_full=y_full,
                    results=results,
                    forecast_timestamps=forecast_timestamps,
                    metric_name='Channel Views',
                    title=f'{selected_channel_name} - View Count Forecast'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Milestone predictions
                st.subheader("Milestone Achievement Predictions")
                
                if results.get('ensemble'):
                    ensemble = results['ensemble']
                    current_views = y_full[-1]
                    
                    milestones = [
                        ("1 Trillion Views", 1_000_000_000_000),
                        ("500 Billion Views", 500_000_000_000),
                        ("250 Billion Views", 250_000_000_000),
                        ("100 Billion Views", 100_000_000_000),
                        ("50 Billion Views", 50_000_000_000),
                        ("25 Billion Views", 25_000_000_000),
                        ("10 Billion Views", 10_000_000_000),
                        ("5 Billion Views", 5_000_000_000),
                        ("1 Billion Views", 1_000_000_000),
                        ("500 Million Views", 500_000_000),
                        ("100 Million Views", 100_000_000)
                    ]
                    
                    for milestone_name, milestone_value in milestones:
                        if current_views < milestone_value:
                            prob_data = calculate_milestone_probability(
                                current_views,
                                milestone_value,
                                current_views + ensemble['forecast'],
                                ensemble['forecast_std'],
                                forecast_timestamps
                            )
                            
                            if prob_data['date']:
                                st.markdown(f"""
                                **{milestone_name}**
                                - Estimated Date: {prob_data['date'].strftime('%Y-%m-%d')}
                                - Probability: {prob_data['probability']*100:.1f}%
                                - Days: {prob_data['days_to_milestone']}
                                """)
                            else:
                                st.caption(f"{milestone_name}: Not achievable in forecast window")
                            
                            st.markdown("---")
                else:
                    st.info("Milestone predictions require ensemble forecast")
    
    # ========================================================================
    # MODE 3: VIDEO INTELLIGENCE
    # ========================================================================
    
    elif intelligence_mode == "Video Intelligence":
        st.subheader("Video-Level Predictions")
        st.caption("Trending analytics and short-term view forecasts for individual videos")
        
        # Trending section
        st.markdown("### Trending Video Analytics")
        
        col1, col2 = st.columns(2)
        with col1:
            days = st.slider("Lookback days:", 1, 14, 7, key="trend_days")
        with col2:
            region = st.selectbox("Region code:", ["US", "GB", "CA", "AU", "IN", "DE", "FR", "JP", "BR", "MX"], key="trend_region")
        
        if st.button("🔍 Load Trending Data"):
            st.cache_data.clear()
        
        try:
            with st.spinner("Loading trending data..."):
                df_trend = get_trending_data(days=days, region=region)
        except Exception as e:
            st.error(f"Unable to load trending data for region '{region}'. This region may not have trending data available.")
            st.info("💡 Try selecting a different region (US, GB, CA are most reliable) or adjusting the lookback days.")
            st.stop()
        
        if df_trend.empty:
            st.warning(f"⚠️ No trending videos found for region '{region}' in the last {days} days.")
            st.info("💡 Try:\n- Selecting a different region (US, GB, CA, AU typically have more data)\n- Increasing the lookback days\n- Checking back later as trending data updates regularly")
            st.stop()
        else:
            latest_ts = df_trend["ingestion_timestamp"].max()
            df_latest = df_trend[df_trend["ingestion_timestamp"] == latest_ts]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Videos", df_latest["video_id"].nunique())
            col2.metric("Channels", df_latest["channel_id"].nunique())
            col3.metric("Total Views", f"{int(df_latest['view_count'].sum()):,}")
            
            st.caption(f"Latest snapshot: {latest_ts}")
            st.markdown("---")
            
            # Creator leaderboard
            st.subheader("Creator Leaderboard (Trending)")
            fig_lb, channel_stats = get_trending_leaderboard_figure(df_trend)
            st.plotly_chart(fig_lb, use_container_width=True)
            
            with st.expander("Show leaderboard table"):
                st.dataframe(channel_stats)
            
            st.markdown("---")
            
            # Fastest growing
            st.subheader("Fastest Growing Videos (View Velocity)")
            
            df_latest_vel = df_latest.dropna(subset=["view_velocity_per_hour"]).copy()
            top_velocity = (
                df_latest_vel.sort_values("view_velocity_per_hour", ascending=False)
                .head(10)[["video_title", "view_count", "view_velocity_per_hour"]]
            )
            
            st.dataframe(top_velocity)
            
            st.markdown("---")
            
            # Regression model
            st.subheader("View Count Prediction (Linear Regression)")
            
            reg_result = fit_regression_model(df_trend, forecast_days=14)
            
            if reg_result:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Train Size", reg_result['train_size'])
                c2.metric("Test Size", reg_result['test_size'])
                c3.metric("MAE", f"{reg_result['mae']:.0f}")
                c4.metric("RMSE", f"{reg_result['rmse']:.0f}")
                
                st.write(f"R²: **{reg_result['r2']:.4f}**")
                
                # Scatter plot
                test_df = reg_result['test_df']
                fig_scatter = px.scatter(
                    test_df,
                    x="predicted_view_count",
                    y="view_count",
                    hover_data=["video_id", "video_title"],
                    title="Predicted vs Actual (Test Set)",
                    labels={"predicted_view_count": "Predicted", "view_count": "Actual"}
                )
                
                min_v = min(test_df["predicted_view_count"].min(), test_df["view_count"].min())
                max_v = max(test_df["predicted_view_count"].max(), test_df["view_count"].max())
                
                fig_scatter.add_shape(
                    type="line",
                    x0=min_v, y0=min_v,
                    x1=max_v, y1=max_v,
                    line=dict(dash="dash", color="red")
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.error("Regression model failed to fit")
    
    # ========================================================================
    # MODE 4: MODEL PERFORMANCE DASHBOARD
    # ========================================================================
    
    elif intelligence_mode == "Model Performance":
        st.subheader("Model Performance Dashboard")
        st.caption("Compare all models across different metrics and scenarios")
        
        # Compute platform forecasts
        forecast_data = compute_platform_forecasts()
        
        if forecast_data is None:
            st.error("Insufficient data")
            st.stop()
        
        results = forecast_data['results']
        
        st.markdown("### Model Comparison Matrix")
        
        comparison_data = []
        for model_name, result in results.items():
            if result:
                comparison_data.append({
                    'Model': model_name.upper(),
                    'RMSE': f"{result['metrics']['train_rmse']:,.0f}",
                    'Final Forecast': f"{format_number(result['forecast'][-1])}",
                    'Status': '✅ Success'
                })
            else:
                comparison_data.append({
                    'Model': model_name.upper(),
                    'RMSE': 'N/A',
                    'Final Forecast': 'N/A',
                    'Status': '❌ Failed'
                })
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("""
        ### Model Selection Guide
        
        **When to use each model:**
        
        - **ARIMA**: Best for data with clear linear trends and seasonality
        - **DLM (Kalman Filter)**: Best for adaptive forecasting with changing trends
        - **Neural Network**: Best for complex non-linear patterns
        - **Ensemble**: Best for most robust predictions (recommended)
        
        **Confidence Intervals:**
        - Wider bands = more uncertainty
        - ARIMA & DLM: 95% confidence intervals
        - NN: 1 standard deviation
        
        **Performance Metrics:**
        - **RMSE**: Root Mean Squared Error (lower is better)
        - **AIC**: Akaike Information Criterion (ARIMA only, lower is better)
        - **R²**: Coefficient of determination (regression only, higher is better)
        """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Project STREAMWATCH - IDS 706 Fall 2025 • Tony Ngari, Matthew Fischer, Can He, Joseph Hong, Trey Chase</p>", unsafe_allow_html=True)
