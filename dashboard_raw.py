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
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

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

# Custom CSS for Social Blade-like styling
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
    .forecast-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .model-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
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
    .best-model-banner {
        background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: #000;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        border: 3px solid #FFD700;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TIME SERIES MODELS
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
        
        # Estimate trend
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


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

@st.cache_resource
def get_db_connection():
    """Connect to Neon database with auto-recovery"""
    db_uri = os.getenv("NEON_DATABASE_URL")
    if not db_uri:
        st.error("⚠️ NEON_DATABASE_URL not found. Please set it in your .env file.")
        st.stop()
    
    db_uri = db_uri.strip()
    if db_uri.startswith("psql "):
        db_uri = db_uri[5:].strip()
    db_uri = db_uri.strip("'\"")
    
    return create_engine(db_uri, pool_pre_ping=True, pool_recycle=1800)


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
def load_timeseries_with_gaps_filled(aggregation_freq='H', fill_method='interpolate'):
    """
    Load aggregated time series and fill gaps
    
    Parameters:
    -----------
    aggregation_freq : str
        'H' for hourly, 'D' for daily, 'W' for weekly
    fill_method : str
        'interpolate' - linear interpolation between points
        'ffill' - forward fill (carry last value forward)
        'zero' - fill gaps with zeros
        'mean' - fill with mean of surrounding values
    """
    engine = get_db_connection()
    
    # Map frequency to SQL date_trunc
    freq_map = {'H': 'hour', 'D': 'day', 'W': 'week'}
    trunc_unit = freq_map.get(aggregation_freq, 'hour')
    
    # Get aggregated data (your existing aggregation)
    query = f"""
    SELECT 
        date_trunc('{trunc_unit}', COALESCE(published_at::timestamp, ingestion_timestamp::timestamp)) as time_bin,
        SUM(view_count) as total_views,
        COUNT(*) as observation_count
    FROM videos_log_v3
    WHERE view_count IS NOT NULL
    GROUP BY time_bin
    ORDER BY time_bin ASC
    """
    
    df = pd.read_sql(query, engine)
    df['time_bin'] = pd.to_datetime(df['time_bin'])
    
    # Set time_bin as index
    df = df.set_index('time_bin')
    
    # Create complete date range from min to max
    complete_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=aggregation_freq
    )
    
    # Reindex to include all timestamps
    df_complete = df.reindex(complete_range)
    
    # Fill gaps based on method
    if fill_method == 'interpolate':
        # Linear interpolation
        df_complete['total_views'] = df_complete['total_views'].interpolate(method='linear', limit_direction='both')
        df_complete['observation_count'] = df_complete['observation_count'].interpolate(method='linear', limit_direction='both')
    elif fill_method == 'ffill':
        # Forward fill
        df_complete['total_views'] = df_complete['total_views'].fillna(method='ffill')
        df_complete['observation_count'] = df_complete['observation_count'].fillna(method='ffill')
    elif fill_method == 'zero':
        # Fill with zeros
        df_complete = df_complete.fillna(0)
    elif fill_method == 'mean':
        # Fill with rolling mean
        df_complete['total_views'] = df_complete['total_views'].fillna(
            df_complete['total_views'].rolling(window=5, min_periods=1, center=True).mean()
        )
        df_complete['observation_count'] = df_complete['observation_count'].fillna(
            df_complete['observation_count'].rolling(window=5, min_periods=1, center=True).mean()
        )
    
    # Reset index
    df_complete = df_complete.reset_index()
    df_complete.columns = ['timestamp', 'total_views', 'observation_count']
    
    return df_complete


# ============================================================================
# PRE-COMPUTE ALL FORECASTS - CACHED
# ============================================================================

@st.cache_data(ttl=3600, show_spinner="Computing forecasts...")
def compute_all_forecasts(aggregation_freq='H', fill_method='interpolate'):
    """Pre-compute all forecast models with gap-filled data"""
    
    # Load data with gaps filled
    ts_data = load_timeseries_with_gaps_filled(aggregation_freq, fill_method)
    
    if len(ts_data) < 30:
        return None
    
    # Prepare data
    timestamps = pd.to_datetime(ts_data['timestamp']).values
    y_full = ts_data['total_views'].values
    
    # Calculate forecast steps
    last_dt = pd.to_datetime(timestamps[-1])
    target_date = pd.Timestamp('2026-01-31')
    
    # Calculate number of periods to forecast
    if aggregation_freq == 'H':
        freq_timedelta = timedelta(hours=1)
        periods_to_forecast = int((target_date - last_dt).total_seconds() / 3600)
    elif aggregation_freq == 'D':
        freq_timedelta = timedelta(days=1)
        periods_to_forecast = (target_date - last_dt).days
    elif aggregation_freq == 'W':
        freq_timedelta = timedelta(weeks=1)
        periods_to_forecast = int((target_date - last_dt).days / 7)
    else:
        freq_timedelta = timedelta(hours=1)
        periods_to_forecast = int((target_date - last_dt).total_seconds() / 3600)
    
    forecast_steps = max(periods_to_forecast, 24)
    forecast_steps = min(forecast_steps, 10000)
    
    # Generate forecast timestamps
    forecast_timestamps = pd.date_range(
        start=last_dt + freq_timedelta,
        periods=forecast_steps,
        freq=aggregation_freq
    )
    
    # Fit all models
    results = {}
    
    # ARIMA
    try:
        results['arima'] = fit_arima_model(y_full, forecast_steps)
    except Exception as e:
        print(f"ARIMA failed: {e}")
        results['arima'] = None
    
    # DLM
    try:
        results['dlm'] = fit_dlm_model(y_full, forecast_steps)
    except Exception as e:
        print(f"DLM failed: {e}")
        results['dlm'] = None
    
    # Neural Network
    try:
        results['nn'] = fit_nn_model(y_full, forecast_steps)
    except Exception as e:
        print(f"NN failed: {e}")
        results['nn'] = None
    
    # Calculate gaps filled info
    original_count = len(ts_data[ts_data['observation_count'] > 0])
    total_count = len(ts_data)
    gaps_filled = total_count - original_count
    
    return {
        'timestamps': timestamps,
        'y_full': y_full,
        'forecast_timestamps': forecast_timestamps,
        'results': results,
        'forecast_steps': forecast_steps,
        'last_dt': last_dt,
        'target_date': target_date,
        'aggregation_freq': aggregation_freq,
        'fill_method': fill_method,
        'gaps_filled': gaps_filled,
        'total_periods': total_count,
        'original_periods': original_count
    }


def calculate_grade(subs, views):
    """Assign a Social Blade Style Letter Grade based on scale"""
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


# ============================================================================
# INDIVIDUAL MODEL FITTING FUNCTIONS
# ============================================================================

def fit_arima_model(y_train, forecast_steps):
    """Fit ARIMA model"""
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
    """Fit DLM model"""
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


# ============================================================================
# INDIVIDUAL MODEL PLOTTING FUNCTION
# ============================================================================

def create_individual_model_plot(timestamps, y_full, result, forecast_timestamps, model_name, color):
    """Create individual plot for a single model"""
    
    fig = go.Figure()
    
    # Convert timestamps
    timestamps_dt = pd.to_datetime(timestamps)
    forecast_timestamps_dt = pd.to_datetime(forecast_timestamps)
    
    # Plot observed data
    fig.add_trace(go.Scatter(
        x=timestamps_dt,
        y=y_full,
        mode='lines',
        name='Observed Data',
        line=dict(color='#2E86AB', width=2),
        opacity=0.7,
        hovertemplate='<b>Observed</b><br>Date: %{x|%Y-%m-%d %H:%M}<br>Views: %{y:,.0f}<extra></extra>'
    ))
    
    # Plot fitted values
    fitted = result.get('fitted')
    if fitted is not None:
        fitted = np.array(fitted)
        valid = ~np.isnan(fitted)
        if np.any(valid):
            fig.add_trace(go.Scatter(
                x=timestamps_dt[valid],
                y=fitted[valid],
                mode='lines',
                name='Fitted Values',
                line=dict(color=color, width=2),
                opacity=0.8,
                hovertemplate=f'<b>Fitted</b><br>Date: %{{x|%Y-%m-%d %H:%M}}<br>Value: %{{y:,.0f}}<extra></extra>'
            ))
    
    # Plot forecast
    forecast = result['forecast']
    forecast_std = result['forecast_std']
    
    fig.add_trace(go.Scatter(
        x=forecast_timestamps_dt,
        y=forecast,
        mode='lines',
        name='Forecast',
        line=dict(color=color, width=3),
        hovertemplate=f'<b>Forecast</b><br>Date: %{{x|%Y-%m-%d %H:%M}}<br>Prediction: %{{y:,.0f}}<extra></extra>'
    ))
    
    # 95% confidence interval
    upper_95 = forecast + 1.96 * forecast_std
    lower_95 = np.maximum(forecast - 1.96 * forecast_std, 0)
    
    # Extract RGB from hex color
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    
    x_combined = list(forecast_timestamps_dt) + list(forecast_timestamps_dt[::-1])
    y_combined = list(upper_95) + list(lower_95[::-1])
    
    fig.add_trace(go.Scatter(
        x=x_combined,
        y=y_combined,
        fill='toself',
        fillcolor=f'rgba({r}, {g}, {b}, 0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% Confidence Interval',
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Add vertical line at forecast start
    fig.add_shape(
        type="line",
        x0=timestamps_dt[-1],
        x1=timestamps_dt[-1],
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=timestamps_dt[-1],
        y=1,
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        yanchor="bottom"
    )
    
    # Layout
    fig.update_layout(
        title=dict(text=f'{model_name} Forecast', font=dict(size=20, color=color)),
        xaxis_title='Date',
        yaxis_title='Total Views',
        hovermode='closest',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white',
        xaxis=dict(
            autorange=True,
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ],
                bgcolor='rgba(150, 150, 150, 0.1)',
                y=1.15,
                x=0.01
            ),
            rangeslider=dict(visible=True, thickness=0.05),
            type="date"
        ),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'forecast_{model_name.lower().replace(" ", "_")}',
            'height': 500,
            'width': 1400,
            'scale': 2
        }
    }
    
    return fig, config


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["🏠 Home", "📊 Channel Leaderboard", "🎬 Video Explorer", 
     "🚀 Milestone Tracker", "🔬 Forecast Lab"]
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
# PAGE 1: HOME
# ============================================================================

if page == "🏠 Home":
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
        total_channels = len(channels_df)
        st.metric("📺 Channels", f"{total_channels}")
    
    with col2:
        total_videos = len(videos_df)
        st.metric("🎬 Videos", format_number(total_videos))
    
    with col3:
        try:
            billionaires = videos_df.filter(pl.col("is_billionaires_watch") == True).height
        except Exception:
            billionaires = 0
        st.metric("💎 1B+ Club", f"{billionaires}")
    
    with col4:
        try:
            approaching = videos_df.filter(pl.col("is_approaching_milestone") == True).height
        except Exception:
            approaching = 0
        st.metric("🎯 Milestones", f"{approaching}")
    
    with col5:
        total_data_points = channels_df.height + videos_df.height
        st.metric("📊 Data Points", format_number(total_data_points))
    
    with col6:
        try:
            top_channel_daily_views = channels_df.sort("view_count", descending=True).select("view_count").head(1).to_numpy()[0][0]
            hourly_estimate = int(top_channel_daily_views / 365 / 24)
            st.metric("⚡ Top/Hour", format_number(hourly_estimate))
        except:
            st.metric("⚡ Top/Hour", "N/A")
    
    with col7:
        try:
            total_views = videos_df.select(pl.col("view_count")).sum().to_numpy()[0][0]
            st.metric("👁️ Total Views", format_number(total_views))
        except:
            st.metric("👁️ Total Views", "N/A")
    
    st.markdown("---")
    
    st.subheader("🏆 Top Channels by Subscribers")
    top_channels = channels_df.sort("subscriber_count", descending=True).head(10)
    
    for idx, row in enumerate(top_channels.iter_rows(named=True)):
        col1, col2 = st.columns([1, 4])
        with col1:
            if row['thumbnail_url']:
                st.image(row['thumbnail_url'], width=80)
            else:
                st.write(f"#{idx+1}")
        with col2:
            st.markdown(f"**{row['channel_title']}** `{row['custom_url']}`")
            st.caption(f"👥 {row['subscriber_count']:,} subscribers | 👀 {row['view_count']:,} total views")
    
    st.markdown("---")
    
    st.subheader("🔥 Highly Viral Videos Right Now")
    try:
        viral_videos = videos_df.filter(pl.col("is_highly_viral") == True).sort("views_per_day", descending=True).head(5)
        
        for row in viral_videos.iter_rows(named=True):
            col1, col2 = st.columns([1, 4])
            with col1:
                if row['thumbnail_url']:
                    st.image(row['thumbnail_url'], width=150)
            with col2:
                st.markdown(f"**{row['video_title']}**")
                st.caption(f"📺 {row.get('channel_title', 'Unknown')} • {row['view_count']:,} views")
                st.caption(f"⚡ {row['views_per_day']:,.0f} views/day • 💬 {row['engagement_rate']:.2f}% engagement")
    except Exception:
        st.info("Viral statistics unavailable")

# ============================================================================
# PAGE 2: CHANNEL LEADERBOARD
# ============================================================================

elif page == "📊 Channel Leaderboard":
    st.title("📊 Channel Leaderboard")
    st.markdown("Compare channels across key metrics with Social Blade-style grades")
    
    # Sorting options
    col1, col2 = st.columns([3, 1])
    with col1:
        sort_metric = st.selectbox(
            "Sort by:",
            ["subscriber_count", "view_count", "video_count"],
            format_func=lambda x: x.replace("_", " ").title()
        )
    with col2:
        sort_order = st.radio("Order:", ["Descending", "Ascending"])
    
    # Sort channels
    ascending = (sort_order == "Ascending")
    sorted_channels = channels_df.sort(sort_metric, descending=not ascending)
    
    st.markdown("---")
    
    # Display channels
    for idx, row in enumerate(sorted_channels.iter_rows(named=True)):
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                if row['thumbnail_url']:
                    st.image(row['thumbnail_url'], width=100)
                else:
                    st.write(f"#{idx+1}")
            
            with col2:
                st.markdown(f"### {row['channel_title']}")
                st.caption(f"`{row['custom_url']}`")
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Subscribers", format_number(row['subscriber_count']))
                with metrics_col2:
                    st.metric("Total Views", format_number(row['view_count']))
                with metrics_col3:
                    st.metric("Videos", f"{row['video_count']:,}")
            
            with col3:
                grade = calculate_grade(row['subscriber_count'], row['view_count'])
                st.markdown(f'<div class="grade-badge">{grade}</div>', unsafe_allow_html=True)
            
            st.markdown("---")

# ============================================================================
# PAGE 3: VIDEO EXPLORER
# ============================================================================

elif page == "🎬 Video Explorer":
    st.title("🎬 Video Explorer")
    st.markdown("Discover and analyze individual videos")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_views = st.number_input("Minimum Views", min_value=0, value=0, step=1000000)
    
    with col2:
        sort_by = st.selectbox(
            "Sort By",
            ["view_count", "like_count", "comment_count", "engagement_rate"],
            format_func=lambda x: x.replace("_", " ").title()
        )
    
    with col3:
        video_limit = st.slider("Number of Videos", 5, 50, 20)
    
    # Filter and sort videos
    filtered_videos = videos_df.filter(pl.col("view_count") >= min_views).sort(sort_by, descending=True).head(video_limit)
    
    st.markdown("---")
    st.subheader(f"Showing {len(filtered_videos)} Videos")
    
    # Display videos
    for idx, row in enumerate(filtered_videos.iter_rows(named=True)):
        with st.expander(f"#{idx+1} - {row['video_title'][:100]}...", expanded=(idx < 3)):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if row['thumbnail_url']:
                    st.image(row['thumbnail_url'], use_column_width=True)
                
                # Badges
                badges = []
                if row.get('is_billionaires_watch'):
                    badges.append('💎 1B+ Club')
                if row.get('is_highly_viral'):
                    badges.append('🔥 Viral')
                if row.get('is_approaching_milestone'):
                    badges.append('🎯 Milestone')
                
                if badges:
                    st.markdown(" ".join([f'<span class="viral-badge">{b}</span>' for b in badges]), unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**{row['video_title']}**")
                st.caption(f"📺 {row.get('channel_title', 'Unknown Channel')}")
                
                metrics_row1 = st.columns(3)
                with metrics_row1[0]:
                    st.metric("👁️ Views", format_number(row['view_count']))
                with metrics_row1[1]:
                    st.metric("👍 Likes", format_number(row['like_count']))
                with metrics_row1[2]:
                    st.metric("💬 Comments", format_number(row['comment_count']))
                
                metrics_row2 = st.columns(3)
                with metrics_row2[0]:
                    st.metric("📈 Engagement", f"{row['engagement_rate']:.2f}%")
                with metrics_row2[1]:
                    if 'views_per_day' in row:
                        st.metric("⚡ Views/Day", format_number(row['views_per_day']))
                with metrics_row2[2]:
                    if 'published_at' in row and row['published_at']:
                        st.caption(f"📅 Published: {row['published_at'][:10]}")

# ============================================================================
# PAGE 4: MILESTONE TRACKER
# ============================================================================

elif page == "🚀 Milestone Tracker":
    st.title("🚀 Milestone Tracker")
    st.markdown("Track videos approaching major view milestones")
    
    # Define milestones
    milestones = [
        (1_000_000_000, "1 Billion"),
        (500_000_000, "500 Million"),
        (100_000_000, "100 Million"),
        (50_000_000, "50 Million"),
        (10_000_000, "10 Million")
    ]
    
    tabs = st.tabs([m[1] for m in milestones])
    
    for tab, (milestone_val, milestone_name) in zip(tabs, milestones):
        with tab:
            # Find videos near this milestone
            lower_bound = milestone_val * 0.8
            upper_bound = milestone_val * 1.2
            
            near_milestone = videos_df.filter(
                (pl.col("view_count") >= lower_bound) & 
                (pl.col("view_count") <= upper_bound)
            ).sort("view_count", descending=True)
            
            st.subheader(f"Videos Near {milestone_name} Views")
            st.caption(f"Showing videos between {format_number(lower_bound)} and {format_number(upper_bound)} views")
            
            if len(near_milestone) == 0:
                st.info(f"No videos currently near the {milestone_name} milestone")
            else:
                for idx, row in enumerate(near_milestone.iter_rows(named=True)):
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
                        if row['thumbnail_url']:
                            st.image(row['thumbnail_url'], width=120)
                    
                    with col2:
                        st.markdown(f"**{row['video_title']}**")
                        st.caption(f"📺 {row.get('channel_title', 'Unknown')}")
                        
                        current_views = row['view_count']
                        progress = (current_views / milestone_val) * 100
                        remaining = milestone_val - current_views
                        
                        st.progress(min(progress / 100, 1.0))
                        st.caption(f"Current: {format_number(current_views)} ({progress:.1f}%)")
                        
                        if remaining > 0:
                            st.caption(f"🎯 {format_number(remaining)} views to {milestone_name}")
                        else:
                            st.caption(f"✅ Surpassed {milestone_name}!")
                    
                    with col3:
                        if 'views_per_day' in row and row['views_per_day'] > 0 and remaining > 0:
                            days_to_milestone = remaining / row['views_per_day']
                            st.metric("Days to Milestone", f"{int(days_to_milestone)}")
                        
                        st.metric("Views/Day", format_number(row.get('views_per_day', 0)))
                    
                    st.markdown("---")

# ============================================================================
# PAGE 5: FORECAST LAB WITH GAP FILLING
# ============================================================================

elif page == "🔬 Forecast Lab":
    st.title("🔬 Forecast Lab - Time Series Analysis")
    
    st.markdown("""
    **Pre-computed time series forecasts** with automatic gap filling:
    
    - **ARIMA**: AutoRegressive Integrated Moving Average
    - **DLM (Kalman Filter)**: Dynamic Linear Model with adaptive trends
    - **Neural Network**: MLP with lag features
    """)
    
    st.markdown("---")
    
    # Settings for gap filling
    with st.expander("⚙️ Forecast Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            aggregation_freq = st.selectbox(
                "Time Aggregation",
                ['H', 'D', 'W'],
                format_func=lambda x: {'H': 'Hourly', 'D': 'Daily', 'W': 'Weekly'}[x],
                index=0
            )
        
        with col2:
            fill_method = st.selectbox(
                "Gap Filling Method",
                ['interpolate', 'ffill', 'zero', 'mean'],
                format_func=lambda x: {
                    'interpolate': 'Linear Interpolation',
                    'ffill': 'Forward Fill',
                    'zero': 'Fill with Zeros',
                    'mean': 'Rolling Mean'
                }[x],
                index=0
            )
        
        # Define the mappings outside the f-string
        freq_labels = {'H': 'Hourly', 'D': 'Daily', 'W': 'Weekly'}
        fill_labels = {
            'interpolate': 'Linear interpolation between data points',
            'ffill': 'Carry last value forward',
            'zero': 'Fill missing periods with zeros',
            'mean': 'Use rolling average'
        }
        
        st.info(f"""
        **Selected Settings:**
        - **Aggregation**: {freq_labels[aggregation_freq]}
        - **Gap Filling**: {fill_labels[fill_method]}
        """)
    
    # PRE-COMPUTE ALL FORECASTS (cached)
    forecast_data = compute_all_forecasts(aggregation_freq, fill_method)
    
    if forecast_data is None:
        st.error("Insufficient data for forecasting. Need at least 30 observations.")
        st.stop()
    
    # Extract pre-computed results
    timestamps = forecast_data['timestamps']
    y_full = forecast_data['y_full']
    forecast_timestamps = forecast_data['forecast_timestamps']
    results = forecast_data['results']
    forecast_steps = forecast_data['forecast_steps']
    last_dt = forecast_data['last_dt']
    target_date = forecast_data['target_date']
    gaps_filled = forecast_data['gaps_filled']
    total_periods = forecast_data['total_periods']
    original_periods = forecast_data['original_periods']
    
    # Display info
    st.success(f"✅ Loaded {len(y_full):,} time periods from {pd.to_datetime(timestamps[0]).strftime('%Y-%m-%d')} to {last_dt.strftime('%Y-%m-%d')}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Periods", f"{total_periods:,}")
    col2.metric("Original Data Points", f"{original_periods:,}")
    col3.metric("Gaps Filled", f"{gaps_filled:,}", delta=f"{(gaps_filled/total_periods*100):.1f}%")
    col4.metric("Forecast To", target_date.strftime("%Y-%m-%d"))
    
    st.markdown("---")
    
    # Determine best model
    model_scores = {}
    for model_key, result in results.items():
        if result and 'metrics' in result:
            rmse = result['metrics'].get('train_rmse', float('inf'))
            if not np.isnan(rmse):
                model_scores[model_key] = rmse
    
    if model_scores:
        best_model_key = min(model_scores, key=model_scores.get)
        best_model_names = {'arima': 'ARIMA', 'dlm': 'DLM (Kalman Filter)', 'nn': 'Neural Network'}
        best_model_name = best_model_names[best_model_key]
        best_rmse = model_scores[best_model_key]
        
        st.markdown(f"""
        <div class="best-model-banner">
            <h2 style='margin: 0; text-align: center;'>🏆 Best Performing Model: {best_model_name}</h2>
            <p style='margin: 10px 0 0 0; text-align: center; font-size: 1.1em;'>
                Training RMSE: {best_rmse:,.0f} (Lower is better)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Create and display individual plots
    if any(results.values()):
        st.subheader("📈 Individual Model Forecasts")
        st.caption("🖱️ **Tip**: Gaps have been filled automatically • Use zoom controls to explore")
        
        model_colors = {'arima': '#FF6B35', 'dlm': '#8B5CF6', 'nn': '#10B981'}
        model_names = {'arima': 'ARIMA', 'dlm': 'DLM (Kalman Filter)', 'nn': 'Neural Network'}
        
        # ARIMA Plot
        if results.get('arima'):
            st.markdown("### 🔶 ARIMA Model")
            fig, config = create_individual_model_plot(
                timestamps=timestamps,
                y_full=y_full,
                result=results['arima'],
                forecast_timestamps=forecast_timestamps,
                model_name='ARIMA',
                color=model_colors['arima']
            )
            st.plotly_chart(fig, use_container_width=True, config=config)
            
            r = results['arima']
            col1, col2, col3 = st.columns(3)
            col1.metric("Training RMSE", f"{r['metrics']['train_rmse']:,.0f}")
            col2.metric("AIC", f"{r['metrics']['aic']:,.0f}")
            col3.metric("Model Order", f"{r.get('order', '(1,1,1)')}")
            st.markdown("---")
        
        # DLM Plot
        if results.get('dlm'):
            st.markdown("### 🔮 DLM (Kalman Filter) Model")
            fig, config = create_individual_model_plot(
                timestamps=timestamps,
                y_full=y_full,
                result=results['dlm'],
                forecast_timestamps=forecast_timestamps,
                model_name='DLM (Kalman Filter)',
                color=model_colors['dlm']
            )
            st.plotly_chart(fig, use_container_width=True, config=config)
            
            r = results['dlm']
            col1, col2 = st.columns(2)
            col1.metric("Training RMSE", f"{r['metrics']['train_rmse']:,.0f}")
            col2.metric("Method", "State-Space (Adaptive)")
            st.markdown("---")
        
        # Neural Network Plot
        if results.get('nn'):
            st.markdown("### 🧠 Neural Network Model")
            fig, config = create_individual_model_plot(
                timestamps=timestamps,
                y_full=y_full,
                result=results['nn'],
                forecast_timestamps=forecast_timestamps,
                model_name='Neural Network',
                color=model_colors['nn']
            )
            st.plotly_chart(fig, use_container_width=True, config=config)
            
            r = results['nn']
            rmse_val = r['metrics']['train_rmse']
            rmse_str = f"{rmse_val:,.0f}" if not np.isnan(rmse_val) else "N/A"
            col1, col2 = st.columns(2)
            col1.metric("Training RMSE", rmse_str)
            col2.metric("Architecture", "MLP (64-32-1)")
            st.markdown("---")
        
        # Model Comparison Table
        st.subheader("📊 Model Performance Comparison")
        
        comparison_data = []
        for model_key in ['arima', 'dlm', 'nn']:
            if results.get(model_key):
                r = results[model_key]
                rmse = r['metrics'].get('train_rmse', np.nan)
                comparison_data.append({
                    'Model': model_names[model_key],
                    'Training RMSE': f"{rmse:,.0f}" if not np.isnan(rmse) else "N/A",
                    'Status': '🏆 Best' if model_key == best_model_key else '✓ Fitted'
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Additional insights
        with st.expander("📖 About Gap Filling & Models"):
            # Define mappings outside f-string
            fill_method_labels = {
                'interpolate': 'Linear Interpolation',
                'ffill': 'Forward Fill',
                'zero': 'Zero Fill',
                'mean': 'Rolling Mean'
            }
            
            st.markdown(f"""
            ### Gap Filling Strategy
            
            **Method Used**: {fill_method_labels[fill_method]}
            
            - **Original Data Points**: {original_periods:,}
            - **Gaps Filled**: {gaps_filled:,} ({(gaps_filled/total_periods*100):.1f}%)
            - **Complete Series**: {total_periods:,} time periods
            
            This ensures models receive a continuous time series without missing periods.
            
            ### Model Descriptions
            
            **ARIMA (AutoRegressive Integrated Moving Average)**
            - Captures linear trends and autocorrelation patterns
            - Order automatically selected using statistical criteria
            - Best for data with clear linear trends
            
            **DLM (Dynamic Linear Model)**
            - Uses Kalman filtering for state-space estimation
            - Adapts to changing trends in real-time
            - Provides smooth, probabilistic forecasts
            
            **Neural Network (MLP)**
            - Learns non-linear patterns from lagged features
            - 64-32 hidden layer architecture
            - Can capture complex temporal dependencies
            
            ### Interpreting Results
            - **RMSE**: Root Mean Squared Error - measures prediction accuracy (lower is better)
            - **Confidence Intervals**: Wider bands indicate higher uncertainty
            - **Best Model**: Selected based on lowest training RMSE
            """)
        
        st.success("✅ All forecasts computed with gaps filled! Refresh to update (cached for 1 hour)")
    else:
        st.error("All models failed to fit. Please check your data and try again.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Project STREAMWATCH - IDS 706 Fall 2025 • Gap-Filled Time Series Forecasting</p>", unsafe_allow_html=True)