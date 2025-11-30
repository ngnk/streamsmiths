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
def load_aggregated_timeseries():
    """Load ALL aggregated time series for the Forecast Lab - EXACTLY like ml_forecasting.py"""
    engine = get_db_connection()
    query = """
    SELECT 
        -- Prefer published_at when available, otherwise fall back to ingestion_timestamp
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
        st.error(f"ARIMA fitting failed: {e}")
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
        st.error(f"DLM fitting failed: {e}")
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
        st.error(f"NN fitting failed: {e}")
        return None


# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def create_forecast_plot(timestamps, y_full, train_size, results, forecast_timestamps, metric_name):
    """Create interactive Plotly forecast visualization - matches ml_forecasting style"""
    
    fig = go.Figure()
    
    # Convert ALL timestamps to pandas datetime immediately
    timestamps_list = pd.to_datetime(timestamps).tolist()
    forecast_timestamps_list = pd.to_datetime(forecast_timestamps).tolist()
    
    # Plot observed data (all historical)
    fig.add_trace(go.Scatter(
        x=timestamps_list,
        y=y_full,
        mode='lines',
        name='Observed Data',
        line=dict(color='#2E86AB', width=2.5),
        hovertemplate='<b>Observed</b><br>Date: %{x|%Y-%m-%d %H:%M}<br>Views: %{y:,.0f}<extra></extra>'
    ))
    
    # Model colors
    model_colors = {'arima': '#FF6B35', 'dlm': '#8B5CF6', 'nn': '#10B981'}
    model_names = {'arima': 'ARIMA', 'dlm': 'DLM (Kalman)', 'nn': 'Neural Network'}
    
    # Plot fitted values for each model (overlay on historical data)
    for model_key in ['arima', 'dlm', 'nn']:
        result = results.get(model_key)
        if result is None:
            continue
        
        fitted = result.get('fitted')
        if fitted is None:
            fitted = result.get('fitted_train')
        
        if fitted is not None:
            color = model_colors[model_key]
            name = model_names[model_key]
            
            # Plot fitted line
            fitted = np.array(fitted)
            valid = ~np.isnan(fitted)
            if np.any(valid):
                hover_template = '<b>' + name + ' Fitted</b><br>Date: %{x|%Y-%m-%d %H:%M}<br>Fitted: %{y:,.0f}<extra></extra>'
                fig.add_trace(go.Scatter(
                    x=[timestamps_list[i] for i in range(len(timestamps_list)) if valid[i]],
                    y=fitted[valid],
                    mode='lines',
                    name=f'{name} Fitted',
                    line=dict(color=color, width=1.5, dash='dot'),
                    opacity=0.7,
                    hovertemplate=hover_template
                ))
    
    # Plot each model's forecast
    for model_key in ['arima', 'dlm', 'nn']:
        result = results.get(model_key)
        if result is None:
            continue
        
        forecast = result['forecast']
        forecast_std = result['forecast_std']
        color = model_colors[model_key]
        name = model_names[model_key]
        
        # Main forecast line - FIXED: Use string template instead of f-string for hovertemplate
        hover_template = '<b>' + name + '</b><br>Date: %{x|%Y-%m-%d %H:%M}<br>Forecast: %{y:,.0f}<extra></extra>'
        
        fig.add_trace(go.Scatter(
            x=forecast_timestamps_list,
            y=forecast,
            mode='lines+markers',
            name=f'{name} Forecast',
            line=dict(color=color, width=2.5, dash='dash'),
            marker=dict(size=4, symbol='circle'),
            hovertemplate=hover_template
        ))
        
        # 95% confidence interval
        upper_95 = forecast + 1.96 * forecast_std
        lower_95 = np.maximum(forecast - 1.96 * forecast_std, 0)
        
        fig.add_trace(go.Scatter(
            x=forecast_timestamps_list + forecast_timestamps_list[::-1],
            y=list(upper_95) + list(lower_95)[::-1],
            fill='toself',
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'{name} 95% CI',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add vertical line at forecast start - use the last timestamp from the list
    fig.add_shape(
        type="line",
        x0=timestamps_list[-1],
        x1=timestamps_list[-1],
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=timestamps_list[-1],
        y=1,
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        yanchor="bottom"
    )
    
    # Layout - Default to showing ALL data
    fig.update_layout(
        title=dict(text=f'{metric_name.replace("_", " ").title()} Forecast', font=dict(size=22)),
        xaxis_title='Date',
        yaxis_title=metric_name.replace('_', ' ').title(),
        hovermode='x unified',
        height=700,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        template='plotly_white',
        xaxis=dict(
            # Set initial range to show ALL data (from first timestamp to last forecast)
            range=[timestamps_list[0], forecast_timestamps_list[-1]],
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ],
                bgcolor='rgba(150, 150, 150, 0.1)'
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
            'filename': f'forecast_{metric_name}',
            'height': 700,
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
# PAGES (HOME, LEADERBOARD, VIDEO EXPLORER, MILESTONE TRACKER)
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

elif page == "🔬 Forecast Lab":
    st.title("🔬 Forecast Lab - AI Time Series Analysis")
    
    st.markdown("""
    Compare three time series forecasting models on aggregated YouTube view data:
    
    - **ARIMA**: AutoRegressive Integrated Moving Average
    - **DLM (Kalman Filter)**: Dynamic Linear Model with adaptive trends
    - **Neural Network**: MLP with lag features
    
    **This matches ml_forecasting.py EXACTLY**: Loads ALL data, forecasts to 2026-01-31
    """)
    
    st.markdown("---")
    
    # Load and prepare data - ALL DATA like ml_forecasting.py
    with st.spinner("Loading ALL aggregated time series data..."):
        ts_data = load_aggregated_timeseries()
    
    if len(ts_data) < 30:
        st.error(f"Insufficient data: only {len(ts_data)} time points. Need at least 30 hours.")
        st.stop()
    
    st.success(f"✅ Loaded {len(ts_data)} hourly observations from {ts_data['time_bin'].min()} to {ts_data['time_bin'].max()}")
    
    # Prepare time series
    timestamps = pd.to_datetime(ts_data['time_bin']).values
    y_full = ts_data['total_views'].values
    
    # No train/test split - using all data for fitting
    train_size = len(y_full)  # All data used for fitting
    
    # Calculate forecast steps EXACTLY like ml_forecasting.py
    last_dt = pd.to_datetime(timestamps[-1])
    target_date = pd.Timestamp('2026-01-31')
    hours_to_forecast = int((target_date - last_dt).total_seconds() / 3600)
    hours_to_forecast = max(hours_to_forecast, 24 * 60)  # At least 60 days
    forecast_steps = hours_to_forecast
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data Points", f"{len(y_full):,}")
    col2.metric("Forecast Steps", f"{forecast_steps:,} hours")
    col3.metric("Forecast To", target_date.strftime("%Y-%m-%d"))
    
    st.markdown("---")
    
    # MODEL SELECTION
    st.subheader("🎯 Select Models to Run")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        run_arima = st.checkbox("🔶 ARIMA", value=True)
    with col2:
        run_dlm = st.checkbox("🔮 DLM (Kalman)", value=True)
    with col3:
        run_nn = st.checkbox("🧠 Neural Network", value=True)
    with col4:
        run_all = st.button("🚀 Run Selected Models", type="primary", use_container_width=True)
    
    if run_all:
        results = {}
        
        # Generate extended forecast timestamps (like ml_forecasting.py)
        last_dt = pd.to_datetime(timestamps[-1])
        forecast_timestamps = pd.date_range(start=last_dt, periods=forecast_steps + 1, freq='h')[1:]
        
        # Fit selected models ON ALL DATA (not just train) for extended forecasts
        if run_arima:
            with st.spinner("Training ARIMA model on full dataset..."):
                results['arima'] = fit_arima_model(y_full, forecast_steps)
        
        if run_dlm:
            with st.spinner("Training DLM (Kalman Filter) model on full dataset..."):
                results['dlm'] = fit_dlm_model(y_full, forecast_steps)
        
        if run_nn:
            with st.spinner("Training Neural Network model on full dataset..."):
                results['nn'] = fit_nn_model(y_full, forecast_steps)
        
        # Create plot if at least one model succeeded
        if any(results.values()):
            st.subheader("📈 Forecast Comparison (Interactive)")
            st.caption("🖱️ Hover, zoom, pan, and use the range selector below the chart")
            
            fig, config = create_forecast_plot(
                timestamps=timestamps,
                y_full=y_full,
                train_size=train_size,
                results=results,
                forecast_timestamps=forecast_timestamps,
                metric_name='total_views'
            )
            
            st.plotly_chart(fig, use_container_width=True, config=config)
            
            # Model performance cards
            st.subheader("📊 Model Performance")
            
            cols = st.columns(3)
            
            if results.get('arima'):
                with cols[0]:
                    r = results['arima']
                    st.markdown(f"""
                    <div class="model-card model-card-arima">
                        <h3>🔶 ARIMA{r.get('order', '(1,1,1)')}</h3>
                        <p><strong>Fit RMSE:</strong> {r['metrics']['train_rmse']:,.0f}</p>
                        <p><strong>AIC:</strong> {r['metrics']['aic']:,.0f}</p>
                        <p><strong>Forecast End:</strong> {r['forecast'][-1]:,.0f} views</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            if results.get('dlm'):
                with cols[1]:
                    r = results['dlm']
                    st.markdown(f"""
                    <div class="model-card model-card-dlm">
                        <h3>🔮 DLM (Kalman)</h3>
                        <p><strong>Fit RMSE:</strong> {r['metrics']['train_rmse']:,.0f}</p>
                        <p><strong>Adaptive:</strong> Yes (state-space)</p>
                        <p><strong>Forecast End:</strong> {r['forecast'][-1]:,.0f} views</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            if results.get('nn'):
                with cols[2]:
                    r = results['nn']
                    rmse_val = r['metrics']['train_rmse']
                    rmse_str = f"{rmse_val:,.0f}" if not np.isnan(rmse_val) else "N/A"
                    st.markdown(f"""
                    <div class="model-card model-card-nn">
                        <h3>🧠 Neural Network</h3>
                        <p><strong>Fit RMSE:</strong> {rmse_str}</p>
                        <p><strong>Architecture:</strong> MLP (64-32)</p>
                        <p><strong>Forecast End:</strong> {r['forecast'][-1]:,.0f} views</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.success("✅ Forecast analysis complete!")
        else:
            st.error("All models failed to fit. Please check your data and try again.")

# Note: Other pages (Channel Leaderboard, Video Explorer, Milestone Tracker) remain the same
# I've abbreviated them here for space. Copy from the full dashboard_ULTIMATE.py if needed.

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Project STREAMWATCH - IDS 706 Fall 2025 • Using ALL available data points</p>", unsafe_allow_html=True)