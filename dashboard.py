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
    """
    Simple NN forecaster that predicts the next value directly.
    No mean reversion to prevent collapse.
    """
    
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
            
            # Keep prediction bounded (no collapse, no explosion)
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
def load_aggregated_timeseries(days: int = 60):
    """Load aggregated time series for the Forecast Lab"""
    engine = get_db_connection()
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    query = f"""
    SELECT 
        date_trunc('hour', ingestion_timestamp::timestamp) as time_bin,
        SUM(view_count) as total_views,
        SUM(like_count) as total_likes,
        SUM(comment_count) as total_comments,
        COUNT(*) as video_count
    FROM videos_log_v3
    WHERE ingestion_timestamp >= '{cutoff}'
    GROUP BY date_trunc('hour', ingestion_timestamp::timestamp)
    ORDER BY time_bin ASC
    """
    return pd.read_sql(query, engine)


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
# FORECASTING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def fit_all_models(_y_train, _y_test, forecast_steps=168):
    """Fit ARIMA, DLM, and NN models and return results"""
    # Convert to regular arrays to avoid caching issues
    y_train = np.array(_y_train)
    y_test = np.array(_y_test)
    
    results = {}
    
    # ARIMA
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
        
        # Align fitted values with training data
        if len(fitted) < len(y_train):
            fitted_full = np.full(len(y_train), np.nan)
            fitted_full[-len(fitted):] = fitted
            fitted = fitted_full
        
        train_rmse = np.sqrt(np.nanmean((y_train - fitted)**2))
        
        results['arima'] = {
            'fitted': fitted,
            'forecast': forecast,
            'forecast_std': forecast_std,
            'order': order,
            'metrics': {'train_rmse': train_rmse, 'aic': model_fit.aic}
        }
    except Exception as e:
        results['arima'] = None
        st.warning(f"ARIMA fitting failed: {e}")
    
    # DLM
    try:
        dlm = DynamicLinearModel()
        dlm.fit(y_train)
        
        fitted = np.array(dlm.smoothed_means)
        forecast, forecast_vars = dlm.forecast(steps=forecast_steps)
        forecast_std = np.sqrt(forecast_vars)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, fitted))
        
        results['dlm'] = {
            'fitted': fitted,
            'forecast': np.array(forecast),
            'forecast_std': np.array(forecast_std),
            'metrics': {'train_rmse': train_rmse}
        }
    except Exception as e:
        results['dlm'] = None
        st.warning(f"DLM fitting failed: {e}")
    
    # Neural Network
    try:
        lags = min(24, len(y_train)//4)  # Ensure enough data for training
        nn = SimpleNNForecaster(lags=lags)
        fitted = nn.fit(y_train)
        
        if fitted is not None:
            forecast, forecast_std = nn.forecast(y_train, forecast_steps)
            
            # Calculate RMSE on non-NaN fitted values
            valid_mask = ~np.isnan(fitted)
            if np.sum(valid_mask) > 0:
                train_rmse = np.sqrt(mean_squared_error(
                    y_train[valid_mask], fitted[valid_mask]
                ))
            else:
                train_rmse = np.nan
            
            results['nn'] = {
                'fitted': fitted,
                'forecast': forecast,
                'forecast_std': forecast_std,
                'lags': lags,
                'metrics': {'train_rmse': train_rmse}
            }
        else:
            results['nn'] = None
    except Exception as e:
        results['nn'] = None
        st.warning(f"NN fitting failed: {e}")
    
    return results


def create_forecast_plot(timestamps, y_full, train_size, results, forecast_timestamps, metric_name):
    """Create a clean Plotly figure with all three model forecasts"""
    
    fig = go.Figure()
    
    timestamps_train = timestamps[:train_size]
    y_train = y_full[:train_size]
    
    # Plot observed data
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=y_full,
        mode='lines',
        name='Observed',
        line=dict(color='#2E86AB', width=2.5),
        hovertemplate='%{x}<br>Views: %{y:,.0f}<extra></extra>'
    ))
    
    # Colors for each model
    model_colors = {
        'arima': '#FF6B35',
        'dlm': '#8B5CF6', 
        'nn': '#10B981'
    }
    
    model_names = {
        'arima': 'ARIMA',
        'dlm': 'DLM (Kalman)',
        'nn': 'Neural Network'
    }
    
    # Plot each model's forecast
    for model_key in ['arima', 'dlm', 'nn']:
        result = results.get(model_key)
        if result is None:
            continue
        
        forecast = result['forecast']
        forecast_std = result['forecast_std']
        color = model_colors[model_key]
        name = model_names[model_key]
        
        # Main forecast line
        fig.add_trace(go.Scatter(
            x=forecast_timestamps,
            y=forecast,
            mode='lines',
            name=f'{name} Forecast',
            line=dict(color=color, width=2.5, dash='dash'),
            hovertemplate=f'{name}<br>%{{x}}<br>Forecast: %{{y:,.0f}}<extra></extra>'
        ))
        
        # 95% confidence interval
        upper_95 = forecast + 1.96 * forecast_std
        lower_95 = forecast - 1.96 * forecast_std
        lower_95 = np.maximum(lower_95, 0)  # Don't go below 0 for counts
        
        fig.add_trace(go.Scatter(
            x=list(forecast_timestamps) + list(forecast_timestamps)[::-1],
            y=list(upper_95) + list(lower_95)[::-1],
            fill='toself',
            fillcolor=color.replace(')', ', 0.15)').replace('rgb', 'rgba') if 'rgb' in color else f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.15)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'{name} 95% CI',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add vertical line at forecast start
    fig.add_vline(
        x=timestamps[-1],
        line_dash="dash",
        line_color="gray",
        annotation_text="Forecast Start",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=dict(
            text=f'{metric_name.replace("_", " ").title()} Forecast Comparison',
            font=dict(size=20)
        ),
        xaxis_title='Date',
        yaxis_title=metric_name.replace('_', ' ').title(),
        hovermode='x unified',
        height=600,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        template='plotly_white'
    )
    
    return fig


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
# PAGES
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

elif page == "📊 Channel Leaderboard":
    st.title("📊 Channel Leaderboard")
    
    if st.session_state.selected_channel_id:
        channel_id = st.session_state.selected_channel_id
        channel_data = channels_df.filter(pl.col("channel_id") == channel_id).to_dicts()[0]
        
        if st.button("← Back to Leaderboard"):
            st.session_state.selected_channel_id = None
            st.rerun()
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if channel_data['thumbnail_url']:
                st.image(channel_data['thumbnail_url'], width=150)
        
        with col2:
            grade = calculate_grade(channel_data['subscriber_count'], channel_data['view_count'])
            st.markdown(f'<div class="grade-badge">GRADE: {grade}</div>', unsafe_allow_html=True)
            st.markdown(f"# {channel_data['channel_title']}")
            st.markdown(f"### `{channel_data['custom_url']}`")
            st.caption(channel_data['description'] if channel_data['description'] else "")
        
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📺 Subscribers", f"{channel_data['subscriber_count']:,}")
        col2.metric("👀 Total Views", f"{channel_data['view_count']:,}")
        col3.metric("🎬 Videos", f"{channel_data['video_count']:,}")
        col4.metric("🌍 Country", channel_data['country'] or "N/A")
        
        st.markdown("---")
        
        st.subheader(f"🎬 Top Videos from {channel_data['channel_title']}")
        channel_videos = videos_df.filter(pl.col("channel_id") == channel_id).sort("view_count", descending=True).head(10)
        
        for vid_row in channel_videos.iter_rows(named=True):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if vid_row['thumbnail_url']:
                    st.image(vid_row['thumbnail_url'], width=180)
            
            with col2:
                if st.button(f"📹 {vid_row['video_title']}", key=f"vid_{vid_row['video_id']}"):
                    st.session_state.selected_video_id = vid_row['video_id']
                    st.rerun()
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Views", f"{vid_row['view_count']:,}")
                col_b.metric("Engagement", f"{vid_row['engagement_rate']:.2f}%")
                col_c.metric("Views/Day", f"{vid_row['views_per_day']:,.0f}")
            
            st.markdown("---")
    
    else:
        sort_by = st.selectbox("Sort by:", ["Subscribers", "Total Views", "Video Count"])
        
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
                if row['thumbnail_url']:
                    st.image(row['thumbnail_url'], width=100)
            
            with col2:
                if st.button(f"📺 {row['channel_title']}", key=f"ch_{row['channel_id']}"):
                    st.session_state.selected_channel_id = row['channel_id']
                    st.rerun()
                
                st.markdown(f"`{row['custom_url']}`")
                st.caption(row['description'][:200] + "..." if row['description'] else "")
            
            with col3:
                st.metric("Subscribers", f"{row['subscriber_count']:,}")
                st.metric("Total Views", f"{row['view_count']:,}")
                st.metric("Videos", f"{row['video_count']:,}")
            
            st.markdown("---")

elif page == "🎬 Video Explorer":
    st.title("🎬 Video Explorer")
    
    if st.session_state.selected_video_id:
        video_id = st.session_state.selected_video_id
        try:
            video_data = videos_df.filter(pl.col("video_id") == video_id).to_dicts()[0]
        except IndexError:
            st.error("Video not found.")
            st.stop()
        
        if st.button("← Back to Explorer"):
            st.session_state.selected_video_id = None
            st.rerun()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if video_data['thumbnail_url']:
                st.image(video_data['thumbnail_url'], width=350)
        
        with col2:
            st.markdown(f"# {video_data['video_title']}")
            c_title = video_data.get('channel_title', 'Unknown Channel')
            st.markdown(f"## {c_title}")
            st.markdown(f"`{video_data.get('custom_url', 'N/A')}`")
            
            if video_data.get('is_billionaires_watch'):
                st.markdown('<span class="milestone-badge">💎 Billionaires Club (1B+)</span>', unsafe_allow_html=True)
            if video_data.get('is_approaching_milestone'):
                st.markdown(f'<span class="viral-badge">🎯 Approaching {video_data["next_milestone"]:,}</span>', unsafe_allow_html=True)
            if video_data.get('is_highly_viral'):
                st.markdown('<span class="viral-badge">🔥 Highly Viral</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("👀 Views", f"{video_data['view_count']:,}")
        col2.metric("👍 Likes", f"{video_data['like_count']:,}")
        col3.metric("💬 Comments", f"{video_data['comment_count']:,}")
        col4.metric("📊 Engagement", f"{video_data['engagement_rate']:.2f}%")
        col5.metric("⚡ Views/Day", f"{video_data['views_per_day']:,.0f}")
        
        st.markdown("---")
        
        # Milestone tracking
        st.subheader(f"🎯 Milestone Progress: {video_data['milestone_tier']}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Views", f"{video_data['view_count']:,}")
        col2.metric("Next Milestone", f"{video_data['next_milestone']:,}")
        col3.metric("Views Needed", f"{video_data['views_to_next_milestone']:,}")
        
        progress_pct = video_data['milestone_progress_pct'] / 100
        st.progress(progress_pct)
        st.caption(f"{video_data['milestone_progress_pct']:.1f}% progress toward {video_data['next_milestone']:,} views")
    
    else:
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
        
        for row in filtered_videos.head(20).iter_rows(named=True):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if row['thumbnail_url']:
                    st.image(row['thumbnail_url'], width=200)
            
            with col2:
                if st.button(f"🎬 {row['video_title']}", key=f"explore_{row['video_id']}"):
                    st.session_state.selected_video_id = row['video_id']
                    st.rerun()
                
                c_title = row.get('channel_title', 'Unknown Channel')
                st.markdown(f"**{c_title}** • `{row.get('custom_url', 'N/A')}`")
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Views", f"{row['view_count']:,}")
                col_b.metric("Engagement", f"{row['engagement_rate']:.2f}%")
                col_c.metric("Views/Day", f"{row['views_per_day']:,.0f}")
            
            st.markdown("---")

elif page == "🚀 Milestone Tracker":
    st.title("🚀 Milestone Tracker")
    
    st.markdown("### Videos Approaching Major Milestones")
    st.caption("Within 5% of reaching their next milestone tier")
    
    approaching_videos = videos_df.filter(pl.col("is_approaching_milestone") == True).sort("milestone_progress_pct", descending=True)
    
    if len(approaching_videos) == 0:
        st.info("No videos are currently approaching milestones (within 5% threshold)")
    else:
        for row in approaching_videos.iter_rows(named=True):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if row['thumbnail_url']:
                    st.image(row['thumbnail_url'], width=180)
            
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

elif page == "🔬 Forecast Lab":
    st.title("🔬 Forecast Lab - AI Time Series Analysis")
    
    st.markdown("""
    Compare three time series forecasting models on aggregated YouTube view data:
    
    - **ARIMA**: AutoRegressive Integrated Moving Average - captures autocorrelation patterns
    - **DLM (Kalman Filter)**: Dynamic Linear Model with forward-backward smoothing - adapts to changing trends
    - **Neural Network**: MLP with trend-aware features - learns complex nonlinear patterns
    """)
    
    st.markdown("---")
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        history_days = st.slider("Historical Data (days)", 14, 90, 60)
    
    with col2:
        forecast_days = st.slider("Forecast Horizon (days)", 1, 14, 7)
    
    with col3:
        train_split = st.slider("Train/Test Split (%)", 70, 95, 85)
    
    # Load and prepare data
    with st.spinner("Loading aggregated time series data..."):
        ts_data = load_aggregated_timeseries(days=history_days)
    
    if len(ts_data) < 30:
        st.error(f"Insufficient data: only {len(ts_data)} time points. Need at least 30 hours.")
        st.stop()
    
    st.success(f"✅ Loaded {len(ts_data)} hourly observations from {ts_data['time_bin'].min()} to {ts_data['time_bin'].max()}")
    
    # Prepare time series
    timestamps = pd.to_datetime(ts_data['time_bin']).values
    y_full = ts_data['total_views'].values
    
    train_size = int(len(y_full) * train_split / 100)
    test_size = len(y_full) - train_size
    
    y_train = y_full[:train_size]
    y_test = y_full[train_size:]
    
    forecast_steps = forecast_days * 24  # Convert to hours
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Points", f"{train_size:,}")
    col2.metric("Test Points", f"{test_size:,}")
    col3.metric("Forecast Steps", f"{forecast_steps:,} hours")
    
    st.markdown("---")
    
    if st.button("🚀 Run All Models", type="primary", use_container_width=True):
        
        with st.spinner("Training models... This may take a moment..."):
            results = fit_all_models(y_train, y_test, forecast_steps)
        
        # Generate forecast timestamps
        last_dt = pd.to_datetime(timestamps[-1])
        forecast_timestamps = pd.date_range(start=last_dt, periods=forecast_steps + 1, freq='h')[1:]
        
        # Create main comparison plot
        st.subheader("📈 Model Comparison")
        
        fig = create_forecast_plot(
            timestamps=timestamps,
            y_full=y_full,
            train_size=train_size,
            results=results,
            forecast_timestamps=forecast_timestamps,
            metric_name='total_views'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance cards
        st.subheader("📊 Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if results.get('arima'):
                r = results['arima']
                st.markdown(f"""
                <div class="model-card model-card-arima">
                    <h3>🔶 ARIMA{r.get('order', '(1,1,1)')}</h3>
                    <p><strong>Train RMSE:</strong> {r['metrics']['train_rmse']:,.0f}</p>
                    <p><strong>AIC:</strong> {r['metrics']['aic']:,.0f}</p>
                    <p><strong>7-Day Forecast:</strong> {r['forecast'][-1]:,.0f} views</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("ARIMA model failed")
        
        with col2:
            if results.get('dlm'):
                r = results['dlm']
                st.markdown(f"""
                <div class="model-card model-card-dlm">
                    <h3>🔮 DLM (Kalman)</h3>
                    <p><strong>Train RMSE:</strong> {r['metrics']['train_rmse']:,.0f}</p>
                    <p><strong>Adaptive:</strong> Yes (state-space)</p>
                    <p><strong>7-Day Forecast:</strong> {r['forecast'][-1]:,.0f} views</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("DLM model failed")
        
        with col3:
            if results.get('nn'):
                r = results['nn']
                rmse_val = r['metrics']['train_rmse']
                rmse_str = f"{rmse_val:,.0f}" if not np.isnan(rmse_val) else "N/A"
                st.markdown(f"""
                <div class="model-card model-card-nn">
                    <h3>🧠 Neural Network</h3>
                    <p><strong>Train RMSE:</strong> {rmse_str}</p>
                    <p><strong>Architecture:</strong> MLP (128-64-32)</p>
                    <p><strong>7-Day Forecast:</strong> {r['forecast'][-1]:,.0f} views</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("NN model failed")
        
        st.markdown("---")
        
        # Individual model plots
        st.subheader("🔍 Individual Model Details")
        
        tabs = st.tabs(["ARIMA", "DLM (Kalman)", "Neural Network"])
        
        with tabs[0]:
            if results.get('arima'):
                r = results['arima']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps, y=y_full,
                    mode='lines', name='Observed',
                    line=dict(color='#2E86AB', width=2)
                ))
                
                fitted_times = timestamps[len(timestamps)-len(r['fitted']):]
                fig.add_trace(go.Scatter(
                    x=fitted_times, y=r['fitted'],
                    mode='lines', name='Fitted',
                    line=dict(color='#FF6B35', width=2, dash='dot')
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_timestamps, y=r['forecast'],
                    mode='lines', name='Forecast',
                    line=dict(color='#FF6B35', width=2.5)
                ))
                
                upper = r['forecast'] + 1.96 * r['forecast_std']
                lower = np.maximum(r['forecast'] - 1.96 * r['forecast_std'], 0)
                
                fig.add_trace(go.Scatter(
                    x=list(forecast_timestamps) + list(forecast_timestamps)[::-1],
                    y=list(upper) + list(lower)[::-1],
                    fill='toself', fillcolor='rgba(255,107,53,0.2)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='95% CI', showlegend=True
                ))
                
                fig.update_layout(title='ARIMA Model Detail', height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"**Model Order:** ARIMA{r.get('order', '(1,1,1)')} | **AIC:** {r['metrics']['aic']:,.0f}")
        
        with tabs[1]:
            if results.get('dlm'):
                r = results['dlm']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps, y=y_full,
                    mode='lines', name='Observed',
                    line=dict(color='#2E86AB', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=timestamps[:train_size], y=r['fitted'],
                    mode='lines', name='Smoothed (Kalman)',
                    line=dict(color='#8B5CF6', width=2, dash='dot')
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_timestamps, y=r['forecast'],
                    mode='lines', name='Forecast',
                    line=dict(color='#8B5CF6', width=2.5)
                ))
                
                upper = r['forecast'] + 1.96 * r['forecast_std']
                lower = np.maximum(r['forecast'] - 1.96 * r['forecast_std'], 0)
                
                fig.add_trace(go.Scatter(
                    x=list(forecast_timestamps) + list(forecast_timestamps)[::-1],
                    y=list(upper) + list(lower)[::-1],
                    fill='toself', fillcolor='rgba(139,92,246,0.2)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='95% CI', showlegend=True
                ))
                
                fig.update_layout(title='DLM (Kalman Filter) Detail', height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("**Method:** Forward-backward Kalman filtering with Rauch-Tung-Striebel smoothing")
        
        with tabs[2]:
            if results.get('nn'):
                r = results['nn']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps, y=y_full,
                    mode='lines', name='Observed',
                    line=dict(color='#2E86AB', width=2)
                ))
                
                # NN fitted values are only for training data
                fitted = r['fitted']
                if fitted is not None:
                    fitted = np.array(fitted)
                    valid_mask = ~np.isnan(fitted)
                    # Use only training timestamps (fitted is length train_size)
                    train_timestamps = timestamps[:len(fitted)]
                    if np.any(valid_mask):
                        fig.add_trace(go.Scatter(
                            x=train_timestamps[valid_mask], 
                            y=fitted[valid_mask],
                            mode='lines', name='Fitted (Training)',
                            line=dict(color='#10B981', width=2, dash='dot')
                        ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_timestamps, y=r['forecast'],
                    mode='lines', name='Forecast',
                    line=dict(color='#10B981', width=2.5)
                ))
                
                upper = r['forecast'] + 1.96 * r['forecast_std']
                lower = np.maximum(r['forecast'] - 1.96 * r['forecast_std'], 0)
                
                fig.add_trace(go.Scatter(
                    x=list(forecast_timestamps) + list(forecast_timestamps)[::-1],
                    y=list(upper) + list(lower)[::-1],
                    fill='toself', fillcolor='rgba(16,185,129,0.2)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='95% CI', showlegend=True
                ))
                
                # Add train/test split line
                fig.add_vline(x=timestamps[train_size-1], line_dash="dash", 
                             line_color="gray", annotation_text="Train End")
                
                fig.update_layout(title='Neural Network Detail', height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("**Method:** MLP (128-64-32) trained on log-transformed data with cyclical hour encoding and mean-reversion regularization")
        
        st.success("✅ Forecast analysis complete!")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Project STREAMWATCH - IDS 706 Fall 2025 • Powered by ARIMA, Kalman Filter & Neural Network Models</p>", unsafe_allow_html=True)