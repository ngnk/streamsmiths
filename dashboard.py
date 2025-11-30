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
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TIME SERIES FORECASTING - ARIMA & DLM
# ============================================================================

class DynamicLinearModel:
    """Dynamic Linear Model using Kalman Filter"""
    
    def __init__(self, obs_variance=None, state_variance=None):
        self.obs_variance = obs_variance
        self.state_variance = state_variance
        self.filtered_means = None
        self.filtered_vars = None
        self.smoothed_means = None
        self.smoothed_vars = None
    
    def _initialize_variances(self, y):
        if self.obs_variance is None:
            self.obs_variance = np.var(y) * 0.1
        if self.state_variance is None:
            diffs = np.diff(y)
            self.state_variance = np.var(diffs) * 0.5 if len(diffs) > 0 else 1.0
    
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
            forecasts[h] = last_mean
            forecast_vars[h] = last_var + (h + 1) * self.state_variance + self.obs_variance
        
        return forecasts, forecast_vars

@st.cache_data(ttl=3600)
def fit_forecasting_models(video_id: str, history_df: pd.DataFrame, forecast_days: int = 7):
    """
    Fit both ARIMA and DLM models for forecasting.
    Returns dict with both model results.
    """
    if len(history_df) < 10:
        return None
    
    # Prepare data
    df = history_df.copy()
    df['ingestion_timestamp'] = pd.to_datetime(df['ingestion_timestamp'])
    df = df.sort_values('ingestion_timestamp')
    
    y = df['view_count'].values
    dates = df['ingestion_timestamp'].values
    
    # Split for validation
    train_size = int(len(y) * 0.85)
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    results = {}
    
    # Fit ARIMA
    try:
        arima = ARIMA(y_train, order=(1, 1, 1))
        arima_fit = arima.fit()
        
        # Forecast
        forecast_obj = arima_fit.get_forecast(steps=forecast_days)
        arima_forecast = forecast_obj.predicted_mean
        arima_std = forecast_obj.se_mean
        
        results['arima'] = {
            'fitted': arima_fit.fittedvalues,
            'forecast': arima_forecast,
            'forecast_std': arima_std,
            'train_size': train_size
        }
    except:
        results['arima'] = None
    
    # Fit DLM
    try:
        dlm = DynamicLinearModel()
        dlm.fit(y_train)
        
        dlm_forecast, dlm_vars = dlm.forecast(steps=forecast_days)
        dlm_std = np.sqrt(dlm_vars)
        
        results['dlm'] = {
            'fitted': dlm.smoothed_means,
            'forecast': dlm_forecast,
            'forecast_std': dlm_std,
            'train_size': train_size
        }
    except:
        results['dlm'] = None
    
    results['dates'] = dates
    results['y_full'] = y
    
    return results

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
    
    # Clean the connection string (remove 'psql' prefix if present)
    db_uri = db_uri.strip()
    if db_uri.startswith("psql "):
        db_uri = db_uri[5:].strip()
    db_uri = db_uri.strip("'\"")
    
    return create_engine(db_uri, pool_pre_ping=True, pool_recycle=1800)

# Data loading functions
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

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["🏠 Home", "📊 Channel Leaderboard", "🎬 Video Explorer", "🚀 Milestone Tracker", "📈 Forecast Lab"]
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
        
        st.markdown("---")
        
        # ML Forecasting Section
        st.subheader("📈 AI-Powered View Count Forecast")
        
        history = load_video_history(video_id, days=30)
        
        if len(history) >= 10:
            df_history = history.to_pandas()
            
            # Fit models
            with st.spinner("🤖 Training ARIMA and DLM models..."):
                forecast_results = fit_forecasting_models(video_id, df_history, forecast_days=14)
            
            if forecast_results:
                # Create comprehensive forecast plot
                fig = go.Figure()
                
                dates = pd.to_datetime(df_history['ingestion_timestamp'])
                y_full = df_history['view_count'].values
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=y_full,
                    mode='lines+markers',
                    name='Actual Views',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=8)
                ))
                
                # Generate future dates for forecast
                last_date = dates.iloc[-1]
                future_dates = pd.date_range(start=last_date, periods=15, freq='D')[1:]
                
                # ARIMA forecast
                if forecast_results['arima']:
                    arima_forecast = forecast_results['arima']['forecast']
                    arima_std = forecast_results['arima']['forecast_std']
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=arima_forecast,
                        mode='lines',
                        name='ARIMA Forecast',
                        line=dict(color='#ff7f0e', width=3, dash='dash')
                    ))
                    
                    # 95% confidence interval
                    fig.add_trace(go.Scatter(
                        x=future_dates.tolist() + future_dates.tolist()[::-1],
                        y=(arima_forecast + 1.96*arima_std).tolist() + (arima_forecast - 1.96*arima_std).tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255,127,14,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='ARIMA 95% CI',
                        showlegend=True
                    ))
                
                # DLM forecast
                if forecast_results['dlm']:
                    dlm_forecast = forecast_results['dlm']['forecast']
                    dlm_std = forecast_results['dlm']['forecast_std']
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=dlm_forecast,
                        mode='lines',
                        name='DLM Forecast',
                        line=dict(color='#d62728', width=3, dash='dot')
                    ))
                    
                    # 95% confidence interval
                    fig.add_trace(go.Scatter(
                        x=future_dates.tolist() + future_dates.tolist()[::-1],
                        y=(dlm_forecast + 1.96*dlm_std).tolist() + (dlm_forecast - 1.96*dlm_std).tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(214,39,40,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='DLM 95% CI',
                        showlegend=True
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
                    title="14-Day View Count Forecast (ARIMA vs DLM)",
                    xaxis_title="Date",
                    yaxis_title="View Count",
                    hovermode='x unified',
                    height=600,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast summary cards
                st.markdown("### 📊 Forecast Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if forecast_results['arima']:
                        arima_14day = forecast_results['arima']['forecast'][-1]
                        st.markdown(f"""
                        <div class="forecast-card">
                            <h3>🔮 ARIMA Model</h3>
                            <h2>{int(arima_14day):,} views</h2>
                            <p>Expected in 14 days</p>
                            <p>Growth: +{int(arima_14day - y_full[-1]):,} views</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if forecast_results['dlm']:
                        dlm_14day = forecast_results['dlm']['forecast'][-1]
                        st.markdown(f"""
                        <div class="forecast-card">
                            <h3>🎯 DLM Model</h3>
                            <h2>{int(dlm_14day):,} views</h2>
                            <p>Expected in 14 days</p>
                            <p>Growth: +{int(dlm_14day - y_full[-1]):,} views</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.info("💡 **Model Info**: ARIMA captures autoregressive patterns, while DLM (Kalman Filter) adapts to changing trends. Compare both forecasts for robust predictions.")
            else:
                st.warning("Unable to fit forecasting models. Need more data points.")
        else:
            st.info("Not enough historical data for AI forecasting (need at least 10 snapshots)")
        
        st.markdown("---")
        
        # Engagement analysis
        st.subheader("💬 Engagement Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Engagement Ratios:**")
            st.metric("Like-to-View", f"{video_data['like_view_ratio']:.4f}")
            st.metric("Comment-to-View", f"{video_data['comment_view_ratio']:.4f}")
            st.metric("Like-to-Comment", f"{video_data['like_comment_ratio']:.2f}")
        
        with col2:
            st.markdown("**Performance Context:**")
            st.metric("Video Age", f"{video_data['days_since_publish']} days")
            st.metric("Duration", f"{video_data['duration_seconds']//60} min {video_data['duration_seconds']%60} sec")
    
    else:
        # Explorer view
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

elif page == "📈 Forecast Lab":
    st.title("📈 Forecast Lab - AI Time Series Analysis")
    
    st.markdown("""
    This page provides advanced time series forecasting using two statistical models:
    - **ARIMA(1,1,1)**: AutoRegressive Integrated Moving Average
    - **DLM**: Dynamic Linear Model (Kalman Filter with forward-backward smoothing)
    """)
    
    st.markdown("---")
    
    # Video selection
    st.subheader("Select a Video for Forecasting")
    
    # Get top videos
    top_videos = videos_df.sort("view_count", descending=True).head(50)
    
    video_options = {}
    for row in top_videos.iter_rows(named=True):
        label = f"{row['video_title'][:60]}... ({row['view_count']:,} views)"
        video_options[label] = row['video_id']
    
    selected_label = st.selectbox("Choose a video:", list(video_options.keys()))
    selected_video_id = video_options[selected_label]
    
    if st.button("🚀 Generate Forecast"):
        video_data = videos_df.filter(pl.col("video_id") == selected_video_id).to_dicts()[0]
        
        st.markdown(f"### 📊 Forecasting: {video_data['video_title']}")
        
        history = load_video_history(selected_video_id, days=30)
        
        if len(history) >= 10:
            df_history = history.to_pandas()
            
            with st.spinner("🤖 Training models... This may take a moment..."):
                forecast_results = fit_forecasting_models(selected_video_id, df_history, forecast_days=14)
            
            if forecast_results:
                # Create faceted visualization
                fig = go.Figure()
                
                dates = pd.to_datetime(df_history['ingestion_timestamp'])
                y_full = df_history['view_count'].values
                
                # Historical
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=y_full,
                    mode='lines+markers',
                    name='Historical Views',
                    line=dict(color='#1f77b4', width=3)
                ))
                
                last_date = dates.iloc[-1]
                future_dates = pd.date_range(start=last_date, periods=15, freq='D')[1:]
                
                # ARIMA
                if forecast_results['arima']:
                    arima_fc = forecast_results['arima']['forecast']
                    arima_std = forecast_results['arima']['forecast_std']
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=arima_fc,
                        mode='lines',
                        name='ARIMA',
                        line=dict(color='#ff7f0e', width=3, dash='dash')
                    ))
                    
                    # Confidence bands
                    for alpha, z in [(0.1, 1.96), (0.2, 1.28), (0.3, 0.67)]:
                        fig.add_trace(go.Scatter(
                            x=future_dates.tolist() + future_dates.tolist()[::-1],
                            y=(arima_fc + z*arima_std).tolist() + (arima_fc - z*arima_std).tolist()[::-1],
                            fill='toself',
                            fillcolor=f'rgba(255,127,14,{alpha})',
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                
                # DLM
                if forecast_results['dlm']:
                    dlm_fc = forecast_results['dlm']['forecast']
                    dlm_std = forecast_results['dlm']['forecast_std']
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=dlm_fc,
                        mode='lines',
                        name='DLM (Kalman)',
                        line=dict(color='#d62728', width=3, dash='dot')
                    ))
                    
                    # Confidence bands
                    for alpha, z in [(0.1, 1.96), (0.2, 1.28), (0.3, 0.67)]:
                        fig.add_trace(go.Scatter(
                            x=future_dates.tolist() + future_dates.tolist()[::-1],
                            y=(dlm_fc + z*dlm_std).tolist() + (dlm_fc - z*dlm_std).tolist()[::-1],
                            fill='toself',
                            fillcolor=f'rgba(214,39,40,{alpha})',
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                
                fig.update_layout(
                    title="14-Day Forecast Comparison",
                    xaxis_title="Date",
                    yaxis_title="View Count",
                    height=700,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Model comparison
                st.markdown("### 📊 Model Comparison")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Current Status**")
                    st.metric("Current Views", f"{y_full[-1]:,}")
                    st.metric("Days of Data", len(y_full))
                
                with col2:
                    if forecast_results['arima']:
                        st.markdown("**ARIMA Forecast**")
                        arima_14 = forecast_results['arima']['forecast'][-1]
                        st.metric("14-Day Forecast", f"{int(arima_14):,}")
                        st.metric("Expected Growth", f"+{int(arima_14 - y_full[-1]):,}")
                
                with col3:
                    if forecast_results['dlm']:
                        st.markdown("**DLM Forecast**")
                        dlm_14 = forecast_results['dlm']['forecast'][-1]
                        st.metric("14-Day Forecast", f"{int(dlm_14):,}")
                        st.metric("Expected Growth", f"+{int(dlm_14 - y_full[-1]):,}")
                
                st.success("✅ Forecast complete! Use these predictions to inform content strategy and milestone planning.")
            else:
                st.error("Unable to fit models. Try a video with more historical data.")
        else:
            st.warning("Not enough data for forecasting. Need at least 10 snapshots (typically 10+ days of data).")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Project STREAMWATCH - IDS 706 Fall 2025 • Powered by ARIMA & Kalman Filter ML Models</p>", unsafe_allow_html=True)