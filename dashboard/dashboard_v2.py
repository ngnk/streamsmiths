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

# Trending tab dependencies
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from typing import Optional


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
</style>
""", unsafe_allow_html=True)

# Database connection
@st.cache_resource
def get_db_connection():
    """Connect to Neon database with auto-recovery"""
    db_uri = os.getenv("NEON_DATABASE_URL")
    if not db_uri:
        st.error("⚠️ NEON_DATABASE_URL not found. Please set it in your environment.")
        st.stop()
    
    # FIX: Added pool_pre_ping and recycle to prevent PendingRollbackError
    return create_engine(db_uri, pool_pre_ping=True, pool_recycle=1800)






# ================= TRENDING DATA FUNCTIONS (NO neon_utils.py) =================

@st.cache_data(show_spinner=True)
def get_trending_data(days: int, region: str) -> pd.DataFrame:
    """
    Load trending videos from the v3 tables and attach channel titles.
    """
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
    df["ingestion_timestamp"] = pd.to_datetime(df["ingestion_timestamp"])
    df = df.sort_values("ingestion_timestamp").reset_index(drop=True)

    # ---- add velocity features ----
    df["view_delta"] = df.groupby("video_id")["view_count"].diff()

    time_diff = df.groupby("video_id")["ingestion_timestamp"].diff()
    df["time_delta_hours"] = time_diff.dt.total_seconds() / 3600.0

    df["view_velocity_per_hour"] = df["view_delta"] / df["time_delta_hours"]
    df.loc[~np.isfinite(df["view_velocity_per_hour"]), "view_velocity_per_hour"] = np.nan

    return df




def get_trending_leaderboard_figure(df: pd.DataFrame):
    latest_ts = df["ingestion_timestamp"].max()
    df_latest = df[df["ingestion_timestamp"] == latest_ts].copy()

    # DO NOT recompute channel_display here – use the one from SQL
    # no df_latest["channel_display"] = ... line

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
        title="Creator leaderboard (latest trending snapshot)",
        labels={"total_views": "Total views", "channel_display": "Channel"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig, channel_stats




def train_and_eval_trending_regression(df: pd.DataFrame):
    max_ts = df["ingestion_timestamp"].max()
    cutoff = max_ts - pd.Timedelta(days=2)

    train_df = df[df["ingestion_timestamp"] < cutoff].copy()
    test_df = df[df["ingestion_timestamp"] >= cutoff].copy()

    feature_cols = [
        "views_per_day",
        "days_since_publish",
        "engagement_rate",
        "duration_seconds",
    ]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["view_count"]

    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["view_count"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    test_df = test_df.copy()
    test_df["predicted_view_count"] = model.predict(X_test)

    errors = y_test - test_df["predicted_view_count"]
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    r2 = 1 - np.sum(errors**2) / np.sum((y_test - y_test.mean())**2)

    fig_pred_vs_actual = px.scatter(
        test_df,
        x="predicted_view_count",
        y="view_count",
        hover_data=["video_id", "video_title"],
        title="Predicted vs actual view count (last 2 days test set)",
        labels={"predicted_view_count": "Predicted", "view_count": "Actual"},
    )

    min_v = min(test_df["predicted_view_count"].min(), test_df["view_count"].min())
    max_v = max(test_df["predicted_view_count"].max(), test_df["view_count"].max())

    fig_pred_vs_actual.add_shape(
        type="line",
        x0=min_v,
        y0=min_v,
        x1=max_v,
        y1=max_v,
    )

    fig_errors = px.histogram(
        x=errors,
        nbins=30,
        title="Prediction errors (actual - predicted, last 2 days)",
        labels={"x": "Error = actual - predicted"},
    )

    return {
        "train_size": len(train_df),
        "test_size": len(test_df),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "fig_pred_vs_actual": fig_pred_vs_actual,
        "fig_errors": fig_errors,
        "test_df": test_df,
    }


def live_trend_figure_trending(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_video_id: Optional[str] = None,
):
    if len(test_df) == 0:
        return None, None

    if target_video_id is None:
        target_video_id = test_df["video_id"].value_counts().idxmax()

    df_video_all = df[df["video_id"] == target_video_id].sort_values(
        "ingestion_timestamp"
    )
    test_video = test_df[test_df["video_id"] == target_video_id].sort_values(
        "ingestion_timestamp"
    )

    fig_live_pred = go.Figure()

    fig_live_pred.add_trace(
        go.Scatter(
            x=df_video_all["ingestion_timestamp"],
            y=df_video_all["view_count"],
            mode="lines+markers",
            name="Actual view count",
        )
    )

    if len(test_video) > 0:
        fig_live_pred.add_trace(
            go.Scatter(
                x=test_video["ingestion_timestamp"],
                y=test_video["predicted_view_count"],
                mode="lines+markers",
                name="Predicted view count",
            )
        )

    title_video = (
        df_video_all["video_title"].iloc[0]
        if len(df_video_all) > 0
        else target_video_id
    )

    fig_live_pred.update_layout(
        title=f"Live view count trends with prediction - {title_video}",
        xaxis_title="Ingestion time",
        yaxis_title="View count",
    )

    return fig_live_pred, title_video




# Data loading functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
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
    
    # Subs Score
    if subs > 50_000_000: score += 50
    elif subs > 10_000_000: score += 40
    elif subs > 1_000_000: score += 30
    elif subs > 100_000: score += 20
    else: score += 10

    # Views Score
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
    ["🏠 Home", "📊 Channel Leaderboard", "🎬 Video Explorer", "🚀 Milestone Tracker", "📈 Trending & Predictions"]
)

# Initialize session state for drill-down navigation
if 'selected_channel_id' not in st.session_state:
    st.session_state.selected_channel_id = None
if 'selected_video_id' not in st.session_state:
    st.session_state.selected_video_id = None

# Load data
channels_df = load_channels()
videos_df = load_videos()

# MAIN PAGES
if page == "🏠 Home":
    # Centered STREAMWATCH header - 40% larger
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <p style='font-size: 4.2rem; font-weight: bold; background: linear-gradient(90deg, #FF0000, #FF6B6B); 
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;'>
                STREAMWATCH
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Expanded key metrics row - 7 metrics total
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
        # Total data points
        total_channel_data_points = channels_df.height
        total_video_data_points = videos_df.height
        total_data_points = total_channel_data_points + total_video_data_points
        st.metric("📊 Data Points", format_number(total_data_points))
    
    with col6:
        # Top channel views in last hour (approximate from views_per_day)
        try:
            top_channel_daily_views = channels_df.sort("view_count", descending=True).select("view_count").head(1).to_numpy()[0][0]
            # Rough estimate: assume 1/24 of daily views per hour
            hourly_estimate = int(top_channel_daily_views / 365 / 24)  # Annual to hourly
            st.metric("⚡ Top/Hour", format_number(hourly_estimate))
        except:
            st.metric("⚡ Top/Hour", "N/A")
    
    with col7:
        # Total views across all videos
        try:
            total_views = videos_df.select(pl.col("view_count")).sum().to_numpy()[0][0]
            st.metric("👁️ Total Views", format_number(total_views))
        except:
            st.metric("👁️ Total Views", "N/A")
    
    st.markdown("---")
    
    # Top channels by subscribers
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
    
    # Viral videos of the moment
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
                st.caption(f"📺 {row['channel_title']} • {row['view_count']:,} views")
                st.caption(f"⚡ {row['views_per_day']:,.0f} views/day • 💬 {row['engagement_rate']:.2f}% engagement")
    except Exception as e:
        st.info("Viral statistics unavailable")

elif page == "📊 Channel Leaderboard":
    st.title("📊 Channel Leaderboard")
    
    # Check if viewing a specific channel
    if 'selected_channel_id' in st.session_state and st.session_state.selected_channel_id:
        # CHANNEL DETAIL VIEW
        channel_id = st.session_state.selected_channel_id
        channel_data = channels_df.filter(pl.col("channel_id") == channel_id).to_dicts()[0]
        
        # Back button
        if st.button("← Back to Leaderboard"):
            st.session_state.selected_channel_id = None
            
        
        # Channel header
        col1, col2 = st.columns([1, 4])
        with col1:
            if channel_data['thumbnail_url']:
                st.image(channel_data['thumbnail_url'], width=150)
        
        with col2:
            # Calculate Grade
            grade = calculate_grade(channel_data['subscriber_count'], channel_data['view_count'])
            
            st.markdown(f"""
                <div class="grade-badge">GRADE: {grade}</div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"# {channel_data['channel_title']}")
            st.markdown(f"### `{channel_data['custom_url']}`")
            st.caption(channel_data['description'] if channel_data['description'] else "")
        
        st.markdown("---")
        
        # Current stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📺 Subscribers", f"{channel_data['subscriber_count']:,}")
        col2.metric("👀 Total Views", f"{channel_data['view_count']:,}")
        col3.metric("🎬 Videos", f"{channel_data['video_count']:,}")
        col4.metric("🌍 Country", channel_data['country'] or "N/A")
        
        st.markdown("---")
        
        # Load historical data
        history = load_channel_history(channel_id, days=30)
        
        # Skip the growth analysis and daily statistics sections
        
        st.markdown("---")
        
        # Top videos from this channel
        st.subheader(f"🎬 Top Videos from {channel_data['channel_title']}")
        
        channel_videos = videos_df.filter(pl.col("channel_id") == channel_id).sort("view_count", descending=True).head(10)
        
        for vid_row in channel_videos.iter_rows(named=True):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if vid_row['thumbnail_url']:
                    st.image(vid_row['thumbnail_url'], width=180)
            
            with col2:
                # Make video title clickable
                if st.button(f"📹 {vid_row['video_title']}", key=f"vid_{vid_row['video_id']}"):
                    st.session_state.selected_video_id = vid_row['video_id']
                    st.session_state.page = "🎬 Video Explorer"
                    
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Views", f"{vid_row['view_count']:,}")
                col_b.metric("Engagement", f"{vid_row['engagement_rate']:.2f}%")
                col_c.metric("Views/Day", f"{vid_row['views_per_day']:,.0f}")
            
            st.markdown("---")
    
    else:
        # LEADERBOARD VIEW (default)
        # Sorting options
        sort_by = st.selectbox(
            "Sort by:",
            ["Subscribers", "Total Views", "Video Count", "Latest Subscriber Count"]
        )
        
        sort_column_map = {
            "Subscribers": "subscriber_count",
            "Total Views": "view_count",
            "Video Count": "video_count",
            "Latest Subscriber Count": "subscriber_count"
        }
        
        sorted_channels = channels_df.sort(sort_column_map[sort_by], descending=True)
        
        # Display channels
        for idx, row in enumerate(sorted_channels.iter_rows(named=True)):
            with st.container():
                col1, col2, col3 = st.columns([1, 5, 2])
                
                with col1:
                    st.markdown(f"### #{idx+1}")
                    if row['thumbnail_url']:
                        st.image(row['thumbnail_url'], width=100)
                
                with col2:
                    # Make channel name clickable
                    if st.button(f"📺 {row['channel_title']}", key=f"ch_{row['channel_id']}"):
                        st.session_state.selected_channel_id = row['channel_id']
                        
                    
                    st.markdown(f"`{row['custom_url']}`")
                    st.caption(row['description'][:200] + "..." if row['description'] else "")
                
                with col3:
                    st.metric("Subscribers", f"{row['subscriber_count']:,}")
                    st.metric("Total Views", f"{row['view_count']:,}")
                    st.metric("Videos", f"{row['video_count']:,}")
            
            st.markdown("---")

elif page == "🎬 Video Explorer":
    st.title("🎬 Video Explorer")
    
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
            
        
        # Video header
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if video_data['thumbnail_url']:
                st.image(video_data['thumbnail_url'], width=350)
        
        with col2:
            st.markdown(f"# {video_data['video_title']}")
            # Use safe get for channel title in case of lingering data issues
            c_title = video_data.get('channel_title', 'Unknown Channel')
            st.markdown(f"## {c_title}")
            st.markdown(f"`{video_data['custom_url']}`")
            
            # Status badges
            badges = []
            if video_data.get('is_billionaires_watch'):
                st.markdown('<span class="milestone-badge">💎 Billionaires Club (1B+)</span>', unsafe_allow_html=True)
            if video_data.get('is_approaching_milestone'):
                st.markdown(f'<span class="viral-badge">🎯 Approaching {video_data["next_milestone"]:,}</span>', unsafe_allow_html=True)
            if video_data.get('is_highly_viral'):
                st.markdown('<span class="viral-badge">🔥 Highly Viral</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Current stats
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("👀 Views", f"{video_data['view_count']:,}")
        col2.metric("👍 Likes", f"{video_data['like_count']:,}")
        col3.metric("💬 Comments", f"{video_data['comment_count']:,}")
        col4.metric("📊 Engagement", f"{video_data['engagement_rate']:.2f}%")
        col5.metric("⚡ Views/Day", f"{video_data['views_per_day']:,.0f}")
        
        st.markdown("---")
        
        # Milestone tracking section
        st.subheader(f"🎯 Milestone Progress: {video_data['milestone_tier']}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Views", f"{video_data['view_count']:,}")
        col2.metric("Next Milestone", f"{video_data['next_milestone']:,}")
        col3.metric("Views Needed", f"{video_data['views_to_next_milestone']:,}")
        
        # Progress bar
        progress_pct = video_data['milestone_progress_pct'] / 100
        st.progress(progress_pct)
        st.caption(f"{video_data['milestone_progress_pct']:.1f}% progress toward {video_data['next_milestone']:,} views")
        
        # Simple milestone projection (no ML)
        views_per_day = video_data.get('views_per_day', 0)
        views_needed = video_data.get('views_to_next_milestone', 0)
        
        col1, col2 = st.columns(2)
        with col1:
            if views_per_day > 0:
                days_to_milestone = int(views_needed / views_per_day)
                st.metric(
                    "⏱️ Days to Next Milestone",
                    f"{days_to_milestone:,} days",
                    help="Linear projection based on current velocity"
                )
            else:
                st.metric("⏱️ Days to Next Milestone", "N/A (low velocity)")
        
        with col2:
            st.metric(
                "📊 Daily Velocity",
                f"{views_per_day:,.0f} views/day",
                help="Average views gained per day"
            )
        
        st.markdown("---")
        
        # Historical view count analysis
        st.subheader("📈 View Count History & Forecast")
        
        history = load_video_history(video_id, days=30)
        
        if len(history) >= 2:
            df_history = history.to_pandas()
            df_history['ingestion_timestamp'] = pd.to_datetime(df_history['ingestion_timestamp'])
            df_history = df_history.sort_values('ingestion_timestamp')
            
            # Calculate growth over period
            first_snapshot = df_history.iloc[0]
            last_snapshot = df_history.iloc[-1]
            days_span = (last_snapshot['ingestion_timestamp'] - first_snapshot['ingestion_timestamp']).days
            
            if days_span > 0:
                view_growth = last_snapshot['view_count'] - first_snapshot['view_count']
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Views Gained (30 days)", f"+{view_growth:,}")
                col2.metric("Avg Daily Growth", f"{view_growth/days_span:,.0f} views/day")
                col3.metric("Growth Rate", f"{(view_growth/first_snapshot['view_count']*100):.2f}%")
            
            # Create comprehensive chart with forecast
            fig = go.Figure()
            
            # Historical data
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
            
            # Removed forecast section - no ML predictions
            
            fig.update_layout(
                title="View Count History",
                xaxis_title="Date",
                yaxis_title="View Count",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough historical data for forecasting (need at least 2 snapshots)")
        
        st.markdown("---")
        
        # Engagement deep dive
        st.subheader("💬 Engagement Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Engagement Ratios:**")
            st.metric("Like-to-View", f"{video_data['like_view_ratio']:.4f}", help="Audience positivity indicator")
            st.metric("Comment-to-View", f"{video_data['comment_view_ratio']:.4f}", help="Controversy/engagement indicator")
            st.metric("Like-to-Comment", f"{video_data['like_comment_ratio']:.2f}", help="Positive sentiment ratio")
        
        with col2:
            st.markdown("**Performance Context:**")
            st.metric("Video Age", f"{video_data['days_since_publish']} days")
            st.metric("Duration", f"{video_data['duration_seconds']//60} min {video_data['duration_seconds']%60} sec")
            
            # Category
            if video_data.get('category_id'):
                category_names = {
                    "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music",
                    "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events",
                    "20": "Gaming", "22": "People & Blogs", "23": "Comedy",
                    "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style",
                    "27": "Education", "28": "Science & Technology"
                }
                category = category_names.get(str(video_data['category_id']), f"Category {video_data['category_id']}")
                st.metric("Category", category)
    
    else:
        # VIDEO EXPLORER VIEW (default)
        # Filters
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
        
        # Apply filters (no min_views filter)
        filtered_videos = videos_df
        
        if filter_type == "Billionaires Watch (1B+)":
            filtered_videos = filtered_videos.filter(pl.col("is_billionaires_watch") == True)
        elif filter_type == "Approaching Milestone":
            filtered_videos = filtered_videos.filter(pl.col("is_approaching_milestone") == True)
        elif filter_type == "Highly Viral":
            filtered_videos = filtered_videos.filter(pl.col("is_highly_viral") == True)
        
        if milestone_tier != "All":
            filtered_videos = filtered_videos.filter(pl.col("milestone_tier") == milestone_tier)
        
        # Sort by views
        filtered_videos = filtered_videos.sort("view_count", descending=True)
        
        st.write(f"**{len(filtered_videos)} videos found**")
        
        # Display videos
        for row in filtered_videos.head(20).iter_rows(named=True):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if row['thumbnail_url']:
                    st.image(row['thumbnail_url'], width=200)
            
            with col2:
                # Make video title clickable
                if st.button(f"🎬 {row['video_title']}", key=f"explore_{row['video_id']}"):
                    st.session_state.selected_video_id = row['video_id']
                    
                
                c_title = row.get('channel_title', 'Unknown Channel')
                st.markdown(f"**{c_title}** • `{row['custom_url']}`")
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Views", f"{row['view_count']:,}")
                col_b.metric("Engagement", f"{row['engagement_rate']:.2f}%")
                col_c.metric("Views/Day", f"{row['views_per_day']:,.0f}")
                
                # Badges
                badges = []
                if row.get('is_billionaires_watch'):
                    badges.append("💎 Billionaires Club")
                if row.get('is_approaching_milestone'):
                    badges.append(f"🎯 {row['next_milestone']:,} approaching")
                if row.get('is_highly_viral'):
                    badges.append("🔥 Highly Viral")
                
                if badges:
                    st.caption(" • ".join(badges))
            
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
                
                # Progress bar
                progress = row['milestone_progress_pct'] / 100
                st.progress(progress)
                st.caption(f"Progress: {row['milestone_progress_pct']:.1f}% to {row['next_milestone']:,} views")
                
                # Stats
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Current Views", f"{row['view_count']:,}")
                col_b.metric("Views Needed", f"{row['views_to_next_milestone']:,}")
                
                if row['days_to_milestone']:
                    col_c.metric("Est. Days", f"{int(row['days_to_milestone'])}")
                else:
                    col_c.metric("Est. Days", "N/A")
            
            st.markdown("---")




elif page == "📈 Trending & Predictions":

    st.title("📈 YouTube Trending Dashboard (Neon + Streamlit)")

    # Sidebar filters
    st.sidebar.subheader("Trending filters")
    days = st.sidebar.slider(
        "Lookback days (trending)", min_value=1, max_value=14, value=7, key="trend_days"
    )
    region = st.sidebar.text_input(
        "Region code (trending)", value="US", key="trend_region"
    )

    if st.sidebar.button("Reload trending data"):
        st.cache_data.clear()

    with st.spinner("Loading trending data from Neon..."):
        df_trend = get_trending_data(days=days, region=region)

    if df_trend.empty:
        st.warning("No trending data for this range / region.")
        st.stop()

    latest_ts = df_trend["ingestion_timestamp"].max()
    df_latest = df_trend[df_trend["ingestion_timestamp"] == latest_ts]

    n_videos = df_latest["video_id"].nunique()
    n_channels = df_latest["channel_id"].nunique()
    total_views = int(df_latest["view_count"].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Videos (latest snapshot)", n_videos)
    col2.metric("Channels (latest snapshot)", n_channels)
    col3.metric("Total views (latest snapshot)", f"{total_views:,}")

    st.caption(f"Latest snapshot timestamp: {latest_ts}")
    st.markdown("---")

    # ---- Creator Leaderboard ----
    st.subheader("Creator leaderboard (trending)")
    fig_lb, channel_stats = get_trending_leaderboard_figure(df_trend)
    st.plotly_chart(fig_lb, use_container_width=True)

    with st.expander("Show leaderboard table"):
        st.dataframe(channel_stats)

    st.markdown("---")

    # ---- Fastest Growing Videos ----
    st.subheader("Fastest growing videos (view velocity per hour)")

    df_latest_vel = df_latest.dropna(subset=["view_velocity_per_hour"]).copy()

    top_velocity = (
        df_latest_vel.sort_values("view_velocity_per_hour", ascending=False)
        .head(10)[
            ["video_title", "view_count", "view_velocity_per_hour"]
        ]
    )

    st.dataframe(top_velocity)

    st.markdown("---")

    # ---- Regression ----
    st.subheader("View count regression (last 2 days as test set)")

    reg_result = train_and_eval_trending_regression(df_trend)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train size", reg_result["train_size"])
    c2.metric("Test size", reg_result["test_size"])
    c3.metric("MAE", f"{reg_result['mae']:.0f}")
    c4.metric("RMSE", f"{reg_result['rmse']:.0f}")

    st.write(f"R²: **{reg_result['r2']:.4f}**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(reg_result["fig_pred_vs_actual"], use_container_width=True)
    with col_b:
        st.plotly_chart(reg_result["fig_errors"], use_container_width=True)

    st.markdown("---")

    # ---- Live Trend ----
    st.subheader("Live view count trend for one video (trending)")

    test_df = reg_result["test_df"]

    if len(test_df) == 0:
        st.info("Test set is empty, cannot show live trend.")
    else:
        video_candidates = (
            test_df[["video_id", "video_title"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        def _make_display(row):
            title_short = (row["video_title"] or "")[:60].replace("\n", " ")
            return f"{title_short} [{row['video_id'][:8]}]"

        video_candidates["display"] = video_candidates.apply(
            _make_display, axis=1
        )

        selected_display = st.selectbox(
            "Select a video from test set (last 2 days):",
            options=video_candidates["display"].tolist(),
        )

        selected_id = video_candidates.loc[
            video_candidates["display"] == selected_display, "video_id"
        ].iloc[0]

        fig_live_pred, title_video = live_trend_figure_trending(
            df_trend, test_df, target_video_id=selected_id
        )

        st.plotly_chart(fig_live_pred, use_container_width=True)
        st.caption(f"Selected video: {title_video}")



# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Project STREAMWATCH - IDS 706 Fall 2025 • Tony Ngari, Matthew Fischer, Can He, Joseph Hong, Trey Chase</p>", unsafe_allow_html=True)