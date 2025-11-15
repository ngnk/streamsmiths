import os
from datetime import datetime, timedelta

import polars as pl
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Neon database connection
DB_CONNECTION_URI = os.getenv("NEON_DATABASE_URL")

if not DB_CONNECTION_URI:
    st.error("NEON_DATABASE_URL not set in .env file")
    st.stop()

# Clean connection string
DB_CONNECTION_URI = DB_CONNECTION_URI.strip()
if DB_CONNECTION_URI.startswith("psql "):
    DB_CONNECTION_URI = DB_CONNECTION_URI[5:].strip()
DB_CONNECTION_URI = DB_CONNECTION_URI.strip("'\"")

st.set_page_config(page_title="YouTube Analytics Pro", layout="wide")


def read_table(name: str) -> pl.DataFrame | None:
    try:
        return pl.read_database_uri(f"SELECT * FROM {name}", uri=DB_CONNECTION_URI)
    except Exception as e:
        st.error(f"Failed to read {name}: {e}")
        return None


def format_number(num):
    """Format large numbers with K/M/B suffixes"""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return str(int(num))


def get_youtube_tier(subs: int) -> dict:
    """Determine YouTube award tier"""
    if subs >= 100_000_000:
        return {"tier": "Red Diamond", "emoji": "💎🔴", "color": "#FF0000"}
    elif subs >= 50_000_000:
        return {"tier": "Ruby", "emoji": "💎", "color": "#E0115F"}
    elif subs >= 10_000_000:
        return {"tier": "Diamond", "emoji": "💎", "color": "#B9F2FF"}
    elif subs >= 1_000_000:
        return {"tier": "Gold", "emoji": "🥇", "color": "#FFD700"}
    elif subs >= 100_000:
        return {"tier": "Silver", "emoji": "🥈", "color": "#C0C0C0"}
    else:
        return {"tier": "Bronze", "emoji": "🥉", "color": "#CD7F32"}


def get_next_tier_target(subs: int) -> int:
    """Get next milestone subscriber count"""
    milestones = [100_000, 1_000_000, 10_000_000, 50_000_000, 100_000_000]
    for milestone in milestones:
        if subs < milestone:
            return milestone
    return 100_000_000


def predict_days_to_milestone(df: pl.DataFrame, current_subs: int, target: int) -> dict:
    """Predict days to reach next milestone using linear regression"""
    if df.height < 2:
        return {"days": None, "date": None, "daily_rate": 0}
    
    # Convert timestamps to days since start
    df = df.sort("ingest_timestamp")
    timestamps = df["ingest_timestamp"].to_numpy()
    subs = df["subscriber_count"].to_numpy()
    
    # Calculate days from first observation
    days = [(t - timestamps[0]).total_seconds() / 86400 for t in timestamps]
    
    # Simple linear regression
    if len(days) > 1:
        coeffs = np.polyfit(days, subs, 1)
        daily_rate = coeffs[0]
        
        if daily_rate > 0:
            subs_needed = target - current_subs
            days_needed = subs_needed / daily_rate
            predicted_date = datetime.now() + timedelta(days=days_needed)
            
            return {
                "days": int(days_needed),
                "date": predicted_date.strftime("%Y-%m-%d"),
                "daily_rate": daily_rate
            }
    
    return {"days": None, "date": None, "daily_rate": 0}


def calculate_social_heat_score(latest_videos: pl.DataFrame) -> float:
    """
    Calculate social heat score based on recent engagement
    Score from 0-100 based on:
    - Like rate (40%)
    - Comment rate (30%) 
    - Upload consistency (30%)
    """
    if latest_videos.height == 0:
        return 0
    
    # Filter videos with views
    videos_with_views = latest_videos.filter(pl.col("view_count") > 0)
    
    if videos_with_views.height == 0:
        return 0
    
    # Calculate engagement rates
    videos_with_engagement = videos_with_views.with_columns([
        ((pl.col("like_count") / pl.col("view_count")) * 100).alias("like_rate"),
        ((pl.col("comment_count") / pl.col("view_count")) * 100).alias("comment_rate")
    ])
    
    # Like rate score (normalize to 0-40, assuming 5% is excellent)
    avg_like_rate = videos_with_engagement["like_rate"].mean()
    like_score = min(40, (avg_like_rate / 5.0) * 40)
    
    # Comment rate score (normalize to 0-30, assuming 0.5% is excellent)
    avg_comment_rate = videos_with_engagement["comment_rate"].mean()
    comment_score = min(30, (avg_comment_rate / 0.5) * 30)
    
    # Upload consistency score (30 points if uploaded in last 30 days)
    if "published_at" in latest_videos.columns:
        try:
            latest_videos = latest_videos.with_columns(
                pl.col("published_at").str.strptime(pl.Datetime, strict=False).alias("pub_date")
            )
            most_recent = latest_videos["pub_date"].max()
            if most_recent:
                days_since_upload = (datetime.now() - most_recent).days
                consistency_score = max(0, 30 * (1 - days_since_upload / 90))
            else:
                consistency_score = 0
        except:
            consistency_score = 15  # Default middle score
    else:
        consistency_score = 15
    
    total_score = like_score + comment_score + consistency_score
    return round(total_score, 1)


def calculate_velocity_score(df: pl.DataFrame) -> dict:
    """Calculate growth velocity and acceleration"""
    if df.height < 3:
        return {"velocity": 0, "acceleration": 0, "trend": "neutral"}
    
    df = df.sort("ingest_timestamp")
    
    # Get last 3 data points
    recent = df.tail(3)
    subs = recent["subscriber_count"].to_list()
    
    # Calculate velocity (average daily change)
    velocity = (subs[-1] - subs[0]) / 2
    
    # Calculate acceleration (is growth speeding up or slowing down?)
    change_1 = subs[1] - subs[0]
    change_2 = subs[2] - subs[1]
    acceleration = change_2 - change_1
    
    if acceleration > velocity * 0.1:
        trend = "accelerating"
    elif acceleration < -velocity * 0.1:
        trend = "decelerating"
    else:
        trend = "steady"
    
    return {
        "velocity": velocity,
        "acceleration": acceleration,
        "trend": trend
    }


try:
    channels_df = read_table("channels_log")
    videos_df = read_table("videos_log")

    if channels_df is None or videos_df is None:
        st.stop()

    if channels_df.height == 0 or videos_df.height == 0:
        st.warning("Database is empty. Run the pipeline first.")
        st.stop()

    # Convert timestamps
    if "ingest_timestamp" in channels_df.columns:
        channels_df = channels_df.with_columns(
            pl.col("ingest_timestamp").str.strptime(pl.Datetime, strict=False)
        )
    if "ingest_timestamp" in videos_df.columns:
        videos_df = videos_df.with_columns(
            pl.col("ingest_timestamp").str.strptime(pl.Datetime, strict=False)
        )

    # Sidebar - Channel Selection
    st.sidebar.header("🎯 Channel Selection")
    
    channel_list = channels_df.select("channel_id", "channel_title").unique().sort("channel_title")
    channel_options = {row["channel_title"]: row["channel_id"] for row in channel_list.iter_rows(named=True)}
    
    selected_channel_name = st.sidebar.selectbox(
        "Select a channel",
        options=list(channel_options.keys()),
        index=0
    )
    selected_channel_id = channel_options[selected_channel_name]

    # Filter data for selected channel
    channel_data = channels_df.filter(pl.col("channel_id") == selected_channel_id).sort("ingest_timestamp")
    channel_videos = videos_df.filter(pl.col("channel_id") == selected_channel_id)

    # Get latest stats
    latest_channel = channel_data[-1]
    current_subs = latest_channel["subscriber_count"]
    current_views = latest_channel["view_count"]
    current_video_count = latest_channel["video_count"]

    # Get latest video snapshot
    latest_videos = channel_videos.sort("ingest_timestamp", descending=True).unique(
        subset=["video_id"], keep="first"
    )

    # HEADER WITH TIER
    tier_info = get_youtube_tier(current_subs)
    st.title(f"{tier_info['emoji']} {selected_channel_name}")
    st.markdown(f"**YouTube {tier_info['tier']} Creator** • `{selected_channel_id}`")
    
    # TIER & PREDICTIONS
    st.header("🏆 Creator Status & Predictions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Tier", tier_info["tier"])
        st.markdown(f"<h1 style='text-align: center; color: {tier_info['color']};'>{tier_info['emoji']}</h1>", unsafe_allow_html=True)
    
    with col2:
        next_milestone = get_next_tier_target(current_subs)
        subs_to_go = next_milestone - current_subs
        progress = (current_subs / next_milestone) * 100
        
        st.metric("Next Milestone", format_number(next_milestone))
        st.progress(progress / 100)
        st.caption(f"{format_number(subs_to_go)} subscribers to go")
    
    with col3:
        prediction = predict_days_to_milestone(channel_data, current_subs, next_milestone)
        if prediction["days"]:
            st.metric("Days to Milestone", f"{prediction['days']:,}")
            st.caption(f"Est. date: {prediction['date']}")
            st.caption(f"+{int(prediction['daily_rate']):,} subs/day")
        else:
            st.metric("Days to Milestone", "N/A")
            st.caption("Need more data")
    
    with col4:
        heat_score = calculate_social_heat_score(latest_videos)
        st.metric("Social Heat Score", f"{heat_score}/100")
        if heat_score >= 70:
            st.success("🔥 On Fire!")
        elif heat_score >= 40:
            st.info("📈 Growing")
        else:
            st.warning("📉 Needs Boost")

    # DERIVED METRICS
    st.header("📊 Derived Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Subscribers per video
    subs_per_video = current_subs / current_video_count if current_video_count > 0 else 0
    col1.metric("Subs/Video", format_number(subs_per_video))
    
    # Views per video
    views_per_video = current_views / current_video_count if current_video_count > 0 else 0
    col2.metric("Views/Video", format_number(views_per_video))
    
    # Subscriber to view ratio
    sub_to_view_ratio = (current_subs / current_views) * 100 if current_views > 0 else 0
    col3.metric("Sub/View Ratio", f"{sub_to_view_ratio:.2f}%")
    
    # Average engagement
    if latest_videos.height > 0:
        avg_engagement = ((latest_videos["like_count"].sum() + latest_videos["comment_count"].sum()) / 
                         latest_videos["view_count"].sum() * 100)
        col4.metric("Avg Engagement", f"{avg_engagement:.2f}%")
    else:
        col4.metric("Avg Engagement", "N/A")
    
    # Growth velocity
    velocity_data = calculate_velocity_score(channel_data)
    col5.metric(
        "Growth Velocity",
        format_number(abs(velocity_data["velocity"])),
        delta=velocity_data["trend"].capitalize()
    )

    # GROWTH TRENDS
    st.header("📈 Growth Analysis")
    
    if channel_data.height > 1:
        # Create growth chart with prediction line
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=channel_data["ingest_timestamp"].to_list(),
            y=channel_data["subscriber_count"].to_list(),
            mode='lines+markers',
            name='Actual Subscribers',
            line=dict(color='#FF0000', width=3),
            marker=dict(size=8)
        ))
        
        # Prediction line
        if prediction["days"] and prediction["daily_rate"] > 0:
            future_dates = [datetime.now() + timedelta(days=i) for i in range(0, prediction["days"] + 1)]
            future_subs = [current_subs + (prediction["daily_rate"] * i) for i in range(0, prediction["days"] + 1)]
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_subs,
                mode='lines',
                name='Projected Growth',
                line=dict(color='#00FF00', width=2, dash='dash')
            ))
            
            # Add milestone marker
            fig.add_trace(go.Scatter(
                x=[datetime.strptime(prediction["date"], "%Y-%m-%d")],
                y=[next_milestone],
                mode='markers',
                name='Next Milestone',
                marker=dict(size=15, color='gold', symbol='star')
            ))
        
        fig.update_layout(
            title="Subscriber Growth & Projection",
            xaxis_title="Date",
            yaxis_title="Subscribers",
            hovermode="x unified",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the pipeline multiple times to see growth trends")

    # ENGAGEMENT BREAKDOWN
    st.header("💬 Engagement Analysis")
    
    if latest_videos.height > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top videos by engagement rate
            videos_with_engagement = latest_videos.filter(pl.col("view_count") > 100).with_columns([
                ((pl.col("like_count") + pl.col("comment_count")) / pl.col("view_count") * 100).alias("engagement_rate")
            ]).sort("engagement_rate", descending=True).head(10)
            
            fig = px.bar(
                videos_with_engagement.to_pandas(),
                y="video_title",
                x="engagement_rate",
                orientation='h',
                title="Top 10 Videos by Engagement Rate",
                labels={"engagement_rate": "Engagement %", "video_title": ""}
            )
            fig.update_traces(marker_color='#FF0000')
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Engagement distribution
            engagement_dist = latest_videos.filter(pl.col("view_count") > 0).with_columns([
                ((pl.col("like_count") / pl.col("view_count")) * 100).alias("like_rate")
            ])
            
            fig = px.histogram(
                engagement_dist.to_pandas(),
                x="like_rate",
                nbins=30,
                title="Like Rate Distribution",
                labels={"like_rate": "Like Rate %"}
            )
            fig.update_traces(marker_color='#FF0000')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # PERFORMANCE LEADERBOARD
    st.header("🏅 Performance Leaderboard")
    
    if latest_videos.height > 0:
        st.subheader("Top 10 Performing Videos")
        
        top_videos = latest_videos.with_columns([
            ((pl.col("like_count") + pl.col("comment_count")) / pl.col("view_count") * 100).alias("engagement_rate")
        ]).sort("view_count", descending=True).head(10)
        
        display_df = top_videos.select([
            pl.col("video_title").str.slice(0, 50).alias("Title"),
            pl.col("view_count").map_elements(lambda x: format_number(x), return_dtype=pl.String).alias("Views"),
            pl.col("like_count").map_elements(lambda x: format_number(x), return_dtype=pl.String).alias("Likes"),
            pl.col("comment_count").map_elements(lambda x: format_number(x), return_dtype=pl.String).alias("Comments"),
            pl.col("engagement_rate").round(2).alias("Engagement %")
        ])
        
        st.dataframe(display_df.to_pandas(), use_container_width=True, hide_index=True)

    # CHANNEL COMPARISON
    st.header("⚖️ Multi-Channel Comparison")
    
    latest_all_channels = channels_df.sort("ingest_timestamp", descending=True).unique(
        subset=["channel_id"], keep="first"
    ).with_columns([
        (pl.col("subscriber_count") / pl.col("video_count")).alias("subs_per_video"),
        (pl.col("view_count") / pl.col("video_count")).alias("views_per_video")
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            latest_all_channels.sort("subscriber_count", descending=True).to_pandas(),
            x="channel_title",
            y="subscriber_count",
            title="Total Subscribers",
            color="subscriber_count",
            color_continuous_scale="Reds"
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Subscribers")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            latest_all_channels.sort("subs_per_video", descending=True).to_pandas(),
            x="channel_title",
            y="subs_per_video",
            title="Subscribers per Video (Efficiency)",
            color="subs_per_video",
            color_continuous_scale="Greens"
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Subs/Video")
        st.plotly_chart(fig, use_container_width=True)

    # RAW DATA
    with st.expander("📋 View Raw Data"):
        tab1, tab2 = st.tabs(["Channel History", "Video Data"])
        with tab1:
            st.dataframe(channel_data.to_pandas(), use_container_width=True)
        with tab2:
            st.dataframe(channel_videos.to_pandas(), use_container_width=True)

except Exception as e:
    st.error(f"Dashboard error: {e}")
    st.exception(e)
