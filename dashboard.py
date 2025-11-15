import os

import polars as pl
import streamlit as st
from dotenv import load_dotenv

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

st.set_page_config(page_title="YouTube Pipeline Status", layout="wide")
st.title("📺 YouTube Channel Tracker Dashboard")
st.markdown("Real-time stats from the Bronze→Silver→Gold pipeline")


def read_table(name: str) -> pl.DataFrame | None:
    try:
        return pl.read_database_uri(f"SELECT * FROM {name}", uri=DB_CONNECTION_URI)
    except Exception as e:
        st.error(f"Failed to read {name}: {e}")
        return None


try:
    channels_df = read_table("channels_log")
    videos_df = read_table("videos_log")

    if channels_df is None or videos_df is None:
        st.stop()

    if channels_df.height == 0 or videos_df.height == 0:
        st.warning("Database is empty. Run the pipeline first.")
        st.stop()

    # Convert ingest_timestamp to datetime if possible
    if "ingest_timestamp" in channels_df.columns:
        channels_df = channels_df.with_columns(
            pl.col("ingest_timestamp").str.strptime(pl.Datetime, strict=False)
        )

    st.subheader("Pipeline Health")

    col1, col2, col3 = st.columns(3)

    unique_channels = channels_df["channel_id"].n_unique()
    total_videos = videos_df["video_id"].n_unique()

    col1.metric("Tracked Channels", unique_channels)
    col2.metric("Tracked Videos", total_videos)

    if "ingest_timestamp" in channels_df.columns:
        last_run_time = (
            channels_df.select(pl.col("ingest_timestamp").max())
            .to_series()
            .item()
        )
        if last_run_time:
            col3.metric(
                "Last Successful Pull (UTC)",
                str(last_run_time).split(".")[0],
            )
        else:
            col3.metric("Last Successful Pull (UTC)", "Unknown")
    else:
        col3.metric("Last Successful Pull (UTC)", "Unknown")

    st.header("📊 Latest Channel Data")
    # Show latest record per channel
    latest_channels = channels_df.sort(
        ["channel_id", "ingest_timestamp"], descending=[False, True]
    ).unique(subset=["channel_id"], keep="first")

    st.dataframe(latest_channels.to_pandas(), use_container_width=True)

    st.header("Latest Video Data (Last 100)")
    videos_df_sorted = videos_df.sort("ingest_timestamp", descending=True)
    st.dataframe(videos_df_sorted.head(100).to_pandas(), use_container_width=True)

except Exception as e:
    st.error(f"Dashboard error: {e}")
    st.info(
        "Make sure NEON_DATABASE_URL is set correctly in your .env file and the pipeline has run at least once."
    )