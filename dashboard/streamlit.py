# streamlit_app.py
import pandas as pd
import numpy as np
from typing import Optional

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression

from neon_utils import load_trending_v2

@st.cache_data(show_spinner=True)
def get_data(days: int, region: str) -> pd.DataFrame:
    df = load_trending_v2(days=days, region=region)
    df = df.sort_values("ingestion_timestamp").reset_index(drop=True)
    return df

def get_leaderboard_figure(df: pd.DataFrame):
    latest_ts = df["ingestion_timestamp"].max()
    df_latest = df[df["ingestion_timestamp"] == latest_ts].copy()

    df_latest["channel_display"] = df_latest["channel_title"].fillna(
        df_latest["channel_id"]
    )

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


def train_and_eval_regression(df: pd.DataFrame):
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
    # R^2
    r2 = 1 - np.sum(errors**2) / np.sum((y_test - y_test.mean())**2)

    # Predicted vs Actual
    fig_pred_vs_actual = px.scatter(
        test_df,
        x="predicted_view_count",
        y="view_count",
        hover_data=["video_id", "video_title"],
        title="Predicted vs actual view count (last 2 days test set)",
        labels={"predicted_view_count": "Predicted", "view_count": "Actual"},
    )

    min_v = min(
        test_df["predicted_view_count"].min(), test_df["view_count"].min()
    )
    max_v = max(
        test_df["predicted_view_count"].max(), test_df["view_count"].max()
    )

    fig_pred_vs_actual.add_shape(
        type="line",
        x0=min_v,
        y0=min_v,
        x1=max_v,
        y1=max_v,
    )

    # Error histogram
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


def live_trend_figure(
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


# ========= Streamlit =========

st.set_page_config(
    page_title="YouTube Trending Dashboard",
    layout="wide",
)

st.title("📈 YouTube Trending Dashboard (Neon + Streamlit)")

# ---- Sidebar  ----
st.sidebar.header("Filters")

days = st.sidebar.slider(
    "Lookback days", min_value=1, max_value=14, value=7
)
region = st.sidebar.text_input("Region code", value="US")

if st.sidebar.button("Reload data"):
    st.cache_data.clear()

with st.spinner("Loading data from Neon..."):
    df = get_data(days=days, region=region)

if df.empty:
    st.warning("No data for this range / region.")
    st.stop()

# ---- summary metrics ----
latest_ts = df["ingestion_timestamp"].max()
df_latest = df[df["ingestion_timestamp"] == latest_ts]

n_videos = df_latest["video_id"].nunique()
n_channels = df_latest["channel_id"].nunique()
total_views = int(df_latest["view_count"].sum())

col1, col2, col3 = st.columns(3)
col1.metric("Videos (latest snapshot)", n_videos)
col2.metric("Channels (latest snapshot)", n_channels)
col3.metric("Total views (latest snapshot)", f"{total_views:,}")

st.caption(f"Latest snapshot timestamp: {latest_ts}")
st.markdown("---")

# ---- 1. Creator leaderboard ----
st.subheader("1. Creator leaderboard")

fig_leaderboard, channel_stats = get_leaderboard_figure(df)
st.plotly_chart(fig_leaderboard, use_container_width=True)

with st.expander("Show underlying table"):
    st.dataframe(channel_stats)

st.markdown("---")

# ---- 2. Regression ----
st.subheader("2. View count regression (last 2 days as test set)")

reg_result = train_and_eval_regression(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Train size", reg_result["train_size"])
c2.metric("Test size", reg_result["test_size"])

st.write(f"R²: **{reg_result['r2']:.4f}**")

col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(reg_result["fig_pred_vs_actual"], use_container_width=True)
with col_b:
    st.plotly_chart(reg_result["fig_errors"], use_container_width=True)

st.markdown("---")

# ---- 3. live trend + prediction----
st.subheader("3. Live view count trend for one video")

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
        title = row["video_title"] or ""
        title_short = title[:60].replace("\n", " ")
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

    fig_live_pred, title_video = live_trend_figure(
        df, test_df, target_video_id=selected_id
    )

    st.plotly_chart(fig_live_pred, use_container_width=True)
    st.caption(f"Selected video: {title_video}")
