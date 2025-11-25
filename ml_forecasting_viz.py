"""ml_forecasting_viz.py

Plot the average engagement-rate time series and overlay fitted / forecast
lines from ARIMA(1,1,1), AR(1), and the DLM implemented in `ml_forecasting.py`.

Usage:
  python3 ml_forecasting_viz.py --horizon 65

Outputs:
  - `outputs/engagement_rate_models.png`
"""
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import polars as pl
from ml_forecasting import prepare_timeseries_data, fit_arima_model, fit_ar1_model, fit_dlm_model


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot engagement rate and fitted model overlays")
    parser.add_argument("--parquet", default="silver_data/videos.parquet", help="Path to videos parquet")
    parser.add_argument("--horizon", type=int, default=65, help="Forecast horizon (test size in points)")
    parser.add_argument("--output", default="outputs/engagement_rate_models.png", help="Output PNG path")
    args = parser.parse_args(argv)

    p = Path(args.parquet)
    if not p.exists():
        raise SystemExit(f"Parquet not found: {p}. Run silver pipeline first.")

    # Prepare aggregated metrics (same logic as `ml_forecasting.py`)
    df_pl = pl.read_parquet(str(p))

    # Adapt column names to match expectations in `prepare_timeseries_data`
    cols = set(df_pl.columns)
    if 'ingestion_timestamp' not in cols and 'ingest_timestamp' in cols:
        df_pl = df_pl.rename({'ingest_timestamp': 'ingestion_timestamp'})

    # Ensure `views_per_day` exists (prepare_timeseries_data aggregates it); set to 0 if missing
    if 'views_per_day' not in set(df_pl.columns):
        df_pl = df_pl.with_columns([pl.lit(0).alias('views_per_day')])

    # Check whether ingestion timestamps provide sufficient time coverage
    pdf = df_pl.to_pandas()
    if 'ingestion_timestamp' in pdf.columns:
        try:
            ts = pd.to_datetime(pdf['ingestion_timestamp'])
            unique_hours = ts.dt.floor('H').nunique()
        except Exception:
            unique_hours = 0
    else:
        unique_hours = 0

    # If ingestion timestamps are concentrated (e.g., a single ingestion run), fall back to published_at daily
    if unique_hours < 10 and 'published_at' in pdf.columns:
        print('\n[PREPARE] ingestion timestamps have low time coverage — falling back to daily published_at aggregation')
        pdf['published_at'] = pd.to_datetime(pdf['published_at'], errors='coerce')
        pdf = pdf.dropna(subset=['published_at'])
        pdf['time_bin'] = pdf['published_at'].dt.floor('D')

        # Compute engagement_rate row-wise if not present
        if 'engagement_rate' not in pdf.columns:
            pdf['engagement_rate'] = ((pdf['like_count'].fillna(0) + pdf['comment_count'].fillna(0)) /
                                      pdf['view_count'].replace(0, 1)) * 100

        agg = pdf.groupby('time_bin').agg({
            'view_count': 'sum',
            'like_count': 'sum',
            'comment_count': 'sum',
            'engagement_rate': 'mean',
            'views_per_day': 'mean',
            'video_id': 'count'
        }).reset_index()

        agg.columns = ['time_bin', 'total_views', 'total_likes', 'total_comments',
                       'avg_engagement_rate', 'avg_views_per_day', 'num_videos']
        data = agg.sort_values('time_bin').reset_index(drop=True)
    else:
        data = prepare_timeseries_data(df_pl, aggregation="hourly")

    # Target series: avg_engagement_rate
    if "avg_engagement_rate" not in data.columns:
        raise SystemExit("avg_engagement_rate column not present in aggregated data")

    # Keep datetime index (ml_forecasting uses 'time_bin' and returns a pandas DataFrame)
    # `prepare_timeseries_data` returns a pandas DataFrame with 'time_bin' and 'avg_engagement_rate'
    data = data[["time_bin", "avg_engagement_rate"]].dropna()
    data = data.sort_values("time_bin").reset_index(drop=True)
    dates = pd.to_datetime(data["time_bin"]).values
    series = data["avg_engagement_rate"].astype(float).values

    # Train/test split similar to run_model_comparison
    split_idx = int(len(series) * 0.7)
    split_idx = max(5, min(split_idx, len(series) - 5))

    dates_train = dates[:split_idx]
    dates_test = dates[split_idx:]
    y_train = series[:split_idx]
    y_test = series[split_idx:]

    # Fit models only if we have enough data points
    arima_res = ar1_res = dlm_res = None
    min_points = 10
    if len(y_train) >= 5 and len(y_train) + len(y_test) >= min_points:
        arima_res = fit_arima_model(y_train, y_test, order=(1, 1, 1))
        ar1_res = fit_ar1_model(y_train, y_test)
        dlm_res = fit_dlm_model(y_train, y_test)
    else:
        print('\n[INFO] Not enough time points to fit models reliably. Observed series will be plotted only.')

    # Build plot
    fig, ax = plt.subplots(figsize=(12, 6))
    # For plotting the full observed series we can use the combined dates
    all_dates = pd.to_datetime(data["time_bin"])
    ax.plot(all_dates, data["avg_engagement_rate"].values, label="Observed (avg engagement rate)", color="#1f77b4")

    # Plot ARIMA fitted and forecast
    if arima_res and arima_res.get("fitted") is not None and arima_res.get("forecast") is not None:
        fitted = np.asarray(arima_res["fitted"])
        forecast = np.asarray(arima_res["forecast"])
        if len(fitted) > 0:
            ax.plot(pd.to_datetime(dates_train[-len(fitted):]), fitted, label="ARIMA fitted", linestyle="--", color="#ff7f0e")
        if len(forecast) > 0:
            ax.plot(pd.to_datetime(dates_test[:len(forecast)]), forecast, label="ARIMA forecast", linestyle=":", color="#ff7f0e")

    # Plot AR1
    if ar1_res and ar1_res.get("fitted") is not None and ar1_res.get("forecast") is not None:
        fitted = np.asarray(ar1_res["fitted"])
        forecast = np.asarray(ar1_res["forecast"])
        if len(fitted) > 0:
            ax.plot(pd.to_datetime(dates_train[-len(fitted):]), fitted, label="AR(1) fitted", linestyle="--", color="#2ca02c")
        if len(forecast) > 0:
            ax.plot(pd.to_datetime(dates_test[:len(forecast)]), forecast, label="AR(1) forecast", linestyle=":", color="#2ca02c")

    # Plot DLM
    if dlm_res and dlm_res.get("fitted") is not None:
        fitted = np.asarray(dlm_res["fitted"])
        forecast = dlm_res.get("forecast")
        if len(fitted) > 0:
            ax.plot(pd.to_datetime(dates_train[-len(fitted):]), fitted, label="DLM fitted", linestyle="--", color="#d62728")
        if forecast is not None:
            forecast = np.asarray(forecast)
            if len(forecast) > 0:
                ax.plot(pd.to_datetime(dates_test[:len(forecast)]), forecast, label="DLM forecast", linestyle=":", color="#d62728")

    ax.set_title("Average Engagement Rate — Observed, fitted, and forecasts")
    ax.set_xlabel("Date")
    ax.set_ylabel("Engagement rate (%)")
    ax.legend()
    ax.grid(alpha=0.2)

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outp, dpi=150)
    print(f"Saved figure to: {outp}")


if __name__ == "__main__":
    main()
