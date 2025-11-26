"""ml_forecasting_viz.py

Plot the average engagement-rate time series and overlay fitted / forecast
lines from ARIMA (auto-selected), AR(1), and the DLM.

Usage:
  python ml_forecasting_viz.py

Outputs:
  - `outputs/engagement_rate_models.png`
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from neon_timeseries_analysis import (
    fetch_videos_from_neon, 
    prepare_timeseries_data, 
    fit_arima_model, 
    fit_ar1_model, 
    fit_dlm_model
)


def main():
    print("="*70)
    print("TIME SERIES VISUALIZATION - Engagement Rate")
    print("="*70)
    
    # Fetch data from Neon
    print("\n[1/4] Fetching data from Neon...")
    df_videos = fetch_videos_from_neon("videos_log_v2")
    
    # Prepare aggregated metrics
    print("\n[2/4] Preparing time series data...")
    minutely_data = prepare_timeseries_data(df_videos, aggregation="minutely")
    
    # If not enough minutely data, try hourly then daily
    if len(minutely_data) < 10:
        print("  Not enough minutely data, trying hourly...")
        hourly_data = prepare_timeseries_data(df_videos, aggregation="hourly")
        if len(hourly_data) < 10:
            print("  Not enough hourly data, trying daily...")
            daily_data = prepare_timeseries_data(df_videos, aggregation="daily")
            data = daily_data
        else:
            data = hourly_data
    else:
        data = minutely_data

    # Target series: avg_engagement_rate
    if "avg_engagement_rate" not in data.columns:
        raise SystemExit("avg_engagement_rate column not present in aggregated data")

    data = data[["time_bin", "avg_engagement_rate"]].dropna()
    data = data.sort_values("time_bin").reset_index(drop=True)
    dates = pd.to_datetime(data["time_bin"]).values
    series = data["avg_engagement_rate"].astype(float).values

    # Train/test split
    split_idx = int(len(series) * 0.7)
    split_idx = max(5, min(split_idx, len(series) - 5))

    dates_train = dates[:split_idx]
    dates_test = dates[split_idx:]
    y_train = series[:split_idx]
    y_test = series[split_idx:]

    print(f"\n[3/4] Fitting models...")
    print(f"  Train: {len(y_train)} points, Test: {len(y_test)} points")
    
    # Fit models
    arima_res = ar1_res = dlm_res = None
    min_points = 10
    
    if len(y_train) >= 5 and len(y_train) + len(y_test) >= min_points:
        print("\n  Fitting ARIMA (auto order selection)...")
        arima_res = fit_arima_model(y_train, y_test, order=None)
        
        print("\n  Fitting AR(1)...")
        ar1_res = fit_ar1_model(y_train, y_test)
        
        print("\n  Fitting DLM...")
        dlm_res = fit_dlm_model(y_train, y_test)
    else:
        print('  Not enough time points to fit models reliably.')

    # Build plot
    print(f"\n[4/4] Creating visualization...")
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Full observed series
    all_dates = pd.to_datetime(data["time_bin"])
    ax.plot(all_dates, data["avg_engagement_rate"].values, 
            label="Observed", color="#1f77b4", linewidth=2, alpha=0.7)

    # Plot ARIMA
    if arima_res:
        fitted = np.asarray(arima_res["fitted"])
        forecast = np.asarray(arima_res["forecast"])
        order = arima_res.get("params", "auto")
        
        if len(fitted) > 0:
            ax.plot(pd.to_datetime(dates_train[-len(fitted):]), fitted, 
                   label=f"ARIMA{order} fitted", linestyle="--", color="#ff7f0e", linewidth=2)
        if len(forecast) > 0:
            ax.plot(pd.to_datetime(dates_test[:len(forecast)]), forecast, 
                   label=f"ARIMA{order} forecast", linestyle=":", color="#ff7f0e", linewidth=2)

    # Plot AR1
    if ar1_res:
        fitted = np.asarray(ar1_res["fitted"])
        forecast = np.asarray(ar1_res["forecast"])
        
        if len(fitted) > 0:
            ax.plot(pd.to_datetime(dates_train[-len(fitted):]), fitted, 
                   label="AR(1) fitted", linestyle="--", color="#2ca02c", linewidth=2)
        if len(forecast) > 0:
            ax.plot(pd.to_datetime(dates_test[:len(forecast)]), forecast, 
                   label="AR(1) forecast", linestyle=":", color="#2ca02c", linewidth=2)

    # Plot DLM
    if dlm_res:
        fitted = np.asarray(dlm_res["fitted"])
        forecast = dlm_res.get("forecast")
        
        if len(fitted) > 0:
            ax.plot(pd.to_datetime(dates_train[-len(fitted):]), fitted, 
                   label="DLM fitted", linestyle="--", color="#d62728", linewidth=2)
        if forecast is not None:
            forecast = np.asarray(forecast)
            if len(forecast) > 0:
                ax.plot(pd.to_datetime(dates_test[:len(forecast)]), forecast, 
                       label="DLM forecast", linestyle=":", color="#d62728", linewidth=2)

    # Add vertical line at train/test split
    ax.axvline(pd.to_datetime(dates[split_idx]), color='black', 
               linestyle='--', alpha=0.3, label='Train/Test Split')

    ax.set_title("Average Engagement Rate — Observed, Fitted, and Forecasts", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Engagement Rate (%)", fontsize=12)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()

    # Save figure
    output_dir = Path('./outputs')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'engagement_rate_models.png'
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved figure to: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    main()