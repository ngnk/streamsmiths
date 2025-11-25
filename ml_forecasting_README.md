# ML Forecasting (ml_forecasting.py)

This document explains how to run `ml_forecasting.py`, summarizes and interprets the terminal results from the most recent run, and points to a visualization helper that plots the average engagement rate with fitted models.

Files added by the project:
- `ml_forecasting.py` — main analysis script that: loads silver parquet data, aggregates to hourly/daily, fits three models (ARIMA(1,1,1), AR(1), and a custom DLM forward-backward smoother), compares performance, and writes results to `./results`.
- `ml_forecasting_viz.py` — visualization helper (in the same folder) that plots the average engagement rate time series and overlays fitted/forecasted model outputs.

How to run

1. Install dependencies (if not already):

```bash
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 -m pip install -r requirements.txt
# optional: if you want pmdarima for auto-order selection
python3 -m pip install pmdarima
```

2. Run the forecasting analysis (this will produce `results/model_comparison.csv` and a pickle `results/model_results.pkl` when possible):

```bash
python3 ml_forecasting.py
```

3. Run the visualization for engagement rate (creates a plot):

```bash
python3 ml_forecasting_viz.py
```

Interpreting the terminal results (from the most recent run)

The script prints summaries for three targets: Total Views, Total Likes, and Average Engagement Rate. Key observations from the run captured on Nov 25, 2025:

- Data: the script fetched ~247,849 raw video rows and aggregated to 216 time points (hourly aggregation), covering 2025-11-15 to 2025-11-25.

- Model candidates: ARIMA(1,1,1), AR(1) (AutoReg), and a custom Dynamic Linear Model (DLM) with forward filtering and backward smoothing.

- Total Views:
  - DLM shows the lowest AIC (5525) and highest train R2 (~0.963), suggesting it fits the historical data best.
  - On the test set, however, ARIMA produced a lower Test RMSE (~79M) than DLM (~159M), and ARIMA's Test R2 (-0.23) is less negative than DLM (-3.97). This indicates DLM may overfit training data (good in-sample fit but poorer out-of-sample generalization).

- Total Likes:
  - DLM again has best AIC and best Train metrics, but test performance shows ARIMA with lower Test RMSE (~933k) than DLM (~1.8M). DLM's test R2 is negative, also pointing to possible overfitting or data non-stationarity.

- Average Engagement Rate:
  - DLM has the best training metrics (low RMSE) and very small Test RMSE reported (~0.0457). However, R2 and MAPE values are unstable/very large for some models on test sets — this can happen when the target has low variance, when values approach zero, or when the R2 denominator is near zero. Use these diagnostics with caution.

Overall interpretation and practical guidance

- The DLM fits historical data very closely (lowest AIC, strong Train R2), but the test results show inconsistent out-of-sample generalization for some targets. This suggests either the model is overfitting, or the time series contains structural breaks/nonstationary behavior in the test window.
- ARIMA(1,1,1) is a reasonable baseline and sometimes generalizes better on the test set (e.g., Total Views). If prediction quality matters more than in-sample fit, prefer simpler models validated by held-out test metrics.
- The engagement-rate target needs careful preprocessing and potentially transformation (log or scaling) before training. Very small denominators or outliers can distort MAPE and R2.

Next steps you may want to try

- Increase train window or use time-series cross-validation to better estimate generalization.
- Add seasonal components (SARIMA) if the series shows daily/weekly cycles.
- Use robust metrics and visualization to inspect forecast residuals and check model assumptions.

Where results are saved

- `./results/model_comparison.csv` — comparison table with AIC/BIC and RMSE for train/test across models.
- `./results/model_results.pkl` — (when present) pickle with the detailed model outputs (fitted arrays, forecasts) used by visualizations.

If you'd like, I can:
- Re-run the forecasting flow and save all model artifacts (ensure the pickle is created).
- Modify the scripts to produce an interactive HTML plot (Plotly) for the engagement-rate overlays.

