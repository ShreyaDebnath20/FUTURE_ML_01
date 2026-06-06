"""
Sales & Demand Forecasting — Superstore Dataset
================================================
FutureInterns ML Task 1 (2026)

This script builds a complete forecasting pipeline on the real
Sample_Superstore.csv dataset using scikit-learn. It covers:

  1. Data loading & cleaning
  2. Time-based feature engineering (trend, seasonality, lag features)
  3. Model training — Linear Regression + Random Forest
  4. Evaluation — MAE, RMSE, R²
  5. Forecasting 6 months ahead
  6. Business-friendly visualisations
  7. Seasonality index report

Run:
    pip install pandas numpy scikit-learn matplotlib
    python sales_forecasting.py

Output files (saved in the same folder):
    forecast_overall.png   — Full sales history + 6-month forecast
    forecast_by_category.png — Per-category breakdowns
    seasonality_report.png — Monthly seasonality index heatmap
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load & Clean
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = "Sample - Superstore.csv"          # ← adjust path if needed
FORECAST_MONTHS = 6                            # how far ahead to predict

print("=" * 60)
print("  Superstore Sales Forecasting Pipeline")
print("=" * 60)

df = pd.read_csv(CSV_PATH, encoding="latin1")

# Parse dates
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Ship Date"]  = pd.to_datetime(df["Ship Date"])

# Drop duplicate rows (if any)
df.drop_duplicates(inplace=True)

# Fill any missing Sales values with 0 (none expected, but safe)
df["Sales"] = df["Sales"].fillna(0)

print(f"\n[Data] {len(df):,} rows | {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
print(f"       Categories : {sorted(df['Category'].unique())}")
print(f"       Regions    : {sorted(df['Region'].unique())}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Aggregate to Monthly Totals
# ─────────────────────────────────────────────────────────────────────────────

def monthly_agg(data: pd.DataFrame) -> pd.DataFrame:
    """
    Resample transaction-level rows to monthly sales totals and
    add all time-based features needed by the model.
    """
    data = data.copy()
    data["YearMonth"] = data["Order Date"].dt.to_period("M")
    monthly = (
        data.groupby("YearMonth")["Sales"]
        .sum()
        .reset_index()
        .rename(columns={"Sales": "total_sales"})
    )
    monthly["date"]        = monthly["YearMonth"].dt.to_timestamp()
    monthly["year"]        = monthly["date"].dt.year
    monthly["month"]       = monthly["date"].dt.month
    monthly["month_index"] = range(len(monthly))   # linear time index

    # Cyclical encoding of month (captures Jan≈Dec proximity)
    monthly["sin_month"]   = np.sin(2 * np.pi * monthly["month"] / 12)
    monthly["cos_month"]   = np.cos(2 * np.pi * monthly["month"] / 12)

    # Lag features — previous 1, 2, and 3 months
    monthly["lag_1"] = monthly["total_sales"].shift(1)
    monthly["lag_2"] = monthly["total_sales"].shift(2)
    monthly["lag_3"] = monthly["total_sales"].shift(3)

    # 3-month rolling mean (trend smoother)
    monthly["rolling_3m"] = monthly["total_sales"].shift(1).rolling(3).mean()

    # Quarter dummy
    monthly["quarter"] = monthly["date"].dt.quarter

    monthly.dropna(inplace=True)   # drop rows with NaN lags
    monthly.reset_index(drop=True, inplace=True)
    return monthly


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Build Features & Labels
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "month_index", "year", "month",
    "sin_month",   "cos_month",
    "lag_1", "lag_2", "lag_3",
    "rolling_3m",  "quarter"
]

def prepare_xy(monthly: pd.DataFrame):
    X = monthly[FEATURES].values
    y = monthly["total_sales"].values
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Train & Evaluate
# ─────────────────────────────────────────────────────────────────────────────
def train_evaluate(monthly: pd.DataFrame, label: str = "Overall"):
    """
    Train-test split (last 6 months = test), fit both models,
    print a metrics table, and return the better model + monthly df.
    """
    X, y = prepare_xy(monthly)

    split = len(monthly) - 6          # last 6 months as hold-out test set
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest    ": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42
        ),
    }

    results = {}
    print(f"\n-- {label} ----------------------------------")
    print(f"{'Model':<24} {'MAE':>10} {'RMSE':>10} {'R2':>8}")
    print("-" * 55)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)

        print(f"{name:<24} ${mae:>9,.0f} ${rmse:>9,.0f} {r2:>7.1%}")
        results[name] = {"model": model, "mae": mae, "rmse": rmse, "r2": r2}

    # Pick the model with higher R²
    best_name = max(results, key=lambda k: results[k]["r2"])
    best      = results[best_name]
    print(f"\n  * Best model: {best_name.strip()}  (R2 = {best['r2']:.1%})")
    return best["model"], best["mae"], best["rmse"]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Forecast Future Months
# ─────────────────────────────────────────────────────────────────────────────
def forecast_future(model, monthly: pd.DataFrame, n_months: int = 6):
    """
    Iterative multi-step forecast: each predicted value is fed back
    as a lag feature for the next step (realistic out-of-sample forecast).
    """
    last_row  = monthly.iloc[-1]
    last_date = last_row["date"]
    last_idx  = last_row["month_index"]

    # Seed the rolling buffer with the most recent actuals
    recent_sales = list(monthly["total_sales"].values[-3:])

    future_rows = []
    for i in range(1, n_months + 1):
        future_date = last_date + pd.DateOffset(months=i)
        idx         = last_idx + i
        month       = future_date.month
        year        = future_date.year

        lag1 = recent_sales[-1]
        lag2 = recent_sales[-2]
        lag3 = recent_sales[-3]
        roll = np.mean(recent_sales[-3:])

        row = [
            idx, year, month,
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            lag1, lag2, lag3,
            roll, (month - 1) // 3 + 1
        ]
        pred = model.predict([row])[0]

        future_rows.append({
            "date":        future_date,
            "total_sales": pred,
            "forecast":    True,
        })
        recent_sales.append(pred)
        if len(recent_sales) > 3:
            recent_sales.pop(0)

    return pd.DataFrame(future_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Seasonality Index
# ─────────────────────────────────────────────────────────────────────────────
def seasonality_index(monthly: pd.DataFrame) -> pd.Series:
    """
    Classic ratio-to-moving-average seasonality index.
    Values > 1 → above-average months; < 1 → below-average months.
    """
    avg_by_month = monthly.groupby("month")["total_sales"].mean()
    grand_mean   = avg_by_month.mean()
    return (avg_by_month / grand_mean).round(3)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
MONTH_NAMES  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
PALETTE      = {"hist": "#378ADD", "forecast": "#1D9E75",
                "band": "#1D9E75", "neutral": "#888780"}

def fmt_k(x, _):
    return f"${x/1000:.0f}K"


def plot_forecast(monthly: pd.DataFrame, future: pd.DataFrame,
                  rmse: float, title: str, ax: plt.Axes):
    """Draw historical sales + forecast ribbon on a given Axes."""
    ax.fill_between(monthly["date"], 0, monthly["total_sales"],
                    alpha=0.08, color=PALETTE["hist"])
    ax.plot(monthly["date"], monthly["total_sales"],
            color=PALETTE["hist"], lw=2, label="Historical sales")

    # Connect last historical point to first forecast
    bridge_dates  = [monthly["date"].iloc[-1], future["date"].iloc[0]]
    bridge_values = [monthly["total_sales"].iloc[-1], future["total_sales"].iloc[0]]
    ax.plot(bridge_dates, bridge_values,
            color=PALETTE["forecast"], lw=2, linestyle="--")

    hi = future["total_sales"] + rmse * 1.5
    lo = (future["total_sales"] - rmse * 1.5).clip(lower=0)
    ax.fill_between(future["date"], lo, hi,
                    alpha=0.2, color=PALETTE["band"], label="Confidence band (±1.5σ)")
    ax.plot(future["date"], future["total_sales"],
            color=PALETTE["forecast"], lw=2.5, linestyle="--",
            marker="o", markersize=5, label="Forecast")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Run the full pipeline
# ─────────────────────────────────────────────────────────────────────────────

# ── 8a. Overall (all categories combined) ────────────────────────────────────
monthly_all    = monthly_agg(df)
model_all, mae_all, rmse_all = train_evaluate(monthly_all, "Overall")
future_all     = forecast_future(model_all, monthly_all, FORECAST_MONTHS)
si_all         = seasonality_index(monthly_all)

# ── 8b. Per category ─────────────────────────────────────────────────────────
categories = df["Category"].unique()
cat_results = {}
for cat in sorted(categories):
    cat_df = df[df["Category"] == cat]
    mdf    = monthly_agg(cat_df)
    mdl, mae_c, rmse_c = train_evaluate(mdf, cat)
    fut    = forecast_future(mdl, mdf, FORECAST_MONTHS)
    cat_results[cat] = {"monthly": mdf, "future": fut,
                        "rmse": rmse_c, "mae": mae_c}

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Plot 1 — Overall Forecast
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                          gridspec_kw={"height_ratios": [2, 1]})
fig.suptitle("Superstore — Sales Forecast Dashboard",
             fontsize=15, fontweight="bold", y=0.98)

# Top panel: forecast line
plot_forecast(monthly_all, future_all, rmse_all,
              "Overall Monthly Sales + 6-Month Forecast", axes[0])
legend_patches = [
    Patch(color=PALETTE["hist"],     alpha=0.7, label="Historical sales"),
    Patch(color=PALETTE["forecast"], alpha=0.7, label="Forecast"),
    Patch(color=PALETTE["band"],     alpha=0.2, label="Confidence band"),
]
axes[0].legend(handles=legend_patches, fontsize=9, framealpha=0.5)

# Bottom panel: seasonality bar chart
colors = [
    "#1D9E75" if v >= 1.15 else
    "#378ADD" if v >= 1.00 else
    "#888780" if v >= 0.90 else
    "#E24B4A"
    for v in si_all.values
]
axes[1].bar([MONTH_NAMES[m - 1] for m in si_all.index],
            si_all.values, color=colors, edgecolor="white", linewidth=0.5)
axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
axes[1].set_title("Monthly Seasonality Index  (1.0 = average)",
                  fontsize=11, fontweight="bold")
axes[1].set_ylabel("Index", fontsize=9)
axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
axes[1].spines[["top", "right"]].set_visible(False)
axes[1].grid(axis="y", alpha=0.3, linestyle=":")

# Annotate bars
for i, (m, v) in enumerate(si_all.items()):
    axes[1].text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold")

legend_si = [
    Patch(color="#1D9E75", label="Peak  (≥1.15×)"),
    Patch(color="#378ADD", label="Above avg (1.0–1.15×)"),
    Patch(color="#888780", label="Below avg (0.9–1.0×)"),
    Patch(color="#E24B4A", label="Low  (<0.9×)"),
]
axes[1].legend(handles=legend_si, fontsize=8, framealpha=0.5,
               loc="upper left", ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("forecast_overall.png", dpi=150, bbox_inches="tight")
print("\n[Saved] forecast_overall.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Plot 2 — Per-Category Forecasts
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5), sharey=False)
fig2.suptitle("Sales Forecast by Product Category",
              fontsize=14, fontweight="bold", y=1.01)

for ax, cat in zip(axes2, sorted(categories)):
    r = cat_results[cat]
    plot_forecast(r["monthly"], r["future"], r["rmse"], cat, ax)

handles = [
    Patch(color=PALETTE["hist"],     alpha=0.7, label="Historical"),
    Patch(color=PALETTE["forecast"], alpha=0.7, label="Forecast"),
    Patch(color=PALETTE["band"],     alpha=0.2, label="Confidence band"),
]
axes2[1].legend(handles=handles, fontsize=8, framealpha=0.5,
                loc="upper left")

plt.tight_layout()
plt.savefig("forecast_by_category.png", dpi=150, bbox_inches="tight")
print("[Saved] forecast_by_category.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 11.  Plot 3 — Seasonality Heatmap (all categories)
# ─────────────────────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(12, 3.5))
fig3.suptitle("Seasonality Index Heatmap — by Category",
              fontsize=13, fontweight="bold")

cat_si = {}
for cat in sorted(categories):
    cat_si[cat] = seasonality_index(cat_results[cat]["monthly"])

si_matrix = pd.DataFrame(cat_si).T                    # shape: (3 cats × 12 months)
si_matrix.columns = MONTH_NAMES

im = ax3.imshow(si_matrix.values, cmap="RdYlGn", aspect="auto",
                vmin=0.6, vmax=1.6)

ax3.set_xticks(range(12))
ax3.set_xticklabels(MONTH_NAMES, fontsize=10)
ax3.set_yticks(range(len(si_matrix)))
ax3.set_yticklabels(si_matrix.index, fontsize=10)

for i in range(si_matrix.shape[0]):
    for j in range(si_matrix.shape[1]):
        val   = si_matrix.values[i, j]
        color = "white" if val > 1.35 or val < 0.75 else "black"
        ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=9, color=color, fontweight="bold")

cbar = plt.colorbar(im, ax=ax3, fraction=0.02, pad=0.04)
cbar.set_label("Seasonality Index", fontsize=9)
ax3.set_title("Green = above average demand  |  Red = below average demand",
              fontsize=9, style="italic", pad=4)

plt.tight_layout()
plt.savefig("seasonality_report.png", dpi=150, bbox_inches="tight")
print("[Saved] seasonality_report.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 12.  Business Summary — printed to console
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  BUSINESS PLANNING SUMMARY")
print("=" * 60)

avg_monthly = monthly_all["total_sales"].mean()
forecast_total = future_all["total_sales"].sum()
last_6_actual  = monthly_all["total_sales"].iloc[-6:].sum()
growth_pct     = (forecast_total - last_6_actual) / last_6_actual * 100

peak_month     = MONTH_NAMES[si_all.idxmax() - 1]
low_month      = MONTH_NAMES[si_all.idxmin() - 1]
peak_idx       = si_all.max()
low_idx        = si_all.min()

print(f"\n  Average monthly sales (historical) : ${avg_monthly:>10,.0f}")
print(f"  Forecast total (next {FORECAST_MONTHS} months)    : ${forecast_total:>10,.0f}")
print(f"  vs. previous {FORECAST_MONTHS} months            :  {growth_pct:>+.1f}%")
print(f"\n  Peak demand month  : {peak_month}  (index {peak_idx:.2f}x  "
      f"-> stock up ~{peak_idx*100-100:.0f}% above normal)")
print(f"  Slow demand month  : {low_month}  (index {low_idx:.2f}x  "
      f"-> reduce orders ~{100-low_idx*100:.0f}% below normal)")
print(f"\n  Model error (RMSE) : ${rmse_all:>10,.0f}")
print(f"  Model error (MAE)  : ${mae_all:>10,.0f}")
print(f"\n  Recommended actions:")
print(f"  - Stock up for {peak_month} ~6 weeks in advance")
print(f"  - Run promotions in {low_month} to clear slow-moving inventory")
print(f"  - Use confidence band (+/-${rmse_all*1.5:,.0f}) as safety-stock buffer")
print("\n" + "=" * 60)
print("  Pipeline complete.  Check the 3 PNG files. *")
print("=" * 60 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# 13. Export forecast data for the interactive dashboard
# ─────────────────────────────────────────────────────────────────────────────
import json

overall_historical = []
for idx, row in monthly_all.iterrows():
    overall_historical.append({
        "date": row["date"].strftime("%Y-%m-%d"),
        "sales": float(row["total_sales"])
    })

overall_forecast = []
for idx, row in future_all.iterrows():
    pred = float(row["total_sales"])
    overall_forecast.append({
        "date": row["date"].strftime("%Y-%m-%d"),
        "sales": pred,
        "lower": max(0.0, pred - rmse_all * 1.5),
        "upper": pred + rmse_all * 1.5
    })

overall_seasonality = []
for m_idx, val in si_all.items():
    overall_seasonality.append({
        "month": MONTH_NAMES[m_idx - 1],
        "index": float(val)
    })

metrics = {
    "avg_monthly": float(avg_monthly),
    "forecast_total": float(forecast_total),
    "growth_pct": float(growth_pct),
    "peak_month": str(peak_month),
    "peak_index": float(peak_idx),
    "low_month": str(low_month),
    "low_index": float(low_idx),
    "rmse": float(rmse_all),
    "mae": float(mae_all)
}

data_to_save = {
    "overall": {
        "historical": overall_historical,
        "forecast": overall_forecast,
        "seasonality": overall_seasonality
    },
    "categories": {},
    "metrics": metrics
}

for cat in sorted(categories):
    r = cat_results[cat]
    mdf = r["monthly"]
    fut = r["future"]
    rmse_c = r["rmse"]
    mae_c = r["mae"]
    
    # Category historical
    cat_hist = []
    for idx, row in mdf.iterrows():
        cat_hist.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "sales": float(row["total_sales"])
        })
        
    # Category forecast
    cat_fore = []
    for idx, row in fut.iterrows():
        pred = float(row["total_sales"])
        cat_fore.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "sales": pred,
            "lower": max(0.0, pred - rmse_c * 1.5),
            "upper": pred + rmse_c * 1.5
        })
        
    # Category seasonality
    c_si = seasonality_index(mdf)
    cat_sea = []
    for m_idx, val in c_si.items():
        cat_sea.append({
            "month": MONTH_NAMES[m_idx - 1],
            "index": float(val)
        })
        
    data_to_save["categories"][cat] = {
        "historical": cat_hist,
        "forecast": cat_fore,
        "seasonality": cat_sea,
        "rmse": float(rmse_c),
        "mae": float(mae_c)
    }

with open("forecast_data.json", "w") as f:
    json.dump(data_to_save, f, indent=2)

print("[Saved] forecast_data.json")
print("=" * 60 + "\n")
