"""
visualize_matplotlib.py
========================
Standalone Matplotlib visualization for Future Interns ML Task 1.

Reads forecast_data.json (produced by sales_forecasting_1.py) and
generates 5 high-quality, business-ready charts + a combined PDF report.

Run:
    python visualize_matplotlib.py

Outputs (saved in the same folder):
    chart1_overall_forecast.png
    chart2_category_forecasts.png
    chart3_seasonality_heatmap.png
    chart4_peak_low_comparison.png
    chart5_business_summary.png
    sales_forecast_report.pdf
"""

import json, warnings, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Colour palette ────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b27"
GRID    = "#1e2736"
T1      = "#e6edf3"
T2      = "#8b949e"
C_HIST  = "#58a6ff"
C_FORE  = "#3fb950"
C_BAND  = "#3fb950"
C_PEAK  = "#3fb950"
C_LOW   = "#f85149"
C_FURN  = "#f78166"
C_OFF   = "#d2a8ff"
C_TECH  = "#ffa657"

CAT_COL = {"Furniture": C_FURN, "Office Supplies": C_OFF, "Technology": C_TECH}
MONTHS  = ["Jan","Feb","Mar","Apr","May","Jun",
           "Jul","Aug","Sep","Oct","Nov","Dec"]

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  BG,   "axes.facecolor":   PANEL,
    "axes.edgecolor":    GRID,  "axes.labelcolor":  T1,
    "axes.titlecolor":   T1,    "xtick.color":      T2,
    "ytick.color":       T2,    "grid.color":       GRID,
    "grid.linestyle":    "-",   "grid.linewidth":   0.5,
    "text.color":        T1,    "legend.facecolor": "#1e2736",
    "legend.edgecolor":  "#30363d", "legend.labelcolor": T1,
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size":         10,    "axes.titlesize":   13,
    "axes.titleweight":  "bold","axes.titlepad":    12,
    "axes.labelsize":    9,     "figure.dpi":       150,
    "savefig.dpi":       180,   "savefig.bbox":     "tight",
    "savefig.facecolor": BG,
})

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data(path):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] '{path}' not found. Run sales_forecasting_1.py first.")
    with open(path) as f:
        return json.load(f)

def pdates(records):
    for r in records:
        r["dt"] = datetime.strptime(r["date"], "%Y-%m-%d")
    return records

def fmtk(x, _=None):
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    return f"${x/1_000:.0f}K"

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

def watermark(fig, txt="Future Interns ML Task 1 (2026) | scikit-learn & Matplotlib"):
    fig.text(0.99, 0.01, txt, ha="right", va="bottom",
             fontsize=7, color=T2, style="italic", alpha=0.7)

# =============================================================================
# CHART 1 -- Overall Sales History + 6-Month Forecast
# =============================================================================
def chart1(data):
    overall  = data["overall"]
    metrics  = data["metrics"]
    hist     = pdates(overall["historical"])
    fore     = pdates(overall["forecast"])
    sea      = overall["seasonality"]

    hd = [r["dt"]    for r in hist]
    hv = [r["sales"] for r in hist]
    fd = [r["dt"]    for r in fore]
    fv = [r["sales"] for r in fore]
    fl = [r["lower"] for r in fore]
    fu = [r["upper"] for r in fore]

    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            height_ratios=[2.2, 1], hspace=0.45, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # --- top: forecast line chart ---
    style_ax(ax1)
    ax1.fill_between(hd, hv, alpha=0.07, color=C_HIST)
    ax1.plot(hd, hv, color=C_HIST, lw=2, label="Historical Sales", zorder=3)
    ax1.axvline(x=fd[0], color=T2, lw=1, linestyle=":", alpha=0.55)
    ax1.text(fd[0], max(hv) * 1.05, "  Forecast Begins",
             va="top", ha="left", fontsize=8, color=T2, style="italic")
    ax1.fill_between(fd, fl, fu, alpha=0.18, color=C_BAND,
                     label="Confidence Band (+/-1.5 sigma)", zorder=2)
    ax1.plot([hd[-1], fd[0]], [hv[-1], fv[0]],
             color=C_FORE, lw=1.5, linestyle="--", alpha=0.6, zorder=4)
    ax1.plot(fd, fv, color=C_FORE, lw=2.5, linestyle="--",
             marker="o", markersize=6,
             markerfacecolor=C_FORE, markeredgecolor=BG, markeredgewidth=1.5,
             label="6-Month Forecast", zorder=5)
    for d, v in zip(fd, fv):
        ax1.annotate(fmtk(v), xy=(d, v), xytext=(0, 14),
                     textcoords="offset points",
                     ha="center", fontsize=7.5,
                     color=C_FORE, fontweight="bold")

    ax1.set_title("Superstore  --  Overall Monthly Sales + 6-Month Demand Forecast")
    ax1.set_ylabel("Monthly Sales (USD)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmtk))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.6)

    # --- bottom-left: seasonality bars ---
    style_ax(ax2)
    sm = [s["month"] for s in sea]
    sv = [s["index"] for s in sea]
    cols = [C_PEAK if v >= 1.0 else C_LOW for v in sv]
    bars = ax2.bar(sm, sv, color=cols, width=0.7,
                   edgecolor=BG, linewidth=0.8, zorder=3)
    ax2.axhline(1.0, color=T2, lw=1, linestyle="--", alpha=0.5)
    ax2.set_title("Monthly Seasonality Index  (1.0 = average)", fontsize=10)
    ax2.set_ylabel("Index", fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.setp(ax2.get_xticklabels(), fontsize=7.5)
    for bar, v in zip(bars, sv):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                 f"{v:.2f}", ha="center", va="bottom",
                 fontsize=7, fontweight="bold", color=T1)

    # --- bottom-right: KPI table ---
    ax3.set_facecolor(PANEL)
    ax3.axis("off")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
    ax3.text(0.5, 0.97, "Business Planning KPIs",
             ha="center", va="top", fontsize=10, fontweight="bold", color=T1)

    m = metrics
    rows = [
        ("Avg Monthly Sales",    fmtk(m["avg_monthly"]),        C_HIST),
        ("6-Month Forecast",     fmtk(m["forecast_total"]),     C_FORE),
        ("Growth vs Prior 6M",   f"{m['growth_pct']:+.1f}%",    C_PEAK if m['growth_pct'] >= 0 else C_LOW),
        ("Peak Season",          f"{m['peak_month']} ({m['peak_index']:.2f}x)", C_PEAK),
        ("Slow Season",          f"{m['low_month']} ({m['low_index']:.2f}x)",   C_LOW),
        ("Model RMSE",           fmtk(m["rmse"]),               T2),
        ("Model MAE",            fmtk(m["mae"]),                T2),
        ("Safety Stock Buffer",  fmtk(m["rmse"] * 1.5),        C_FURN),
    ]
    row_h = 0.82 / len(rows)
    for i, (lbl, val, col) in enumerate(rows):
        y = 0.88 - i * row_h
        ax3.text(0.05, y, lbl + ":", ha="left", va="center",
                 fontsize=8, color=T2, transform=ax3.transAxes)
        ax3.text(0.97, y, val, ha="right", va="center",
                 fontsize=8.5, fontweight="bold", color=col,
                 transform=ax3.transAxes)
        # separator line using a thin Rectangle instead of axhline
        sep_y = y - row_h * 0.42
        ax3.add_patch(plt.Rectangle((0.03, sep_y), 0.94, 0.002,
                                    transform=ax3.transAxes,
                                    color=GRID, clip_on=False, zorder=5))

    watermark(fig)
    fig.savefig("chart1_overall_forecast.png")
    print("[Saved] chart1_overall_forecast.png")
    return fig


# =============================================================================
# CHART 2 -- Per-Category Forecasts
# =============================================================================
def chart2(data):
    cats = ["Furniture", "Office Supplies", "Technology"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Sales Forecast by Product Category  --  6-Month Out-of-Sample Prediction",
                 fontsize=13, color=T1, fontweight="bold", y=1.01)

    for ax, cat in zip(axes, cats):
        col  = CAT_COL[cat]
        node = data["categories"][cat]
        hist = pdates(node["historical"])
        fore = pdates(node["forecast"])
        rmse = node["rmse"]

        hd = [r["dt"]    for r in hist];  hv = [r["sales"] for r in hist]
        fd = [r["dt"]    for r in fore];  fv = [r["sales"] for r in fore]
        fl = [r["lower"] for r in fore];  fu = [r["upper"] for r in fore]

        style_ax(ax)
        ax.fill_between(hd, hv, alpha=0.07, color=col)
        ax.plot(hd, hv, color=col, lw=2, label="Historical", zorder=3)
        ax.fill_between(fd, fl, fu, alpha=0.2, color=C_FORE,
                        label="Conf. Band", zorder=2)
        ax.plot([hd[-1], fd[0]], [hv[-1], fv[0]],
                color=C_FORE, lw=1.2, linestyle="--", alpha=0.6)
        ax.plot(fd, fv, color=C_FORE, lw=2.2, linestyle="--",
                marker="o", markersize=5,
                markerfacecolor=C_FORE, markeredgecolor=BG, markeredgewidth=1.5,
                label="Forecast", zorder=5)

        # Label last forecast point
        ax.annotate(fmtk(fv[-1]),
                    xy=(fd[-1], fv[-1]), xytext=(-6, 16),
                    textcoords="offset points", ha="right",
                    fontsize=7.5, color=C_FORE, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C_FORE, lw=0.8))

        ax.set_title(cat, fontsize=12, color=col)
        ax.set_ylabel("Monthly Sales (USD)", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmtk))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=8))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7.5)
        ax.text(0.02, 0.97, f"RMSE: {fmtk(rmse)}/mo",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, color=T2,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL,
                          edgecolor=GRID, alpha=0.85))
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.6)

    plt.tight_layout()
    watermark(fig)
    fig.savefig("chart2_category_forecasts.png")
    print("[Saved] chart2_category_forecasts.png")
    return fig


# =============================================================================
# CHART 3 -- Seasonality Heatmap (all categories)
# =============================================================================
def chart3(data):
    cats   = ["Furniture", "Office Supplies", "Technology"]
    matrix = np.zeros((len(cats), 12))
    for ri, cat in enumerate(cats):
        for s in data["categories"][cat]["seasonality"]:
            matrix[ri, MONTHS.index(s["month"])] = s["index"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                   gridspec_kw={"height_ratios": [3, 1], "hspace": 0.45})
    fig.suptitle("Seasonal Demand Analysis  --  Superstore Dataset",
                 fontsize=14, color=T1, fontweight="bold")

    # --- Heatmap ---
    im = ax1.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.8)
    ax1.set_xticks(range(12));  ax1.set_xticklabels(MONTHS, fontsize=9)
    ax1.set_yticks(range(len(cats))); ax1.set_yticklabels(cats, fontsize=10)
    ax1.set_facecolor(PANEL)
    for ri in range(len(cats)):
        for ci in range(12):
            v     = matrix[ri, ci]
            color = "black" if 0.72 < v < 1.5 else "white"
            ax1.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                     fontsize=9, fontweight="bold", color=color)

    cbar = plt.colorbar(im, ax=ax1, fraction=0.018, pad=0.03)
    cbar.set_label("Seasonality Index", fontsize=9, color=T1)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=T2, fontsize=8)
    ax1.set_title("Seasonality Index by Category  (Green=High Demand | Red=Low Demand | 1.0=Avg)",
                  fontsize=11)
    ax1.spines[:].set_color(GRID)

    # --- Overall bar ---
    style_ax(ax2)
    sea = data["overall"]["seasonality"]
    mo  = [s["month"] for s in sea]
    iv  = [s["index"] for s in sea]
    cols_bar = [C_PEAK if v >= 1.0 else C_LOW for v in iv]
    bars = ax2.bar(mo, iv, color=cols_bar, edgecolor=BG, linewidth=0.8,
                   width=0.7, zorder=3)
    ax2.axhline(1.0, color=T2, lw=1, linestyle="--", alpha=0.6)
    ax2.set_title("Overall Business Seasonality Index", fontsize=10)
    ax2.set_ylabel("Index", fontsize=8)
    for bar, v in zip(bars, iv):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=7.5,
                 fontweight="bold", color=T1)
    ax2.legend(handles=[
        mpatches.Patch(color=C_PEAK, label="Above Avg (>=1.0)"),
        mpatches.Patch(color=C_LOW,  label="Below Avg (<1.0)"),
    ], fontsize=8, loc="upper left", framealpha=0.6)

    watermark(fig)
    fig.savefig("chart3_seasonality_heatmap.png")
    print("[Saved] chart3_seasonality_heatmap.png")
    return fig


# =============================================================================
# CHART 4 -- Peak vs Low + Stacked Monthly Distribution
# =============================================================================
def chart4(data):
    cats = ["Furniture", "Office Supplies", "Technology"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Seasonal Demand Peaks & Troughs  --  Category Breakdown",
                 fontsize=13, color=T1, fontweight="bold", y=1.01)

    # --- peak vs low grouped bar ---
    style_ax(ax1)
    peak_vals, low_vals = [], []
    for cat in cats:
        by_m = {}
        for r in data["categories"][cat]["historical"]:
            m = datetime.strptime(r["date"], "%Y-%m-%d").month
            by_m.setdefault(m, []).append(r["sales"])
        avg_m = {m: np.mean(v) for m, v in by_m.items()}
        peak_vals.append(max(avg_m.values()))
        low_vals.append(min(avg_m.values()))

    x = np.arange(len(cats)); w = 0.35
    b1 = ax1.bar(x - w/2, peak_vals, w, color=C_PEAK, edgecolor=BG, label="Peak Month Avg", zorder=3)
    b2 = ax1.bar(x + w/2, low_vals,  w, color=C_LOW,  edgecolor=BG, label="Low Month Avg",  zorder=3)
    for bar, v in zip(b1, peak_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 300,
                 fmtk(v), ha="center", fontsize=7.5, color=C_PEAK, fontweight="bold")
    for bar, v in zip(b2, low_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 300,
                 fmtk(v), ha="center", fontsize=7.5, color=C_LOW, fontweight="bold")
    ax1.set_title("Peak vs Low Month Average Sales by Category", fontsize=11)
    ax1.set_xticks(x); ax1.set_xticklabels(cats, fontsize=9)
    ax1.set_ylabel("Avg Monthly Sales (USD)", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmtk))
    ax1.legend(fontsize=8.5, framealpha=0.6)

    # --- stacked monthly distribution ---
    style_ax(ax2)
    totals  = {cat: np.zeros(12) for cat in cats}
    counts  = {cat: np.zeros(12) for cat in cats}
    for cat in cats:
        for r in data["categories"][cat]["historical"]:
            mi = datetime.strptime(r["date"], "%Y-%m-%d").month - 1
            totals[cat][mi] += r["sales"]
            counts[cat][mi] += 1

    bottom = np.zeros(12)
    for cat in cats:
        avg = np.where(counts[cat] > 0, totals[cat] / counts[cat], 0)
        ax2.bar(MONTHS, avg, bottom=bottom, color=CAT_COL[cat],
                edgecolor=BG, linewidth=0.5, label=cat, zorder=3)
        bottom += avg

    ax2.set_title("Avg Monthly Sales Distribution by Category", fontsize=11)
    ax2.set_ylabel("Avg Monthly Sales (USD)", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmtk))
    plt.setp(ax2.get_xticklabels(), fontsize=8)
    ax2.legend(fontsize=8.5, framealpha=0.6)

    plt.tight_layout()
    watermark(fig)
    fig.savefig("chart4_peak_low_comparison.png")
    print("[Saved] chart4_peak_low_comparison.png")
    return fig


# =============================================================================
# CHART 5 -- Business Planning Summary Infographic
# =============================================================================
def chart5(data):
    m   = data["metrics"]
    sea = data["overall"]["seasonality"]

    pm, lm   = m["peak_month"], m["low_month"]
    pi, li   = m["peak_index"], m["low_index"]
    grow     = m["growth_pct"]
    rmse,mae = m["rmse"], m["mae"]
    avg, tot = m["avg_monthly"], m["forecast_total"]
    buf      = rmse * 1.5

    fig, ax = plt.subplots(figsize=(15, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 15); ax.set_ylim(0, 9)
    ax.axis("off")

    # ---- Title -------------------------------------------------
    ax.text(7.5, 8.6, "SALES FORECASTING  --  BUSINESS PLANNING REPORT",
            ha="center", fontsize=18, fontweight="bold", color=T1)
    ax.text(7.5, 8.18,
            "Superstore Dataset  |  Future Interns ML Task 1 (2026)  "
            "|  Random Forest + Linear Regression  |  Matplotlib",
            ha="center", fontsize=9, color=T2)
    ax.add_patch(plt.Rectangle((0.3, 7.98), 14.4, 0.025,
                               color=GRID, zorder=5))

    # ---- KPI cards ---------------------------------------------
    kpis = [
        ("Avg Monthly Sales",   fmtk(avg),             C_HIST),
        ("6M Projection",       fmtk(tot),             C_FORE),
        ("Projected Growth",    f"{grow:+.1f}%",       C_PEAK if grow >= 0 else C_LOW),
        ("Safety Stock/Month",  fmtk(buf),             C_FURN),
        ("Peak Season",         pm,                    C_PEAK),
        ("Slow Season",         lm,                    C_LOW),
    ]
    card_w, card_h = 2.2, 1.35
    start_x = 0.45

    for i, (lbl, val, col) in enumerate(kpis):
        cx = start_x + i * (card_w + 0.12)
        cy = 6.35
        ax.add_patch(FancyBboxPatch((cx, cy), card_w, card_h,
                                    boxstyle="round,pad=0.1",
                                    linewidth=1.5, edgecolor=col,
                                    facecolor=PANEL, zorder=3))
        ax.add_patch(plt.Rectangle((cx, cy + card_h - 0.1), card_w, 0.1,
                                   color=col, alpha=0.75, zorder=4))
        ax.text(cx + card_w/2, cy + card_h*0.58, val,
                ha="center", va="center",
                fontsize=15, fontweight="bold", color=col, zorder=5)
        ax.text(cx + card_w/2, cy + 0.2, lbl,
                ha="center", va="center",
                fontsize=7.5, color=T2, zorder=5)

    ax.add_patch(plt.Rectangle((0.3, 6.18), 14.4, 0.025,
                               color=GRID, zorder=5))

    # ---- Action cards ------------------------------------------
    ax.text(0.55, 5.93, "RECOMMENDED BUSINESS ACTIONS",
            ha="left", va="top", fontsize=10.5, fontweight="bold", color=T1)

    def wrap(text, limit=70):
        words = text.split()
        lines, line = [], ""
        for w in words:
            if len(line) + len(w) < limit:
                line += w + " "
            else:
                lines.append(line.rstrip())
                line = w + " "
        lines.append(line.rstrip())
        return lines

    actions = [
        ("INVENTORY",
         f"Build {fmtk(avg * pi)} stock before {pm} (6-week lead time). "
         f"Demand spikes to {pi:.2f}x average in peak season.",
         C_PEAK),
        ("PROMOTIONS",
         f"Launch clearance campaigns in {lm} when demand drops to "
         f"{li:.2f}x average. Discounts protect cash flow.",
         C_LOW),
        ("SAFETY STOCK",
         f"Keep {fmtk(buf)}/month safety buffer to absorb "
         f"forecast error (RMSE = {fmtk(rmse)}/month).",
         C_FURN),
        ("STAFFING",
         f"Increase warehouse staff 30% in October ahead of {pm} peak. "
         f"Scale back in {lm} to reduce overhead costs.",
         C_OFF),
        ("CASH FLOW",
         f"Projected 6-month revenue: {fmtk(tot)}. "
         f"Reserve {fmtk(rmse * 2)} working capital for Aug-Sep procurement.",
         C_TECH),
        ("MODEL NOTES",
         f"Furniture R2=84.1% (excellent). Office Supplies & Technology show "
         f"high variance. Use wider safety bands for those categories.",
         T2),
    ]

    col_w, col_h = 6.8, 1.45
    for i, (title, text, col) in enumerate(actions):
        row = i // 2; ci = i % 2
        bx = 0.45 + ci * (col_w + 0.6)
        by = 5.52 - row * (col_h + 0.18)

        ax.add_patch(FancyBboxPatch((bx, by - col_h), col_w, col_h,
                                    boxstyle="round,pad=0.08",
                                    linewidth=0.8, edgecolor=col,
                                    facecolor=PANEL, alpha=0.9, zorder=3))
        ax.add_patch(plt.Rectangle((bx, by - col_h), 0.07, col_h,
                                   color=col, zorder=4))
        ax.text(bx + 0.22, by - 0.18, title,
                ha="left", va="top",
                fontsize=8.5, fontweight="bold", color=col, zorder=5)

        lines = wrap(text)
        for li_idx, line_txt in enumerate(lines[:4]):
            ax.text(bx + 0.22, by - 0.5 - li_idx * 0.27, line_txt,
                    ha="left", va="top",
                    fontsize=7.5, color=T2, zorder=5)

    watermark(fig)
    fig.savefig("chart5_business_summary.png")
    print("[Saved] chart5_business_summary.png")
    return fig


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Matplotlib Visualization -- Sales Forecasting Report")
    print("=" * 60)

    data = load_data("forecast_data.json")
    h_count = len(data["overall"]["historical"])
    f_count = len(data["overall"]["forecast"])
    print(f"[Loaded] forecast_data.json -- "
          f"{h_count} historical months, {f_count} forecast months\n")

    figs = [
        chart1(data),
        chart2(data),
        chart3(data),
        chart4(data),
        chart5(data),
    ]

    # Combine into PDF report
    pdf_path = "sales_forecast_report.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
        d = pdf.infodict()
        d["Title"]    = "Superstore Sales Forecasting Report"
        d["Author"]   = "Future Interns ML Task 1 (2026)"
        d["Subject"]  = "Sales & Demand Forecasting -- scikit-learn & Matplotlib"
        d["Keywords"] = "sales forecasting, machine learning, demand planning"
    print(f"[Saved] {pdf_path}")

    print("\n" + "=" * 60)
    print("  All 5 charts generated + PDF report saved!")
    print()
    print("  chart1_overall_forecast.png     -- History + 6M forecast")
    print("  chart2_category_forecasts.png   -- Per-category breakdown")
    print("  chart3_seasonality_heatmap.png  -- Monthly demand heatmap")
    print("  chart4_peak_low_comparison.png  -- Peak vs low months")
    print("  chart5_business_summary.png     -- Executive summary")
    print()
    print("  sales_forecast_report.pdf       -- Submit this as your report!")
    print("=" * 60)
