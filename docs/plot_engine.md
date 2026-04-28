# plot_engine

The `plot_engine` module is the visualization core of `haashi_pkg`. It provides three classes that operate at different levels of abstraction — from full manual control all the way to complete dashboard composition — while sharing the same underlying engine.

---

## Overview

| Class | Purpose | Code Required |
|---|---|---|
| `PlotEngine` | Full control over every aspect of a plot | High |
| `QuickPlot` | Single-method plots with sensible defaults | Low |
| `PowerCanvas` | Multi-panel dashboard composition | Medium |

All three classes are available from the same module:

```python
from haashi_pkg.plot_engine import PlotEngine, QuickPlot, PowerCanvas
```

---

## PlotEngine

The base class. Every other class in this module is built on top of it. `PlotEngine` wraps matplotlib and seaborn into a structured, consistent API that handles theming, figure creation, drawing, decoration, and export.

Use `PlotEngine` directly when you need fine-grained control — custom grid layouts, multiple series on one axes, manual tick formatting, or anything that doesn't fit a standard template.

### Initialization

```python
from haashi_pkg.plot_engine import PlotEngine

pe = PlotEngine()
```

An optional `Logger` instance can be passed in. If not provided, one is created internally.

```python
from haashi_pkg.utility import Logger
import logging

logger = Logger(level=logging.INFO)
pe = PlotEngine(logger=logger)
```

### Color Palettes

Four built-in palettes are available as instance attributes:

```python
pe.colors_01      # Professional: blue, orange, green, red, purple
pe.colors_02      # Deep: deep blue, warm orange, green, muted red, violet
pe.colors_03      # Soft: soft blue, pink, green, gold, purple
pe.colors_vibrant # Vibrant: coral, teal, yellow, mint, vibrant purple
```

These can be passed directly anywhere a color list is accepted.

---

### Standard Workflow

Every `PlotEngine` plot follows the same sequence:

```
create_figure  →  draw  →  decorate  →  save_or_show
```

```python
pe = PlotEngine()

fig, ax = pe.create_figure(figsize=(12, 7))

pe.draw(ax, x=[1, 2, 3, 4], y=[10, 15, 13, 17], plot_type="line",
        color="#4E79A7", linewidth=2.5, marker="o")

pe.decorate(ax,
    title="Monthly Trend",
    xlabel="Month",
    ylabel="Value",
    title_color="white",
    label_color="white",
    tick_color="white")

pe.set_background_color(fig, ax, fig_color="#1a1a1a", ax_color="#2a2a2a")

pe.save_or_show(fig, save_path="trend.png", dpi=300)
```

---

### Methods

#### `create_figure`

Creates a matplotlib figure and axes with seaborn theming applied.

```python
fig, ax = pe.create_figure(
    figsize=(10, 6),
    seaborn_theme="darkgrid",   # darkgrid | whitegrid | dark | white | ticks
    seaborn_context="notebook"  # paper | notebook | talk | poster
)

# Grid of subplots — positional args passed to plt.subplots()
fig, axes = pe.create_figure(2, 3, figsize=(18, 10))
```

Returns a `(Figure, Axes)` tuple. For grids, the second element is a numpy array of `Axes`.

---

#### `create_custom_grid`

Creates a `GridSpec`-based layout for complex, unequal subplot arrangements.

```python
fig, gs = pe.create_custom_grid(
    2, 3,
    figsize=(20, 10),
    height_ratios=[2, 1],       # First row twice as tall
    width_ratios=[3, 1, 1],     # First column three times as wide
    hspace=0.4,
    wspace=0.3
)

ax_top  = fig.add_subplot(gs[0, :])   # Spans full top row
ax_bl   = fig.add_subplot(gs[1, 0])
ax_bm   = fig.add_subplot(gs[1, 1])
ax_br   = fig.add_subplot(gs[1, 2])
```

---

#### `draw`

Draws data onto an existing axes. Supports four plot types.

```python
# Line
pe.draw(ax, x=months, y=values, plot_type="line",
        color="#4E79A7", linewidth=2, marker="o", label="Series A")

# Bar
pe.draw(ax, x=categories, y=counts, plot_type="bar",
        color="#59A14F", edgecolor="#1a1a1a")

# Scatter
pe.draw(ax, x=x_data, y=y_data, plot_type="scatter",
        color="#F28E2B", s=80, alpha=0.7)

# Pie
pe.draw(ax, x=None, y=[40, 30, 20, 10], plot_type="pie",
        labels=["A", "B", "C", "D"], autopct="%1.1f%%", startangle=140)
```

All additional keyword arguments are forwarded directly to the underlying matplotlib function, so any matplotlib parameter is valid here.

---

#### `decorate`

Adds title, axis labels, and tick styling to an axes in one call.

```python
pe.decorate(ax,
    title="Sales Report",
    xlabel="Month",
    ylabel="Revenue ($)",
    title_fontsize=18,
    label_fontsize=13,
    title_color="white",
    label_color="white",
    tick_color="white",
    ylim="zero",           # Start y-axis at 0
    xlim=(0, 12)
)
```

`ylim` accepts a `(min, max)` tuple, the string `"zero"` to force the axis to start at zero, or `None` for automatic scaling.

---

#### `set_background_color`

Sets background colors for the figure and axes, with grid line control.

```python
# Single axes
pe.set_background_color(fig, ax,
    fig_color="#1a1a1a",
    ax_color="#2a2a2a",
    grid_color="white",
    grid_alpha=0.15)

# Array of axes (from subplots)
pe.set_background_color(fig, axes,
    fig_color="#f5f5f5",
    ax_color="#ffffff",
    apply_to_all=True)
```

---

#### `add_reference_line`

Adds a horizontal or vertical reference line — targets, thresholds, averages.

```python
# Horizontal (y value)
pe.add_reference_line(ax, y=70000, label="Target",
                      color="red", linestyle="--", linewidth=1.5)

# Vertical (x value)
pe.add_reference_line(ax, x=6, label="Midpoint",
                      color="gray", linestyle="-.", alpha=0.6)
```

Exactly one of `y` or `x` must be provided, not both.

---

#### `add_value_labels_on_bars`

Adds text labels on top of each bar in a bar chart.

```python
# Integer labels
pe.add_value_labels_on_bars(ax)

# Currency
pe.add_value_labels_on_bars(ax, format_string="${:,.0f}",
                             color="white", fontsize=10)

# Percentage
pe.add_value_labels_on_bars(ax, format_string="{:.1f}%", fontweight="normal")
```

---

#### `create_stats_text_box`

Converts an axes into a styled statistics display box.

```python
stats = {
    "Total Revenue": 128500,
    "Average Order": 54.3,
    "Return Rate": "2.1%"
}

pe.create_stats_text_box(ax, stats,
    title="Q4 Summary",
    box_color="#2a2a2a",
    text_color="white",
    border_color="#4ECDC4",
    fontsize=12)
```

The axes is turned off and replaced with a rounded text box. Pass an axes that would otherwise be empty.

---

#### `format_y_axis`

Formats y-axis tick labels with currency symbols and thousands separators.

```python
pe.format_y_axis(ax, currency="$", decimals=0)
# Output: $1,234  $5,678  $9,012

pe.format_y_axis(ax, currency="€", decimals=2)
# Output: €1,234.00

pe.format_y_axis(ax, currency="", decimals=1)
# Output: 1,234.5
```

---

#### `force_xticks`

Manually sets x-axis tick positions and labels.

```python
months = np.array(["Jan", "Feb", "Mar", "Apr"])
pe.force_xticks(ax, labels=months, rotation=45)

# Custom positions
pe.force_xticks(ax,
    labels=np.array(["Q1", "Q2", "Q3", "Q4"]),
    positions=np.array([0, 3, 6, 9]),
    rotation=0)
```

---

#### `force_yticks`

Controls y-axis tick density and minimum value.

```python
pe.force_yticks(ax, nbins=5, bottom=0)
```

---

#### `add_margins`

Adds padding between data and axes edges.

```python
pe.add_margins(ax, xpad=0.05, ypad=0.2)
```

---

#### `set_legend`

Adds and positions a legend.

```python
pe.set_legend(ax, loc="upper right", fontsize=12)

pe.set_legend(ax, loc="best", fontsize=11,
              frameon=True, title="Series")
```

---

#### `set_suptitle`

Sets the overall figure title when working with subplots.

```python
pe.set_suptitle(fig, "Annual Report 2024",
                fontsize=20, color="white", fontweight="bold")
```

---

#### `save_or_show`

Finalizes layout, saves to file, and/or displays the figure. Always called last.

```python
# Display only
pe.save_or_show(fig)

# Save and display
pe.save_or_show(fig, save_path="chart.png", dpi=300)

# Save without displaying (headless / batch)
pe.save_or_show(fig, save_path="chart.pdf", dpi=600, show=False)
```

Supported export formats: `.png`, `.jpg`, `.pdf`, `.svg`, `.eps`

---

### Complete Example

```python
import numpy as np
from haashi_pkg.plot_engine import PlotEngine

pe = PlotEngine()

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
revenue = [52000, 58000, 54000, 61000, 67000, 72000]
expenses = [38000, 41000, 39000, 44000, 47000, 50000]

fig, ax = pe.create_figure(figsize=(12, 7), seaborn_theme="darkgrid")

pe.set_background_color(fig, ax, fig_color="#1a1a1a", ax_color="#2a2a2a")

pe.draw(ax, x=months, y=revenue,  plot_type="line",
        color="#4ECDC4", linewidth=2.5, marker="o", label="Revenue")
pe.draw(ax, x=months, y=expenses, plot_type="line",
        color="#FF6B6B", linewidth=2.0, linestyle="--", label="Expenses")

pe.add_reference_line(ax, y=60000, label="Target", color="gray")

pe.decorate(ax,
    title="Revenue vs Expenses — H1 2024",
    xlabel="Month", ylabel="Amount ($)",
    title_color="white", label_color="white", tick_color="white",
    ylim=(0, 80000))

pe.format_y_axis(ax, currency="$", decimals=0)
pe.set_legend(ax, loc="upper left", fontsize=11)
pe.save_or_show(fig, save_path="h1_report.png", dpi=300, show=False)
```

---

## QuickPlot

`QuickPlot` wraps the entire `PlotEngine` workflow into a single method call per plot type. It is the right choice when you need a clean, styled plot quickly without managing figure lifecycle, theming, or decoration manually.

### Initialization

```python
from haashi_pkg.plot_engine import QuickPlot

qp = QuickPlot(theme="dark")   # dark | light
```

Two themes are available:

| Theme | Background | Text | Grid |
|---|---|---|---|
| `dark` | `#1a1a1a` | White | Subtle white |
| `light` | `#f5f5f5` | Dark gray | Light gray |

---

### Methods

#### `line`

```python
qp.line(
    x=months,
    y=revenue,
    title="Monthly Revenue",
    xlabel="Month",
    ylabel="Revenue ($)",
    color="#4ECDC4",         # Optional — defaults to theme color
    linewidth=2.5,
    marker="o",
    ref_y=60000,             # Optional reference line
    ref_label="Target",
    ylim=(0, 90000),
    save_path="trend.png",
    show=False
)
```

---

#### `bar`

```python
qp.bar(
    x=["North", "South", "East", "West"],
    y=[134000, 98000, 112000, 145000],
    title="Sales by Region",
    ylabel="Sales ($)",
    value_labels=True,
    value_format="${:,.0f}",
    rotation=0,
    ylim=(0, 170000),
    save_path="sales.png",
    show=False
)
```

---

#### `scatter`

```python
qp.scatter(
    x=ad_spend,
    y=conversions,
    title="Ad Spend vs Conversions",
    xlabel="Ad Spend ($)",
    ylabel="Conversions",
    size=80,
    alpha=0.75,
    ref_y=conversions.mean(),
    ref_label="Average",
    show=False
)
```

---

#### `pie`

```python
qp.pie(
    values=[40, 30, 20, 10],
    labels=["Enterprise", "Mid-Market", "SMB", "Other"],
    title="Customer Segments",
    explode=[0.05, 0, 0, 0],
    save_path="segments.png",
    show=False
)
```

---

### PlotEngine vs QuickPlot

The following two blocks produce the same plot:

**PlotEngine (9 lines):**
```python
pe = PlotEngine()
fig, ax = pe.create_figure(figsize=(10, 6), seaborn_theme="darkgrid")
pe.set_background_color(fig, ax, fig_color="#1a1a1a", ax_color="#2a2a2a")
pe.draw(ax, x=months, y=revenue, plot_type="line",
        color="#4E79A7", linewidth=2.5, marker="o")
pe.add_reference_line(ax, y=70000, label="Target", color="gray")
pe.decorate(ax, title="Revenue Trend", xlabel="Month", ylabel="$",
            title_color="white", label_color="white", tick_color="white")
pe.set_legend(ax, fontsize=12)
pe.save_or_show(fig, save_path="trend.png", show=False)
```

**QuickPlot (6 lines):**
```python
qp = QuickPlot(theme="dark")
qp.line(
    x=months, y=revenue,
    title="Revenue Trend", xlabel="Month", ylabel="$",
    ref_y=70000, ref_label="Target",
    save_path="trend.png", show=False
)
```

Use `PlotEngine` when you need multi-series overlays, custom grid layouts, or behaviour that falls outside the standard templates. Use `QuickPlot` for everything else.

---

## PowerCanvas

`PowerCanvas` is a Power BI-style dashboard builder. It composes multiple panels — KPI cards, sparklines, charts, and stats boxes — into a single polished figure using either preset layouts or a fully flexible grid system.

### Initialization

```python
from haashi_pkg.plot_engine import PowerCanvas

pc = PowerCanvas(
    title="Sales Dashboard 2024",
    theme="light",              # dark | light
    figsize=(22, 13)
)
```

---

### Layout Modes

`PowerCanvas` offers two layout modes. Choose one per dashboard — they cannot be mixed.

#### Preset Layouts

Three presets are available, each ready with one method call.

**`preset_kpi_top`** — KPI card row across the top, chart panels below. The most common Power BI layout.

```python
pc.preset_kpi_top(
    kpi_count=4,          # Number of KPI cards in the top row
    chart_rows=2,         # Chart rows below
    chart_cols=2,         # Chart columns
    kpi_height_ratio=0.28 # Fraction of height for the KPI row
)
```

Panel indexing: KPI cards are `0` through `kpi_count - 1`. Chart panels follow sequentially left-to-right, top-to-bottom.

```
[  KPI 0  ] [  KPI 1  ] [  KPI 2  ] [  KPI 3  ]
[  Panel 4           ] [  Panel 5             ]
[  Panel 6           ] [  Panel 7             ]
```

---

**`preset_split`** — Wide chart or content panel on the left, stacked KPI cards on the right.

```python
pc.preset_split(
    kpi_count=3,
    left_width_ratio=0.65
)
# Panel 0: left wide panel
# Panels 1, 2, 3: right stacked panels
```

---

**`preset_full_grid`** — Even N×M grid of chart panels, indexed left-to-right, top-to-bottom.

```python
pc.preset_full_grid(rows=2, cols=3)
# Panels 0-5, indexed left-to-right top-to-bottom
```

---

#### Flexible Layout

`create_canvas` gives full control over rows, columns, sizing ratios, and panel placement. Panels are addressed with `(row, col)` tuples and support `col_span` and `row_span` for spanning.

```python
pc.create_canvas(
    rows=4,
    cols=6,
    height_ratios=[0.16, 0.28, 0.28, 0.28],
    hspace=0.55,
    wspace=0.40
)

# Place panels with (row, col) tuples
pc.add_kpi_card((0, 0), label="Revenue", value="$8.4M")
pc.add_kpi_card((0, 1), label="Profit",  value="$2.1M")

# Span multiple columns
pc.add_line((1, 0), x=months, y=revenue,
            title="Revenue Over Time", col_span=3)

# Place in specific cell
pc.add_pie((1, 3), values=seg_vals, labels=seg_labels,
           title="Segments", col_span=2)
```

---

### Panel Methods

#### `add_kpi_card`

Displays a large metric value with optional label, delta indicator, and sparkline.

```python
pc.add_kpi_card(
    0,                           # Panel index (int for preset, tuple for flexible)
    label="Total Revenue",
    value="$8.42M",
    delta="+12%",
    delta_up=True,               # True = green arrow, False = red arrow
    sparkline_data=[52, 58, 54, 61, 67, 72, 69, 75, 78, 83, 91],
    icon="$"                     # Optional character shown next to label
)
```

The delta arrow is green (`▲`) when `delta_up=True` and red (`▼`) when `delta_up=False`. When `sparkline_data` is provided, a mini trend chart appears on the right side of the card.

---

#### `add_bar`

```python
pc.add_bar(
    4,
    x=regions,
    y=sales,
    title="Sales by Region",
    value_labels=True,
    value_format="${:,.0f}",
    ylim=(0, 170000),
    rotation=0,
    ref_y=100000,
    ref_label="Target"
)
```

---

#### `add_line`

```python
pc.add_line(
    5,
    x=months,
    y=revenue,
    title="Revenue Over Time",
    fill=True,                  # Fills area under line
    color="#4ECDC4",
    ref_y=70000,
    ref_label="Target",
    ylim=(0, 95000)
)
```

---

#### `add_scatter`

```python
pc.add_scatter(
    6,
    x=ad_spend,
    y=conversions,
    title="Spend vs Conversions",
    xlabel="Ad Spend ($)",
    ylabel="Conversions",
    ref_y=conversions.mean()
)
```

---

#### `add_pie`

Supports both standard pie and donut charts.

```python
# Standard pie
pc.add_pie(2, values=cat_vals, labels=categories, title="Category Share")

# Donut
pc.add_pie(2, values=cat_vals, labels=categories,
           title="Category Share", donut=True)
```

---

#### `add_stats_panel`

Renders a dictionary of key-value metrics in a styled text box.

```python
pc.add_stats_panel(
    7,
    stats={
        "ARR":       "$78.9M",
        "MRR":       "$6.6M",
        "CAC":       "$1,240",
        "LTV":       "$18,700",
        "LTV : CAC": "15.1x"
    },
    title="SaaS Metrics"
)
```

---

#### `render`

Finalizes and outputs the dashboard. Always called last.

```python
pc.render(
    save_path="dashboard.png",
    dpi=180,
    show=False
)
```

---

### Complete Example — Preset Layout

```python
import numpy as np
from haashi_pkg.plot_engine import PowerCanvas

months  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
revenue = [52000,58000,54000,61000,67000,72000,69000,75000,78000,71000,83000,91000]
regions = ["North","South","East","West","Central"]
sales   = [134000,98000,112000,145000,87000]
spark   = [52,58,54,61,67,72,69,75,78,71,83,91]

pc = PowerCanvas(title="Sales Overview", theme="light", figsize=(22, 13))

pc.preset_kpi_top(kpi_count=4, chart_rows=1, chart_cols=2)

pc.add_kpi_card(0, label="Annual Revenue", value="$8.4M",
                delta="+19%", delta_up=True, sparkline_data=spark)
pc.add_kpi_card(1, label="Net Profit",     value="$2.1M",
                delta="+23%", delta_up=True, sparkline_data=spark)
pc.add_kpi_card(2, label="Active Users",   value="24,842",
                delta="+8%",  delta_up=True, sparkline_data=spark)
pc.add_kpi_card(3, label="Churn Rate",     value="2.3%",
                delta="-1.1%",delta_up=True, sparkline_data=list(reversed(spark)))

pc.add_bar(4, x=regions, y=sales,
           title="Sales by Region",
           value_labels=True, value_format="${:,.0f}",
           ylim=(0, int(max(sales) * 1.2)), rotation=0)

pc.add_line(5, x=months, y=revenue,
            title="Revenue Over Time",
            fill=True, ref_y=70000, ref_label="Target",
            ylim=(0, int(max(revenue) * 1.15)))

pc.render(save_path="sales_overview.png", dpi=180, show=False)
```

---

### Complete Example — Flexible Layout

```python
pc = PowerCanvas(title="Operations Report", theme="dark", figsize=(24, 15))

pc.create_canvas(rows=3, cols=4, height_ratios=[0.18, 0.41, 0.41])

# Row 0 — four KPI cards
pc.add_kpi_card((0,0), label="Uptime",       value="99.8%", delta="+0.2%", delta_up=True)
pc.add_kpi_card((0,1), label="Incidents",    value="3",     delta="-41%",  delta_up=True)
pc.add_kpi_card((0,2), label="Avg Response", value="142ms", delta="-18%",  delta_up=True)
pc.add_kpi_card((0,3), label="Active Users", value="12,441",delta="+22%",  delta_up=True)

# Row 1 — wide line chart + donut
pc.add_line((1,0), x=months, y=revenue, title="Traffic Over Time",
            fill=True, col_span=3)
pc.add_pie((1,3), values=[42,28,16,14], donut=True,
           labels=["Web","Mobile","API","CLI"], title="Traffic by Source")

# Row 2 — bar + scatter + stats
pc.add_bar((2,0), x=regions, y=sales, title="Load by Region",
           value_labels=True, col_span=2)
pc.add_scatter((2,2), x=np.random.randint(100,1000,30),
               y=np.random.uniform(50,300,30),
               title="Latency Distribution", xlabel="RPS", ylabel="ms")
pc.add_stats_panel((2,3), stats={"P99 Latency":"287ms","Error Rate":"0.04%",
                                  "Cache Hit":"91%"}, title="SLO Status")

pc.render(save_path="ops_report.png", dpi=180, show=False)
```

---

## Error Reference

All exceptions inherit from `PlotEngineError` and can be caught at the base level.

```python
from haashi_pkg.plot_engine import PlotEngineError

try:
    pe.draw(ax, x=data, y=values, plot_type="histogram")
except PlotEngineError as e:
    print(e)
```

| Exception | When it is raised |
|---|---|
| `PlotEngineError` | Base class for all plot engine errors |
| `InvalidPlotTypeError` | A plot type other than `line`, `bar`, `scatter`, or `pie` is passed to `draw()` |
| `InvalidDataError` | Y data is empty or None; x/y length mismatch in tick methods; both or neither of `x`/`y` provided to `add_reference_line()` |
| `ConfigurationError` | Figure creation fails; invalid color values; grid dimension mismatch |

---

## Type Reference

```python
from haashi_pkg.plot_engine import (
    PlotType,    # Literal["line", "bar", "scatter", "pie"]
    ThemeType,   # Literal["darkgrid", "whitegrid", "dark", "white", "ticks"]
    ContextType, # Literal["paper", "notebook", "talk", "poster"]
    QuickTheme,  # Literal["dark", "light"]
    DashTheme,   # Literal["dark", "light"]
    PanelIndex,  # Union[int, Tuple[int, int]]
    XData,       # Optional numeric, string, datetime, Series, or ndarray iterable
    YData,       # Numeric iterable, ndarray, or Series
)
```
