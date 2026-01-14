# plotengine.py


from __future__ import annotations

import logging
from datetime import datetime
from typing import (
    Tuple,
    Literal,
    Dict,
    Union,
    Optional,
    Iterable,
    Sequence,
    Any,
    List
)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.ticker as ticker
from matplotlib.ticker import StrMethodFormatter
import matplotlib.gridspec as gridspec
from pandas import Series, Timestamp

from haashi_pkg.utility.utils import Utility

# ----------------------------
# TYPE ALIASES
# ----------------------------

Numeric = Union[int, float]

XData = Optional[
    Union[
        Iterable[int],
        Iterable[float],
        Iterable[str],
        Iterable[datetime],
        Iterable[Timestamp],
        Series,
        np.ndarray,
    ]
]

YData = Union[
    Iterable[Numeric],
    np.ndarray,
    Series,
]

PlotType = Literal["line", "bar", "scatter", "pie"]

ThemeType = Literal["darkgrid", "whitegrid", "dark", "white", "ticks"]

ContextType = Literal["paper", "notebook", "talk", "poster"]

Kwargs = Dict[str, Union[
    str, int, float, bool,
    Sequence[int], Sequence[float],
    Optional[str], Optional[int],
    Tuple[int, ...], Tuple[float, ...]
]]


# ----------------------------
# ENGINE
# ----------------------------

class PlotEngine:
    def __init__(self) -> None:

        self.ut: Utility = Utility(level=logging.INFO)

        self.colors_01: list[str] = [
            "#4E79A7",  # blue
            "#F28E2B",  # orange
            "#59A14F",  # green
            "#E15759",  # red
            "#B07AA1",  # purple
        ]

        self.colors_02: list[str] = [
            "#1f77b4",  # deep blue
            "#ff7f0e",  # warm orange
            "#2ca02c",  # green
            "#d62728",  # muted red
            "#9467bd",  # violet
        ]

        self.colors_03: list[str] = [
            "#5DA5DA",  # soft blue
            "#F17CB0",  # pink
            "#60BD68",  # green
            "#B2912F",  # gold
            "#B276B2",  # purple
        ]

        # Vibrant modern palette
        self.colors_vibrant: list[str] = [
            "#FF6B6B",  # coral
            "#4ECDC4",  # teal
            "#FFE66D",  # yellow
            "#95E1D3",  # mint
            "#9D4EDD",  # vibrant purple
        ]

    # ----------------------------
    # SETUP
    # ----------------------------

    def create_figure(
        self,
        *args: Any,
        figsize: Tuple[int, int] = (10, 6),
        seaborn_theme: ThemeType = "darkgrid",
        seaborn_context: ContextType = "notebook",
        **kwargs: Any,
    ) -> Tuple[Figure, Any]:
        """
        Create a matplotlib figure with seaborn styling.

        Args:
            *args: Positional arguments for plt.subplots (nrows, ncols)
            figsize: Figure size as (width, height)
            seaborn_theme: Seaborn theme style
            seaborn_context: Seaborn context for scaling
            **kwargs: Additional keyword arguments for plt.subplots

        Returns:
            Tuple of (Figure, Axes or array of Axes)
        """
        sns.set_theme(
            style=seaborn_theme,
            context=seaborn_context,
        )

        return plt.subplots(
            *args, figsize=figsize, **kwargs
        )  # type: ignore[return-value]

    def create_custom_grid(
        self,
        rows: int = 2,
        cols: int = 3,
        *,
        figsize: Tuple[int, int] = (20, 10),
        height_ratios: Optional[List[float]] = None,
        width_ratios: Optional[List[float]] = None,
        hspace: float = 0.3,
        wspace: float = 0.3,
        seaborn_theme: ThemeType = "darkgrid",
        seaborn_context: ContextType = "notebook",
    ) -> Tuple[Figure, Any]:
        """
        Create a flexible custom grid layout using GridSpec.

        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            figsize: Figure size as (width, height)
            height_ratios: Relative heights of rows (e.g., [2, 1] makes first row twice as tall)
            width_ratios: Relative widths of columns (e.g., [2, 1, 1])
            hspace: Vertical spacing between subplots
            wspace: Horizontal spacing between subplots
            seaborn_theme: Seaborn theme style
            seaborn_context: Seaborn context for scaling

        Returns:
            Tuple of (Figure, GridSpec object)

        Example usage:
            # 1 wide top, 3 bottom
            fig, gs = pe.create_custom_grid(2, 3)
            ax1 = fig.add_subplot(gs[0, :])    # Top, span all columns
            ax2 = fig.add_subplot(gs[1, 0])    # Bottom left
            ax3 = fig.add_subplot(gs[1, 1])    # Bottom middle
            ax4 = fig.add_subplot(gs[1, 2])    # Bottom right

            # 2x2 equal grid
            fig, gs = pe.create_custom_grid(2, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])

            # Complex layout with custom ratios
            fig, gs = pe.create_custom_grid(3, 2, height_ratios=[2, 1, 1], width_ratios=[3, 1])
            ax1 = fig.add_subplot(gs[0, :])    # First row, full width
            ax2 = fig.add_subplot(gs[1, 0])    
            ax3 = fig.add_subplot(gs[1, 1])
            ax4 = fig.add_subplot(gs[2, :])    # Last row, full width
        """

        sns.set_theme(style=seaborn_theme, context=seaborn_context)

        fig = plt.figure(figsize=figsize)

        gs = gridspec.GridSpec(
            rows, cols,
            figure=fig,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            hspace=hspace,
            wspace=wspace
        )

        return fig, gs

    # ----------------------------
    # DRAW
    # ----------------------------

    def _validate_plot_type(self, parse_plot_type: str) -> str:
        """Validate and normalize plot type string."""
        parse_plot_type = parse_plot_type.lower()
        if parse_plot_type in ["line", "bar", "scatter", "pie"]:
            return parse_plot_type
        else:
            raise ValueError(f"Unsupported plot type: {parse_plot_type}")

    def draw(
        self,
        ax: Axes,
        x: XData,
        y: YData,
        plot_type: PlotType = "line",
        **kwargs: Any,
    ) -> None:
        """
        Draw a plot on the provided Axes.

        Args:
            ax: The axes to draw on
            x: X-axis data
            y: Y-axis data
            plot_type: Type of plot ('line', 'bar', 'scatter', 'pie')
            **kwargs: Forwarded to matplotlib plotting function
                Common examples: color, marker, markersize, linewidth, alpha, label
        """
        plot_type = self._validate_plot_type(
            plot_type)  # type: ignore[assignment]

        if plot_type == "line":
            ax.plot(x, y, **kwargs)
        elif plot_type == "bar":
            ax.bar(x, y, **kwargs)
        elif plot_type == "scatter":
            ax.scatter(x, y, **kwargs)
        elif plot_type == "pie":
            ax.pie(y, **kwargs)

    # ----------------------------
    # DECORATE
    # ----------------------------

    def set_background_color(
        self,
        fig: Figure,
        ax: Union[Axes, np.ndarray, Tuple[Axes]],
        *,
        fig_color: str = "#1a1a1a",
        ax_color: str = "#2a2a2a",
        grid_color: str = "white",
        grid_alpha: float = 0.2,
        apply_to_all: bool = True,
    ) -> None:
        """
        Set background colors for figure and axes.

        Args:
            fig: The figure to modify
            ax: The axes (single, array, or tuple of axes) to modify
            fig_color: Figure background color (hex or named color)
            ax_color: Axes background color (hex or named color)
            grid_color: Grid line color
            grid_alpha: Grid line transparency (0-1)
            apply_to_all: If True, apply to all axes in array/tuple
        """
        # Set figure background
        fig.patch.set_facecolor(fig_color)

        # Set axes background
        if isinstance(ax, (np.ndarray, tuple, list)):  # Handle tuple/list
            axes_to_modify = ax.flat if isinstance(ax, np.ndarray) else ax

            if apply_to_all:
                for a in axes_to_modify:
                    a.set_facecolor(ax_color)
                    a.grid(True, alpha=grid_alpha, color=grid_color)
            else:
                axes_to_modify[0].set_facecolor(ax_color)
                axes_to_modify[0].grid(
                    True, alpha=grid_alpha, color=grid_color
                )
        else:
            ax.set_facecolor(ax_color)
            ax.grid(True, alpha=grid_alpha, color=grid_color)

    def set_text_colors(
        self,
        ax: Union[Axes, np.ndarray, Tuple[Axes]],
        *,
        title_color: str = "white",
        label_color: str = "white",
        tick_color: str = "#aaaaaa",
        apply_to_all: bool = True,
    ) -> None:
        """
        Set colors for all text elements on axes.

        Args:
            ax: The axes (single, array, or tuple) to modify
            title_color: Color for title text
            label_color: Color for axis labels
            tick_color: Color for tick labels
            apply_to_all: If True, apply to all axes
        """
        axes_list = []
        if isinstance(ax, (np.ndarray, tuple, list)):
            axes_list = list(ax.flat) if isinstance(
                ax, np.ndarray) else list(ax)
        else:
            axes_list = [ax]

        if not apply_to_all:
            axes_list = axes_list[:1]

        for a in axes_list:
            # Set title color
            a.title.set_color(title_color)

            # Set axis label colors
            a.xaxis.label.set_color(label_color)
            a.yaxis.label.set_color(label_color)

            # Set tick label colors
            a.tick_params(axis='x', colors=tick_color)
            a.tick_params(axis='y', colors=tick_color)

            # Set spine (border) colors
            for spine in a.spines.values():
                spine.set_color(tick_color)

    def set_suptitle(
        self,
        title: str,
        fontsize: int = 24,
        fontweight: str = "bold",
        color: str = "black"
    ) -> None:
        """
        Set the figure-level super title.

        Args:
            title: Title text
            fontsize: Font size for title
            fontweight: Font weight ('normal', 'bold', 'semibold', etc.)
        """
        plt.suptitle(
            title, fontsize=fontsize, weight=fontweight, color=color
        )

    def set_labels_and_title(
        self,
        ax: Axes,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title_fontsize: int = 18,
        label_fontsize: int = 14,
        title_weight: str = "semibold",
        label_weight: str = "semibold",
    ) -> None:
        """
        Set title and axis labels for an axes.

        Args:
            ax: The axes to modify
            title: Axes title
            xlabel: X-axis label
            ylabel: Y-axis label
            title_fontsize: Font size for title
            label_fontsize: Font size for labels
            title_weight: Font weight for title
            label_weight: Font weight for labels
        """
        if title is not None:
            ax.set_title(
                title,
                fontsize=title_fontsize,
                fontweight=title_weight,
            )

        if xlabel is not None:
            ax.set_xlabel(
                xlabel,
                fontsize=label_fontsize,
                fontweight=label_weight,
            )

        if ylabel is not None:
            ax.set_ylabel(
                ylabel,
                fontsize=label_fontsize,
                fontweight=label_weight,
            )

    def set_axis_limits(
        self,
        ax: Axes,
        *,
        xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
        ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
        x_from_zero: bool = False,
        y_from_zero: bool = False,
    ) -> None:
        """
        Set axis limits. Convenience method for forcing zero origins.

        Args:
            ax: The axes to modify
            xlim: Explicit x-axis limits as (min, max)
            ylim: Explicit y-axis limits as (min, max)
            x_from_zero: Force x-axis to start at 0
            y_from_zero: Force y-axis to start at 0
        """
        if x_from_zero:
            ax.set_xlim(left=0)
        elif xlim is not None:
            ax.set_xlim(xlim)

        if y_from_zero:
            ax.set_ylim(bottom=0)
        elif ylim is not None:
            ax.set_ylim(ylim)

    def add_reference_line(
        self,
        ax: Axes,
        *,
        y: Optional[float] = None,
        x: Optional[float] = None,
        label: Optional[str] = None,
        color: str = "gray",
        linestyle: str = "--",
        linewidth: float = 2,
        alpha: float = 0.5,
    ) -> None:
        """
        Add a horizontal or vertical reference line (e.g., goal line, threshold).

        Args:
            ax: The axes to draw on
            y: Y-value for horizontal line
            x: X-value for vertical line
            label: Label for the line (appears in legend)
            color: Line color
            linestyle: Line style ('--', '-.', ':', '-')
            linewidth: Line width
            alpha: Transparency (0-1)
        """
        if y is not None:
            ax.axhline(
                y=y,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                label=label
            )
        elif x is not None:
            ax.axvline(
                x=x,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                label=label
            )

    def add_value_labels_on_bars(
        self,
        ax: Axes,
        *,
        format_string: str = "{:.0f}",
        fontsize: int = 11,
        fontweight: str = "bold",
        color: str = "black",
        padding: int = 3,
    ) -> None:
        """
        Add value labels on top of bar charts.

        Args:
            ax: The axes with bar chart
            format_string: Format string for values (e.g., '{:.1f}', '${:,.0f}')
            fontsize: Font size for labels
            fontweight: Font weight
            color: Text color
            padding: Padding between bar and label in points
        """
        for container in ax.containers:
            labels = [
                format_string.format(v.get_height()) for v in container
            ]
            ax.bar_label(
                container,
                labels=labels,
                fontsize=fontsize,
                fontweight=fontweight,
                color=color,
                padding=padding
            )

    def create_stats_text_box(
        self,
        ax: Axes,
        stats: Dict[str, Union[str, int, float]],
        *,
        title: Optional[str] = None,
        fontsize: int = 12,
        title_fontsize: int = 14,
        box_color: str = "#2a2a2a",
        text_color: str = "white",
        border_color: str = "white",
        border_width: float = 2,
    ) -> None:
        """
        Create a text box with statistics on an axis.

        Args:
            ax: The axes to draw on (will be turned off)
            stats: Dictionary of stat_name: value pairs
            title: Optional title for the stats box
            fontsize: Font size for stats
            title_fontsize: Font size for title
            box_color: Background color
            text_color: Text color
            border_color: Border color
            border_width: Border width
        """
        ax.axis('off')

        # Build text
        lines = []
        if title:
            lines.append(title)
            lines.append("")

        for key, value in stats.items():
            if isinstance(value, float):
                lines.append(f"{key}\n{value:.1f}")
            elif isinstance(value, int):
                lines.append(f"{key}\n{value:,}")
            else:
                lines.append(f"{key}\n{value}")
            lines.append("")

        text = "\n".join(lines)

        ax.text(
            0.5, 0.5, text,
            ha='center', va='center',
            fontsize=fontsize,
            color=text_color,
            fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=1',
                facecolor=box_color,
                edgecolor=border_color,
                linewidth=border_width
            ),
            linespacing=1.8,
            transform=ax.transAxes
        )

    def format_y_axis(
        self,
        ax: Axes,
        currency: str = "$",
        style: str = "comma",
        decimals: int = 0
    ) -> None:
        """
        Format y-axis tick labels with currency and/or comma separators.

        Args:
            ax: The axes to modify
            currency: Currency symbol (use "" for no currency)
            style: Formatting style ('comma' for thousands separators)
            decimals: Number of decimal places
        """
        if style == "comma":
            ax.yaxis.set_major_formatter(
                StrMethodFormatter(f"{currency}{{x:,.{decimals}f}}")
            )

    def force_xticks(
        self,
        ax: Axes,
        labels: np.ndarray,
        positions: Optional[np.ndarray] = None,
        rotation: int = 45
    ) -> None:
        """
        Manually set x-axis tick positions and labels.

        Args:
            ax: The axes to modify
            labels: Tick labels
            positions: Tick positions (defaults to range(len(labels)))
            rotation: Label rotation in degrees
        """
        if positions is None:
            positions = np.arange(len(labels))

        if len(labels) != len(positions):
            raise ValueError("labels and positions must be the same length")

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=rotation, ha="right")

    def force_yticks(
        self,
        ax: Axes,
        nbins: int = 10,
        bottom: Optional[int] = None
    ) -> None:
        """
        Control y-axis tick density and minimum value.

        Args:
            ax: The axes to modify
            nbins: Maximum number of y-axis ticks
            bottom: Minimum y-axis value (force start point)
        """
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=nbins))
        if bottom is not None:
            ax.set_ylim(bottom=bottom)

    def add_margins(
        self,
        ax: Axes,
        xpad: float = 0.05,
        ypad: float = 0.2
    ) -> None:
        """
        Add padding/margins around the data in the plot.

        Args:
            ax: The axes to modify
            xpad: X-axis padding as fraction of data range
            ypad: Y-axis padding as fraction of data range
        """
        ax.margins(x=xpad, y=ypad)

    def set_legend(
        self,
        ax: Axes,
        *,
        loc: str = "upper right",
        fontsize: int = 14,
        **kwargs: Any,
    ) -> None:
        """
        Configure and display legend.

        Args:
            ax: The axes to modify
            loc: Legend location
            fontsize: Legend font size
        """
        ax.legend(loc=loc, fontsize=fontsize, **kwargs)

    # ----------------------------
    # FINALIZE
    # ----------------------------

    def save_or_show(
        self,
        fig: Figure,
        *,
        dpi: int = 300,
        save_path: Optional[str] = None,
        top: float = 0.95,
        bottom: float = 0.08,
        show: bool = True,
        use_tight_layout: bool = True
    ) -> None:
        """
        Finalize, optionally save, and display the figure.

        Args:
            fig: The figure to finalize
            dpi: DPI for saved image
            save_path: Path to save figure (None to skip saving)
            show: Whether to display the figure
        """

        fig.subplots_adjust(top=top, bottom=bottom)

        if use_tight_layout:
            fig.tight_layout()

        if save_path is not None:
            safe_save_path = self.ut.ensure_writable_path(save_path)
            fig.savefig(safe_save_path, dpi=dpi)
            self.ut.info(f"Saved figure to {safe_save_path}")

        if show:
            plt.show()

        plt.close(fig)
