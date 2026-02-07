

# plotengine.py

"""
PlotEngine: A comprehensive plotting utility built on matplotlib and seaborn.

This module provides a high-level interface for creating, customizing, and managing
matplotlib plots with sensible defaults and extensive customization options.

Classes:
    PlotEngineError: Base exception class for PlotEngine errors
    InvalidPlotTypeError: Raised when an invalid plot type is specified
    InvalidDataError: Raised when data validation fails
    ConfigurationError: Raised when configuration/setup fails
    PlotEngine: Main plotting engine class

Example:
    >>> pe = PlotEngine()
    >>> fig, ax = pe.create_figure(figsize=(12, 8))
    >>> pe.draw(ax, x=[1, 2, 3], y=[4, 5, 6], plot_type='line')
    >>> pe.decorate(ax, title='My Plot', xlabel='X', ylabel='Y')
    >>> pe.save_or_show(fig, save_path='plot.png')
"""

from __future__ import annotations

import logging
import warnings
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

from haashi_pkg.utility.utils import Logger

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
# CUSTOM EXCEPTIONS
# ----------------------------

class PlotEngineError(Exception):
    """Base exception class for all PlotEngine errors.

    All custom exceptions in PlotEngine inherit from this class,
    allowing for easy catching of all PlotEngine-specific errors.
    """
    pass


class InvalidPlotTypeError(PlotEngineError):
    """Raised when an invalid plot type is specified.

    Valid plot types are: 'line', 'bar', 'scatter', 'pie'.

    Args:
        plot_type: The invalid plot type that was provided
        message: Optional custom error message
    """

    def __init__(self, plot_type: str, message: Optional[str] = None):
        self.plot_type = plot_type
        if message is None:
            message = (
                f"Invalid plot type: '{plot_type}'. "
                f"Valid types are: 'line', 'bar', 'scatter', 'pie'"
            )
        super().__init__(message)


class InvalidDataError(PlotEngineError):
    """Raised when data validation fails.

    This can occur when data dimensions don't match, data is empty,
    or data types are incompatible.

    Args:
        message: Description of the data validation error
    """
    pass


class ConfigurationError(PlotEngineError):
    """Raised when configuration or setup operations fail.

    This includes errors in figure creation, grid setup, or
    invalid parameter combinations.

    Args:
        message: Description of the configuration error
    """
    pass


# ----------------------------
# ENGINE
# ----------------------------

class PlotEngine:
    """
    A comprehensive plotting engine built on matplotlib and seaborn.

    PlotEngine provides a high-level interface for creating publication-quality
    plots with sensible defaults while maintaining full customization capabilities.
    It handles figure creation, data visualization, styling, and export.

    Attributes:
        ut (Utility): Utility instance for logging and file operations
        colors_01 (list[str]): Professional color palette (blue, orange, green, red, purple)
        colors_02 (list[str]): Deep color palette (deep blue, warm orange, etc.)
        colors_03 (list[str]): Soft color palette (soft blue, pink, etc.)
        colors_vibrant (list[str]): Vibrant modern palette (coral, teal, etc.)

    Example:
        >>> pe = PlotEngine()
        >>> fig, ax = pe.create_figure(figsize=(10, 6))
        >>> pe.draw(ax, x=[1, 2, 3, 4], y=[10, 15, 13, 17], plot_type='line')
        >>> pe.decorate(ax, title='Sales Trend', xlabel='Month', ylabel='Revenue')
        >>> pe.save_or_show(fig, save_path='sales.png')
    """

    def __init__(self, logger: None | Logger = None) -> None:
        """
        Initialize the PlotEngine with default settings and color palettes.

        Sets up the utility logger and defines four different color palettes
        for various visualization needs.
        """
        self.logger: Logger = Logger(
            level=logging.DEBUG) if logger is None else logger

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

        This method creates a new figure and axes with seaborn theming applied.
        It's a wrapper around plt.subplots() with sensible defaults and theming.

        Args:
            *args: Positional arguments for plt.subplots (nrows, ncols).
                   For example: (2, 3) creates a 2x3 grid of subplots.
            figsize: Figure size as (width, height) in inches. Default is (10, 6).
            seaborn_theme: Seaborn theme style. Options:
                - 'darkgrid': Dark background with white grid (default)
                - 'whitegrid': White background with gray grid
                - 'dark': Dark background, no grid
                - 'white': White background, no grid
                - 'ticks': White background with tick marks
            seaborn_context: Seaborn context for scaling plot elements. Options:
                - 'paper': Smallest, for papers/publications
                - 'notebook': Default, for notebooks
                - 'talk': Larger, for presentations
                - 'poster': Largest, for posters
            **kwargs: Additional keyword arguments passed to plt.subplots().
                      Common options: sharex, sharey, subplot_kw, gridspec_kw

        Returns:
            Tuple[Figure, Axes | ndarray]: A tuple containing:
                - Figure: The matplotlib Figure object
                - Axes or array of Axes: Single Axes if no grid specified,
                  otherwise array of Axes objects

        Raises:
            ConfigurationError: If figure creation fails due to invalid parameters

        Example:
            >>> # Single plot
            >>> fig, ax = pe.create_figure(figsize=(12, 8))

            >>> # 2x2 grid of subplots
            >>> fig, axes = pe.create_figure(2, 2, figsize=(12, 10))

            >>> # Custom styling
            >>> fig, ax = pe.create_figure(
            ...     figsize=(10, 6),
            ...     seaborn_theme='white',
            ...     seaborn_context='talk'
            ... )
        """
        try:
            sns.set_theme(
                style=seaborn_theme,
                context=seaborn_context,
            )

            return plt.subplots(
                *args, figsize=figsize, **kwargs
            )  # type: ignore[return-value]

        except Exception as e:
            raise ConfigurationError(
                f"Failed to create figure: {str(e)}"
            ) from e

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

        This method provides fine-grained control over subplot layouts, allowing
        for unequal sizing, spanning, and custom spacing. It's more powerful than
        create_figure() for complex layouts.

        Args:
            rows: Number of rows in the grid. Must be >= 1.
            cols: Number of columns in the grid. Must be >= 1.
            figsize: Figure size as (width, height) in inches. Default is (20, 10).
            height_ratios: Relative heights of rows. For example:
                - [2, 1] makes first row twice as tall as second
                - [3, 1, 1] makes first row 3x taller than others
                - None uses equal heights (default)
            width_ratios: Relative widths of columns. For example:
                - [2, 1, 1] makes first column twice as wide
                - None uses equal widths (default)
            hspace: Vertical spacing between subplots as fraction of axes height.
                    Default is 0.3 (30% of subplot height).
            wspace: Horizontal spacing between subplots as fraction of axes width.
                    Default is 0.3 (30% of subplot width).
            seaborn_theme: Seaborn theme style. See create_figure() for options.
            seaborn_context: Seaborn context. See create_figure() for options.

        Returns:
            Tuple[Figure, GridSpec]: A tuple containing:
                - Figure: The matplotlib Figure object
                - GridSpec: GridSpec object used to create subplots via
                  fig.add_subplot(gs[row, col])

        Raises:
            ConfigurationError: If grid creation fails or parameters are invalid
            InvalidDataError: If height_ratios/width_ratios don't match dimensions

        Example:
            >>> # 1 wide plot on top, 3 plots on bottom
            >>> fig, gs = pe.create_custom_grid(2, 3)
            >>> ax1 = fig.add_subplot(gs[0, :])    # Top row, span all columns
            >>> ax2 = fig.add_subplot(gs[1, 0])    # Bottom left
            >>> ax3 = fig.add_subplot(gs[1, 1])    # Bottom middle
            >>> ax4 = fig.add_subplot(gs[1, 2])    # Bottom right

            >>> # 2x2 equal grid
            >>> fig, gs = pe.create_custom_grid(2, 2)
            >>> ax1 = fig.add_subplot(gs[0, 0])
            >>> ax2 = fig.add_subplot(gs[0, 1])
            >>> ax3 = fig.add_subplot(gs[1, 0])
            >>> ax4 = fig.add_subplot(gs[1, 1])

            >>> # Complex layout with custom ratios
            >>> fig, gs = pe.create_custom_grid(
            ...     3, 2,
            ...     height_ratios=[2, 1, 1],
            ...     width_ratios=[3, 1]
            ... )
            >>> ax1 = fig.add_subplot(gs[0, :])    # First row, full width
            >>> ax2 = fig.add_subplot(gs[1, 0])    # Second row, left
            >>> ax3 = fig.add_subplot(gs[1, 1])    # Second row, right
            >>> ax4 = fig.add_subplot(gs[2, :])    # Last row, full width
        """
        # Validate inputs
        if rows < 1 or cols < 1:
            raise ConfigurationError(
                f"Grid dimensions must be >= 1, got rows={rows}, cols={cols}"
            )

        if height_ratios is not None and len(height_ratios) != rows:
            raise InvalidDataError(
                f"height_ratios length ({len(height_ratios)}) must match "
                f"number of rows ({rows})"
            )

        if width_ratios is not None and len(width_ratios) != cols:
            raise InvalidDataError(
                f"width_ratios length ({len(width_ratios)}) must match "
                f"number of columns ({cols})"
            )

        try:
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

        except Exception as e:
            raise ConfigurationError(
                f"Failed to create custom grid: {str(e)}"
            ) from e

    # ----------------------------
    # DRAW
    # ----------------------------

    def _validate_plot_type(self, parse_plot_type: str) -> str:
        """
        Validate and normalize plot type string.

        This internal method ensures the plot type is valid and returns it in
        lowercase. It's used by draw() to validate user input.

        Args:
            parse_plot_type: Plot type string to validate

        Returns:
            str: Validated and normalized (lowercase) plot type

        Raises:
            InvalidPlotTypeError: If plot type is not one of the valid options

        Note:
            Valid plot types are: 'line', 'bar', 'scatter', 'pie'
        """
        parse_plot_type = parse_plot_type.lower()
        valid_types = ["line", "bar", "scatter", "pie"]

        if parse_plot_type not in valid_types:
            raise InvalidPlotTypeError(parse_plot_type)

        return parse_plot_type

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

        This is the main method for adding data visualizations to an axes.
        It supports multiple plot types and forwards all styling options to
        the underlying matplotlib functions.

        Args:
            ax: The matplotlib Axes object to draw on
            x: X-axis data. Can be:
                - Array-like of numbers, strings, or datetime objects
                - pandas Series
                - numpy array
                - None (for pie charts, uses automatic positioning)
            y: Y-axis data. Must be numeric array-like:
                - List/tuple of numbers
                - pandas Series
                - numpy array
            plot_type: Type of plot to create. Options:
                - 'line': Line plot (default)
                - 'bar': Bar chart
                - 'scatter': Scatter plot
                - 'pie': Pie chart (x is ignored)
            **kwargs: Additional keyword arguments passed to matplotlib plotting
                      function. Common options:
                - color: Color of the plot elements
                - label: Label for legend
                - linewidth: Width of lines (line plot)
                - marker: Marker style (line/scatter plots)
                - markersize: Size of markers
                - alpha: Transparency (0-1)
                - linestyle: Line style ('--', '-.', ':', '-')
                For bar charts: width, edgecolor, etc.
                For pie charts: labels, autopct, startangle, etc.

        Raises:
            InvalidPlotTypeError: If plot_type is not valid
            InvalidDataError: If data is incompatible with the plot type

        Example:
            >>> fig, ax = pe.create_figure()

            >>> # Line plot
            >>> pe.draw(ax, x=[1, 2, 3, 4], y=[10, 15, 13, 17],
            ...         plot_type='line', color='blue', linewidth=2)

            >>> # Bar chart with labels
            >>> pe.draw(ax, x=['A', 'B', 'C'], y=[23, 45, 56],
            ...         plot_type='bar', color='green', label='Sales')

            >>> # Scatter plot
            >>> pe.draw(ax, x=[1, 2, 3, 4], y=[10, 15, 13, 17],
            ...         plot_type='scatter', marker='o', s=100, alpha=0.6)

            >>> # Pie chart
            >>> pe.draw(ax, x=None, y=[30, 25, 20, 25],
            ...         plot_type='pie', labels=['A', 'B', 'C', 'D'],
            ...         autopct='%1.1f%%')
        """
        try:
            plot_type_validated = self._validate_plot_type(
                plot_type)  # type: ignore[arg-type]

            # Validate data
            if y is None or (hasattr(y, '__len__') and len(y) == 0):
                raise InvalidDataError("Y data cannot be None or empty")

            if plot_type_validated == "line":
                ax.plot(x, y, **kwargs)
            elif plot_type_validated == "bar":
                ax.bar(x, y, **kwargs)
            elif plot_type_validated == "scatter":
                ax.scatter(x, y, **kwargs)
            elif plot_type_validated == "pie":
                ax.pie(y, **kwargs)

        except InvalidPlotTypeError:
            raise
        except Exception as e:
            raise PlotEngineError(
                f"Failed to draw {plot_type} plot: {str(e)}"
            ) from e

    # ----------------------------
    # DECORATE
    # ----------------------------

    def set_background_color(
        self,
        fig: Figure,
        ax: Union[Axes, np.ndarray, Tuple[Axes, ...]],
        *,
        fig_color: str = "#1a1a1a",
        ax_color: str = "#2a2a2a",
        grid_color: str = "white",
        grid_alpha: float = 0.2,
        apply_to_all: bool = True,
    ) -> None:
        """
        Set background colors for figure and axes.

        This method allows customization of the background colors for both the
        figure (outer area) and axes (plot area), as well as grid styling.

        Args:
            fig: The matplotlib Figure object to modify
            ax: The axes to modify. Can be:
                - Single Axes object
                - numpy array of Axes objects (from subplots)
                - Tuple of Axes objects
            fig_color: Figure background color. Accepts:
                - Hex color (e.g., '#1a1a1a')
                - Named color (e.g., 'black', 'white')
                - RGB tuple (e.g., (0.1, 0.1, 0.1))
                Default is dark gray.
            ax_color: Axes background color. Same format as fig_color.
                      Default is slightly lighter gray.
            grid_color: Grid line color. Default is 'white'.
            grid_alpha: Grid line transparency, range 0-1 where:
                - 0 = fully transparent (invisible)
                - 1 = fully opaque
                Default is 0.2 (subtle).
            apply_to_all: If True and ax is an array, apply to all axes.
                         If False, only apply to the first axes.
                         Default is True.

        Raises:
            ConfigurationError: If color values are invalid

        Example:
            >>> fig, ax = pe.create_figure()
            >>> pe.set_background_color(
            ...     fig, ax,
            ...     fig_color='#0a0a0a',
            ...     ax_color='#1a1a1a',
            ...     grid_color='gray',
            ...     grid_alpha=0.3
            ... )

            >>> # For subplots
            >>> fig, axes = pe.create_figure(2, 2)
            >>> pe.set_background_color(
            ...     fig, axes,
            ...     fig_color='white',
            ...     ax_color='#f5f5f5',
            ...     apply_to_all=True
            ... )
        """
        try:
            fig.patch.set_facecolor(fig_color)

            if isinstance(ax, np.ndarray):
                if apply_to_all:
                    for single_ax in ax.flat:
                        single_ax.set_facecolor(ax_color)
                        single_ax.grid(color=grid_color, alpha=grid_alpha)
                else:
                    ax.flat[0].set_facecolor(ax_color)
                    ax.flat[0].grid(color=grid_color, alpha=grid_alpha)
            elif isinstance(ax, (tuple, list)):
                if apply_to_all:
                    for single_ax in ax:
                        single_ax.set_facecolor(ax_color)
                        single_ax.grid(color=grid_color, alpha=grid_alpha)
                else:
                    ax[0].set_facecolor(ax_color)
                    ax[0].grid(color=grid_color, alpha=grid_alpha)
            else:
                ax.set_facecolor(ax_color)
                ax.grid(color=grid_color, alpha=grid_alpha)

        except Exception as e:
            raise ConfigurationError(
                f"Failed to set background colors: {str(e)}"
            ) from e

    def decorate(
        self,
        ax: Axes,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title_fontsize: int = 18,
        label_fontsize: int = 14,
        title_color: str = "white",
        label_color: str = "white",
        tick_color: str = "white",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Union[Tuple[float, float], str]] = None,
    ) -> None:
        """
        Add title, labels, and styling to an axes.

        This method provides a convenient way to add all the main decorative
        elements to a plot in a single call, with consistent styling.

        Args:
            ax: The matplotlib Axes object to decorate
            title: Plot title. None for no title.
            xlabel: X-axis label. None for no label.
            ylabel: Y-axis label. None for no label.
            title_fontsize: Font size for title in points. Default is 18.
            label_fontsize: Font size for axis labels in points. Default is 14.
            title_color: Color for title text. Default is 'white'.
            label_color: Color for axis labels. Default is 'white'.
            tick_color: Color for tick marks and tick labels. Default is 'white'.
            xlim: X-axis limits as (min, max) tuple. None for automatic.
            ylim: Y-axis limits. Can be:
                - (min, max) tuple for explicit range
                - "zero" to start from 0 with automatic max
                - None for fully automatic (default)

        Raises:
            ConfigurationError: If decoration parameters are invalid

        Example:
            >>> fig, ax = pe.create_figure()
            >>> pe.draw(ax, x=[1, 2, 3], y=[10, 20, 15], plot_type='line')
            >>> pe.decorate(
            ...     ax,
            ...     title='Monthly Sales',
            ...     xlabel='Month',
            ...     ylabel='Revenue ($)',
            ...     title_fontsize=20,
            ...     label_fontsize=14,
            ...     ylim='zero'  # Start y-axis at 0
            ... )

            >>> # Custom colors for dark theme
            >>> pe.decorate(
            ...     ax,
            ...     title='Data Analysis',
            ...     xlabel='Time',
            ...     ylabel='Value',
            ...     title_color='cyan',
            ...     label_color='lightgray',
            ...     tick_color='gray'
            ... )
        """
        try:
            if title:
                ax.set_title(title, fontsize=title_fontsize, color=title_color)

            if xlabel:
                ax.set_xlabel(xlabel, fontsize=label_fontsize,
                              color=label_color)

            if ylabel:
                ax.set_ylabel(ylabel, fontsize=label_fontsize,
                              color=label_color)

            ax.tick_params(axis='both', colors=tick_color)

            if xlim is not None:
                ax.set_xlim(xlim)

            if ylim == "zero":
                ax.set_ylim(bottom=0)
            elif ylim is not None:
                ax.set_ylim(ylim)

        except Exception as e:
            raise ConfigurationError(
                f"Failed to decorate axes: {str(e)}"
            ) from e

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

        Reference lines are useful for highlighting targets, thresholds, averages,
        or other important values in a plot.

        Args:
            ax: The matplotlib Axes object to draw on
            y: Y-value for horizontal line. Provide this OR x, not both.
            x: X-value for vertical line. Provide this OR y, not both.
            label: Label for the line (appears in legend when legend is shown).
                   None for no label.
            color: Line color. Default is 'gray'.
            linestyle: Line style. Options:
                - '--': Dashed (default)
                - '-.': Dash-dot
                - ':': Dotted
                - '-': Solid
            linewidth: Line width in points. Default is 2.
            alpha: Transparency, range 0-1. Default is 0.5 (semi-transparent).

        Raises:
            InvalidDataError: If neither y nor x is provided, or both are provided

        Example:
            >>> fig, ax = pe.create_figure()
            >>> pe.draw(ax, x=[1, 2, 3, 4], y=[10, 15, 13, 17], plot_type='line')

            >>> # Add goal line
            >>> pe.add_reference_line(
            ...     ax,
            ...     y=20,
            ...     label='Target',
            ...     color='red',
            ...     linestyle='--'
            ... )

            >>> # Add vertical marker
            >>> pe.add_reference_line(
            ...     ax,
            ...     x=2.5,
            ...     label='Milestone',
            ...     color='green',
            ...     linestyle='-.',
            ...     alpha=0.7
            ... )
        """
        if y is None and x is None:
            raise InvalidDataError(
                "Must provide either 'y' for horizontal line or 'x' for vertical line"
            )

        if y is not None and x is not None:
            raise InvalidDataError(
                "Cannot provide both 'y' and 'x'. Choose one for either "
                "horizontal or vertical line"
            )

        try:
            if y is not None:
                ax.axhline(
                    y=y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                    label=label
                )
            else:  # x is not None
                ax.axvline(
                    x=x,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                    label=label
                )
        except Exception as e:
            raise PlotEngineError(
                f"Failed to add reference line: {str(e)}"
            ) from e

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

        This method automatically adds text labels showing the value of each bar,
        positioned above the bar. Very useful for making exact values clear.

        Args:
            ax: The matplotlib Axes object with bar chart
            format_string: Python format string for values. Examples:
                - '{:.0f}': Integer (default)
                - '{:.1f}': One decimal place
                - '{:.2f}': Two decimal places
                - '${:,.0f}': Currency with thousands separator
                - '{:.1%}': Percentage with one decimal
            fontsize: Font size for labels in points. Default is 11.
            fontweight: Font weight. Options:
                - 'normal': Regular weight
                - 'bold': Bold (default)
                - 'light': Light weight
                - Numeric: 100-900
            color: Text color. Default is 'black'.
            padding: Padding between bar top and label in points. Default is 3.

        Raises:
            PlotEngineError: If the axes doesn't contain bar containers

        Example:
            >>> fig, ax = pe.create_figure()
            >>> pe.draw(ax, x=['A', 'B', 'C'], y=[23, 45, 56], plot_type='bar')

            >>> # Add integer labels
            >>> pe.add_value_labels_on_bars(ax)

            >>> # Add currency labels
            >>> pe.add_value_labels_on_bars(
            ...     ax,
            ...     format_string='${:,.0f}',
            ...     fontsize=12,
            ...     color='white'
            ... )

            >>> # Add percentage labels
            >>> pe.add_value_labels_on_bars(
            ...     ax,
            ...     format_string='{:.1f}%',
            ...     fontweight='normal'
            ... )
        """
        try:
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
        except Exception as e:
            raise PlotEngineError(
                f"Failed to add value labels on bars: {str(e)}"
            ) from e

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

        This method converts an axes into a statistics display box, useful for
        showing summary metrics alongside plots in a multi-axes figure.

        Args:
            ax: The matplotlib Axes object to convert (will be turned off)
            stats: Dictionary of stat_name: value pairs. For example:
                   {'Total': 1234, 'Average': 45.6, 'Max': 100}
            title: Optional title displayed at the top of the box
            fontsize: Font size for statistics in points. Default is 12.
            title_fontsize: Font size for title in points. Default is 14.
            box_color: Background color of the box. Default is dark gray.
            text_color: Color of the text. Default is 'white'.
            border_color: Color of the border. Default is 'white'.
            border_width: Width of the border in points. Default is 2.

        Raises:
            InvalidDataError: If stats dictionary is empty

        Example:
            >>> fig, axes = pe.create_figure(1, 2)

            >>> # Left plot: actual chart
            >>> pe.draw(axes[0], x=[1, 2, 3], y=[10, 20, 15], plot_type='line')
            >>> pe.decorate(axes[0], title='Trend')

            >>> # Right plot: stats box
            >>> stats = {
            ...     'Total': 45,
            ...     'Average': 15.0,
            ...     'Max': 20,
            ...     'Min': 10
            ... }
            >>> pe.create_stats_text_box(
            ...     axes[1],
            ...     stats,
            ...     title='Summary Statistics'
            ... )

            >>> # Custom styling
            >>> pe.create_stats_text_box(
            ...     ax,
            ...     stats={'Revenue': 125000, 'Growth': '23.5%'},
            ...     title='Q4 Results',
            ...     box_color='navy',
            ...     text_color='gold',
            ...     border_color='gold',
            ...     fontsize=14
            ... )
        """
        if not stats:
            raise InvalidDataError("Stats dictionary cannot be empty")

        try:
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
        except InvalidDataError:
            raise
        except Exception as e:
            raise PlotEngineError(
                f"Failed to create stats text box: {str(e)}"
            ) from e

    def format_y_axis(
        self,
        ax: Axes,
        currency: str = "$",
        style: str = "comma",
        decimals: int = 0
    ) -> None:
        """
        Format y-axis tick labels with currency and/or comma separators.

        This method provides easy formatting for financial and numerical data,
        automatically adding currency symbols and thousands separators.

        Args:
            ax: The matplotlib Axes object to modify
            currency: Currency symbol to prepend. Use "" for no currency.
                     Common examples: '$', '€', '£', '¥'
                     Default is '$'.
            style: Formatting style. Currently supported:
                - 'comma': Add thousands separators (default)
                  Examples: 1,234 or $1,234.56
            decimals: Number of decimal places to display. Default is 0.
                     Examples:
                     - 0: 1,234
                     - 1: 1,234.5
                     - 2: 1,234.56

        Example:
            >>> fig, ax = pe.create_figure()
            >>> pe.draw(ax, x=[1, 2, 3], y=[1234, 5678, 9012], plot_type='bar')

            >>> # Format as currency
            >>> pe.format_y_axis(ax, currency='$', decimals=0)
            >>> # Result: $1,234  $5,678  $9,012

            >>> # Format as plain numbers with decimals
            >>> pe.format_y_axis(ax, currency='', decimals=2)
            >>> # Result: 1,234.00  5,678.00  9,012.00

            >>> # Format as euros
            >>> pe.format_y_axis(ax, currency='€', decimals=2)
            >>> # Result: €1,234.00  €5,678.00  €9,012.00
        """
        try:
            if style == "comma":
                ax.yaxis.set_major_formatter(
                    StrMethodFormatter(f"{currency}{{x:,.{decimals}f}}")
                )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to format y-axis: {str(e)}"
            ) from e

    def force_xticks(
        self,
        ax: Axes,
        labels: np.ndarray,
        positions: Optional[np.ndarray] = None,
        rotation: int = 45
    ) -> None:
        """
        Manually set x-axis tick positions and labels.

        This method gives you full control over x-axis ticks, useful when
        matplotlib's automatic tick placement doesn't meet your needs.

        Args:
            ax: The matplotlib Axes object to modify
            labels: Array of tick labels (strings or convertible to strings)
            positions: Array of tick positions. If None, uses range(len(labels)),
                      which places ticks at 0, 1, 2, etc.
            rotation: Label rotation in degrees. Default is 45.
                     Common values:
                     - 0: Horizontal
                     - 45: Diagonal (default, good for long labels)
                     - 90: Vertical

        Raises:
            InvalidDataError: If labels and positions have different lengths

        Example:
            >>> fig, ax = pe.create_figure()
            >>> months = np.array(['Jan', 'Feb', 'Mar', 'Apr'])
            >>> values = [10, 15, 13, 17]
            >>> pe.draw(ax, x=range(len(months)), y=values, plot_type='line')

            >>> # Use month names as labels
            >>> pe.force_xticks(ax, labels=months, rotation=0)

            >>> # Custom positions
            >>> pe.force_xticks(
            ...     ax,
            ...     labels=np.array(['Q1', 'Q2', 'Q3', 'Q4']),
            ...     positions=np.array([0, 3, 6, 9]),
            ...     rotation=0
            ... )
        """
        if positions is None:
            positions = np.arange(len(labels))

        if len(labels) != len(positions):
            raise InvalidDataError(
                f"labels and positions must have the same length. "
                f"Got labels: {len(labels)}, positions: {len(positions)}"
            )

        try:
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=rotation, ha="right")
        except Exception as e:
            raise ConfigurationError(
                f"Failed to set x-axis ticks: {str(e)}"
            ) from e

    def force_yticks(
        self,
        ax: Axes,
        nbins: int = 10,
        bottom: Optional[int] = None
    ) -> None:
        """
        Control y-axis tick density and minimum value.

        This method helps control the number of y-axis ticks and optionally
        sets a minimum value, useful for cleaner axes or emphasizing ranges.

        Args:
            ax: The matplotlib Axes object to modify
            nbins: Maximum number of y-axis ticks. The actual number may be
                  less to maintain "nice" numbers. Default is 10.
            bottom: Minimum y-axis value (forces the y-axis to start here).
                   None for automatic. Common use: set to 0 to start from zero.

        Example:
            >>> fig, ax = pe.create_figure()
            >>> pe.draw(ax, x=[1, 2, 3], y=[10, 15, 13], plot_type='line')

            >>> # Limit to 5 ticks, start at 0
            >>> pe.force_yticks(ax, nbins=5, bottom=0)

            >>> # More ticks for detailed view
            >>> pe.force_yticks(ax, nbins=20)

            >>> # Start from specific value
            >>> pe.force_yticks(ax, bottom=10)
        """
        try:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=nbins))
            if bottom is not None:
                ax.set_ylim(bottom=bottom)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to set y-axis ticks: {str(e)}"
            ) from e

    def add_margins(
        self,
        ax: Axes,
        xpad: float = 0.05,
        ypad: float = 0.2
    ) -> None:
        """
        Add padding/margins around the data in the plot.

        Margins create whitespace between the data and the axes edges,
        preventing data points from touching the plot boundaries.

        Args:
            ax: The matplotlib Axes object to modify
            xpad: X-axis padding as fraction of data range.
                 For example, 0.05 adds 5% padding on each side.
                 Default is 0.05 (5%).
            ypad: Y-axis padding as fraction of data range.
                 Default is 0.2 (20%).

        Example:
            >>> fig, ax = pe.create_figure()
            >>> pe.draw(ax, x=[1, 2, 3, 4], y=[10, 20, 15, 25], plot_type='scatter')

            >>> # Add generous margins
            >>> pe.add_margins(ax, xpad=0.1, ypad=0.3)

            >>> # Tight fit with minimal margins
            >>> pe.add_margins(ax, xpad=0.02, ypad=0.05)

            >>> # More margin on y-axis
            >>> pe.add_margins(ax, xpad=0.05, ypad=0.25)
        """
        try:
            ax.margins(x=xpad, y=ypad)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to add margins: {str(e)}"
            ) from e

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

        Legends identify different data series in plots where multiple series
        are shown (using the 'label' parameter in draw()).

        Args:
            ax: The matplotlib Axes object to modify
            loc: Legend location. Options:
                - 'upper right' (default)
                - 'upper left'
                - 'lower left'
                - 'lower right'
                - 'upper center'
                - 'lower center'
                - 'center left'
                - 'center right'
                - 'center'
                - 'best' (automatically choose best position)
            fontsize: Legend font size in points. Default is 14.
            **kwargs: Additional keyword arguments passed to ax.legend().
                     Common options:
                     - frameon: bool, draw frame around legend
                     - fancybox: bool, rounded corners
                     - shadow: bool, add shadow
                     - ncol: int, number of columns
                     - title: str, legend title

        Example:
            >>> fig, ax = pe.create_figure()
            >>> pe.draw(ax, x=[1,2,3], y=[10,15,13], plot_type='line', label='Series A')
            >>> pe.draw(ax, x=[1,2,3], y=[12,14,16], plot_type='line', label='Series B')

            >>> # Simple legend
            >>> pe.set_legend(ax)

            >>> # Custom position and size
            >>> pe.set_legend(ax, loc='lower right', fontsize=12)

            >>> # With additional styling
            >>> pe.set_legend(
            ...     ax,
            ...     loc='best',
            ...     fontsize=12,
            ...     frameon=True,
            ...     shadow=True,
            ...     title='Data Series'
            ... )
        """
        try:
            ax.legend(loc=loc, fontsize=fontsize, **kwargs)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to set legend: {str(e)}"
            ) from e

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

        This method should be called last, after all plotting and decoration.
        It handles layout adjustment, saving to file, displaying, and cleanup.

        Args:
            fig: The matplotlib Figure object to finalize
            dpi: DPI (dots per inch) for saved image. Higher = better quality.
                Common values:
                - 72-96: Screen viewing
                - 150-200: Draft printing
                - 300: High-quality printing (default)
                - 600: Publication quality
            save_path: Path to save figure. Can be absolute or relative.
                      Supported formats (auto-detected from extension):
                      - .png: Portable Network Graphics
                      - .jpg/.jpeg: JPEG
                      - .pdf: PDF (vector)
                      - .svg: SVG (vector)
                      - .eps: Encapsulated PostScript
                      None to skip saving (default).
            top: Top margin as fraction of figure height. Default is 0.95.
            bottom: Bottom margin as fraction of figure height. Default is 0.08.
            show: Whether to display the figure in a window. Default is True.
                 Set to False when running headless or batch processing.
            use_tight_layout: Whether to use tight_layout for automatic spacing.
                             Default is True. Set to False if you need manual
                             control via subplots_adjust.

        Raises:
            PlotEngineError: If saving fails or path is invalid

        Example:
            >>> fig, ax = pe.create_figure()
            >>> pe.draw(ax, x=[1, 2, 3], y=[10, 15, 13], plot_type='line')
            >>> pe.decorate(ax, title='My Plot')

            >>> # Just display
            >>> pe.save_or_show(fig)

            >>> # Save and display
            >>> pe.save_or_show(fig, save_path='plot.png')

            >>> # Save high-res without displaying
            >>> pe.save_or_show(
            ...     fig,
            ...     dpi=600,
            ...     save_path='publication_figure.pdf',
            ...     show=False
            ... )

            >>> # Custom margins
            >>> pe.save_or_show(
            ...     fig,
            ...     top=0.92,
            ...     bottom=0.10,
            ...     save_path='figure.png'
            ... )
        """
        try:
            fig.subplots_adjust(top=top, bottom=bottom)

            if use_tight_layout:
                fig.tight_layout()

            if save_path is not None:
                safe_save_path = self.ut.ensure_writable_path(save_path)
                fig.savefig(safe_save_path, dpi=dpi)
                self.logger.debug(f"Saved figure to {safe_save_path}")

            if show:
                plt.show()

            plt.close(fig)

        except Exception as e:
            raise PlotEngineError(
                f"Failed to save or show figure: {str(e)}"
            ) from e
