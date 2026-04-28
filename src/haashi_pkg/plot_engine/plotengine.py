

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
from matplotlib.patches import FancyBboxPatch
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
    List,
    cast
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

from haashi_pkg.utility import Logger, FileHandler

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

QuickTheme = Literal["dark", "light"]
DashTheme = Literal["dark", "light"]
PanelIndex = Union[int, Tuple[int, int]]


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

    def __init__(self, logger: Logger | None = None) -> None:
        """
        Initialize the PlotEngine with default settings and color palettes.

        Sets up the utility logger and defines four different color palettes
        for various visualization needs.
        """
        self.logger: Logger = logger if logger else Logger(
            level=logging.DEBUG)
        self.handler: FileHandler = FileHandler(logger=self.logger)

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

    def _set_theme_safe(
        self,
        style: ThemeType = "darkgrid",
        context: ContextType = "notebook",
    ) -> None:
        """
        Safely set seaborn theme without breaking logging configuration.

        Seaborn's set_theme() can interfere with logging handlers.
        This method preserves the logging configuration.

        Args:
            style: Seaborn theme style
            context: Seaborn context for scaling
        """
        # Save current logging configuration
        root_logger = logging.getLogger()
        original_level = root_logger.level
        original_handlers = root_logger.handlers.copy()

        # Set seaborn theme
        sns.set_theme(style=style, context=context)

        # Restore logging configuration
        root_logger.setLevel(original_level)
        root_logger.handlers = original_handlers

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
            self._set_theme_safe(
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
            self._set_theme_safe(
                style=seaborn_theme, context=seaborn_context
            )

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

    def set_suptitle(
        self,
        fig: Figure,
        title: str,
        *,
        fontsize: int = 16,
        fontweight: str = "bold",
        color: str = "black",
        **kwargs: Any,
    ) -> None:
        """
        Set the main title for the entire figure (suptitle).

        This is different from ax.set_title() which sets the title for a single
        axes. Use this for the overall figure title when you have multiple subplots.

        Args:
            fig: The matplotlib Figure object to modify
            title: The title text to display
            fontsize: Font size for the title in points. Default is 16.
            fontweight: Font weight. Options:
                - 'normal': Regular weight
                - 'bold': Bold (default)
                - 'light': Light weight
                - Numeric: 100-900
            color: Title text color. Default is 'black'.
            **kwargs: Additional keyword arguments passed to fig.suptitle().
                     Common options:
                     - x: float, x position (0-1)
                     - y: float, y position (0-1)
                     - ha: horizontal alignment ('left', 'center', 'right')
                     - va: vertical alignment ('top', 'center', 'bottom')

        Example:
            >>> fig, axes = pe.create_figure(2, 2)
            >>> 
            >>> # Set individual subplot titles
            >>> axes[0, 0].set_title('Subplot 1')
            >>> axes[0, 1].set_title('Subplot 2')
            >>> 
            >>> # Set overall figure title
            >>> pe.set_suptitle(fig, 'Overall Dashboard Title')

            >>> # Custom styling
            >>> pe.set_suptitle(
            ...     fig,
            ...     'My Dashboard',
            ...     fontsize=20,
            ...     color='navy',
            ...     fontweight='bold'
            ... )

            >>> # Custom positioning
            >>> pe.set_suptitle(
            ...     fig,
            ...     'Report Title',
            ...     y=0.98,  # Higher position
            ...     ha='left',
            ...     x=0.1
            ... )
        """
        try:
            fig.suptitle(
                title,
                fontsize=fontsize,
                fontweight=fontweight,
                color=color,
                **kwargs
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to set figure suptitle: {str(e)}"
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
                safe_save_path = self.handler.ensure_writable_path(
                    save_path)
                fig.savefig(safe_save_path, dpi=dpi)
                self.logger.debug(f"Saved figure to {safe_save_path}")

            if show:
                plt.show()

            plt.close(fig)

        except Exception as e:
            raise PlotEngineError(
                f"Failed to save or show figure: {str(e)}"
            ) from e


# ----------------------------
# QUICK PLOT
# ----------------------------


class QuickPlot:
    """
    A rapid visualization class built on top of PlotEngine.

    QuickPlot collapses the full PlotEngine workflow — create_figure,
    draw, decorate, set_background_color, save_or_show — into a single
    method call per plot type. Plots still look polished with sensible
    defaults, but you can override anything via keyword arguments.

    Two built-in themes are available:
    - "dark"  : Dark background, white text, subtle grid (default)
    - "light" : White background, dark text, gray grid

    Attributes:
        _engine (PlotEngine): The underlying PlotEngine instance
        theme (QuickTheme): Active theme ('dark' or 'light')

    Example:
        >>> qp = QuickPlot(theme="dark")

        >>> # Minimal — just data and a title
        >>> qp.line(x=[1, 2, 3, 4], y=[10, 15, 13, 17], title="Trend")

        >>> # With more options
        >>> qp.bar(
        ...     x=["Jan", "Feb", "Mar"],
        ...     y=[5000, 7200, 6800],
        ...     title="Monthly Revenue",
        ...     ylabel="USD",
        ...     color="#4ECDC4",
        ...     value_labels=True,
        ...     save_path="revenue.png"
        ... )

        >>> # Scatter with reference line
        >>> qp.scatter(
        ...     x=[1, 2, 3, 4, 5],
        ...     y=[3, 7, 2, 9, 5],
        ...     title="Distribution",
        ...     ref_y=5.0,
        ...     ref_label="Average"
        ... )

        >>> # Pie chart
        >>> qp.pie(
        ...     values=[40, 30, 20, 10],
        ...     labels=["A", "B", "C", "D"],
        ...     title="Market Share"
        ... )
    """

    # ----------------------------
    # THEME CONFIGS
    # ----------------------------

    _THEMES: Dict[str, Dict[str, Union[str, float]]] = {
        "dark": {
            "seaborn_style": "darkgrid",
            "fig_color":     "#1a1a1a",
            "ax_color":      "#2a2a2a",
            "grid_color":    "white",
            "grid_alpha":    0.15,
            "title_color":   "white",
            "label_color":   "white",
            "tick_color":    "white",
            "default_color": "#4E79A7",
        },
        "light": {
            "seaborn_style": "whitegrid",
            "fig_color":     "#f5f5f5",
            "ax_color":      "#ffffff",
            "grid_color":    "#cccccc",
            "grid_alpha":    0.6,
            "title_color":   "#1a1a1a",
            "label_color":   "#333333",
            "tick_color":    "#444444",
            "default_color": "#1f77b4",
        },
    }

    def __init__(
        self,
        theme: QuickTheme = "dark",
        logger: Optional[Logger] = None,
    ) -> None:
        """
        Initialize QuickPlot with a theme and an internal PlotEngine.

        Args:
            theme: Visual theme. Options:
                - 'dark'  : Dark background, white labels (default)
                - 'light' : Light background, dark labels
            logger: Optional Logger instance. If None, PlotEngine creates its own.

        Example:
            >>> qp = QuickPlot()                   # dark theme
            >>> qp = QuickPlot(theme="light")      # light theme
            >>> qp = QuickPlot(theme="dark", logger=my_logger)
        """
        if theme not in self._THEMES:
            raise ValueError(
                f"Invalid theme '{theme}'. Choose from: {list(self._THEMES)}"
            )

        self.theme: QuickTheme = theme
        self._engine: PlotEngine = PlotEngine(logger=logger)
        self._cfg = self._THEMES[theme]

    # ----------------------------
    # INTERNAL HELPERS
    # ----------------------------

    def _seaborn_style(self) -> ThemeType:
        return self._cfg["seaborn_style"]  # type: ignore[return-value]

    def _apply_background(self, fig: Figure, ax: Axes) -> None:
        """Apply theme background colors to fig and ax."""
        self._engine.set_background_color(
            fig, ax,
            fig_color=cast(str, self._cfg["fig_color"]),
            ax_color=cast(str, self._cfg["ax_color"]),
            grid_color=cast(str, self._cfg["grid_color"]),
            grid_alpha=cast(float, self._cfg["grid_alpha"]),
        )

    def _apply_decorate(
        self,
        ax: Axes,
        title: Optional[str],
        xlabel: Optional[str],
        ylabel: Optional[str],
        xlim: Optional[Tuple[float, float]],
        ylim,
        title_fontsize: int,
        label_fontsize: int,
    ) -> None:
        """Apply labels and axis settings using theme colors."""
        self._engine.decorate(
            ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            title_color=cast(str, self._cfg["title_color"]),
            label_color=cast(str, self._cfg["label_color"]),
            tick_color=cast(str, self._cfg["tick_color"]),
            xlim=xlim,
            ylim=ylim,
        )

    def _finalize(
        self,
        fig: Figure,
        save_path: Optional[str],
        dpi: int,
        show: bool,
        top: float = 0.93,
        bottom: float = 0.1,
    ) -> None:
        """Finalize and optionally save/show the figure."""
        self._engine.save_or_show(
            fig,
            dpi=dpi,
            save_path=save_path,
            top=top,
            bottom=bottom,
            show=show,
        )

    # ----------------------------
    # PUBLIC PLOT METHODS
    # ----------------------------

    def line(
        self,
        x: XData,
        y: YData,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
        linewidth: float = 2.5,
        marker: Optional[str] = "o",
        markersize: int = 6,
        label: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        xlim: Optional[Tuple[float, float]] = None,
        ylim=None,
        ref_y: Optional[float] = None,
        ref_label: Optional[str] = None,
        ref_color: str = "gray",
        title_fontsize: int = 18,
        label_fontsize: int = 13,
        legend: bool = False,
        save_path: Optional[str] = None,
        dpi: int = 150,
        show: bool = False,
        context: ContextType = "notebook",
    ) -> None:
        """
        Create a quick line plot.

        Args:
            x: X-axis data (numbers, strings, datetimes, or pandas Series).
            y: Y-axis numeric data.
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            color: Line color. Defaults to theme's primary color.
            linewidth: Width of the line. Default is 2.5.
            marker: Marker style. Default is 'o'. Use None for no markers.
            markersize: Size of markers. Default is 6.
            label: Series label (shows in legend if legend=True).
            figsize: Figure size as (width, height). Default is (10, 6).
            xlim: X-axis limits as (min, max). None for automatic.
            ylim: Y-axis limits. (min, max) tuple, "zero", or None.
            ref_y: Optional horizontal reference line value (e.g. target, mean).
            ref_label: Label for the reference line (shown in legend).
            ref_color: Color of the reference line. Default is 'gray'.
            title_fontsize: Title font size. Default is 18.
            label_fontsize: Axis label font size. Default is 13.
            legend: Whether to show a legend. Default is False.
            save_path: File path to save the figure (e.g. 'plot.png'). None to skip.
            dpi: Resolution for saved image. Default is 150.
            show: Whether to display the figure. Default is True.
            context: Seaborn context ('paper', 'notebook', 'talk', 'poster').

        Example:
            >>> qp = QuickPlot()
            >>> qp.line(
            ...     x=[1, 2, 3, 4, 5],
            ...     y=[10, 14, 12, 18, 15],
            ...     title="Weekly Trend",
            ...     xlabel="Week",
            ...     ylabel="Value",
            ...     ref_y=13,
            ...     ref_label="Average"
            ... )
        """
        clr = color or self._cfg["default_color"]

        fig, ax = self._engine.create_figure(
            figsize=figsize,
            seaborn_theme=self._seaborn_style(),
            seaborn_context=context,
        )
        self._apply_background(fig, ax)

        self._engine.draw(
            ax, x=x, y=y, plot_type="line",
            color=clr,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
            **({"label": label} if label else {}),
        )

        if ref_y is not None:
            self._engine.add_reference_line(
                ax, y=ref_y, label=ref_label, color=ref_color
            )

        self._apply_decorate(
            ax, title=title, xlabel=xlabel, ylabel=ylabel,
            xlim=xlim, ylim=ylim,
            title_fontsize=title_fontsize, label_fontsize=label_fontsize,
        )

        if legend or label or ref_label:
            self._engine.set_legend(ax, fontsize=12)

        self._finalize(fig, save_path=save_path, dpi=dpi, show=show)

    def bar(
        self,
        x: XData,
        y: YData,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
        edgecolor: Optional[str] = None,
        value_labels: bool = False,
        value_format: str = "{:.0f}",
        value_color: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        xlim: Optional[Tuple[float, float]] = None,
        ylim=None,
        rotation: int = 45,
        ref_y: Optional[float] = None,
        ref_label: Optional[str] = None,
        ref_color: str = "gray",
        title_fontsize: int = 18,
        label_fontsize: int = 13,
        save_path: Optional[str] = None,
        dpi: int = 150,
        show: bool = False,
        context: ContextType = "notebook",
    ) -> None:
        """
        Create a quick bar chart.

        Args:
            x: Category labels for bars.
            y: Bar height values (numeric).
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            color: Bar color. Defaults to theme's primary color.
            edgecolor: Bar edge color. Defaults to theme's ax background color.
            value_labels: If True, adds value labels on top of bars. Default is False.
            value_format: Python format string for value labels. Default is '{:.0f}'.
                          Examples: '${:,.0f}', '{:.1f}%', '{:,.2f}'
            value_color: Color of value labels. Defaults to theme title color.
            figsize: Figure size as (width, height). Default is (10, 6).
            xlim: X-axis limits. None for automatic.
            ylim: Y-axis limits. (min, max) tuple, "zero", or None.
            rotation: X tick label rotation in degrees. Default is 45.
            ref_y: Optional horizontal reference line value.
            ref_label: Label for the reference line.
            ref_color: Color of the reference line. Default is 'gray'.
            title_fontsize: Title font size. Default is 18.
            label_fontsize: Axis label font size. Default is 13.
            save_path: File path to save the figure. None to skip.
            dpi: Resolution for saved image. Default is 150.
            show: Whether to display the figure. Default is True.
            context: Seaborn context ('paper', 'notebook', 'talk', 'poster').

        Example:
            >>> qp = QuickPlot(theme="light")
            >>> qp.bar(
            ...     x=["Jan", "Feb", "Mar", "Apr"],
            ...     y=[5200, 7100, 6300, 8800],
            ...     title="Monthly Sales",
            ...     ylabel="Revenue ($)",
            ...     value_labels=True,
            ...     value_format="${:,.0f}",
            ...     save_path="sales.png"
            ... )
        """
        clr = color or self._cfg["default_color"]
        edge = edgecolor or self._cfg["ax_color"]
        val_clr = cast(str, value_color or self._cfg["title_color"])

        fig, ax = self._engine.create_figure(
            figsize=figsize,
            seaborn_theme=self._seaborn_style(),
            seaborn_context=context,
        )
        self._apply_background(fig, ax)

        self._engine.draw(
            ax, x=x, y=y, plot_type="bar",
            color=clr,
            edgecolor=edge,
        )

        if value_labels:
            self._engine.add_value_labels_on_bars(
                ax,
                format_string=value_format,
                color=val_clr,
                fontsize=11,
            )

        if ref_y is not None:
            self._engine.add_reference_line(
                ax, y=ref_y, label=ref_label, color=ref_color
            )

        # Rotate x tick labels if needed
        ax.tick_params(axis='x', rotation=rotation)

        self._apply_decorate(
            ax, title=title, xlabel=xlabel, ylabel=ylabel,
            xlim=xlim, ylim=ylim,
            title_fontsize=title_fontsize, label_fontsize=label_fontsize,
        )

        if ref_label:
            self._engine.set_legend(ax, fontsize=12)

        self._finalize(fig, save_path=save_path, dpi=dpi, show=show)

    def scatter(
        self,
        x: XData,
        y: YData,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        color: Optional[str] = None,
        size: int = 80,
        alpha: float = 0.75,
        marker: str = "o",
        label: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6),
        xlim: Optional[Tuple[float, float]] = None,
        ylim=None,
        ref_y: Optional[float] = None,
        ref_x: Optional[float] = None,
        ref_label: Optional[str] = None,
        ref_color: str = "gray",
        title_fontsize: int = 18,
        label_fontsize: int = 13,
        legend: bool = False,
        save_path: Optional[str] = None,
        dpi: int = 150,
        show: bool = False,
        context: ContextType = "notebook",
    ) -> None:
        """
        Create a quick scatter plot.

        Args:
            x: X-axis data.
            y: Y-axis numeric data.
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            color: Marker color. Defaults to theme's primary color.
            size: Marker size. Default is 80.
            alpha: Marker transparency (0-1). Default is 0.75.
            marker: Marker shape. Default is 'o'.
            label: Series label for legend.
            figsize: Figure size as (width, height). Default is (10, 6).
            xlim: X-axis limits. None for automatic.
            ylim: Y-axis limits. (min, max) tuple, "zero", or None.
            ref_y: Optional horizontal reference line value.
            ref_x: Optional vertical reference line value.
            ref_label: Label for the reference line.
            ref_color: Color of the reference line. Default is 'gray'.
            title_fontsize: Title font size. Default is 18.
            label_fontsize: Axis label font size. Default is 13.
            legend: Whether to show a legend. Default is False.
            save_path: File path to save the figure. None to skip.
            dpi: Resolution for saved image. Default is 150.
            show: Whether to display the figure. Default is True.
            context: Seaborn context ('paper', 'notebook', 'talk', 'poster').

        Example:
            >>> qp = QuickPlot()
            >>> qp.scatter(
            ...     x=[1, 2, 3, 4, 5, 6],
            ...     y=[5, 9, 3, 7, 2, 8],
            ...     title="Score Distribution",
            ...     ref_y=5.0,
            ...     ref_label="Median"
            ... )
        """
        clr = color or self._cfg["default_color"]

        fig, ax = self._engine.create_figure(
            figsize=figsize,
            seaborn_theme=self._seaborn_style(),
            seaborn_context=context,
        )
        self._apply_background(fig, ax)

        self._engine.draw(
            ax, x=x, y=y, plot_type="scatter",
            color=clr,
            s=size,
            alpha=alpha,
            marker=marker,
            **({"label": label} if label else {}),
        )

        if ref_y is not None:
            self._engine.add_reference_line(
                ax, y=ref_y, label=ref_label, color=ref_color
            )
        if ref_x is not None:
            self._engine.add_reference_line(
                ax, x=ref_x, label=ref_label, color=ref_color
            )

        self._apply_decorate(
            ax, title=title, xlabel=xlabel, ylabel=ylabel,
            xlim=xlim, ylim=ylim,
            title_fontsize=title_fontsize, label_fontsize=label_fontsize,
        )

        if legend or label or ref_label:
            self._engine.set_legend(ax, fontsize=12)

        self._finalize(fig, save_path=save_path, dpi=dpi, show=show)

    def pie(
        self,
        values: YData,
        *,
        labels: Optional[Iterable[str]] = None,
        title: Optional[str] = None,
        colors: Optional[Iterable[str]] = None,
        autopct: str = "%1.1f%%",
        startangle: int = 140,
        explode: Optional[Iterable[float]] = None,
        shadow: bool = False,
        figsize: Tuple[int, int] = (8, 8),
        title_fontsize: int = 18,
        save_path: Optional[str] = None,
        dpi: int = 150,
        show: bool = False,
        context: ContextType = "notebook",
    ) -> None:
        """
        Create a quick pie chart.

        Args:
            values: Numeric values for each slice.
            labels: Slice labels. None for no labels.
            title: Chart title.
            colors: List of slice colors. Defaults to the engine's colors_01 palette.
            autopct: Format string for percentage labels. Default is '%1.1f%%'.
                     Use None to hide percentages.
            startangle: Starting angle in degrees. Default is 140.
            explode: Offset for each slice (e.g. [0.1, 0, 0, 0] to pop out first slice).
                     None for no explosion.
            shadow: Whether to add a shadow effect. Default is False.
            figsize: Figure size as (width, height). Default is (8, 8).
            title_fontsize: Title font size. Default is 18.
            save_path: File path to save the figure. None to skip.
            dpi: Resolution for saved image. Default is 150.
            show: Whether to display the figure. Default is True.
            context: Seaborn context ('paper', 'notebook', 'talk', 'poster').

        Example:
            >>> qp = QuickPlot(theme="light")
            >>> qp.pie(
            ...     values=[40, 30, 20, 10],
            ...     labels=["Product A", "Product B", "Product C", "Other"],
            ...     title="Revenue Breakdown",
            ...     explode=[0.05, 0, 0, 0]
            ... )
        """
        clrs = colors or self._engine.colors_01

        fig, ax = self._engine.create_figure(
            figsize=figsize,
            seaborn_theme=self._seaborn_style(),
            seaborn_context=context,
        )

        # For pie, set the figure background only (ax background is irrelevant)
        fig.patch.set_facecolor(cast(str, self._cfg["fig_color"]))
        ax.set_facecolor(self._cfg["fig_color"])

        self._engine.draw(
            ax, x=None, y=values, plot_type="pie",
            labels=labels,
            colors=clrs,
            autopct=autopct,
            startangle=startangle,
            **({"explode": explode} if explode is not None else {}),
            shadow=shadow,
        )

        if title:
            ax.set_title(
                title,
                fontsize=title_fontsize,
                color=self._cfg["title_color"],
                pad=16,
            )

        # Style the percentage and label text to match theme
        for text in ax.texts:
            text.set_color(self._cfg["title_color"])

        self._finalize(fig, save_path=save_path, dpi=dpi, show=show)


# ----------------------------
# POWERCANVAS
# ----------------------------


class PowerCanvas:
    """
    A Power BI-style dashboard builder built on top of PlotEngine.

    PowerCanvas lets you compose full multi-panel dashboards — KPI cards,
    sparklines, bar/line/scatter/pie/donut charts, and stats panels — all
    inside a single polished figure. Two layout modes are available:

    - **Preset layouts**: Call one method to get a ready-made grid structure
      (KPI row on top + charts below, split layout, full grid, etc.)
    - **Flexible layout**: Define your own rows/cols and place panels anywhere,
      including spanning multiple columns or rows.

    Attributes:
        title (str): Dashboard title shown at the top.
        theme (DashTheme): Active theme ('dark' or 'light').
        _engine (PlotEngine): Underlying PlotEngine instance.
        _fig (Figure): The active matplotlib Figure.
        _panels (dict): Mapping of panel index → Axes object.

    Example:
        >>> pc = PowerCanvas(title="Sales Dashboard 2024", theme="light")
        >>> pc.preset_kpi_top(kpi_count=4, chart_rows=1, chart_cols=2)
        >>> pc.add_kpi_card(0, label="Revenue", value="$8.4M", delta="+12%", delta_up=True)
        >>> pc.add_kpi_card(1, label="Orders",  value="24,842")
        >>> pc.add_kpi_card(2, label="Customers", value="5,320")
        >>> pc.add_kpi_card(3, label="Profit",  value="$1.6M", delta="+5%", delta_up=True)
        >>> pc.add_bar(4, x=["North","South","East","West"], y=[134,98,112,145],
        ...            title="Sales by Region")
        >>> pc.add_line(5, x=months, y=revenue, title="Revenue Trend")
        >>> pc.render(save_path="sales_dashboard.png")
    """

    # ----------------------------
    # THEME CONFIGS
    # ----------------------------

    _THEMES: dict = {
        "dark": {
            "fig_color":        "#0f0f0f",
            "canvas_color":     "#1a1a1a",
            "panel_color":      "#242424",
            "panel_edge":       "#3a3a3a",
            "header_color":     "#1f1f1f",
            "title_color":      "#ffffff",
            "label_color":      "#cccccc",
            "tick_color":       "#aaaaaa",
            "grid_color":       "white",
            "grid_alpha":       0.08,
            "kpi_value_color":  "#ffffff",
            "kpi_label_color":  "#aaaaaa",
            "delta_up_color":   "#00c48c",
            "delta_down_color": "#ff6b6b",
            "accent":           "#4ECDC4",
            "seaborn_style":    "darkgrid",
            "default_colors": [
                "#4ECDC4", "#45B7D1", "#96CEB4",
                "#FFEAA7", "#DDA0DD", "#98D8C8",
            ],
        },
        "light": {
            "fig_color":        "#f0f2f5",
            "canvas_color":     "#f0f2f5",
            "panel_color":      "#ffffff",
            "panel_edge":       "#e0e0e0",
            "header_color":     "#1a3a5c",
            "title_color":      "#ffffff",
            "label_color":      "#333333",
            "tick_color":       "#555555",
            "grid_color":       "#cccccc",
            "grid_alpha":       0.5,
            "kpi_value_color":  "#1a3a5c",
            "kpi_label_color":  "#666666",
            "delta_up_color":   "#00a86b",
            "delta_down_color": "#e63946",
            "accent":           "#1a3a5c",
            "seaborn_style":    "whitegrid",
            "default_colors": [
                "#1a3a5c", "#2196F3", "#4CAF50",
                "#FF9800", "#9C27B0", "#00BCD4",
            ],
        },
    }

    def __init__(
        self,
        title: str = "Dashboard",
        theme: DashTheme = "light",
        figsize: Tuple[int, int] = (20, 12),
        logger=None,
    ) -> None:
        """
        Initialize PowerCanvas.

        Args:
            title: Dashboard title displayed at the top. Default is 'Dashboard'.
            theme: Visual theme — 'dark' or 'light'. Default is 'light'.
            figsize: Overall figure size (width, height) in inches. Default (20, 12).
            logger: Optional Logger instance.

        Example:
            >>> pc = PowerCanvas(title="Q4 Report", theme="dark", figsize=(24, 14))
        """
        if theme not in self._THEMES:
            raise ValueError(
                f"Invalid theme '{theme}'. Choose from: {list(self._THEMES)}"
            )

        self.title:   str = title
        self.theme:   DashTheme = theme
        self.figsize: Tuple = figsize
        self._engine: PlotEngine = PlotEngine(logger=logger)
        self._cfg:    dict = self._THEMES[theme]

        # State
        self._fig:    Optional[Figure] = None
        self._panels: Dict[int, Axes] = {}   # flat index → ax
        self._panel_count: int = 0
        self._layout_mode: str = "none"  # 'preset' | 'flexible'

    # ----------------------------
    # INTERNAL HELPERS
    # ----------------------------

    def _style_panel(self, ax: Axes) -> None:
        """Apply theme styling to a panel axes."""
        ax.set_facecolor(self._cfg["panel_color"])
        ax.tick_params(colors=self._cfg["tick_color"])
        ax.grid(color=self._cfg["grid_color"], alpha=self._cfg["grid_alpha"])
        for spine in ax.spines.values():
            spine.set_edgecolor(self._cfg["panel_edge"])

    def _style_chart_ax(
        self,
        ax: Axes,
        title:  Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title_fontsize: int = 13,
        label_fontsize: int = 10,
    ) -> None:
        """Apply labels and tick colors to a chart panel."""
        self._style_panel(ax)
        if title:
            ax.set_title(
                title,
                fontsize=title_fontsize,
                color=self._cfg["label_color"],
                fontweight="bold",
                pad=8,
            )
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=label_fontsize,
                          color=self._cfg["label_color"])
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=label_fontsize,
                          color=self._cfg["label_color"])
        ax.tick_params(axis="both", colors=self._cfg["tick_color"],
                       labelsize=9)

    def _get_panel(self, index: PanelIndex) -> Axes:
        """Retrieve an axes panel by index."""
        if isinstance(index, tuple):
            key = index[0] * 100 + index[1]  # row*100+col as flat key
        else:
            key = index
        if key not in self._panels:
            raise KeyError(
                f"Panel {index} not found. "
                "Did you call a layout method first?"
            )
        return self._panels[key]

    def _register_panel(self, key: int, ax: Axes) -> None:
        """Register an axes under a flat key."""
        self._panels[key] = ax

    def _draw_header(self, fig: Figure, title: str) -> None:
        """Draw the dashboard title header band at the top."""
        header_ax = fig.add_axes([0, 0.955, 1, 0.045])
        header_ax.set_facecolor(self._cfg["header_color"])
        header_ax.axis("off")
        header_ax.text(
            0.02, 0.5, title,
            transform=header_ax.transAxes,
            fontsize=16,
            fontweight="bold",
            color=self._cfg["title_color"],
            va="center",
        )

    # ----------------------------
    # LAYOUT — PRESETS
    # ----------------------------

    def preset_kpi_top(
        self,
        kpi_count:  int = 4,
        chart_rows: int = 1,
        chart_cols: int = 2,
        kpi_height_ratio: float = 0.28,
    ) -> "PowerCanvas":
        """
        Preset layout: a row of KPI cards across the top, charts below.

        This is the most common Power BI layout — summary numbers on top,
        detail charts underneath.

        Panel indexing:
        - KPI cards:   0, 1, 2, ... (kpi_count - 1)
        - Chart panels: kpi_count, kpi_count+1, ... left-to-right, top-to-bottom

        Args:
            kpi_count: Number of KPI cards in the top row. Default is 4.
            chart_rows: Number of chart rows below KPIs. Default is 1.
            chart_cols: Number of chart columns. Default is 2.
            kpi_height_ratio: Fraction of height given to KPI row. Default 0.28.

        Returns:
            self (for optional chaining)

        Example:
            >>> pc = PowerCanvas(title="Sales Overview", theme="light")
            >>> pc.preset_kpi_top(kpi_count=4, chart_rows=2, chart_cols=3)
            >>> pc.add_kpi_card(0, label="Revenue", value="$8.4M")
            >>> pc.add_bar(4, x=cats, y=vals, title="By Category")
        """
        cols = max(kpi_count, chart_cols)
        total_rows = 1 + chart_rows

        height_ratios = [kpi_height_ratio] + [
            (1 - kpi_height_ratio) / chart_rows
        ] * chart_rows

        self._fig = plt.figure(figsize=self.figsize)
        self._fig.patch.set_facecolor(self._cfg["fig_color"])

        gs = gridspec.GridSpec(
            total_rows, cols,
            figure=self._fig,
            height_ratios=height_ratios,
            hspace=0.45,
            wspace=0.35,
            top=0.94, bottom=0.06,
            left=0.04, right=0.97,
        )

        # KPI panels
        kpi_col_span = cols // kpi_count
        for i in range(kpi_count):
            col_start = i * kpi_col_span
            col_end = col_start + kpi_col_span
            ax = self._fig.add_subplot(gs[0, col_start:col_end])
            ax.axis("off")
            ax.set_facecolor(self._cfg["panel_color"])
            self._register_panel(i, ax)

        # Chart panels
        chart_idx = kpi_count
        for r in range(chart_rows):
            for c in range(chart_cols):
                if c < cols:
                    ax = self._fig.add_subplot(gs[r + 1, c])
                    self._style_panel(ax)
                    self._register_panel(chart_idx, ax)
                    chart_idx += 1

        self._panel_count = chart_idx
        self._layout_mode = "preset"
        self._draw_header(self._fig, self.title)
        return self

    def preset_split(
        self,
        kpi_count: int = 3,
        left_width_ratio: float = 0.65,
    ) -> "PowerCanvas":
        """
        Preset layout: wide chart/content on the left, KPI stack on the right.

        Panel indexing:
        - 0: Left wide panel (full height chart)
        - 1, 2, 3...: Right-side KPI/chart panels stacked vertically

        Args:
            kpi_count: Number of stacked panels on the right. Default is 3.
            left_width_ratio: Fraction of width for the left panel. Default 0.65.

        Returns:
            self

        Example:
            >>> pc = PowerCanvas(title="Overview", theme="dark")
            >>> pc.preset_split(kpi_count=3)
            >>> pc.add_bar(0, x=months, y=revenue, title="Revenue by Month")
            >>> pc.add_kpi_card(1, label="Total", value="$1.2M")
            >>> pc.add_kpi_card(2, label="Growth", value="+18%", delta_up=True)
            >>> pc.add_kpi_card(3, label="Customers", value="4,201")
        """
        right_ratio = 1 - left_width_ratio

        self._fig = plt.figure(figsize=self.figsize)
        self._fig.patch.set_facecolor(self._cfg["fig_color"])

        # Left panel
        left_ax = self._fig.add_axes(
            [0.03, 0.07, left_width_ratio - 0.06, 0.85]
        )
        self._style_panel(left_ax)
        self._register_panel(0, left_ax)

        # Right stacked panels
        panel_h = 0.85 / kpi_count
        for i in range(kpi_count):
            bottom = 0.07 + (kpi_count - 1 - i) * panel_h
            ax = self._fig.add_axes([
                left_width_ratio + 0.01,
                bottom + 0.01,
                right_ratio - 0.04,
                panel_h - 0.02,
            ])
            ax.axis("off")
            ax.set_facecolor(self._cfg["panel_color"])
            self._register_panel(i + 1, ax)

        self._panel_count = kpi_count + 1
        self._layout_mode = "preset"
        self._draw_header(self._fig, self.title)
        return self

    def preset_full_grid(
        self,
        rows: int = 2,
        cols: int = 3,
    ) -> "PowerCanvas":
        """
        Preset layout: even N×M grid of chart panels.

        Panels are indexed left-to-right, top-to-bottom starting at 0.

        Args:
            rows: Number of rows. Default is 2.
            cols: Number of columns. Default is 3.

        Returns:
            self

        Example:
            >>> pc = PowerCanvas(title="Full Grid Dashboard", theme="light")
            >>> pc.preset_full_grid(rows=2, cols=3)
            >>> pc.add_bar(0, x=cats, y=vals, title="Sales")
            >>> pc.add_line(1, x=months, y=revenue, title="Trend")
            >>> pc.add_pie(2, values=[40,30,20,10], title="Share")
        """
        self._fig = plt.figure(figsize=self.figsize)
        self._fig.patch.set_facecolor(self._cfg["fig_color"])

        gs = gridspec.GridSpec(
            rows, cols,
            figure=self._fig,
            hspace=0.45,
            wspace=0.35,
            top=0.93, bottom=0.07,
            left=0.05, right=0.97,
        )

        idx = 0
        for r in range(rows):
            for c in range(cols):
                ax = self._fig.add_subplot(gs[r, c])
                self._style_panel(ax)
                self._register_panel(idx, ax)
                idx += 1

        self._panel_count = idx
        self._layout_mode = "preset"
        self._draw_header(self._fig, self.title)
        return self

    # ----------------------------
    # LAYOUT — FLEXIBLE
    # ----------------------------

    def create_canvas(
        self,
        rows: int = 3,
        cols: int = 3,
        height_ratios: Optional[List[float]] = None,
        width_ratios:  Optional[List[float]] = None,
        hspace: float = 0.45,
        wspace: float = 0.35,
    ) -> "PowerCanvas":
        """
        Flexible layout: define your own grid and place panels manually.

        Use fig.add_subplot(gs[row, col]) style via add_* methods with
        (row, col) tuple indices. Supports col_span and row_span in panel methods.

        Args:
            rows: Number of rows in the grid.
            cols: Number of columns in the grid.
            height_ratios: Relative row heights. None for equal.
            width_ratios: Relative column widths. None for equal.
            hspace: Vertical spacing between panels. Default 0.45.
            wspace: Horizontal spacing between panels. Default 0.35.

        Returns:
            self

        Example:
            >>> pc = PowerCanvas(title="Custom Layout", theme="dark")
            >>> pc.create_canvas(rows=3, cols=4,
            ...                  height_ratios=[0.2, 0.4, 0.4])
            >>> pc.add_kpi_card((0,0), label="Revenue", value="$8.4M")
            >>> pc.add_bar((1,0), x=cats, y=vals, title="Sales", col_span=2)
        """
        self._fig = plt.figure(figsize=self.figsize)
        self._fig.patch.set_facecolor(self._cfg["fig_color"])

        self._gs = gridspec.GridSpec(
            rows, cols,
            figure=self._fig,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            hspace=hspace,
            wspace=wspace,
            top=0.93, bottom=0.07,
            left=0.04, right=0.97,
        )
        self._flex_rows = rows
        self._flex_cols = cols
        self._layout_mode = "flexible"
        self._draw_header(self._fig, self.title)
        return self

    def _get_or_create_flex_panel(
        self,
        index: PanelIndex,
        col_span: int = 1,
        row_span: int = 1,
        kpi: bool = False,
    ) -> Axes:
        """Get or create a panel in flexible mode."""
        if isinstance(index, tuple):
            row, col = index
            key = row * 100 + col
        else:
            raise ValueError(
                "Flexible layout requires (row, col) tuple as panel index."
            )

        if key not in self._panels:
            r_slice = slice(row, row + row_span)
            c_slice = slice(col, col + col_span)
            ax = self._fig.add_subplot(self._gs[r_slice, c_slice])
            if kpi:
                ax.axis("off")
                ax.set_facecolor(self._cfg["panel_color"])
            else:
                self._style_panel(ax)
            self._register_panel(key, ax)

        return self._panels[key]

    # ----------------------------
    # PANEL — KPI CARD
    # ----------------------------

    def add_kpi_card(
        self,
        index: PanelIndex,
        *,
        label: str,
        value: str,
        delta: Optional[str] = None,
        delta_up: Optional[bool] = None,
        sparkline_data: Optional[Iterable] = None,
        icon: Optional[str] = None,
        col_span: int = 1,
        row_span: int = 1,
    ) -> "PowerCanvas":
        """
        Add a KPI card to a panel.

        KPI cards display a large metric value with an optional label,
        delta indicator (↑ / ↓), and mini sparkline trend chart.

        Args:
            index: Panel index. Int for preset layouts, (row, col) for flexible.
            label: Metric name displayed above the value (e.g. 'Total Revenue').
            value: The main metric value as a string (e.g. '$8.42M', '24,842').
            delta: Optional change indicator string (e.g. '+12%', '-3%').
            delta_up: True if delta is positive (green), False if negative (red).
                      Required when delta is provided.
            sparkline_data: Optional iterable of numbers for a mini trend line.
            icon: Optional single emoji/character shown next to the label.
            col_span: Column span for flexible layouts. Default is 1.
            row_span: Row span for flexible layouts. Default is 1.

        Returns:
            self

        Example:
            >>> pc.add_kpi_card(
            ...     0,
            ...     label="Total Revenue",
            ...     value="$8.42M",
            ...     delta="+12%",
            ...     delta_up=True,
            ...     sparkline_data=[52, 58, 54, 61, 67, 72, 69, 75]
            ... )
        """
        if self._layout_mode == "flexible":
            ax = self._get_or_create_flex_panel(
                index, col_span=col_span, row_span=row_span, kpi=True
            )
        else:
            ax = self._get_panel(index)

        ax.axis("off")
        ax.set_facecolor(self._cfg["panel_color"])

        # Draw card background with rounded look
        bg = FancyBboxPatch(
            (0.03, 0.05), 0.94, 0.90,
            boxstyle="round,pad=0.02",
            facecolor=self._cfg["panel_color"],
            edgecolor=self._cfg["panel_edge"],
            linewidth=1.5,
            transform=ax.transAxes,
            zorder=0,
        )
        ax.add_patch(bg)

        # Accent bar on left edge
        accent_bar = FancyBboxPatch(
            (0.03, 0.05), 0.025, 0.90,
            boxstyle="round,pad=0.01",
            facecolor=self._cfg["accent"],
            edgecolor="none",
            transform=ax.transAxes,
            zorder=1,
        )
        ax.add_patch(accent_bar)

        # Layout: with sparkline → left text, right mini chart
        has_spark = sparkline_data is not None

        # Label
        label_text = f"{icon}  {label}" if icon else label
        ax.text(
            0.13, 0.80, label_text,
            transform=ax.transAxes,
            fontsize=10,
            color=self._cfg["kpi_label_color"],
            fontweight="normal",
            va="top",
        )

        # Main value
        ax.text(
            0.13, 0.52, value,
            transform=ax.transAxes,
            fontsize=22,
            color=self._cfg["kpi_value_color"],
            fontweight="bold",
            va="center",
        )

        # Delta
        if delta is not None:
            arrow = "▲" if delta_up else "▼"
            d_color = (
                self._cfg["delta_up_color"]
                if delta_up
                else self._cfg["delta_down_color"]
            )
            ax.text(
                0.13, 0.22,
                f"{arrow} {delta}",
                transform=ax.transAxes,
                fontsize=10,
                color=d_color,
                fontweight="bold",
                va="bottom",
            )

        # Sparkline (inset axes on the right side of the card)
        if has_spark:
            spark_ax = ax.inset_axes([0.62, 0.15, 0.34, 0.60])
            data = list(sparkline_data)
            spark_ax.plot(
                data,
                color=self._cfg["accent"],
                linewidth=1.8,
                solid_capstyle="round",
            )
            spark_ax.fill_between(
                range(len(data)), data,
                alpha=0.15,
                color=self._cfg["accent"],
            )
            spark_ax.axis("off")
            spark_ax.set_facecolor("none")

        return self

    # ----------------------------
    # PANEL — CHARTS
    # ----------------------------

    def add_bar(
        self,
        index: PanelIndex,
        x: XData,
        y: YData,
        *,
        title:        Optional[str] = None,
        xlabel:       Optional[str] = None,
        ylabel:       Optional[str] = None,
        color:        Optional[str] = None,
        value_labels: bool = False,
        value_format: str = "{:.0f}",
        rotation:     int = 45,
        ref_y:        Optional[float] = None,
        ref_label:    Optional[str] = None,
        ylim=None,
        col_span: int = 1,
        row_span: int = 1,
    ) -> "PowerCanvas":
        """
        Add a bar chart to a panel.

        Args:
            index: Panel index. Int for preset, (row, col) for flexible.
            x: Category labels.
            y: Bar height values.
            title: Panel title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            color: Bar color. Defaults to theme accent.
            value_labels: Show values on top of bars. Default False.
            value_format: Format string for value labels. Default '{:.0f}'.
            rotation: X tick label rotation. Default 45.
            ref_y: Optional horizontal reference line.
            ref_label: Label for reference line.
            ylim: Y-axis limits. Tuple, 'zero', or None.
            col_span: Column span for flexible layouts.
            row_span: Row span for flexible layouts.

        Returns:
            self

        Example:
            >>> pc.add_bar(
            ...     4,
            ...     x=["North", "South", "East", "West"],
            ...     y=[134000, 98000, 112000, 145000],
            ...     title="Sales by Region",
            ...     value_labels=True,
            ...     value_format="${:,.0f}",
            ...     ylim=(0, 170000),
            ... )
        """
        clr = color or self._cfg["accent"]

        if self._layout_mode == "flexible":
            ax = self._get_or_create_flex_panel(
                index, col_span=col_span, row_span=row_span
            )
        else:
            ax = self._get_panel(index)

        self._engine.draw(ax, x=x, y=y, plot_type="bar",
                          color=clr, edgecolor=self._cfg["panel_color"])

        if value_labels:
            self._engine.add_value_labels_on_bars(
                ax,
                format_string=value_format,
                color=self._cfg["label_color"],
                fontsize=9,
            )

        if ref_y is not None:
            self._engine.add_reference_line(
                ax, y=ref_y, label=ref_label,
                color=self._cfg["grid_color"], linewidth=1.5
            )

        ax.tick_params(axis="x", rotation=rotation)
        if ylim is not None:
            if ylim == "zero":
                ax.set_ylim(bottom=0)
            else:
                ax.set_ylim(ylim)

        self._style_chart_ax(ax, title=title, xlabel=xlabel, ylabel=ylabel)
        return self

    def add_line(
        self,
        index: PanelIndex,
        x: XData,
        y: YData,
        *,
        title:      Optional[str] = None,
        xlabel:     Optional[str] = None,
        ylabel:     Optional[str] = None,
        color:      Optional[str] = None,
        linewidth:  float = 2.0,
        marker:     Optional[str] = None,
        markersize: int = 5,
        fill:       bool = True,
        ref_y:      Optional[float] = None,
        ref_label:  Optional[str] = None,
        ylim=None,
        col_span: int = 1,
        row_span: int = 1,
    ) -> "PowerCanvas":
        """
        Add a line chart to a panel.

        Args:
            index: Panel index.
            x: X-axis data.
            y: Y-axis numeric data.
            title: Panel title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            color: Line color. Defaults to theme accent.
            linewidth: Line width. Default 2.0.
            marker: Marker style. None for no markers.
            markersize: Marker size. Default 5.
            fill: Fill area under the line. Default True.
            ref_y: Optional horizontal reference line.
            ref_label: Label for reference line.
            ylim: Y-axis limits.
            col_span: Column span for flexible layouts.
            row_span: Row span for flexible layouts.

        Returns:
            self

        Example:
            >>> pc.add_line(
            ...     5,
            ...     x=months,
            ...     y=revenue,
            ...     title="Revenue Over Time",
            ...     fill=True,
            ...     ref_y=70000,
            ...     ref_label="Target",
            ... )
        """
        clr = color or self._cfg["accent"]

        if self._layout_mode == "flexible":
            ax = self._get_or_create_flex_panel(
                index, col_span=col_span, row_span=row_span
            )
        else:
            ax = self._get_panel(index)

        self._engine.draw(
            ax, x=x, y=y, plot_type="line",
            color=clr,
            linewidth=linewidth,
            **({"marker": marker, "markersize": markersize}
               if marker else {}),
        )

        if fill:
            ax.fill_between(
                range(len(list(y))) if not hasattr(
                    x, '__len__') else range(len(x)),
                list(y),
                alpha=0.12,
                color=clr,
            )

        if ref_y is not None:
            self._engine.add_reference_line(
                ax, y=ref_y, label=ref_label,
                color="gray", linewidth=1.5
            )
            if ref_label:
                self._engine.set_legend(ax, fontsize=9)

        if ylim is not None:
            if ylim == "zero":
                ax.set_ylim(bottom=0)
            else:
                ax.set_ylim(ylim)

        self._style_chart_ax(ax, title=title, xlabel=xlabel, ylabel=ylabel)
        return self

    def add_scatter(
        self,
        index: PanelIndex,
        x: XData,
        y: YData,
        *,
        title:   Optional[str] = None,
        xlabel:  Optional[str] = None,
        ylabel:  Optional[str] = None,
        color:   Optional[str] = None,
        size:    int = 60,
        alpha:   float = 0.7,
        ref_y:   Optional[float] = None,
        col_span: int = 1,
        row_span: int = 1,
    ) -> "PowerCanvas":
        """
        Add a scatter plot to a panel.

        Args:
            index: Panel index.
            x: X-axis data.
            y: Y-axis numeric data.
            title: Panel title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
            color: Marker color. Defaults to theme accent.
            size: Marker size. Default 60.
            alpha: Marker transparency. Default 0.7.
            ref_y: Optional horizontal reference line.
            col_span: Column span for flexible layouts.
            row_span: Row span for flexible layouts.

        Returns:
            self
        """
        clr = color or self._cfg["accent"]

        if self._layout_mode == "flexible":
            ax = self._get_or_create_flex_panel(
                index, col_span=col_span, row_span=row_span
            )
        else:
            ax = self._get_panel(index)

        self._engine.draw(ax, x=x, y=y, plot_type="scatter",
                          color=clr, s=size, alpha=alpha)

        if ref_y is not None:
            self._engine.add_reference_line(ax, y=ref_y, color="gray",
                                            linewidth=1.5)

        self._style_chart_ax(ax, title=title, xlabel=xlabel, ylabel=ylabel)
        return self

    def add_pie(
        self,
        index: PanelIndex,
        values: YData,
        *,
        labels:     Optional[Iterable[str]] = None,
        title:      Optional[str] = None,
        colors:     Optional[Iterable[str]] = None,
        autopct:    str = "%1.1f%%",
        donut:      bool = False,
        explode:    Optional[Iterable[float]] = None,
        col_span: int = 1,
        row_span: int = 1,
    ) -> "PowerCanvas":
        """
        Add a pie or donut chart to a panel.

        Args:
            index: Panel index.
            values: Numeric values for each slice.
            labels: Slice labels.
            title: Panel title.
            colors: Slice colors. Defaults to theme palette.
            autopct: Percentage format string. Default '%1.1f%%'.
            donut: If True, renders as a donut chart. Default False.
            explode: Slice offsets for emphasis.
            col_span: Column span for flexible layouts.
            row_span: Row span for flexible layouts.

        Returns:
            self

        Example:
            >>> pc.add_pie(
            ...     2,
            ...     values=[38, 24, 18, 13, 7],
            ...     labels=["Electronics","Clothing","Food","Home","Other"],
            ...     title="Revenue by Category",
            ...     donut=True,
            ... )
        """
        clrs = colors or self._cfg["default_colors"]

        if self._layout_mode == "flexible":
            ax = self._get_or_create_flex_panel(
                index, col_span=col_span, row_span=row_span
            )
        else:
            ax = self._get_panel(index)

        wedge_props = {"width": 0.5} if donut else {}

        self._engine.draw(
            ax, x=None, y=values, plot_type="pie",
            labels=labels,
            colors=clrs,
            autopct=autopct,
            startangle=140,
            **({"explode": explode} if explode is not None else {}),
            wedgeprops=wedge_props if donut else {},
        )

        # Style text to match theme
        for text in ax.texts:
            text.set_color(self._cfg["label_color"])
            text.set_fontsize(8)

        ax.set_facecolor(self._cfg["panel_color"])

        if title:
            ax.set_title(
                title,
                fontsize=13,
                color=self._cfg["label_color"],
                fontweight="bold",
                pad=8,
            )

        return self

    def add_stats_panel(
        self,
        index: PanelIndex,
        stats: Dict[str, Union[str, int, float]],
        *,
        title:    Optional[str] = None,
        col_span: int = 1,
        row_span: int = 1,
    ) -> "PowerCanvas":
        """
        Add a statistics text panel.

        Displays a dictionary of key-value metrics in a styled text box.
        Useful for summary panels alongside charts.

        Args:
            index: Panel index.
            stats: Dict of metric_name: value pairs.
            title: Optional panel title.
            col_span: Column span for flexible layouts.
            row_span: Row span for flexible layouts.

        Returns:
            self

        Example:
            >>> pc.add_stats_panel(
            ...     3,
            ...     stats={"Total Sales": 145000, "Avg Order": 58.3, "Returns": "2.1%"},
            ...     title="Summary"
            ... )
        """
        if self._layout_mode == "flexible":
            ax = self._get_or_create_flex_panel(
                index, col_span=col_span, row_span=row_span, kpi=True
            )
        else:
            ax = self._get_panel(index)

        self._engine.create_stats_text_box(
            ax, stats,
            title=title,
            box_color=self._cfg["panel_color"],
            text_color=self._cfg["kpi_value_color"],
            border_color=self._cfg["accent"],
            border_width=2,
            fontsize=11,
            title_fontsize=13,
        )
        return self

    # ----------------------------
    # FINALIZE
    # ----------------------------

    def render(
        self,
        save_path: Optional[str] = None,
        dpi:       int = 150,
        show:      bool = True,
    ) -> None:
        """
        Finalize and render the dashboard.

        Call this last, after all panels have been populated.
        Handles tight layout, saving, displaying, and cleanup.

        Args:
            save_path: File path to save (e.g. 'dashboard.png'). None to skip.
            dpi: Image resolution. Default 150.
            show: Whether to display the figure. Default True.

        Example:
            >>> pc.render(save_path="dashboard.png", dpi=200, show=False)
        """
        if self._fig is None:
            raise RuntimeError(
                "No canvas created. Call a layout method first "
                "(e.g. preset_kpi_top(), create_canvas())."
            )

        if save_path is not None:
            safe_path = self._engine.handler.ensure_writable_path(save_path)
            self._fig.savefig(
                safe_path, dpi=dpi,
                facecolor=self._cfg["fig_color"],
                bbox_inches="tight",
            )

        if show:
            plt.show()

        plt.close(self._fig)
