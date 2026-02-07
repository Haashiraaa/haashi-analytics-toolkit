# __init__.py
"""
Plot Engine Module for haashi_pkg
==================================

A comprehensive plotting utility built on matplotlib and seaborn for creating
publication-quality visualizations with sensible defaults and extensive
customization options.

Main Class:
    PlotEngine: High-level interface for creating, customizing, and managing plots

Custom Exceptions:
    PlotEngineError: Base exception for all plot engine errors
    InvalidPlotTypeError: Invalid plot type specified
    InvalidDataError: Data validation failures
    ConfigurationError: Configuration/setup failures

Workflow:
    1. Setup: Create figures and configure layout
    2. Draw: Add data visualizations to axes
    3. Decorate: Add titles, labels, legends, styling
    4. Finalize: Save and/or display the figure

Features:
    - Multiple color palettes (professional, vibrant, soft, deep)
    - Flexible grid layouts with custom ratios
    - Built-in theming via seaborn
    - Automatic formatting (currency, dates, etc.)
    - Value labels on bar charts
    - Reference lines for targets/thresholds
    - Statistics text boxes
    - Background color customization

Supported Plot Types:
    - Line plots
    - Bar charts
    - Scatter plots
    - Pie charts

Usage Example:
    >>> from haashi_pkg.plot_engine import PlotEngine
    >>> 
    >>> pe = PlotEngine()
    >>> 
    >>> # Create figure
    >>> fig, ax = pe.create_figure(figsize=(12, 8))
    >>> 
    >>> # Draw data
    >>> pe.draw(ax, x=[1, 2, 3, 4], y=[10, 15, 13, 17], plot_type='line')
    >>> 
    >>> # Decorate
    >>> pe.decorate(
    ...     ax,
    ...     title='Sales Trend',
    ...     xlabel='Month',
    ...     ylabel='Revenue ($)'
    ... )
    >>> 
    >>> # Add reference line
    >>> pe.add_reference_line(ax, y=20, label='Target', color='red')
    >>> 
    >>> # Finalize
    >>> pe.save_or_show(fig, save_path='sales_trend.png')

Advanced Example (Custom Grid):
    >>> # Create custom layout: 1 wide plot on top, 3 plots on bottom
    >>> fig, gs = pe.create_custom_grid(
    ...     rows=2,
    ...     cols=3,
    ...     height_ratios=[2, 1],
    ...     figsize=(20, 12)
    ... )
    >>> 
    >>> # Top plot spanning all columns
    >>> ax_main = fig.add_subplot(gs[0, :])
    >>> pe.draw(ax_main, x=dates, y=revenue, plot_type='line')
    >>> 
    >>> # Bottom plots
    >>> ax_stats = fig.add_subplot(gs[1, 0])
    >>> pe.create_stats_text_box(ax_stats, stats={'Total': 125000, 'Avg': 15625})
    >>> 
    >>> ax_left = fig.add_subplot(gs[1, 1])
    >>> pe.draw(ax_left, x=categories, y=counts, plot_type='bar')
    >>> 
    >>> ax_right = fig.add_subplot(gs[1, 2])
    >>> pe.draw(ax_right, x=None, y=distribution, plot_type='pie')
    >>> 
    >>> pe.save_or_show(fig, save_path='dashboard.png')
"""

from haashi_pkg.plot_engine.plotengine import (
    # Main class
    PlotEngine,
    
    # Exceptions
    PlotEngineError,
    InvalidPlotTypeError,
    InvalidDataError,
    ConfigurationError,
)

__all__ = [
    'PlotEngine',
    'PlotEngineError',
    'InvalidPlotTypeError',
    'InvalidDataError',
    'ConfigurationError',
]

__version__ = '1.0.0'
