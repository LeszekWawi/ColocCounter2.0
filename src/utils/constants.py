"""
Constants and configuration values
"""

# Professional color palettes
COLORS = {
    'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
    'categorical': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7'],
    'sequential': ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#084594'],
    'diverging': ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6']
}

# Matplotlib configuration
MATPLOTLIB_CONFIG = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 100,
    'savefig.dpi': 200,
    'figure.max_open_warning': 50,
    'axes.titlepad': 15,
    'axes.labelpad': 8,
    'xtick.major.pad': 5,
    'ytick.major.pad': 5,
    'figure.constrained_layout.use': True,
    'figure.constrained_layout.h_pad': 0.1,
    'figure.constrained_layout.w_pad': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.5,
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8
}

# Application metadata
APP_NAME = "Granular Co-localization Analysis Pipeline"
APP_VERSION = "2.0.0 - Refactored"
APP_AUTHOR = "Advanced Microscopy Analysis Framework"
