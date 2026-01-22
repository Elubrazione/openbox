"""
OpenBox Advanced Visualization Module

This module provides visualization capabilities for compression results:
- Basic mode: Generate static HTML files (no server required)
- Advanced mode: Start a local HTTP server with full interactivity

Example:
    >>> from openbox.visualization.advance import visualize_compression
    >>> 
    >>> # Basic mode - static HTML
    >>> html_path = visualize_compression('./results', mode='basic')
    >>> 
    >>> # Advanced mode - local server
    >>> url = visualize_compression('./results', mode='advanced')
"""

from .server import VisualizationServer, start_visualization_server, visualize_compression
from .html_generator import generate_static_html

__all__ = [
    'VisualizationServer',
    'start_visualization_server',
    'visualize_compression',
    'generate_static_html',
]
