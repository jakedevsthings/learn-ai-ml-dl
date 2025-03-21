# /learn-ai-ml-dl/phase0/linear_algebra/src/vector_viz.py

"""
Vector visualization utilities.
This module provides functions to visualize vectors and transformations.
"""

import matplotlib.pyplot as plt
import numpy as np

def setup_2d_plot(x_range=(-10, 10), y_range=(-10, 10), figsize=(10, 10), grid=True):
    """Set up a 2D plot with specified ranges and options."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set axis ranges
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    
    # Add grid
    if grid:
        ax.grid(True)
    
    # Add x and y axes
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Set equal aspect so circles look like circles
    ax.set_aspect('equal')
    
    return fig, ax

def plot_vector(ax, vector, origin=[0, 0], color='r', label=None, width=0.005):
    """Plot a 2D vector as an arrow."""
    return ax.arrow(origin[0], origin[1], vector[0], vector[1], 
                   head_width=0.2, head_length=0.3, fc=color, ec=color,
                   length_includes_head=True, label=label)

def plot_vector_addition(ax, v1, v2, origin=[0, 0], colors=['r', 'g', 'b']):
    """Visualize vector addition v1 + v2."""
    # Plot first vector
    plot_vector(ax, v1, origin=origin, color=colors[0], label='v1')
    
    # Plot second vector, starting from the end of the first
    plot_vector(ax, v2, origin=[origin[0] + v1[0], origin[1] + v1[1]], 
               color=colors[1], label='v2')
    
    # Plot the resultant vector
    result = [v1[0] + v2[0], v1[1] + v2[1]]
    plot_vector(ax, result, origin=origin, color=colors[2], label='v1 + v2')
    
    return result