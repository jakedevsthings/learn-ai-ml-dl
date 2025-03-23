# /learn-ai-ml-dl/phase0/linear_algebra/matrix_viz.py

"""
Matrix visualization utilities.
This module provides functions to visualize matrices and linear transformations.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

def plot_matrix_as_grid(matrix, ax=None, cmap='viridis', show_values=True):
    """
    Visualize a matrix as a colored grid.
    
    Parameters:
    -----------
    matrix : list of lists or numpy array
        The matrix to visualize
    ax : matplotlib axis, optional
        The axis to plot on. If not provided, a new one is created.
    cmap : str, optional
        The colormap to use
    show_values : bool, optional
        Whether to display the matrix values on the grid
    
    Returns:
    --------
    fig : matplotlib figure
        The figure containing the plot
    ax : matplotlib axis
        The axis containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Convert to numpy array if not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    rows, cols = matrix.shape
    
    # Create heatmap
    im = ax.imshow(matrix, cmap=cmap)
    
    # Show values
    if show_values:
        for i in range(rows):
            for j in range(cols):
                text_color = 'white' if abs(matrix[i, j]) > 0.5 * np.max(np.abs(matrix)) else 'black'
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=text_color)
    
    # Set ticks
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    
    # Set tick labels
    ax.set_xticklabels([f"j={j}" for j in range(cols)])
    ax.set_yticklabels([f"i={i}" for i in range(rows)])
    
    plt.colorbar(im, ax=ax)
    ax.set_title("Matrix Visualization")
    
    return fig, ax

def plot_linear_transformation(ax, transformation_matrix, grid_lines=True):
    """
    Visualize the effect of a linear transformation on the unit square.
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to plot on
    transformation_matrix : 2x2 matrix (list of lists or numpy array)
        The transformation matrix to visualize
    grid_lines : bool, optional
        Whether to draw grid lines
    
    Returns:
    --------
    ax : matplotlib axis
        The axis with the plot
    """
    # Convert to numpy array if not already
    if not isinstance(transformation_matrix, np.ndarray):
        transformation_matrix = np.array(transformation_matrix)
    
    # Set up the axis
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.grid(grid_lines)
    
    # Draw the original x and y axes
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Unit vectors
    i_hat = np.array([1, 0])
    j_hat = np.array([0, 1])
    
    # Transformed unit vectors
    i_hat_transformed = transformation_matrix @ i_hat
    j_hat_transformed = transformation_matrix @ j_hat
    
    # Plot original unit vectors
    ax.arrow(0, 0, i_hat[0], i_hat[1], head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5)
    ax.arrow(0, 0, j_hat[0], j_hat[1], head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5)
    
    # Plot transformed unit vectors
    ax.arrow(0, 0, i_hat_transformed[0], i_hat_transformed[1], head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    ax.arrow(0, 0, j_hat_transformed[0], j_hat_transformed[1], head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    # Draw the unit square and its transformation
    unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    
    # Transform each point of the unit square
    transformed_square = np.array([transformation_matrix @ point for point in unit_square])
    
    # Plot the unit square
    ax.add_patch(Polygon(unit_square, closed=True, alpha=0.2, fill=True, color='green'))
    
    # Plot the transformed square
    ax.add_patch(Polygon(transformed_square, closed=True, alpha=0.2, fill=True, color='purple'))
    
    ax.set_title("Linear Transformation")
    ax.set_aspect('equal')
    
    return ax