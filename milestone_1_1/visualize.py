# milestone_1_1/visualize.py

import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Visualize decision boundary of a trained model.
    
    Parameters:
        model: instance of your NeuralNetwork class with .forward(x) method
        X: input points (shape: [num_samples, 2])
        y: true binary labels (shape: [num_samples, 1])
    """
    # Convert inputs to numpy arrays if they aren't already
    X = np.array(X)
    y = np.array(y)
    
    h = 0.01  # Step size in mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Create mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]  # Shape: [num_points, 2]

    # Predict over grid
    Z = model.forward(grid)
    Z = Z.reshape(xx.shape)  # Reshape to match mesh

    # Plot contour map (decision surface)
    plt.contourf(xx, yy, Z, levels=100, cmap="RdBu", alpha=0.6)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)


    # Overlay true points
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="bwr", edgecolors='k')
    plt.title(title)
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.grid(True)
    plt.show()