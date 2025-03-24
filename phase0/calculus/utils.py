# src/utils.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_function_1d(f, x_min, x_max, num_points=1000, title="Function Plot"):
    """
    Plot a 1D function.
    
    Args:
        f: Function to plot
        x_min: Minimum x value
        x_max: Maximum x value
        num_points: Number of points to plot
        title: Plot title
    """
    x = np.linspace(x_min, x_max, num_points)
    y = np.array([f(xi) for xi in x])
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()

def plot_function_2d(f, x_min, x_max, y_min, y_max, num_points=100, 
                     plot_type='surface', title="Function Plot"):
    """
    Plot a 2D function.
    
    Args:
        f: Function to plot
        x_min, x_max: Min/max x values
        y_min, y_max: Min/max y values
        num_points: Number of points in each dimension
        plot_type: 'surface' or 'contour'
        title: Plot title
    """
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(num_points):
        for j in range(num_points):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    
    fig = plt.figure(figsize=(12, 10))
    
    if plot_type == 'surface':
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
    elif plot_type == 'contour':
        plt.contourf(X, Y, Z, 50, cmap='viridis')
        plt.colorbar()
        plt.contour(X, Y, Z, 20, colors='black', linewidths=0.5, alpha=0.5)
        
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def plot_gradient_descent_1d(f, df, x0, learning_rate=0.1, max_iter=20):
    """
    Visualize gradient descent optimization on a 1D function.
    
    Args:
        f: Function to optimize
        df: Derivative of the function
        x0: Initial point
        learning_rate: Learning rate
        max_iter: Maximum number of iterations
    """
    # Find reasonable plot range
    x_min = min(x0 - 2, -2)
    x_max = max(x0 + 2, 2)
    
    # Generate points for function plot
    x = np.linspace(x_min, x_max, 1000)
    y = np.array([f(xi) for xi in x])
    
    # Run gradient descent
    x_history = [x0]
    for i in range(max_iter):
        grad = df(x_history[-1])
        x_new = x_history[-1] - learning_rate * grad
        x_history.append(x_new)
    
    # Convert to numpy array for easier indexing
    x_history = np.array(x_history)
    y_history = np.array([f(xi) for xi in x_history])
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'b-', label='f(x)')
    plt.plot(x_history, y_history, 'ro-', label='Gradient Descent Path')
    
    # Add arrows to show direction
    for i in range(len(x_history) - 1):
        plt.annotate("", xy=(x_history[i+1], y_history[i+1]), 
                    xytext=(x_history[i], y_history[i]),
                    arrowprops=dict(arrowstyle="->", color="red"))
    
    plt.scatter(x_history[-1], y_history[-1], color='g', s=100, 
               label=f'Final: x={x_history[-1]:.4f}, f(x)={y_history[-1]:.4f}')
    
    plt.title(f'Gradient Descent Optimization (Learning Rate: {learning_rate})')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_contour_with_path(f, history, x_min, x_max, y_min, y_max, num_points=100,
                          title="Optimization Path", show_points=True):
    """
    Plot contour of a 2D function with optimization path.
    
    Args:
        f: Function to plot
        history: History of points from optimization
        x_min, x_max, y_min, y_max: Plot range
        num_points: Number of points for contour plot
        title: Plot title
        show_points: Whether to show points along the path
    """
    # Create meshgrid for contour plot
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(num_points):
        for j in range(num_points):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, 50, cmap='viridis', alpha=0.8)
    plt.colorbar(label='f(x, y)')
    plt.contour(X, Y, Z, 20, colors='black', linewidths=0.5, alpha=0.3)
    
    # Plot optimization path
    plt.plot(history[:, 0], history[:, 1], 'r-', linewidth=2, label='Optimization Path')
    
    if show_points:
        plt.scatter(history[:, 0], history[:, 1], c='red', s=30, alpha=0.5)
    
    plt.scatter(history[0, 0], history[0, 1], c='blue', s=100, 
               label=f'Start: ({history[0, 0]:.2f}, {history[0, 1]:.2f})')
    plt.scatter(history[-1, 0], history[-1, 1], c='green', s=100, 
               label=f'End: ({history[-1, 0]:.2f}, {history[-1, 1]:.2f})')
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_optimizer_comparison(f, df, x0, optimizers, names, num_iter=100):
    """
    Compare different optimization algorithms on the same function.
    
    Args:
        f: Function to optimize
        df: Gradient of the function
        x0: Initial point
        optimizers: List of optimizer functions
        names: List of optimizer names
        num_iter: Number of iterations
    """
    plt.figure(figsize=(12, 8))
    
    for i, (optimizer, name) in enumerate(zip(optimizers, names)):
        # Record function values during optimization
        f_values = []
        
        def callback(iteration, x, f_val, grad):
            f_values.append(f_val)
        
        # Run optimization
        result, history = optimizer(f, df, x0.copy(), max_iter=num_iter, callback=callback)
        
        # Plot convergence
        plt.plot(range(len(f_values)), f_values, label=name, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title('Optimizer Convergence Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.show()