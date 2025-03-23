# /learn-ai-ml-dl/phase0/calculus/function_approx.py

"""
Function approximation and curve fitting using gradient descent.
"""

import numpy as np
from .gradient_descent import gradient_descent

def linear_function(x, params):
    """Linear function: f(x) = m*x + b"""
    m, b = params
    return m * x + b

def quadratic_function(x, params):
    """Quadratic function: f(x) = a*x^2 + b*x + c"""
    a, b, c = params
    return a * x**2 + b * x + c

def polynomial_function(x, params):
    """General polynomial function: f(x) = sum(a_i * x^i)"""
    result = 0
    for i, a in enumerate(params):
        result += a * x**i
    return result

def mean_squared_error(y_true, y_pred):
    """Compute the mean squared error between true and predicted values."""
    return np.mean((np.array(y_true) - np.array(y_pred))**2)

def fit_linear_model(x_data, y_data, init_params=None, learning_rate=0.01, n_iterations=1000):
    """
    Fit a linear model to data using gradient descent.
    
    Parameters:
    -----------
    x_data : array-like
        Independent variable values
    y_data : array-like
        Dependent variable values
    init_params : array-like, optional
        Initial parameter values [m, b]. If None, random values are used.
    learning_rate : float
        Learning rate for gradient descent
    n_iterations : int
        Maximum number of iterations
        
    Returns:
    --------
    optimal_params : array-like
        Optimized parameter values [m, b]
    loss_history : list
        MSE values during optimization
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    if init_params is None:
        init_params = np.random.randn(2)
    
    # Define the loss function (MSE)
    def loss_function(params):
        y_pred = linear_function(x_data, params)
        return mean_squared_error(y_data, y_pred)
    
    # Define the gradient function
    def gradient_function(params):
        m, b = params
        n = len(x_data)
        y_pred = m * x_data + b
        error = y_pred - y_data
        
        # Gradient of MSE with respect to m and b
        grad_m = (2/n) * np.sum(error * x_data)
        grad_b = (2/n) * np.sum(error)
        
        return np.array([grad_m, grad_b])
    
    # Perform gradient descent
    optimal_params, param_history = gradient_descent(
        gradient_function, init_params, learning_rate, n_iterations)
    
    # Compute loss history
    loss_history = [loss_function(params) for params in param_history]
    
    return optimal_params, loss_history