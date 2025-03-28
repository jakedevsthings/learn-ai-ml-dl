# src/differentiation.py
import numpy as np

def finite_difference(f, x, h=1e-5):
    """
    Compute the derivative of function f at point x using finite difference method.
    
    Args:
        f: Function to differentiate
        x: Point at which to compute the derivative
        h: Step size
        
    Returns:
        Approximate derivative of f at x
    """
    return (f(x + h) - f(x)) / h

def central_difference(f, x, h=1e-5):
    """
    Compute the derivative of function f at point x using central difference method.
    
    Args:
        f: Function to differentiate
        x: Point at which to compute the derivative
        h: Step size
        
    Returns:
        Approximate derivative of f at x
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def second_derivative(f, x, h=1e-5):
    """
    Compute the second derivative of function f at point x.
    
    Args:
        f: Function to differentiate
        x: Point at which to compute the derivative
        h: Step size
        
    Returns:
        Approximate second derivative of f at x
    """
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)

def partial_derivative(f, x, i, h=1e-5):
    """
    Compute the partial derivative of function f with respect to variable i at point x.
    
    Args:
        f: Function to differentiate
        x: Point (vector) at which to compute the derivative
        i: Index of the variable to differentiate with respect to
        h: Step size
        
    Returns:
        Approximate partial derivative of f with respect to x_i at x
    """
    x_plus_h = x.copy()
    x_plus_h[i] += h
    
    return (f(x_plus_h) - f(x)) / h

def gradient(f, x, h=1e-5):
    """
    Compute the gradient of function f at point x.
    
    Args:
        f: Function to differentiate
        x: Point (vector) at which to compute the gradient
        h: Step size
        
    Returns:
        Gradient vector of f at x
    """
    n = len(x)
    grad = np.zeros_like(x)
    
    for i in range(n):
        grad[i] = partial_derivative(f, x, i, h)
        
    return grad

def jacobian(f, x, h=1e-5):
    """
    Compute the Jacobian matrix of vector-valued function f at point x.
    
    Args:
        f: Vector-valued function to differentiate
        x: Point (vector) at which to compute the Jacobian
        h: Step size
        
    Returns:
        Jacobian matrix of f at x
    """
    n = len(x)
    y = f(x)
    m = len(y)
    
    J = np.zeros((m, n))
    
    for i in range(n):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        y_plus_h = f(x_plus_h)
        
        J[:, i] = (y_plus_h - y) / h
        
    return J

def hessian(f, x, h=1e-5):
    """
    Compute the Hessian matrix of function f at point x.
    
    Args:
        f: Function to differentiate
        x: Point (vector) at which to compute the Hessian
        h: Step size
        
    Returns:
        Hessian matrix of f at x
    """
    n = len(x)
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                # Diagonal elements: use second derivative
                x_copy = x.copy()
                H[i, j] = second_derivative(lambda xi: f_at_point(f, x_copy, i, xi), x[i], h)
            else:
                # Off-diagonal elements: mixed partial derivatives
                H[i, j] = mixed_partial_derivative(f, x, i, j, h)
                
    return H

def f_at_point(f, x, i, xi):
    """Helper function to compute f with only one variable changing."""
    x[i] = xi
    return f(x)

def mixed_partial_derivative(f, x, i, j, h=1e-5):
    """
    Compute the mixed partial derivative of f with respect to x_i and x_j.
    """
    x_copy = x.copy()
    
    def df_dj(xi):
        x_copy[i] = xi
        return partial_derivative(f, x_copy, j, h)
    
    return partial_derivative(df_dj, x_copy, i, h)