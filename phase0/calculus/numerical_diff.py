# /learn-ai-ml-dl/phase0/calculus/numerical_diff.py

"""
Numerical differentiation tools for machine learning applications.
"""

def numerical_derivative(f, x, h=1e-5):
    """
    Compute the numerical derivative of function f at point x.
    
    Parameters:
    -----------
    f : function
        Function to differentiate
    x : float
        Point at which to evaluate the derivative
    h : float, optional
        Step size for numerical approximation
        
    Returns:
    --------
    float
        Approximation of f'(x)
    """
    return (f(x + h) - f(x)) / h

def central_difference(f, x, h=1e-5):
    """
    Compute the numerical derivative using the central difference method.
    This is generally more accurate than the standard numerical derivative.
    
    Parameters:
    -----------
    f : function
        Function to differentiate
    x : float
        Point at which to evaluate the derivative
    h : float, optional
        Step size for numerical approximation
        
    Returns:
    --------
    float
        Approximation of f'(x)
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def partial_derivative(f, x, i, h=1e-5):
    """
    Compute the partial derivative of a multivariate function f with respect to 
    the i-th variable at point x.
    
    Parameters:
    -----------
    f : function
        Multivariate function to differentiate
    x : list or array
        Point at which to evaluate the derivative
    i : int
        Index of the variable to differentiate with respect to
    h : float, optional
        Step size for numerical approximation
        
    Returns:
    --------
    float
        Approximation of the partial derivative
    """
    x_plus_h = x.copy()
    x_plus_h[i] += h
    
    return (f(x_plus_h) - f(x)) / h