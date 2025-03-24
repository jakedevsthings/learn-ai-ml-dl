# src/optimization.py
import numpy as np
from .differentiation import gradient

def gradient_descent(f, df, x0, learning_rate=0.1, tol=1e-6, max_iter=1000, callback=None):
    """
    Gradient descent optimization algorithm.
    
    Args:
        f: Function to minimize
        df: Gradient of f
        x0: Initial point
        learning_rate: Step size
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        callback: Function to call at each iteration
        
    Returns:
        Optimized point and history of points
    """
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(max_iter):
        grad = df(x)
        
        if np.linalg.norm(grad) < tol:
            break
            
        x = x - learning_rate * grad
        history.append(x.copy())
        
        if callback:
            callback(i, x, f(x), grad)
    
    return x, np.array(history)

def gradient_descent_with_momentum(f, df, x0, learning_rate=0.1, momentum=0.9, 
                                  tol=1e-6, max_iter=1000, callback=None):
    """
    Gradient descent with momentum optimization algorithm.
    
    Args:
        f: Function to minimize
        df: Gradient of f
        x0: Initial point
        learning_rate: Step size
        momentum: Momentum coefficient
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        callback: Function to call at each iteration
        
    Returns:
        Optimized point and history of points
    """
    x = x0.copy()
    history = [x.copy()]
    velocity = np.zeros_like(x)
    
    for i in range(max_iter):
        grad = df(x)
        
        if np.linalg.norm(grad) < tol:
            break
        
        velocity = momentum * velocity - learning_rate * grad
        x = x + velocity
        history.append(x.copy())
        
        if callback:
            callback(i, x, f(x), grad)
    
    return x, np.array(history)

def adagrad(f, df, x0, learning_rate=0.01, epsilon=1e-8, 
           tol=1e-6, max_iter=1000, callback=None):
    """
    Adagrad optimization algorithm.
    
    Args:
        f: Function to minimize
        df: Gradient of f
        x0: Initial point
        learning_rate: Step size
        epsilon: Small constant for numerical stability
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        callback: Function to call at each iteration
        
    Returns:
        Optimized point and history of points
    """
    x = x0.copy()
    history = [x.copy()]
    cache = np.zeros_like(x)
    
    for i in range(max_iter):
        grad = df(x)
        
        if np.linalg.norm(grad) < tol:
            break
        
        cache += np.square(grad)
        x = x - learning_rate * grad / (np.sqrt(cache) + epsilon)
        history.append(x.copy())
        
        if callback:
            callback(i, x, f(x), grad)
    
    return x, np.array(history)

def rmsprop(f, df, x0, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8,
           tol=1e-6, max_iter=1000, callback=None):
    """
    RMSprop optimization algorithm.
    
    Args:
        f: Function to minimize
        df: Gradient of f
        x0: Initial point
        learning_rate: Step size
        decay_rate: Decay rate for moving average
        epsilon: Small constant for numerical stability
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        callback: Function to call at each iteration
        
    Returns:
        Optimized point and history of points
    """
    x = x0.copy()
    history = [x.copy()]
    cache = np.zeros_like(x)
    
    for i in range(max_iter):
        grad = df(x)
        
        if np.linalg.norm(grad) < tol:
            break
        
        cache = decay_rate * cache + (1 - decay_rate) * np.square(grad)
        x = x - learning_rate * grad / (np.sqrt(cache) + epsilon)
        history.append(x.copy())
        
        if callback:
            callback(i, x, f(x), grad)
    
    return x, np.array(history)

def adam(f, df, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
        tol=1e-6, max_iter=1000, callback=None):
    """
    Adam optimization algorithm.
    
    Args:
        f: Function to minimize
        df: Gradient of f
        x0: Initial point
        learning_rate: Step size
        beta1: Decay rate for moment estimates
        beta2: Decay rate for squared gradients
        epsilon: Small constant for numerical stability
        tol: Tolerance for convergence
        max_iter: Maximum number of iterations
        callback: Function to call at each iteration
        
    Returns:
        Optimized point and history of points
    """
    x = x0.copy()
    history = [x.copy()]
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    
    for i in range(max_iter):
        grad = df(x)
        
        if np.linalg.norm(grad) < tol:
            break
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * np.square(grad)
        
        m_hat = m / (1 - beta1 ** (i + 1))
        v_hat = v / (1 - beta2 ** (i + 1))
        
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        history.append(x.copy())
        
        if callback:
            callback(i, x, f(x), grad)
    
    return x, np.array(history)