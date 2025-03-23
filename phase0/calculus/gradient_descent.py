# /learn-ai-ml-dl/phase0/calculus/gradient_descent.py

"""
Gradient descent implementations for optimization problems.
"""

import numpy as np

def gradient_descent(gradient_func, init_params, learning_rate=0.01, n_iterations=100, tolerance=1e-6):
    """
    Perform gradient descent optimization.
    
    Parameters:
    -----------
    gradient_func : function
        Function that computes the gradient at a point
    init_params : array-like
        Initial parameter values
    learning_rate : float
        Step size for parameter updates
    n_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence threshold for the gradient norm
        
    Returns:
    --------
    params : array-like
        Optimized parameter values
    trajectory : list
        List of parameter values throughout optimization
    """
    params = np.array(init_params, dtype=float)
    trajectory = [params.copy()]
    
    for i in range(n_iterations):
        # Compute gradient
        grad = np.array(gradient_func(params))
        
        # Check for convergence
        if np.linalg.norm(grad) < tolerance:
            break
        
        # Update parameters
        params = params - learning_rate * grad
        trajectory.append(params.copy())
    
    return params, trajectory

def stochastic_gradient_descent(gradient_func, init_params, data_points, batch_size=1, 
                               learning_rate=0.01, n_iterations=100, tolerance=1e-6):
    """
    Perform stochastic gradient descent optimization.
    
    Parameters:
    -----------
    gradient_func : function
        Function that computes the gradient for a batch of data
    init_params : array-like
        Initial parameter values
    data_points : array-like
        Training data points
    batch_size : int
        Number of data points to use for each update
    learning_rate : float
        Step size for parameter updates
    n_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence threshold for the gradient norm
        
    Returns:
    --------
    params : array-like
        Optimized parameter values
    trajectory : list
        List of parameter values throughout optimization
    """
    params = np.array(init_params, dtype=float)
    trajectory = [params.copy()]
    n_data = len(data_points)
    
    for i in range(n_iterations):
        # Shuffle data
        indices = np.random.permutation(n_data)
        
        # Process mini-batches
        for start_idx in range(0, n_data, batch_size):
            batch_indices = indices[start_idx:min(start_idx + batch_size, n_data)]
            batch = [data_points[j] for j in batch_indices]
            
            # Compute gradient on batch
            grad = np.array(gradient_func(params, batch))
            
            # Check for convergence
            if np.linalg.norm(grad) < tolerance:
                break
            
            # Update parameters
            params = params - learning_rate * grad
            trajectory.append(params.copy())
    
    return params, trajectory