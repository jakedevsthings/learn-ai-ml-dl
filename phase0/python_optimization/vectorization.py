# /learn-ai-ml-dl/phase0/python_optimization/vectorization.py

"""
Demonstration of vectorized operations vs. loop-based implementations.
This module shows performance comparisons for common operations.
"""

import time
import numpy as np

def time_function(func, *args, **kwargs):
    """
    Measure the execution time of a function.
    
    Parameters:
    -----------
    func : function
        The function to time
    *args, **kwargs
        Arguments to pass to the function
        
    Returns:
    --------
    result
        The result of the function call
    execution_time : float
        The time taken to execute the function in seconds
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return result, end_time - start_time

def vector_sum_loops(vectors):
    """
    Sum a list of vectors using loops.
    
    Parameters:
    -----------
    vectors : list of lists
        List of vectors to sum
        
    Returns:
    --------
    result : list
        Sum of all vectors
    """
    if not vectors:
        return []
    
    n_dims = len(vectors[0])
    result = [0] * n_dims
    
    for vector in vectors:
        for i in range(n_dims):
            result[i] += vector[i]
    
    return result

def vector_sum_numpy(vectors):
    """
    Sum a list of vectors using NumPy.
    
    Parameters:
    -----------
    vectors : list of lists or numpy array
        List of vectors to sum
        
    Returns:
    --------
    result : numpy array
        Sum of all vectors
    """
    return np.sum(np.array(vectors), axis=0)