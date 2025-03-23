# /learn-ai-ml-dl/phase0/python_optimization/profiling.py

"""
Profiling tools for identifying performance bottlenecks in Python code.
"""

import time
import cProfile
import pstats
from functools import wraps
import numpy as np

def timer_decorator(func):
    """
    Decorator to measure function execution time.
    
    Parameters:
    -----------
    func : function
        The function to measure
        
    Returns:
    --------
    wrapper : function
        Decorated function that prints execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.6f} seconds to execute")
        return result
    return wrapper

def profile_function(func, *args, **kwargs):
    """
    Profile a function execution and return statistics.
    
    Parameters:
    -----------
    func : function
        The function to profile
    *args, **kwargs
        Arguments to pass to the function
        
    Returns:
    --------
    stats : pstats.Stats
        Profiling statistics
    """
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    
    return stats

def benchmark_loop_vs_vectorized(size=1000000):
    """
    Benchmark loop-based vs. vectorized operations.
    
    Parameters:
    -----------
    size : int
        Size of the arrays to operate on
        
    Returns:
    --------
    dict
        Dictionary containing timing results for different operations
    """
    # Create arrays
    array1 = np.random.random(size)
    array2 = np.random.random(size)
    
    results = {}
    
    # Test element-wise addition
    start_time = time.time()
    result_loop = np.zeros(size)
    for i in range(size):
        result_loop[i] = array1[i] + array2[i]
    loop_time = time.time() - start_time
    
    start_time = time.time()
    result_vectorized = array1 + array2
    vectorized_time = time.time() - start_time
    
    results['addition'] = {
        'loop_time': loop_time,
        'vectorized_time': vectorized_time,
        'speedup': loop_time / vectorized_time
    }
    
    # Test element-wise multiplication
    start_time = time.time()
    result_loop = np.zeros(size)
    for i in range(size):
        result_loop[i] = array1[i] * array2[i]
    loop_time = time.time() - start_time
    
    start_time = time.time()
    result_vectorized = array1 * array2
    vectorized_time = time.time() - start_time
    
    results['multiplication'] = {
        'loop_time': loop_time,
        'vectorized_time': vectorized_time,
        'speedup': loop_time / vectorized_time
    }
    
    # Test sum
    start_time = time.time()
    result_loop = 0
    for i in range(size):
        result_loop += array1[i]
    loop_time = time.time() - start_time
    
    start_time = time.time()
    result_vectorized = np.sum(array1)
    vectorized_time = time.time() - start_time
    
    results['sum'] = {
        'loop_time': loop_time,
        'vectorized_time': vectorized_time,
        'speedup': loop_time / vectorized_time
    }
    
    return results