# /learn-ai-ml-dl/phase0/python_optimization/memory_management.py

"""
Memory management techniques for efficient machine learning operations.
"""

import sys
import numpy as np
import gc

def get_size(obj, seen=None):
    """
    Recursively determine the size of an object in bytes.
    
    Parameters:
    -----------
    obj : any
        The object to measure
    seen : set, optional
        Set of objects already seen (for handling recursive structures)
        
    Returns:
    --------
    int
        Size of the object in bytes
    """
    # Initialize the set of seen objects if needed
    if seen is None:
        seen = set()
    
    # Get object id to avoid duplicate counting
    obj_id = id(obj)
    
    # If we've already seen this object, skip it
    if obj_id in seen:
        return 0
    
    # Add the object to seen
    seen.add(obj_id)
    
    # Get the size of the object itself
    size = sys.getsizeof(obj)
    
    # Handle containers
    if isinstance(obj, (list, tuple, set, frozenset)):
        # Add size of each element
        size += sum(get_size(item, seen) for item in obj)
    
    elif isinstance(obj, dict):
        # Add size of keys and values
        size += sum(get_size(k, seen) + get_size(v, seen) for k, v in obj.items())
    
    elif isinstance(obj, np.ndarray):
        # Add size of array data
        size = obj.nbytes
    
    return size

def optimize_dtype(array, verbose=False):
    """
    Optimize the data type of a NumPy array to use less memory.
    
    Parameters:
    -----------
    array : numpy.ndarray
        The array to optimize
    verbose : bool, optional
        Whether to print information about the optimization
        
    Returns:
    --------
    numpy.ndarray
        The array with an optimized data type
    """
    original_size = array.nbytes
    original_dtype = array.dtype
    
    # Find the min and max values
    min_val = array.min()
    max_val = array.max()
    
    # Determine if the array contains integers or floats
    if np.issubdtype(array.dtype, np.integer):
        # For integer data, find the smallest integer type that can hold the data
        if min_val >= 0:  # unsigned
            if max_val <= np.iinfo(np.uint8).max:
                new_dtype = np.uint8
            elif max_val <= np.iinfo(np.uint16).max:
                new_dtype = np.uint16
            elif max_val <= np.iinfo(np.uint32).max:
                new_dtype = np.uint32
            else:
                new_dtype = np.uint64
        else:  # signed
            if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
                new_dtype = np.int8
            elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
                new_dtype = np.int16
            elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
                new_dtype = np.int32
            else:
                new_dtype = np.int64
    
    elif np.issubdtype(array.dtype, np.floating):
        # For floating-point data, consider float16 or float32 if precision is sufficient
        if min_val >= np.finfo(np.float16).min and max_val <= np.finfo(np.float16).max:
            new_dtype = np.float16
        elif min_val >= np.finfo(np.float32).min and max_val <= np.finfo(np.float32).max:
            new_dtype = np.float32
        else:
            new_dtype = np.float64
    
    else:
        # For other types, keep the original dtype
        new_dtype = array.dtype
    
    # Convert to the new dtype
    optimized_array = array.astype(new_dtype)
    new_size = optimized_array.nbytes
    
    if verbose:
        print(f"Original dtype: {original_dtype}, size: {original_size} bytes")
        print(f"New dtype: {new_dtype}, size: {new_size} bytes")
        print(f"Memory savings: {original_size - new_size} bytes ({(1 - new_size/original_size)*100:.2f}%)")
    
    return optimized_array