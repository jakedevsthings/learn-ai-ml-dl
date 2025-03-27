# /learn-ai-ml-dl/phase0/python_optimization/batch_processing.py

"""
Efficient batch processing techniques for machine learning datasets.
"""

import numpy as np
import time
from functools import wraps
import itertools
from collections import deque

class BatchGenerator:
    """
    Memory-efficient batch generator for large datasets.
    """
    
    def __init__(self, data, batch_size, shuffle=True):
        """
        Initialize the batch generator.
        
        Parameters:
        -----------
        data : array-like
            The data to generate batches from
        batch_size : int
            Size of batches to generate
        shuffle : bool
            Whether to shuffle the data before generating batches
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(data)
        self.current_idx = 0
        
        # Generate sample indices
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        """
        Return the iterator object itself.
        """
        return self
    
    def __next__(self):
        """
        Get the next batch.
        """
        if self.current_idx >= self.num_samples:
            # End of epoch, reset and shuffle
            self.current_idx = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
            raise StopIteration
        
        # Get batch indices
        batch_indices = self.indices[self.current_idx:
                                     min(self.current_idx + self.batch_size, 
                                         self.num_samples)]
        
        # Update current index
        self.current_idx += self.batch_size
        
        # Return the batch
        return [self.data[i] for i in batch_indices]

def minibatch_generator(data, batch_size, shuffle=True, max_epochs=None):
    """
    Generator function for creating mini-batches from a dataset.
    
    Parameters:
    -----------
    data : array-like
        The dataset to divide into mini-batches
    batch_size : int
        Size of each mini-batch
    shuffle : bool
        Whether to shuffle the data before each epoch
    max_epochs : int or None
        Maximum number of epochs (iterations through the dataset).
        If None, continue indefinitely.
        
    Yields:
    -------
    mini_batch : array-like
        A mini-batch of data
    """
    num_samples = len(data)
    indices = np.arange(num_samples)
    epoch = 0
    
    while max_epochs is None or epoch < max_epochs:
        # Shuffle data at the start of each epoch if requested
        if shuffle:
            np.random.shuffle(indices)
        
        # Create mini-batches
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            # Extract the mini-batch
            mini_batch = [data[i] for i in batch_indices]
            yield mini_batch
        
        epoch += 1

def time_batch_processing(func):
    """
    Decorator to time batch processing operations.
    
    Parameters:
    -----------
    func : callable
        The batch processing function to time
        
    Returns:
    --------
    wrapper : callable
        Decorated function that measures and reports processing time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        print(f"Batch processing took {end_time - start_time:.6f} seconds")
        
        # If result is a tuple, add timing information
        if isinstance(result, tuple):
            return result + (end_time - start_time,)
        else:
            return result, end_time - start_time
    
    return wrapper

class StreamingBatchProcessor:
    """
    Processes data in a streaming fashion with a fixed memory footprint.
    """
    
    def __init__(self, buffer_size, batch_size, processing_func):
        """
        Initialize the streaming processor.
        
        Parameters:
        -----------
        buffer_size : int
            Maximum number of elements to keep in memory
        batch_size : int
            Number of items to process in each batch
        processing_func : callable
            Function to apply to each batch
        """
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.processing_func = processing_func
        self.processed_count = 0
    
    def add_item(self, item):
        """
        Add an item to the buffer and process batches if needed.
        
        Parameters:
        -----------
        item : any
            The item to add to the buffer
            
        Returns:
        --------
        processed_items : list or None
            Processed items if a batch was completed, otherwise None
        """
        self.buffer.append(item)
        self.processed_count += 1
        
        # Process batch if we have enough items
        if len(self.buffer) >= self.batch_size:
            # Get batch from buffer
            batch = list(itertools.islice(self.buffer, self.batch_size))
            
            # Process the batch
            processed_batch = self.processing_func(batch)
            
            # Remove processed items from buffer
            for _ in range(self.batch_size):
                self.buffer.popleft()
            
            return processed_batch
        
        return None
    
    def flush(self):
        """
        Process any remaining items in the buffer.
        
        Returns:
        --------
        processed_items : list or None
            Processed items if any were in the buffer, otherwise None
        """
        if not self.buffer:
            return None
        
        # Process all remaining items
        batch = list(self.buffer)
        processed_batch = self.processing_func(batch)
        
        # Clear the buffer
        self.buffer.clear()
        
        return processed_batch

def compare_batch_sizes(data, process_func, batch_sizes):
    """
    Compare processing time for different batch sizes.
    
    Parameters:
    -----------
    data : array-like
        Data to process
    process_func : callable
        Function to apply to each batch
    batch_sizes : list of int
        Batch sizes to compare
        
    Returns:
    --------
    results : dict
        Dictionary mapping batch sizes to processing times
    """
    results = {}
    
    for batch_size in batch_sizes:
        start_time = time.time()
        
        # Process data in batches
        for start_idx in range(0, len(data), batch_size):
            end_idx = min(start_idx + batch_size, len(data))
            batch = data[start_idx:end_idx]
            process_func(batch)
        
        end_time = time.time()
        results[batch_size] = end_time - start_time
    
    return results