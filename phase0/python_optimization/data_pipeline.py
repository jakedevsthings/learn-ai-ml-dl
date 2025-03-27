# /learn-ai-ml-dl/phase0/python_optimization/data_pipeline.py

"""
Efficient data processing pipeline implementation for machine learning applications.
"""

import numpy as np
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import os
import psutil

class DataPipeline:
    """
    An efficient data processing pipeline that implements batch processing
    and parallelization techniques.
    """
    
    def __init__(self, batch_size=32, num_workers=None, use_multiprocessing=False):
        """
        Initialize the data pipeline.
        
        Parameters:
        -----------
        batch_size : int
            The number of samples to process at once
        num_workers : int or None
            Number of parallel workers to use. If None, uses CPU count - 1.
        use_multiprocessing : bool
            Whether to use process-based parallelism instead of threads
        """
        self.batch_size = batch_size
        
        if num_workers is None:
            # Default to number of CPU cores - 1 (leave one core free)
            self.num_workers = max(1, os.cpu_count() - 1)
        else:
            self.num_workers = num_workers
            
        self.use_multiprocessing = use_multiprocessing
        self.transforms = []
    
    def add_transform(self, transform_func):
        """
        Add a transformation function to the pipeline.
        
        Parameters:
        -----------
        transform_func : callable
            A function that takes data as input and returns transformed data
        """
        self.transforms.append(transform_func)
        return self
    
    def _process_batch(self, batch):
        """
        Apply all transformations to a batch of data.
        
        Parameters:
        -----------
        batch : array-like
            Batch of data to process
            
        Returns:
        --------
        processed_batch : array-like
            The processed batch after all transformations
        """
        result = batch
        for transform in self.transforms:
            result = transform(result)
        return result
    
    def process_data(self, data_iterable):
        """
        Process all data through the pipeline.
        
        Parameters:
        -----------
        data_iterable : iterable
            The data to process
            
        Returns:
        --------
        processed_data : list
            The processed data after all transformations
        """
        # Convert to list if it's not already
        data_list = list(data_iterable)
        num_samples = len(data_list)
        
        # Create batches
        batches = [
            data_list[i:i + self.batch_size] 
            for i in range(0, num_samples, self.batch_size)
        ]
        
        start_time = time.time()
        
        # Process batches in parallel
        if self.num_workers > 1:
            # Choose the executor based on configuration
            executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
            
            with executor_class(max_workers=self.num_workers) as executor:
                processed_batches = list(executor.map(self._process_batch, batches))
        else:
            # Process sequentially if only one worker
            processed_batches = [self._process_batch(batch) for batch in batches]
        
        end_time = time.time()
        
        # Flatten the batches back into a single list
        processed_data = []
        for batch in processed_batches:
            processed_data.extend(batch)
        
        processing_time = end_time - start_time
        return processed_data, processing_time

def monitor_memory_usage(func):
    """
    Decorator to monitor memory usage during function execution.
    
    Parameters:
    -----------
    func : callable
        The function to monitor
        
    Returns:
    --------
    wrapped : callable
        Wrapped function that reports memory usage
    """
    def wrapped(*args, **kwargs):
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Call the function
        result = func(*args, **kwargs)
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Memory usage: Before: {mem_before:.2f} MB, After: {mem_after:.2f} MB, Diff: {mem_after - mem_before:.2f} MB")
        
        return result
    
    return wrapped

def create_lazy_processor(transform_func, batch_size=1000):
    """
    Creates a generator-based lazy data processor that only processes data as needed.
    
    Parameters:
    -----------
    transform_func : callable
        The transformation function to apply
    batch_size : int
        Size of batches to process at once
        
    Returns:
    --------
    processor : callable
        A function that returns a generator for lazy processing
    """
    def processor(data_iterable):
        """
        Process data lazily.
        
        Parameters:
        -----------
        data_iterable : iterable
            The data to process
            
        Yields:
        -------
        item : Any
            Processed data items one at a time
        """
        batch = []
        
        for item in data_iterable:
            batch.append(item)
            
            # Process in batches for efficiency
            if len(batch) >= batch_size:
                processed_batch = transform_func(batch)
                for processed_item in processed_batch:
                    yield processed_item
                batch = []
        
        # Process any remaining items
        if batch:
            processed_batch = transform_func(batch)
            for processed_item in processed_batch:
                yield processed_item
    
    return processor

# Demo batch processor for image augmentation (simulated)
def demo_batch_image_augmentation(images, flip_probability=0.5, rotation_max=30):
    """
    Simulate batch image augmentation operations.
    
    Parameters:
    -----------
    images : list or array
        List of image arrays to augment
    flip_probability : float
        Probability of flipping an image horizontally
    rotation_max : int
        Maximum rotation in degrees
        
    Returns:
    --------
    augmented_images : list
        List of augmented images
    """
    # In a real implementation, this would use actual image processing
    # Here we'll simulate the operations for demonstration
    augmented_images = []
    
    for img in images:
        # Simulate horizontal flip
        if np.random.random() < flip_probability:
            # In real code: img = np.fliplr(img)
            pass
            
        # Simulate rotation
        rotation_angle = np.random.uniform(-rotation_max, rotation_max)
        # In real code: img = rotate_image(img, rotation_angle)
        
        # Add random noise to simulate other transformations
        if isinstance(img, np.ndarray):
            augmented_img = img + np.random.normal(0, 0.01, img.shape)
        else:
            augmented_img = img  # If not numpy array, just pass through
            
        augmented_images.append(augmented_img)
        
    return augmented_images