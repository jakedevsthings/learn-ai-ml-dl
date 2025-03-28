import sys
import numpy as np
import gc # Garbage Collector interface

# --- Basic Memory Usage with Python Lists ---

def create_large_list(size):
    """Creates a large list of integers."""
    print(f"\nCreating a list with {size:,} integers...")
    large_list = list(range(size))
    memory_usage_bytes = sys.getsizeof(large_list)
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)
    print(f"Approximate memory usage of the list object: {memory_usage_bytes:,} bytes (~{memory_usage_mb:.2f} MB)")
    # Note: sys.getsizeof() might not account for the memory of the items *within* the list accurately,
    # especially for complex objects. It gives the size of the container itself.
    # For a list of simple integers, it's a reasonable starting point.
    return large_list

list_size = 10_000_000
my_list = create_large_list(list_size)

# Observe memory - you might need OS tools (like top/htop on Linux, Task Manager on Windows)
# input("Press Enter after observing memory usage...")

# --- Memory Usage with NumPy Arrays ---

def create_large_numpy_array(size):
    """Creates a large NumPy array of integers."""
    print(f"\nCreating a NumPy array with {size:,} integers...")
    large_array = np.arange(size, dtype=np.int64) # Use a specific dtype
    memory_usage_bytes = large_array.nbytes
    memory_usage_mb = memory_usage_bytes / (1024 * 1024)
    print(f"Accurate memory usage of the NumPy array data: {memory_usage_bytes:,} bytes (~{memory_usage_mb:.2f} MB)")
    # NumPy's .nbytes gives the total bytes consumed by the array's elements.
    return large_array

array_size = 10_000_000
my_array = create_large_numpy_array(array_size)

# Compare memory usage (NumPy arrays are generally more memory-efficient for numerical data)
print("\nComparing list vs NumPy array for the same number of elements:")
print(f"List object size: {sys.getsizeof(my_list) / (1024*1024):.2f} MB (approximate)")
print(f"NumPy array data size: {my_array.nbytes / (1024*1024):.2f} MB (accurate)")

# --- Releasing Memory ---
print("\nReleasing references to large objects...")
del my_list
del my_array

# Suggest garbage collection (Python does this automatically, but can be triggered)
print("Suggesting garbage collection...")
gc.collect()
print("Garbage collection suggested. Observe memory usage again.")

# input("Press Enter after observing memory usage reduction...")

# --- Memory-Efficient Processing (Example: Generators) ---

def process_data_generator(data_source):
    """Processes data yielded by a generator, one item at a time."""
    print("\nProcessing data using a generator (memory-efficient)...")
    total_sum = 0
    count = 0
    for item in data_source:
        # Simulate processing
        total_sum += item * 2
        count += 1
        # if count % 1_000_000 == 0:
        #     print(f"Processed {count} items...")
    print(f"Finished processing {count} items. Final sum: {total_sum}")

def data_generator(size):
    """A generator function that yields numbers without storing them all."""
    for i in range(size):
        yield i

large_data_size = 20_000_000 # Even larger size

# Process using the generator - avoids creating a huge list/array in memory
process_data_generator(data_generator(large_data_size))

print("\nNote: For truly massive datasets that don't fit in RAM, techniques like memory mapping (numpy.memmap) or chunking with libraries like Dask or Pandas are necessary.")
