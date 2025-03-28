import numpy as np
import time

# Define the size of the arrays/lists
data_size = 1_000_000

# --- Loop-based approach ---
def loop_addition(list1, list2):
    """Adds elements of two lists using a Python loop."""
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")
    result = [0] * len(list1)
    for i in range(len(list1)):
        result[i] = list1[i] + list2[i]
    return result

# Generate sample data
list_a = list(range(data_size))
list_b = list(range(data_size))

# Time the loop-based approach
start_time_loop = time.time()
result_loop = loop_addition(list_a, list_b)
end_time_loop = time.time()
loop_duration = end_time_loop - start_time_loop

print(f"Loop-based addition took: {loop_duration:.6f} seconds")

# --- Vectorized approach (NumPy) ---
def vectorized_addition(arr1, arr2):
    """Adds elements of two NumPy arrays using vectorization."""
    return arr1 + arr2

# Generate sample data as NumPy arrays
array_a = np.arange(data_size)
array_b = np.arange(data_size)

# Time the vectorized approach
start_time_vec = time.time()
result_vec = vectorized_addition(array_a, array_b)
end_time_vec = time.time()
vec_duration = end_time_vec - start_time_vec

print(f"Vectorized addition took: {vec_duration:.6f} seconds")

# --- Comparison ---
if vec_duration > 0:
    speedup = loop_duration / vec_duration
    print(f"\nVectorization speedup: {speedup:.2f}x")
else:
    print("\nVectorized operation was too fast to measure significant speedup.")

# Verify results (optional, check a few elements)
# print(f"Loop result sample: {result_loop[:5]}")
# print(f"Vectorized result sample: {result_vec[:5]}")
# print(f"Results match: {np.array_equal(result_loop, result_vec)}") # Note: Comparing list and array
