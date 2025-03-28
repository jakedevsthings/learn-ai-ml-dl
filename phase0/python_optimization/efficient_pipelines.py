import time
import random
import sys

# --- Inefficient Pipeline Example (Intermediate Lists) ---

def generate_random_numbers_list(n):
    """Generates a list of n random numbers."""
    return [random.random() for _ in range(n)]

def filter_numbers_list(numbers, threshold):
    """Filters numbers in a list greater than a threshold."""
    return [num for num in numbers if num > threshold]

def square_numbers_list(numbers):
    """Squares numbers in a list."""
    return [num * num for num in numbers]

def inefficient_pipeline(data_size, threshold):
    """Pipeline that creates intermediate lists at each step."""
    print(f"\n--- Running Inefficient Pipeline (data_size={data_size:,}) ---")
    start_time = time.time()

    # Step 1: Generate numbers
    random_nums = generate_random_numbers_list(data_size)
    print(f"Step 1 Memory (approx list size): {sys.getsizeof(random_nums)/(1024*1024):.2f} MB")

    # Step 2: Filter numbers
    filtered_nums = filter_numbers_list(random_nums, threshold)
    print(f"Step 2 Memory (approx list size): {sys.getsizeof(filtered_nums)/(1024*1024):.2f} MB")
    del random_nums # Try to free memory

    # Step 3: Square numbers
    squared_nums = square_numbers_list(filtered_nums)
    print(f"Step 3 Memory (approx list size): {sys.getsizeof(squared_nums)/(1024*1024):.2f} MB")
    del filtered_nums # Try to free memory

    # Step 4: Calculate sum (final result)
    total_sum = sum(squared_nums)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Inefficient pipeline finished in {duration:.4f} seconds.")
    print(f"Final sum: {total_sum:.4f}")
    return total_sum

# --- Efficient Pipeline Example (Generators) ---

def generate_random_numbers_gen(n):
    """Generates n random numbers one by one."""
    for _ in range(n):
        yield random.random()

def filter_numbers_gen(numbers_gen, threshold):
    """Filters numbers from a generator."""
    for num in numbers_gen:
        if num > threshold:
            yield num

def square_numbers_gen(numbers_gen):
    """Squares numbers from a generator."""
    for num in numbers_gen:
        yield num * num

def efficient_pipeline(data_size, threshold):
    """Pipeline using generators to process data lazily."""
    print(f"\n--- Running Efficient Pipeline (data_size={data_size:,}) ---")
    start_time = time.time()

    # Chain the generators together
    random_gen = generate_random_numbers_gen(data_size)
    filtered_gen = filter_numbers_gen(random_gen, threshold)
    squared_gen = square_numbers_gen(filtered_gen)

    # The actual computation happens here, as sum() pulls items through the pipeline
    # Memory usage remains low as only one item is processed at a time
    total_sum = sum(squared_gen)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Efficient pipeline finished in {duration:.4f} seconds.")
    print(f"Final sum: {total_sum:.4f}")
    # Note: Measuring memory usage precisely for generators is tricky as they hold minimal state.
    # The key benefit is avoiding large intermediate data structures.
    return total_sum

# --- Run Comparison ---
data_size = 5_000_000 # Adjust size as needed
filter_threshold = 0.5

sum_inefficient = inefficient_pipeline(data_size, filter_threshold)
sum_efficient = efficient_pipeline(data_size, filter_threshold)

print(f"\nResults match: {abs(sum_inefficient - sum_efficient) < 1e-9}")
print("Observe the difference in reported memory usage and execution time.")
