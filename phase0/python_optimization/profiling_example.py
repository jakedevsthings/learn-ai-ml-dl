import cProfile
import pstats
import io
import time

# Define the size of the lists
data_size = 1_000_000

def loop_addition(list1, list2):
    """Adds elements of two lists using a Python loop."""
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")
    result = [0] * len(list1)
    # Simulate some work inside the loop
    for i in range(len(list1)):
        result[i] = list1[i] + list2[i]
        # Adding a small delay to make the loop part more prominent in profiling
        # In real code, this might be a more complex calculation
        # time.sleep(0.0000001) # Uncomment if needed, but can slow down significantly
    return result

# Generate sample data
list_a = list(range(data_size))
list_b = list(range(data_size))

# --- Profiling the function ---

# Create a Profile object
profiler = cProfile.Profile()

# Run the code that you want to profile
print(f"Starting profiling for data size: {data_size}...")
profiler.enable()
result = loop_addition(list_a, list_b) # Call the function under profiling
profiler.disable()
print("Profiling finished.")

# --- Analyzing the results ---

# Create a stream to capture the stats output
stream = io.StringIO()

# Sort the stats by cumulative time spent in the function
# Other sort keys include: 'calls', 'time', 'cumulative', 'filename', 'lineno', 'name', 'nfl' (name/file/line)
sorted_stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')

# Print the statistics
# print_stats(n) limits the output to the top n lines
print("\n--- Profiling Results (Top 10 by cumulative time) ---")
sorted_stats.print_stats(10)
print(stream.getvalue())

# You can also dump the stats to a file for later analysis
# profiler.dump_stats('loop_addition_profile.prof')
# print("\nProfiling stats saved to loop_addition_profile.prof")
# To view this file later, you can use tools like snakeviz:
# pip install snakeviz
# snakeviz loop_addition_profile.prof
