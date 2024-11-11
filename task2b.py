import time
import random
import matplotlib.pyplot as plt
import math
from clrsPython import dijkstra, AdjacencyListGraph

# Parameters for the experiment
network_sizes = range(1100, 2001, 100)
average_times = []
theoretical_times = []

# Function to generate a tube network with n stations based on the number of stops
def generate_network(n):
    edges = []
    for i in range(n - 1):
        # Connecting each station to the next station (1 stop between each pair)
        edges.append((i, i + 1, 1))
        # Optionally, connect more edges to simulate a realistic network
        if i < n - 2:
            edges.append((i, i + 2, 1))
    return edges

# Measure time complexity empirically
for n in network_sizes:
    # Generate a graph with n stations
    edges = generate_network(n)
    graph = AdjacencyListGraph(n, True, True)
    for u, v, weight in edges:
        try:
            graph.insert_edge(u, v, weight)
        except RuntimeError:
            # Skip if edge already exists
            continue

    # Measure the execution time for finding the shortest path
    num_pairs = 10  # Number of station pairs to test
    total_time = 0
    for _ in range(num_pairs):
        start, end = random.sample(range(n), 2)  # Random pair of stations
        start_time = time.time()
        dijkstra(graph, start)
        total_time += (time.time() - start_time) * 1000  # Convert to milliseconds

    # Average execution time for this network size
    average_time = total_time / num_pairs
    average_times.append(average_time)
    
    # Calculate the theoretical time based on O(n log n)
    theoretical_time = n * math.log(n)
    theoretical_times.append(theoretical_time)

    print(f"Average execution time for n={n}: {average_time:.2f} ms")

# Normalize theoretical times to align the scale with empirical results
max_empirical = max(average_times)
max_theoretical = max(theoretical_times)
normalized_theoretical_times = [t * (max_empirical / max_theoretical) for t in theoretical_times]

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(network_sizes, average_times, marker='o', label='Empirical Execution Time')
plt.plot(network_sizes, normalized_theoretical_times, marker='x', linestyle='--', color='orange', label=r'Theoretical $O(n \log n)$ Time')
plt.xlabel('Network Size (n)')
plt.ylabel('Average Execution Time (ms)')
plt.title("Average Execution Time of Dijkstra's Algorithm (Number of Stops)")
plt.legend()
plt.grid(True)
plt.show()