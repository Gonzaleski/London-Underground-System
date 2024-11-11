import random
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from clrsPython import dijkstra, AdjacencyListGraph

def generate_tube_network(n):
    """Generates an artificial tube network with n stations, avoiding duplicate edges."""
    edges = set()  # Use a set to keep track of edges (no duplicates)
    stations = [f"Station_{i}" for i in range(n)]
    edge_check = {}  # Dictionary to track added edges and prevent duplicates

    # Create a base path graph to ensure connectivity
    for i in range(n - 1):
        u, v, weight = stations[i], stations[i + 1], random.randint(1, 20)
        edges.add((u, v, weight))
        edge_check[(u, v)] = True

    # Add additional random edges to increase connectivity, avoiding duplicates
    while len(edges) < n + n // 2:  # Adjust density as needed
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u != v:
            # Check if the edge or its reverse already exists
            if (stations[u], stations[v]) not in edge_check and (stations[v], stations[u]) not in edge_check:
                weight = random.randint(1, 20)
                edges.add((stations[u], stations[v], weight))
                edge_check[(stations[u], stations[v])] = True

    return stations, list(edges)


def measure_dijkstra_execution_time(graph, stations, num_trials=10):
    """Measures the execution time of Dijkstra's algorithm."""
    total_time = 0
    for _ in range(num_trials):
        source = random.choice(stations)
        start_time = time.time()
        dijkstra(graph, stations.index(source))
        end_time = time.time()
        total_time += (end_time - start_time) * 1000  # Convert to milliseconds
    average_time = total_time / num_trials
    return average_time

# Experiment with different values of n
network_sizes = [100 * i for i in range(1, 11)]
average_times = []

for n in network_sizes:
    stations, edges = generate_tube_network(n)
    # Create the graph for Dijkstra's algorithm
    graph = AdjacencyListGraph(n, True, True)
    for edge in edges:
        u, v, weight = edge
        graph.insert_edge(stations.index(u), stations.index(v), weight)
    
    # Measure the execution time and store the result
    avg_time = measure_dijkstra_execution_time(graph, stations)
    average_times.append(avg_time)
    print(f"Average execution time for n = {n}: {avg_time:.4f} ms")

# Theoretical time complexity curve (scaled for comparison)
theoretical_times = [n * np.log(n) for n in network_sizes]
# Scale to match empirical data range for visual comparison
scale_factor = average_times[-1] / theoretical_times[-1]
theoretical_times = [t * scale_factor for t in theoretical_times]

# Plot the empirical and theoretical results
plt.plot(network_sizes, average_times, marker='o', color='b', label="Empirical Time")
plt.plot(network_sizes, theoretical_times, color='r', linestyle='--', label="Theoretical O(n log n) Time")
plt.xlabel("Network Size (n)")
plt.ylabel("Average Execution Time (ms)")
plt.title("Empirical vs. Theoretical Time Complexity of Dijkstra's Algorithm")
plt.legend()
plt.grid(True)
plt.show()
