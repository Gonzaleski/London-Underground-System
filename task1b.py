import random
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from clrsPython import dijkstra, AdjacencyListGraph

def generate_tube_network(n):
    """Generates an artificial tube network with n stations, avoiding duplicate edges."""
    edges = set()  # Use a set to keep track of edges (no duplicates)
    stations = [f"Station_{i}" for i in range(n)]  # Create station names
    edge_check = {}  # Dictionary to track added edges and prevent duplicates

    # Create a base path graph to ensure connectivity
    for i in range(n - 1):
        u, v, weight = stations[i], stations[i + 1], random.randint(1, 20)  # Set edge weight between consecutive stations
        edges.add((u, v, weight))
        edge_check[(u, v)] = True

    # Add additional random edges to increase connectivity, avoiding duplicates
    while len(edges) < n + n // 2:  # Ensure the graph has enough edges
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
        source = random.choice(stations)  # Randomly select source station
        start_time = time.time()  # Record start time
        dijkstra(graph, stations.index(source))  # Run Dijkstra's algorithm from source
        end_time = time.time()  # Record end time
        total_time += (end_time - start_time) * 1000  # Convert to milliseconds
    average_time = total_time / num_trials  # Compute average time
    return average_time

# Experiment with different values of n (network sizes)
network_sizes = [100 * i for i in range(1, 11)]  # Experimenting with network sizes from 100 to 1000 stations
average_times = []  # List to store average execution times for each network size

for n in network_sizes:
    stations, edges = generate_tube_network(n)  # Generate a network with n stations
    # Create the graph for Dijkstra's algorithm
    graph = AdjacencyListGraph(n, True, True)
    for edge in edges:
        u, v, weight = edge
        graph.insert_edge(stations.index(u), stations.index(v), weight)  # Insert edges into graph
    
    # Measure the execution time and store the result
    avg_time = measure_dijkstra_execution_time(graph, stations)
    average_times.append(avg_time)
    print(f"Average execution time for n = {n}: {avg_time:.4f} ms")

# Theoretical time complexity curve (scaled for comparison)
theoretical_times = [n * np.log(n) for n in network_sizes]  # Theoretical O(n log n) time complexity
# Scale to match empirical data range for visual comparison
scale_factor = average_times[-1] / theoretical_times[-1]
theoretical_times = [t * scale_factor for t in theoretical_times]

# Plot the empirical and theoretical results
plt.plot(network_sizes, average_times, marker='o', color='b', label="Empirical Time")
plt.plot(network_sizes, theoretical_times, color='r', linestyle='--', label="Theoretical O(n log n) Time")
plt.xlabel("Network Size (n)")  # X-axis label
plt.ylabel("Average Execution Time (ms)")  # Y-axis label
plt.title("Empirical vs. Theoretical Time Complexity of Dijkstra's Algorithm")  # Plot title
plt.legend()  # Show legend
plt.grid(True)  # Enable grid for better readability
plt.show()  # Display the plot
