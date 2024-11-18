import time
import random
import matplotlib.pyplot as plt
import math
from clrsPython import dijkstra, AdjacencyListGraph

# Parameters for the experiment: defining range of network sizes and lists to store execution times
network_sizes = range(1100, 2001, 100)  # Network sizes from 1100 to 2000, with steps of 100
average_times = []  # To store the empirical average execution times for each network size
theoretical_times = []  # To store the calculated theoretical times based on O(n log n)

# Function to generate a tube network with n stations based on the number of stops
def generate_network(n):
    edges = []  # List to store edges between stations
    for i in range(n - 1):
        # Connecting each station to the next station (1 stop between each pair)
        edges.append((i, i + 1, 1))  # Add edge between consecutive stations with weight 1
        # Connect more edges to simulate a realistic network
        if i < n - 2:
            edges.append((i, i + 2, 1))
    return edges

# Measure time complexity empirically
for n in network_sizes:
    # Generate a graph with n stations
    edges = generate_network(n)
    graph = AdjacencyListGraph(n, True, True)  # Initialize the graph with n stations and directed, weighted edges
    for u, v, weight in edges:
        try:
            graph.insert_edge(u, v, weight)  # Insert each edge into the graph
        except RuntimeError:
            # Skip if edge already exists (ensuring no duplicate edges)
            continue

    # Measure the execution time for finding the shortest path for random station pairs
    num_pairs = 10  # Number of station pairs to test
    total_time = 0  # To accumulate the total time for this network size
    for _ in range(num_pairs):
        start, end = random.sample(range(n), 2)  # Randomly select two stations
        start_time = time.time()  # Record start time
        dijkstra(graph, start)  # Run Dijkstra's algorithm from the start station
        total_time += (time.time() - start_time) * 1000  # Calculate elapsed time in milliseconds

    # Average execution time for this network size
    average_time = total_time / num_pairs  # Average time for the selected number of pairs
    average_times.append(average_time)  # Add the average time to the list

    # Calculate the theoretical time based on O(n log n) complexity
    theoretical_time = n * math.log(n)  # Theoretical time using O(n log n) formula
    theoretical_times.append(theoretical_time)  # Store the theoretical time for comparison

    # Print the empirical average execution time for the current network size
    print(f"Average execution time for n={n}: {average_time:.2f} ms")

# Normalize theoretical times to align the scale with empirical results
max_empirical = max(average_times)  # Find the maximum empirical time
max_theoretical = max(theoretical_times)  # Find the maximum theoretical time
normalized_theoretical_times = [t * (max_empirical / max_theoretical) for t in theoretical_times]  # Normalize

# Plotting results: comparison between empirical and normalized theoretical times
plt.figure(figsize=(10, 6))
plt.plot(network_sizes, average_times, marker='o', label='Empirical Execution Time')  # Empirical data points
plt.plot(network_sizes, normalized_theoretical_times, marker='x', linestyle='--', color='orange', label=r'Theoretical $O(n \log n)$ Time')  # Normalized theoretical curve
plt.xlabel('Network Size (n)')  # X-axis label
plt.ylabel('Average Execution Time (ms)')  # Y-axis label
plt.title("Average Execution Time of Dijkstra's Algorithm (Number of Stops)")  # Plot title
plt.legend()  # Display legend
plt.grid(True)  # Enable grid
plt.show()  # Display the plot