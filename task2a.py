import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from clrsPython import dijkstra, AdjacencyListGraph

# Define the stations and edges for both tasks
stations = ['A', 'B', 'C', 'D', 'E']
edges_time = [('A', 'B', 10), ('A', 'D', 5), ('B', 'C', 1), ('B', 'D', 3), ('C', 'E', 6),
              ('D', 'B', 3), ('D', 'C', 9), ('D', 'E', 2), ('E', 'A', 7), ('E', 'C', 6)]
edges_stops = [(edge[0], edge[1], 1) for edge in edges_time]  # Each connection = 1 stop

# Function to visualize graphs
def visualize_graphs(edges_time, edges_stops, seed=39):
    # Create the graphs
    G_time = nx.DiGraph()
    G_stops = nx.DiGraph()
    for edge in edges_time:
        G_time.add_edge(edge[0], edge[1], weight=edge[2])
    for edge in edges_stops:
        G_stops.add_edge(edge[0], edge[1], weight=edge[2])
    
    # Define consistent layout
    pos = nx.spring_layout(G_time, seed=seed)
    
    # Create a single figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot the graph with travel times
    nx.draw(G_time, pos, ax=axes[0], with_labels=True, node_color='lightblue',
            node_size=2000, font_size=10, font_weight='bold', arrowsize=20)
    edge_labels_time = nx.get_edge_attributes(G_time, 'weight')
    nx.draw_networkx_edge_labels(G_time, pos, edge_labels=edge_labels_time, font_color='red', ax=axes[0])
    axes[0].set_title("Tube Network (Travel Times)")
    
    # Plot the graph with stops
    nx.draw(G_stops, pos, ax=axes[1], with_labels=True, node_color='lightgreen',
            node_size=2000, font_size=10, font_weight='bold', arrowsize=20)
    edge_labels_stops = nx.get_edge_attributes(G_stops, 'weight')
    nx.draw_networkx_edge_labels(G_stops, pos, edge_labels=edge_labels_stops, font_color='blue', ax=axes[1])
    axes[1].set_title("Tube Network (Number of Stops)")
    
    plt.tight_layout()
    plt.show()

# Visualize the graphs
visualize_graphs(edges_time, edges_stops)

# Function to reconstruct paths from the predecessor array
def get_path(pi, start_idx, end_idx):
    """Reconstruct the shortest path from the predecessor array."""
    path = []
    current = end_idx
    while current != start_idx and current is not None:
        path.append(current)
        current = pi[current]
    path.append(start_idx)
    return path[::-1]

# Function to compute shortest paths and build a matrix
def compute_shortest_paths(edges, stations, weight_type):
    """Compute shortest paths using Dijkstra's algorithm and create a matrix."""
    number_of_stations = len(stations)  # Determine the number of stations
    graph = AdjacencyListGraph(number_of_stations, True, True)  # Initialise a directed graph

    # Add edges to the graph with station indices and weights
    for u, v, weight in edges:
        graph.insert_edge(stations.index(u), stations.index(v), weight)
    
    # Initialise a matrix to store shortest path distances
    matrix = pd.DataFrame(float('inf'), index=stations, columns=stations)
    paths = {}  # Dictionary to store paths for comparison later
    
    # Loop over each station to calculate the shortest path to all other stations
    for start_station in stations:
        start_idx = stations.index(start_station)  # Get the index of the start station
        d, pi = dijkstra(graph, start_idx)  # Run Dijkstra's algorithm to get distances and predecessors
        paths[start_station] = {}  # Initialize the path storage for the current start station
        
        # Loop over each end station to reconstruct the shortest path
        for end_idx in range(number_of_stations):
            end_station = stations[end_idx]  # Get the end station
            if d[end_idx] != float('inf'):  # If a path exists
                path = get_path(pi, start_idx, end_idx)  # Reconstruct the path using the predecessors
                matrix.loc[start_station, end_station] = d[end_idx]  # Update the matrix with the shortest distance
                paths[start_station][end_station] = [stations[i] for i in path]  # Store the reconstructed path
    
    return matrix, paths  # Return the shortest path matrix and the path dictionary

# Compute shortest paths based on journey time
time_matrix, time_paths = compute_shortest_paths(edges_time, stations, "time")
print("Shortest path travel time matrix:\n", time_matrix)

# Compute shortest paths based on number of stops
stops_matrix, stops_paths = compute_shortest_paths(edges_stops, stations, "stops")
print("Shortest path number of stops matrix:\n", stops_matrix)

# Create the heatmaps for travel times and stops
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot the heatmap for travel times
sns.heatmap(time_matrix, annot=True, cmap="YlGnBu", ax=axes[0])
axes[0].set_title("Shortest Path Travel Times Between Stations")
axes[0].set_xlabel("Destination")
axes[0].set_ylabel("Source")

# Plot the heatmap for the number of stops
sns.heatmap(stops_matrix, annot=True, cmap="YlOrBr", ax=axes[1])
axes[1].set_title("Shortest Path Number of Stops Between Stations")
axes[1].set_xlabel("Destination")
axes[1].set_ylabel("Source")

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# Comparison of paths
comparison_results = []
for start_station in stations:
    for end_station in stations:
        if start_station != end_station:
            time_path = time_paths[start_station].get(end_station, [])
            stops_path = stops_paths[start_station].get(end_station, [])
            identical = time_path == stops_path
            comparison_results.append({
                "Start": start_station,
                "End": end_station,
                "Time Path": " -> ".join(time_path),
                "Stops Path": " -> ".join(stops_path),
                "Identical": identical
            })

# Display comparison results
comparison_df = pd.DataFrame(comparison_results)
print("\nComparison of Shortest Paths Based on Time and Stops:")
print(comparison_df)

# Display paths with differences
differences = comparison_df[~comparison_df["Identical"]]
print("\nPaths with differences:")
print(differences)
