import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from clrsPython import dijkstra, AdjacencyListGraph

# Define the stations and edges
stations = ['A', 'B', 'C', 'D', 'E']
edges = [('A', 'B', 1), ('A', 'D', 1), ('B', 'C', 1), ('B', 'D', 1), ('C', 'E', 1),
         ('D', 'B', 1), ('D', 'C', 1), ('D', 'E', 1), ('E', 'A', 1), ('E', 'C', 1)]

# Adjusted data for stops-based calculation (each adjacent connection is 1 stop)
stops_data = {
    "source": [edge[0] for edge in edges],
    "destination": [edge[1] for edge in edges],
    "stops": [1 for _ in edges]  # Each journey has a "stop" count of 1
}
stops_df = pd.DataFrame(stops_data)
print("Journey Data Based on Number of Stops:\n", stops_df, end="\n\n")

# Create the graph for the number of stops
number_of_stations = len(stations)
stops_graph = AdjacencyListGraph(number_of_stations, True, True)
for u, v, weight in zip(stops_data["source"], stops_data["destination"], stops_data["stops"]):
    stops_graph.insert_edge(stations.index(u), stations.index(v), weight)

# Function to reconstruct the path from the predecessor array
def get_path(pi, start_idx, end_idx):
    path = []
    current = end_idx
    while current != start_idx and current is not None:
        path.append(current)
        current = pi[current]
    path.append(start_idx)
    return path[::-1]

# Initialize a matrix for stops
stops_matrix = pd.DataFrame(float('inf'), index=stations, columns=stations)

# Calculate paths and populate matrix for the number of stops
for start_station in stations:
    start_idx = stations.index(start_station)
    
    # Dijkstra for number of stops
    d_stops, pi_stops = dijkstra(stops_graph, start_idx)
    
    print(f"Shortest paths from {start_station}:")
    
    for end_idx in range(number_of_stations):
        end_station = stations[end_idx]
        
        # Get path and update stops matrix
        if d_stops[end_idx] != float('inf'):
            path_stops = get_path(pi_stops, start_idx, end_idx)
            path_stops_names = [stations[i] for i in path_stops]
            stops_matrix.loc[start_station, end_station] = d_stops[end_idx]
            print(f"  Path to {end_station}: {' -> '.join(path_stops_names)}, Stops: {d_stops[end_idx]}")
    
    print("\n")

# Display matrix for number of stops
print("Number of Stops Matrix:\n", stops_matrix)

# Plot heatmap for number of stops
plt.figure(figsize=(6, 5))
sns.heatmap(stops_matrix, annot=True, cmap="YlOrBr", fmt=".1f")
plt.title("Shortest Path Based on Number of Stops Between Stations")
plt.xlabel("Destination")
plt.ylabel("Source")
plt.show()