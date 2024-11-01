import pandas as pd
from clrsPython import dijkstra, AdjacencyListGraph

# Define the stations and edges
stations = ['A', 'B', 'C', 'D', 'E']
edges = [('A', 'B', 10), ('A', 'D', 5), ('B', 'C', 1), ('B', 'D', 2), ('C', 'E', 4),
         ('D', 'B', 3), ('D', 'C', 9), ('D', 'E', 2), ('E', 'A', 7), ('E', 'C', 6)]

# Create DataFrame for journey times (minutes) as in Task 1a
data = {
    "source": [edge[0] for edge in edges],
    "destination": [edge[1] for edge in edges],
    "estimated time": [edge[2] for edge in edges]  # in minutes
}
df = pd.DataFrame(data)
print("Journey Time Data (minutes):\n", df, end="\n\n")

# Create the graph for journey time in minutes
number_of_stations = len(stations)
time_graph = AdjacencyListGraph(number_of_stations, True, True)
for edge in edges:
    time_graph.insert_edge(stations.index(edge[0]), stations.index(edge[1]), edge[2])

# Adjusted data for stops-based calculation (each adjacent connection is 1 stop)
stops_data = {
    "source": [edge[0] for edge in edges],
    "destination": [edge[1] for edge in edges],
    "stops": [1 for _ in edges]  # Each journey has a "stop" count of 1
}
stops_df = pd.DataFrame(stops_data)
print("Journey Data Based on Number of Stops:\n", stops_df, end="\n\n")

# Create the graph for the number of stops
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

# Select a pair of stations to compare paths (e.g., 'A' to 'E')
start, end = 'A', 'E'

# Calculate shortest path based on journey time from Task 1a
d_time, pi_time = dijkstra(time_graph, stations.index(start))
path_time = get_path(pi_time, stations.index(start), stations.index(end))
path_time_names = [stations[i] for i in path_time]

# Calculate shortest path based on number of stops for Task 2a
d_stops, pi_stops = dijkstra(stops_graph, stations.index(start))
path_stops = get_path(pi_stops, stations.index(start), stations.index(end))
path_stops_names = [stations[i] for i in path_stops]

# Print results for journey time and number of stops
print(f"Shortest path from {start} to {end} based on journey time (minutes): {path_time_names}")
print(f"Shortest path from {start} to {end} based on number of stops: {path_stops_names}")

# Analysis: Check if paths are identical
if path_time_names == path_stops_names:
    print("The paths are identical for both journey time and number of stops.")
else:
    print("The paths differ between journey time and number of stops.")