import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from clrsPython import AdjacencyListGraph, dijkstra, kruskal

# Load dataset and remove rows without destinations
df = pd.read_excel('London Underground data.xlsx', names=['Lane', 'Source', 'Destination', 'Duration (minutes)'])
df = df.dropna(subset=['Destination'])  # Remove rows without destinations

# Collect unique stations and map them to indices
stations = pd.concat([df['Source'], df['Destination']]).unique()
station_indices = {station: idx for idx, station in enumerate(stations)}

# Create the full graph and add edges with durations as weights
number_of_stations = len(stations)
full_graph = AdjacencyListGraph(number_of_stations, directed=False, weighted=True)
edges = []
existing_edges = set()  # Keep track of added edges to avoid duplicates

for _, row in df.iterrows():
    source_idx = station_indices[row['Source']]
    dest_idx = station_indices[row['Destination']]
    duration = row['Duration (minutes)'] if not pd.isna(row['Duration (minutes)']) else 1
    edge = (min(source_idx, dest_idx), max(source_idx, dest_idx))  # Sort to avoid duplicates

    # Add edge if it hasn't been added before
    if edge not in existing_edges:
        full_graph.insert_edge(source_idx, dest_idx, duration)
        edges.append((source_idx, dest_idx, duration))
        existing_edges.add(edge)

# Create MST for reduced network using Kruskal's algorithm
mst_graph = kruskal(full_graph)

# Function to calculate shortest paths and journey durations
def calculate_shortest_paths(graph, stations, station_indices):
    # Initialise a matrix to store the shortest path between all stations
    shortest_path_matrix = pd.DataFrame(float('inf'), index=stations, columns=stations)
    
    # Calculate shortest paths using Dijkstra's algorithm for each station
    for station in stations:
        d, _ = dijkstra(graph, station_indices[station])
        for i in range(len(stations)):
            shortest_path_matrix.loc[station, stations[i]] = d[i]
    
    return shortest_path_matrix

# Shortest path matrices for original and reduced networks
full_shortest_path_matrix = calculate_shortest_paths(full_graph, stations, station_indices)
reduced_shortest_path_matrix = calculate_shortest_paths(mst_graph, stations, station_indices)

# Extract journey durations for both networks
def extract_journey_durations(shortest_path_matrix):
    # Flatten the shortest path matrix and return only valid journey durations (ignoring infinities)
    journey_durations = shortest_path_matrix.values.flatten()
    return journey_durations[journey_durations < float('inf')]

full_journey_durations = extract_journey_durations(full_shortest_path_matrix)
reduced_journey_durations = extract_journey_durations(reduced_shortest_path_matrix)

# Plot histograms side by side
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

# Full Network
sns.histplot(full_journey_durations, bins=50, kde=True, color='blue', ax=axes[0])
axes[0].set_title("Histogram of Journey Durations (Full Network)")
axes[0].set_xlabel("Duration (minutes)")
axes[0].set_ylabel("Frequency")

# Reduced Network
sns.histplot(reduced_journey_durations, bins=50, kde=True, color='green', ax=axes[1])
axes[1].set_title("Histogram of Journey Durations (Reduced Network)")
axes[1].set_xlabel("Duration (minutes)")
axes[1].set_ylabel("Frequency")

# Display the plots
plt.tight_layout()
plt.show()

# Function to find the longest journey and trace the path
def find_longest_journey(shortest_path_matrix, stations, station_indices, graph):
    # Identify the maximum journey duration in the matrix
    max_duration = shortest_path_matrix.values.flatten().max()
    longest_path = None
    for i in range(len(stations)):
        for j in range(len(stations)):
            if shortest_path_matrix.iloc[i, j] == max_duration:
                longest_path = (stations[i], stations[j])

    if longest_path:
        start_station = longest_path[0]
        end_station = longest_path[1]
        
        # Use Dijkstra's algorithm to find the shortest path from start_station to end_station
        start_idx = station_indices[start_station]
        end_idx = station_indices[end_station]
        d, pi = dijkstra(graph, start_idx)
        
        # Reconstruct the path by tracing back from the end station
        path = []
        current = end_idx
        while current is not None:
            path.insert(0, stations[current])  # Insert each station at the beginning to form the correct path
            current = pi[current]  # Move to the predecessor station
        
        return max_duration, path
    return None, None

# Longest journey in original network
max_full_duration, full_longest_path = find_longest_journey(full_shortest_path_matrix, stations, station_indices, full_graph)
print("Longest journey duration in full network:", max_full_duration, "minutes")
print(f"Path for longest journey in full network: {' -> '.join(full_longest_path)}")

# Longest journey in reduced network
max_reduced_duration, reduced_longest_path = find_longest_journey(reduced_shortest_path_matrix, stations, station_indices, mst_graph)
print("Longest journey duration in reduced network:", max_reduced_duration, "minutes")
print(f"Path for longest journey in reduced network: {' -> '.join(reduced_longest_path)}")
