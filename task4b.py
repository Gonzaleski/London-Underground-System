import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from clrsPython import AdjacencyListGraph, dijkstra, kruskal

# Load dataset
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

# Shortest path matrix for original network
full_shortest_path_matrix = pd.DataFrame(float('inf'), index=stations, columns=stations)
for station in stations:
    d, _ = dijkstra(full_graph, station_indices[station])
    for i in range(len(stations)):
        full_shortest_path_matrix.loc[station, stations[i]] = d[i]

# Shortest path matrix for reduced (MST) network
reduced_shortest_path_matrix = pd.DataFrame(float('inf'), index=stations, columns=stations)
for station in stations:
    d, _ = dijkstra(mst_graph, station_indices[station])
    for i in range(len(stations)):
        reduced_shortest_path_matrix.loc[station, stations[i]] = d[i]

# Extract journey durations and plot histogram for original network
full_journey_durations = full_shortest_path_matrix.values.flatten()
full_journey_durations = full_journey_durations[full_journey_durations < float('inf')]
plt.figure(figsize=(12, 6))
sns.histplot(full_journey_durations, bins=50, kde=True, color='blue')
plt.title("Histogram of Journey Durations (Full Network)")
plt.xlabel("Duration (minutes)")
plt.ylabel("Frequency")
plt.show()

# Extract journey durations and plot histogram for reduced network
reduced_journey_durations = reduced_shortest_path_matrix.values.flatten()
reduced_journey_durations = reduced_journey_durations[reduced_journey_durations < float('inf')]
plt.figure(figsize=(12, 6))
sns.histplot(reduced_journey_durations, bins=50, kde=True, color='green')
plt.title("Histogram of Journey Durations (Reduced Network)")
plt.xlabel("Duration (minutes)")
plt.ylabel("Frequency")
plt.show()

# Find longest journey in original network
max_full_duration = full_journey_durations.max()
full_longest_path = None
for i in range(len(stations)):
    for j in range(len(stations)):
        if full_shortest_path_matrix.iloc[i, j] == max_full_duration:
            full_longest_path = (stations[i], stations[j])

# Find longest journey in reduced network
max_reduced_duration = reduced_journey_durations.max()
reduced_longest_path = None
for i in range(len(stations)):
    for j in range(len(stations)):
        if reduced_shortest_path_matrix.iloc[i, j] == max_reduced_duration:
            reduced_longest_path = (stations[i], stations[j])

print("Longest journey duration in full network:", max_full_duration, "minutes")
print("Path for longest journey in full network:", full_longest_path)

print("Longest journey duration in reduced network:", max_reduced_duration, "minutes")
print("Path for longest journey in reduced network:", reduced_longest_path)
