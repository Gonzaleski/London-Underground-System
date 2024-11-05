import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from clrsPython import dijkstra, AdjacencyListGraph

# Load dataset
df = pd.read_excel('London Underground data.xlsx', names=['Lane', 'Source', 'Destination', 'Duration (minutes)'])

# Drop rows where Destination is NaN (assuming they only list stations)
df = df.dropna(subset=['Destination'])

# Resolve duplicate edges by selecting the minimum duration for each (Source, Destination) pair
df_min_durations = df.groupby(['Source', 'Destination'])['Duration (minutes)'].min().reset_index()

# Collect unique stations and map them to indices
stations = pd.concat([df_min_durations['Source'], df_min_durations['Destination']]).unique()
station_indices = {station: idx for idx, station in enumerate(stations)}

# Create a graph
number_of_stations = len(stations)
graph = AdjacencyListGraph(number_of_stations, directed=False, weighted=True)

# Track added edges to prevent duplicates
added_edges = set()

# Insert edges with standardized (minimum) durations
for _, row in df_min_durations.iterrows():
    source_idx = station_indices[row['Source']]
    dest_idx = station_indices[row['Destination']]
    duration = row['Duration (minutes)']
    
    # Check if this edge or its reverse has already been added
    if (source_idx, dest_idx) not in added_edges and (dest_idx, source_idx) not in added_edges:
        graph.insert_edge(source_idx, dest_idx, duration)
        added_edges.add((source_idx, dest_idx))

# Calculate journey durations and store results
journey_durations = []

for station in stations:
    station_idx = station_indices[station]
    d, _ = dijkstra(graph, station_idx)  # 'd' gives shortest distances from this station
    for i in range(station_idx + 1, number_of_stations):  # Avoid duplicates by only storing upper-triangle entries
        if d[i] < float('inf'):  # If reachable
            journey_durations.append(d[i])

# Plot histogram of journey durations
plt.figure(figsize=(10, 6))
plt.hist(journey_durations, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Journey Durations Between Stations')
plt.xlabel('Journey Duration (minutes)')
plt.ylabel('Frequency')
plt.show()

# Find the longest journey duration and its path
longest_duration = max(journey_durations)
longest_pair = None

# Re-run to find the path with the longest journey duration
for station in stations:
    station_idx = station_indices[station]
    d, pi = dijkstra(graph, station_idx)
    for i in range(station_idx + 1, number_of_stations):
        if d[i] == longest_duration:
            start_station = station
            end_station = stations[i]
            # Trace the path back using pi (predecessors)
            path = [end_station]
            current = i
            while pi[current] is not None:
                path.insert(0, stations[pi[current]])
                current = pi[current]
            path.insert(0, start_station)
            longest_pair = (start_station, end_station, path)
            break

# Display longest journey details
print(f"The longest journey duration is {longest_duration} minutes.")
print(f"Path: {' -> '.join(longest_pair[2])}")
