import pandas as pd
import matplotlib.pyplot as plt
from clrsPython import dijkstra, AdjacencyListGraph

# Load dataset
df = pd.read_excel('London Underground data.xlsx', names=['Lane', 'Source', 'Destination', 'Duration (minutes)'])

# Drop rows where Destination is NaN
df = df.dropna(subset=['Destination'])

# Collect unique stations and map them to indices
stations = pd.concat([df['Source'], df['Destination']]).unique()
station_indices = {station: idx for idx, station in enumerate(stations)}

# Create a graph where each edge weight is set to 1 to represent a stop
number_of_stations = len(stations)
graph = AdjacencyListGraph(number_of_stations, directed=False, weighted=True)

# Track added edges to prevent duplicates
added_edges = set()

# Iterate through rows and create edges between stations
for _, row in df.iterrows():
    source_idx = station_indices[row['Source']]
    dest_idx = station_indices[row['Destination']]
    
    # Insert edge with weight of 1 (each edge represents one stop)
    if (source_idx, dest_idx) not in added_edges and (dest_idx, source_idx) not in added_edges:
        graph.insert_edge(source_idx, dest_idx, 1)
        added_edges.add((source_idx, dest_idx))

# Calculate the number of stops for journeys and store results
journey_stops = []

# For each station, compute the shortest path to all other stations using Dijkstra's algorithm
for station in stations:
    station_idx = station_indices[station]
    d, _ = dijkstra(graph, station_idx)  # 'd' gives shortest paths in terms of stops from this station
    for i in range(station_idx + 1, number_of_stations):  # Avoid duplicates
        if d[i] < float('inf'):  # If reachable
            journey_stops.append(d[i])

# Plot histogram of journey durations by stops
plt.figure(figsize=(10, 6))
plt.hist(journey_stops, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Journey Durations by Number of Stops')
plt.xlabel('Number of Stops')
plt.ylabel('Frequency')
plt.show()

# Display the total number of journey durations calculated
print(f"Total number of journey durations calculated: {len(journey_stops)}")

# Find the longest journey in terms of stops and its path
longest_stops = max(journey_stops)
longest_path = None

# Find the path with the most stops
for station in stations:
    station_idx = station_indices[station]
    d, pi = dijkstra(graph, station_idx)  # Use Dijkstra's algorithm again to get the shortest paths and predecessors
    for i in range(station_idx + 1, number_of_stations):
        if d[i] == longest_stops:  # Check if the journey matches the longest stops found earlier
            start_station = station
            end_station = stations[i]
            # Trace the path back using pi (predecessors)
            path = []
            current = i
            while current is not None:  # Reconstruct the path from destination to start using predecessors
                path.insert(0, stations[current])  # Add stations to the front of the path
                current = pi[current]
            # Remove the start station if it's already in the path
            if path[0] == start_station:
                path = path[1:]
            path.insert(0, start_station)  # Add start station only once
            longest_path = (start_station, end_station, path)
            break

# Display longest journey details by stops
print(f"The longest journey in terms of stops is {longest_stops} stops.")
print(f"Path: {' -> '.join(longest_path[2])}")
