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

# Find the longest journey duration by determining the maximum value in the journey_durations list
longest_duration = max(journey_durations)

# Initialise variable to store the journey pair with the longest duration
longest_pair = None

# Iterate over all stations to calculate the longest journey duration and the corresponding path
for station in stations:
    station_idx = station_indices[station]
    
    # Perform Dijkstra's algorithm to find the shortest distances (d) and predecessors (pi) from the current station
    d, pi = dijkstra(graph, station_idx)
    
    # Iterate over all other stations to find the pair with the longest journey duration
    for i in range(station_idx + 1, number_of_stations):
        if d[i] == longest_duration:
            # Store the start and end stations for the longest journey
            start_station = station
            end_station = stations[i]
            
            # Trace the path back from the destination station using the predecessors (pi) array
            path = []
            current = i
            while current is not None:
                path.insert(0, stations[current])  # Insert each station at the beginning to form the correct path
                current = pi[current]  # Move to the predecessor station
            
            # Store the journey pair (start station, end station, path) as the longest journey
            longest_pair = (start_station, end_station, path)
            break

# Total number of journey durations calculated
total_journey_durations = len(journey_durations)
print(f"Total number of journey durations calculated: {total_journey_durations}")

# Display longest journey details
print(f"The longest journey duration is {longest_duration} minutes.")
print(f"Path: {' -> '.join(longest_pair[2])}")
