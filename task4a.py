import pandas as pd
from clrsPython import AdjacencyListGraph, kruskal

# Load the dataset
df = pd.read_excel('London Underground data.xlsx', names=['Lane', 'Source', 'Destination', 'Duration (minutes)'])
df = df.dropna(subset=['Destination'])  # Remove rows without destinations

# Collect unique stations and map them to indices
stations = pd.concat([df['Source'], df['Destination']]).unique()
station_indices = {station: idx for idx, station in enumerate(stations)}

# Create the graph and add edges with duration as weights
number_of_stations = len(stations)
graph = AdjacencyListGraph(number_of_stations, directed=False, weighted=True)

# Dictionary to store the minimum duration for each unique edge
edge_durations = {}

# Populate edge_durations with the minimum duration for each unique station pair
for _, row in df.iterrows():
    source_idx = station_indices[row['Source']]
    dest_idx = station_indices[row['Destination']]
    duration = row['Duration (minutes)'] if not pd.isna(row['Duration (minutes)']) else 1
    
    # Sort indices to treat (source, dest) and (dest, source) as the same edge
    edge = (min(source_idx, dest_idx), max(source_idx, dest_idx))
    
    # Only keep the minimum duration for each unique edge
    if edge not in edge_durations or duration < edge_durations[edge]:
        edge_durations[edge] = duration

# Insert edges into the graph and prepare the edges list for Kruskal's algorithm
edges = []
for (source_idx, dest_idx), duration in edge_durations.items():
    graph.insert_edge(source_idx, dest_idx, duration)
    edges.append((source_idx, dest_idx, duration))  # Store for Kruskal's algorithm

# Get the MST using Kruskal's algorithm
mst_graph = kruskal(graph)

# Extract edges from the MST graph
mst_edges = set()
for u in range(mst_graph.get_card_V()):
    for edge in mst_graph.get_adj_list(u):
        v = edge.get_v()
        # Use a set to ensure each undirected edge is only added once
        if (min(u, v), max(u, v)) not in mst_edges:
            mst_edges.add((min(u, v), max(u, v)))

# Identify edges not in the MST (these can be closed)
original_edges_set = {(min(u, v), max(u, v)) for u, v, _ in edges}
closable_edges = original_edges_set - mst_edges

# List the closable line sections
print("The following line sections can be closed without losing connectivity:")
for u, v in closable_edges:
    print(f"{stations[u]} -- {stations[v]}")

