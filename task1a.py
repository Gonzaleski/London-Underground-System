import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from clrsPython import dijkstra, AdjacencyListGraph

stations = ['A', 'B', 'C', 'D', 'E']
edges = [('A', 'B', 10), ('A', 'D', 5), ('B', 'C', 1), ('B', 'D', 2), ('C', 'E', 4),
        ('D', 'B', 3), ('D', 'C', 9), ('D', 'E', 2), ('E', 'A', 7), ('E', 'C', 6)]

# Create an artificial dataset for a tube network with 5 stations (A, B, C, D, E)
data = {
    "source": [edge[0] for edge in edges],
    "destination": [edge[1] for edge in edges],
    "estimated time": [edge[2] for edge in edges]  # in minutes
}

# Create DataFrame
df = pd.DataFrame(data)
print(df, end="\n\n")

# Create a graph
number_of_stations = len(stations)
graph = AdjacencyListGraph(number_of_stations, True, True)
for edge in edges:
    graph.insert_edge(stations.index(edge[0]), stations.index(edge[1]), edge[2])

# Shortest path determination using Dijkstra's algorithm
for start_station in stations:
    start_index = stations.index(start_station)
    d, pi = dijkstra(graph, start_index)
    
    print(f"Shortest paths from {start_station}:")
    for end_index in range(number_of_stations):
        end_station = stations[end_index]
        # Reconstruct the path from start_station to end_station
        if d[end_index] == float('inf'):
            print(f"  No path to {end_station}")
            continue
        
        path = []
        current = end_index
        while current is not None:
            path.append(stations[current])
            current = pi[current]
        path.reverse()  # To get the path from start to end
        
        # Print path and distance
        print(f"  Path to {end_station}: {' -> '.join(path)}, Distance: {d[end_index]} minutes")
    print("\n")

# Initialize a distance matrix with infinity for shortest paths
shortest_path_matrix = pd.DataFrame(float('inf'), index=stations, columns=stations)

# Populate the shortest path matrix using Dijkstra's algorithm
for station in stations:
    d, _ = dijkstra(graph, stations.index(station))  # 'd' gives shortest distances from this station
    for i in range(len(stations)):
        shortest_path_matrix.loc[station, stations[i]] = d[i]  # Update the matrix with shortest path distances

print("Shortest path travel time matrix:\n", shortest_path_matrix)

# Plotting the travel time heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(shortest_path_matrix, annot=True, cmap="YlGnBu")  # Use a color map suitable for distances
plt.title("Shortest Path Travel Times Between Stations")
plt.xlabel("Destination")
plt.ylabel("Source")
plt.show()
