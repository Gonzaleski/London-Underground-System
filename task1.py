import pandas as pd
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

# Shortest path determination using the Dijkstra's algorithm
for station in stations:
    d, pi = dijkstra(graph, stations.index(station))
    print("Shortest path from ", station, " to:")
    for i in range(number_of_stations):
        print(stations[i] + ": distance = " + str(d[i]) + ",\tpredecessor = " + ("None" if pi[i] is None else stations[pi[i]]))
    print("\n")

import seaborn as sns
import matplotlib.pyplot as plt

# Initialize a distance matrix with infinity for shortest paths
shortest_path_matrix = pd.DataFrame(float('inf'), index=stations, columns=stations)

# Populate the shortest path matrix using Dijkstra's algorithm
for station in stations:
    d, _ = dijkstra(graph, stations.index(station))  # 'd' gives shortest distances from this station
    for i in range(len(stations)):
        shortest_path_matrix.loc[station, stations[i]] = d[i]  # Update the matrix with shortest path distances

print(shortest_path_matrix)

# Plotting the correlation map
plt.figure(figsize=(8, 6))
sns.heatmap(shortest_path_matrix, annot=True, vmin=-1, vmax=1)
plt.title("Correlation Map of Shortest Path Travel Times Between Stations")
plt.show()