import matplotlib.pyplot as plt
import networkx as nx
from clrsPython import dijkstra, AdjacencyListGraph

stations = ['A', 'B', 'C', 'D', 'E']
edges = [('A', 'B', 10), ('A', 'D', 5), ('B', 'C', 1), ('B', 'D', 2), ('C', 'E', 4),
         ('D', 'B', 3), ('D', 'C', 9), ('D', 'E', 2), ('E', 'A', 7), ('E', 'C', 6)]

# Visualize the graph with nodes and edges
G = nx.DiGraph()  # Directed graph

# Add nodes
for station in stations:
    G.add_node(station)

# Add edges with weights
for edge in edges:
    G.add_edge(edge[0], edge[1], weight=edge[2])

# Draw the graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)  # Positions nodes in a visually pleasing manner
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', arrowsize=20)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.title("Tube Network Representation with Travel Times")
plt.show()

# Create a graph using AdjacencyListGraph for shortest path computation
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
        if d[end_index] == float('inf'):
            print(f"  No path to {end_station}")
            continue
        
        path = []
        current = end_index
        while current is not None:
            path.append(stations[current])
            current = pi[current]
        path.reverse()  # To get the path from start to end
        
        print(f"  Path to {end_station}: {' -> '.join(path)}, Distance: {d[end_index]} minutes")
    print("\n")

# Prompt the user for source and destination inputs and compute the shortest path
while True:
    source = input("Enter the source station: ").strip().upper()
    destination = input("Enter the destination station: ").strip().upper()

    if source in stations and destination in stations:
        start_index = stations.index(source)
        d, pi = dijkstra(graph, start_index)
        end_index = stations.index(destination)

        if d[end_index] == float('inf'):
            print(f"No path exists between {source} and {destination}.")
        else:
            path = []
            current = end_index
            while current is not None:
                path.append(stations[current])
                current = pi[current]
            path.reverse()  # Get the path from source to destination
            print(f"Shortest path from {source} to {destination}: {' -> '.join(path)}")
            print(f"Travel time: {d[end_index]} minutes")
    else:
        print("Invalid source or destination station.")

    # Ask the user if they want to try again
    try_again = input("Do you want to find another path? (y/n): ").strip().lower()
    if try_again != 'y':
        print("Thank you for using the Tube Network Path Finder. Goodbye!")
        break
