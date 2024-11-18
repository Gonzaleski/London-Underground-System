import matplotlib.pyplot as plt
try:
    import networkx as nx
except ImportError:
    print("To see the graph, please install the 'networkx' library using pip.")
    nx = None

from clrsPython import dijkstra, AdjacencyListGraph

# Define the stations and edges of the artificial network
stations = ['A', 'B', 'C', 'D', 'E']
edges = [('A', 'B', 10), ('A', 'D', 5), ('B', 'C', 1), ('B', 'D', 3), ('C', 'E', 6),
         ('D', 'B', 3), ('D', 'C', 9), ('D', 'E', 2), ('E', 'A', 7), ('E', 'C', 6)]

# Visualise the graph with nodes and edges
if nx:
    G = nx.DiGraph()  # Create a directed graph

    # Add nodes to the graph
    for station in stations:
        G.add_node(station)

    # Add edges with weights to the graph
    for edge in edges:
        G.add_edge(edge[0], edge[1], weight=edge[2])

    # Draw the graph with positions, labels, and weights
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed = 40)  # Use spring layout for better visualisation
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, 'weight')  # Extract edge weights
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')  # Display edge weights

    plt.title("Tube Network Representation with Travel Times")  # Title for the graph
    plt.show()

# Create a graph using AdjacencyListGraph for shortest path computation
number_of_stations = len(stations)  # Total number of stations in the graph
graph = AdjacencyListGraph(number_of_stations, True, True)  # Create a directed, weighted adjacency list graph
for edge in edges:
    graph.insert_edge(stations.index(edge[0]), stations.index(edge[1]), edge[2])  # Add edges with weights

# Interactive section to allow users to query shortest paths
while True:
    # Prompt user for source and destination stations
    source = input("Enter the source station: ").strip().upper()
    destination = input("Enter the destination station: ").strip().upper()

    if source in stations and destination in stations:  # Validate station inputs
        start_index = stations.index(source)  # Index of the source station
        d, pi = dijkstra(graph, start_index)  # Get shortest path distances and predecessors
        end_index = stations.index(destination)  # Index of the destination station

        if d[end_index] == float('inf'):  # Check if no path exists
            print(f"No path exists between {source} and {destination}.")
        else:
            # Trace the shortest path from source to destination
            path = []
            current = end_index
            while current is not None:
                path.append(stations[current])
                current = pi[current]
            path.reverse()  # Reverse to display the path in correct order
            print(f"Shortest path from {source} to {destination}: {' -> '.join(path)}")
            print(f"Travel time: {d[end_index]} minutes")
    else:
        print("Invalid source or destination station.")  # Handle invalid input

    # Ask the user if they want to try again
    try_again = input("Do you want to find another path? (put y if yes): ").strip().lower()
    if try_again != 'y':  # Exit if the user doesn't want to continue
        print("Thank you for using the Tube Network Path Finder. Goodbye!")
        break