# Introduction to Algorithms, Fourth edition
# Linda Xiao

#########################################################################
#                                                                       #
# Copyright 2022 Massachusetts Institute of Technology                  #
#                                                                       #
# Permission is hereby granted, free of charge, to any person obtaining #
# a copy of this software and associated documentation files (the       #
# "Software"), to deal in the Software without restriction, including   #
# without limitation the rights to use, copy, modify, merge, publish,   #
# distribute, sublicense, and/or sell copies of the Software, and to    #
# permit persons to whom the Software is furnished to do so, subject to #
# the following conditions:                                             #
#                                                                       #
# The above copyright notice and this permission notice shall be        #
# included in all copies or substantial portions of the Software.       #
#                                                                       #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       #
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    #
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND                 #
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS   #
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN    #
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN     #
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE      #
# SOFTWARE.                                                             #
#                                                                       #
#########################################################################

import numpy as np

"""Base class for MaxHeap and MinHeap."""


class Heap:
    def __init__(self, compare, array, get_key_func=None, dict=None):
        """Initialize a heap with an array and heap size.

        Arguments:
        compare -- comparison function: greater-than for a max-heap, less-than for a min-heap
        array -- array of heap elements.
        get_key_func -- an optional function that returns the key for the
        objects stored. If given, may be a static function in the object class. If
        omitted, then the identity function is used.
        dict -- an optional dictionary mapping objects in the max-heap to indices.
        """
        self.compare = compare
        self.array = array
        # heap_size is the number of elements in the heap that are stored
        # in the array, defaults to all elements in array.
        self.heap_size = len(array)
        if get_key_func is None:
            self.get_key = lambda x: x
        else:
            self.get_key = get_key_func

        # If there is a dictionary mapping objects to indices, initialize it.
        # It should be empty to start.
        self.dict = dict
        if self.dict is not None:
            if len(self.dict) > 0:
                raise RuntimeError("Dictionary argument to constructor must be None or an empty dictionary.")
            for i in range(self.heap_size):
                dict[self.array[i]] = i

    def get_heap_size(self):
        """Return the size of this heap."""
        return self.heap_size

    def is_full(self):
        """Return True if this heap is full, False if not full."""
        return self.heap_size >= len(self.array)

    def get_array(self):
        """Return the array implementation of this heap."""
        return self.array

    def set_heap_size(self, size):
        """Set heap size to given size."""
        self.heap_size = size

    def parent(self, i):
        """Return the index of the parent node of i."""
        return (i-1) // 2

    def left(self, i):
        """Return the index of the left child of i."""
        return 2*i + 1

    def right(self, i):
        """Return the index of the right child of i. """
        return 2*i + 2

    def swap(self, i, j):
        """Swap two elements in an array."""
        if self.dict is not None:
            self.dict[self.array[i]] = j
            self.dict[self.array[j]] = i
        self.array[i], self.array[j] = self.array[j], self.array[i]

    def heapify(self, i):
        """Maintain the heap property.

        Argument:
        i -- index of the element in the heap.
        """
        l = self.left(i)
        r = self.right(i)

        if l < self.heap_size and self.compare(self.get_key(self.array[l]), self.get_key(self.array[i])):
            swap_with = l
        else:
            swap_with = i

        if r < self.heap_size and self.compare(self.get_key(self.array[r]), self.get_key(self.array[swap_with])):
            swap_with = r

        if swap_with != i:
            self.swap(i, swap_with)
            self.heapify(swap_with)

    def build_heap(self):
        """Convert a list or numpy array into a heap."""
        # Run heapify on all roots of the tree, from ((heap_size // 2) - 1) to 0.
        self.heap_size = len(self.array)
        for i in range((len(self.array) // 2) - 1, -1, -1):
            self.heapify(i)

    def __str__(self):
        """Return the heap as an array."""
        return ", ".join(str(x) for x in self.array[:self.heap_size])

    def is_heap(self):
        """Verify that the array or list represents a heap."""
        # From root node to last internal node.
        for i in range(0, self.heap_size // 2):
            # Check the left child.
            if self.compare(self.get_key(self.array[self.left(i)]), self.get_key(self.array[i])):
                return False
            # If there is a right child, check it.
            if self.right(i) < self.heap_size and \
                    self.compare(self.get_key(self.array[self.right(i)]), self.get_key(self.array[i])):
                return False

        return True

"""Base class for MaxHeapPriorityQueue and MinHeapPriorityQueue."""

class HeapPriorityQueue:

    def __init__(self, compare, temp_insert_value, get_key_func, set_key_func=None):
        """Initialize minimum priority queue implemented with a heap.

        Arguments:
        compare -- comparison function: greater-than for a max-heap priority queue,
        less-than for a min-heap priority queue
        temp_insert_value -- temporary value given to objects upon insertion, then
        changed to the actual value of the object
        get_key_func -- required function that returns the key for the
        objects stored. May be a static function in the object class.
        set_key_func -- optional function that sets the key for the objects
        stored. May be a static function in the object class.
        """

        # Dictionary to map array objects to array indices.
        # Mapping might not take worst-case time O(1).
        self.dict = {}

        # self.get_key function used to get key of object.
        self.get_key = get_key_func

        # self.set_key function used to set key of object.
        self.set_key = set_key_func

        # Initialize to empty heap.
        self.heap = Heap(compare, [], self.get_key, self.dict)
        self.compare = compare
        self.temp_insert_value = temp_insert_value

    def get_heap(self):
        """Return heap, used in testing."""
        return self.heap

    def get_size(self):
        """Return the number of objects in the priority queue."""
        return self.heap.get_heap_size()

    def top_of_heap(self):
        """Return the object at the top of the heap."""
        if self.heap.get_heap_size() <= 0:  # error if heap is empty
            raise RuntimeError("Heap underflow.")
        return self.heap.get_array()[0]

    def extract_top(self):
        """Return and delete the top element in a heap."""
        top = self.top_of_heap()

        # Move the last object in heap to the root position.
        last_obj = self.heap.get_array()[self.heap.get_heap_size()-1]
        self.heap.get_array()[0] = last_obj
        self.dict[last_obj] = 0

        # Remove the old top object.
        del self.dict[top]
        self.heap.set_heap_size(self.heap.get_heap_size() - 1)

        # Restore the heap property.
        self.heap.heapify(0)

        # Return the top item, which was extracted.
        return top

    def update_key(self, x, k):
        """Update the key of object x to value k.
        Assumption: The caller has already verified that the new value is OK.

        Arguments:
        x -- object whose key has been changed
        k -- new key of x
        """
        if self.set_key is not None:
            self.set_key(x, k)

        # Get the index from the dictionary.
        i = self.dict[x]

        # Compare the value with parents up the heap to place in the correct position.
        while i > 0 and \
                self.compare(self.get_key(self.heap.get_array()[i]),
                             self.get_key(self.heap.get_array()[self.heap.parent(i)])):
            # Exchange positions and continue if the element should head toward the root.
            self.heap.swap(i, self.heap.parent(i))
            i = self.heap.parent(i)

    def insert(self, x):
        """Insert x into the heap.  Grows the heap as necessary.

        Arguments:
        x -- object to insert
        """

        # Increment the heap size.
        self.heap.set_heap_size(self.heap.get_heap_size() + 1)

        k = self.get_key(x)

        if self.set_key is not None:
            self.set_key(x, self.temp_insert_value)

        # Insert x into the array and the dictionary.
        self.heap.get_array().insert(self.heap.get_heap_size() - 1, x)
        self.dict[x] = self.heap.get_heap_size() - 1

        # Maintain the heap property.
        self.update_key(x, k)

    def is_heap(self):
        """Verify that the array or list represents a heap."""
        return self.heap.is_heap()

    def __str__(self):
        """Return the heap as an array."""
        return str(self.heap)



class MinHeapPriorityQueue(HeapPriorityQueue):

    def __init__(self, get_key_func, set_key_func=None):
        """Initialize a minimum priority queue implemented with a heap.

        Arguments:
        get_key_func -- required function that returns the key for the
        objects stored. May be a static function in the object class.
        set_key_func -- optional function that sets the key for the objects
        stored. May be a static function in the object class.
        """
        HeapPriorityQueue.__init__(self, lambda x, y: x < y, float('inf'), get_key_func, set_key_func)

    def minimum(self):
        """Return the object with the minimum key in a heap."""
        return self.top_of_heap()

    def extract_min(self):
        """Return and delete the object with the minimum value in a heap."""
        return self.extract_top()

    def decrease_key(self, x, k):
        """Decrease the key of object x to value k.  Error if k is greater than x's current key.
            Update the heap structure appropriately.

        Arguments:
        x -- object whose key has been decreased
        k -- new key of x
        """

        if k > self.get_key(x):
            raise RuntimeError("Error in decrease_key: new key " + str(k)
                               + " is greater than current key " + str(x.get_key()))

        # Make the changes in the heap.
        self.update_key(x, k)

    def insert(self, x):
        """Insert x into the min heap.  Grows the heap as necessary."""
        HeapPriorityQueue.insert(self, x)

def initialize_single_source(G, s):
	"""Initialize distance and predecessor values for vertices in graph. 

	Arguments:
	G -- a graph
	s -- index of the source vertex for shortest paths
	"""
	# Initialize d and pred.
	card_V = G.get_card_V()
	# d[v] is an upper bound on the weight of a shortest path from source s to v.
	d = [float('inf')] * card_V
	pi = [None] * card_V
	d[s] = 0
	return d, pi


def relax(u, v, w, d, pi, relax_func=None):
	"""Improve the shortest path to v found so far.

	Arguments:
	u, v -- relaxing the edge (u, v))
	w -- weight of the edge (u, v)
	d -- upper bound on the weight of a shortest path from source s to v
	pi -- list of predecessors
	relax_func -- function called after relaxing an edge, default is to do nothing
	"""
	if d[v] > d[u] + w:
		d[v] = d[u] + w  # reduce v's shortest path weight
		pi[v] = u        # update v's predecessor predecessor
		if relax_func is not None:
			relax_func(v)


def dijkstra(G, s):
	"""Solve single-source shortest-paths problem with no negative-weight edges.

	Arguments:
	G -- a directed, weighted graph
	s -- index of source vertex
	Assumption:
	All weights are nonnegative

	Returns:
	d -- distances from source vertex s
	pi -- predecessors
	"""

	card_V = G.get_card_V()

	d, pi = initialize_single_source(G, s)

	# Key function for the priority queue is distance.
	queue = MinHeapPriorityQueue(lambda u: d[u])
	for u in range(card_V):
		queue.insert(u)

	while queue.get_size() > 0:  # while the priority queue is not empty
		u = queue.extract_min()  # extract a vertex with the minimum distance

		# Relax each edge and update d and pi.
		for edge in G.get_adj_list(u):
			v = edge.get_v()
			# Upon each relaxation, decrease the key in the priority queue.
			relax(u, v, edge.get_weight(), d, pi,
					lambda v: queue.decrease_key(v, d[u] + edge.get_weight()))

	return d, pi

class LinkedListNode:

	def __init__(self, data):
		"""Initialize a node of a circular doubly linked list with a sentinel with the given data."""
		self.prev = None
		self.next = None
		self.data = data

	def get_data(self):
		"""Return data."""
		return self.data

	def __str__(self):
		"""Return data as a string."""
		return str(self.data)


class DLLSentinel:

	def __init__(self, get_key_func=None):
		"""Initialize the sentinel of a circular doubly linked list with a sentinel.

		Arguments:
		get_key_func -- an optional function that returns the key for the
		objects stored. May be a static function in the object class. If 
		omitted, then identity function is used.
		"""
		self.sentinel = LinkedListNode(None)  # holds None as data
		self.sentinel.next = self.sentinel  # the sentinel points to itself in an empty list
		self.sentinel.prev = self.sentinel

		if get_key_func is None:
			self.get_key = lambda x: x   # return self
		else:
			self.get_key = get_key_func  # return key defined by user

	def search(self, k):
		"""Search a circular doubly linked list with a sentinel for a node with key k.

		Returns:
		x -- node with key k or None if not found
		"""
		x = self.sentinel.next
		# Go down the list until key k is found.
		# Need to test for the sentinel to avoid calling get_key(None) when x is the sentinel.
		while x is not self.sentinel and self.get_key(x.data) != k:
			x = x.next

		if x is self.sentinel:  # went through the whole list?
			return None         # yes, so no node had key
		else:
			return x            # found it!

	def insert(self, data, y):
		"""Insert a node with data after node y.  Return the new node."""
		x = LinkedListNode(data)   # construct a node x
		x.next = y.next            # x's successor is y's successor
		x.prev = y                 # x's predecessor is y
		y.next.prev = x            # x comes before y's successor
		y.next = x                 # x is now y's successor
		return x

	def prepend(self, data):
		"""Insert a node with data as the head of a circular doubly linked list with a sentinel.
		Return the new node."""
		return self.insert(data, self.sentinel)

	def append(self, data):
		"""Append a node with data to the tail of a circular doubly linked list with a sentinel.
		Return the new node."""
		return self.insert(data, self.sentinel.prev)

	def delete(self, x):
		"""Remove a node x from the a circular doubly linked list with a sentinel.

		Assumption:
		x is a node in the linked list. 
		"""
		if x is self.sentinel:
			raise RuntimeError("Cannot delete sentinel.")
		x.prev.next = x.next  # point prev to next
		x.next.prev = x.prev  # point next to prev

	def delete_all(self):
		"""Delete all nodes in a circular doubly linked list with a sentinel."""
		self.sentinel.next = self.sentinel
		self.sentinel.prev = self.sentinel

	def iterator(self):
		"""Iterator from the head of a circular doubly linked list with a sentinel."""
		x = self.sentinel.next
		while x is not self.sentinel:
			yield x.get_data()
			x = x.next

	def copy(self):
		"""Return a copy of this circular doubly linked list with a sentinel."""
		c = DLLSentinel(self.get_key)      # c is the copy
		x = self.sentinel.next
		while x != self.sentinel:
			c.append(x.data)   # append a node with x's data to c
			x = x.next
		return c

	def __str__(self):
		"""Return this circular doubly linked list with a sentinel formatted as a list."""
		x = self.sentinel.next
		string = "["
		while x != self.sentinel:
			string += (str(x) + ", ")
			x = x.next
		string += (str(x) + "]")
		return string

class Edge:

	def __init__(self, v, weight=None):
		"""Initialize an edge to add to the adjacency list of another vertex.

		Arguments:
		v -- the other vertex that the edge is incident on
		weight -- optional parameter for weighted graphs
		"""
		self.v = v
		if weight is not None:
			self.weight = weight

	def get_v(self):
		"""Return the vertex index."""
		return self.v

	def get_weight(self):
		"""Return the weight of this edge."""
		return self.weight

	def set_weight(self, weight):
		"""Set the weight of this edge."""
		self.weight = weight

	def __str__(self):
		"""String version of the vertex with optional weight in parentheses."""
		return self.strmap(lambda v: v)

	def strmap(self, mapping_func):
		"""String version of the vertex with optional weight in parentheses.
		Vertex numbers are mapped according to a mapping function."""
		string = str(mapping_func(self.v))
		if hasattr(self, "weight"):
			string += " (" + str(self.weight) + ")"
		return string
	
class AdjacencyMatrixGraph:

	def __init__(self, card_V, directed=True, weighted=False):
		"""Initialize a graph implemented by an adjacency matrix. 

		Arguments:
		card_V -- number of vertices in this graph
		directed -- boolean whether or not graph is directed
		weighted -- boolean whether or not edges are weighted
		"""
		self.directed = directed
		if weighted:
			# For weighted graphs, adj_matrix will default to infinity for no edge.
			self.adj_matrix = np.ndarray((card_V, card_V))
			self.no_edge = float('inf')
			self.adj_matrix.fill(self.no_edge)
		else:
			# For unweighted graphs, adj_matrix will default to 0 for no edge.
			self.adj_matrix = np.zeros(shape=(card_V, card_V), dtype=int)
			self.no_edge = 0
		self.card_V = card_V
		self.weighted = weighted
		self.card_E = 0

	def get_card_V(self):
		"""Return the number of vertices in this graph."""
		return self.card_V

	def get_card_E(self):
		"""Return the number of edges in this graph."""
		return self.card_E

	def get_adj_matrix(self):
		"""Return the adjacency matrix for this graph."""
		return self.adj_matrix

	def is_directed(self):
		"""Return a boolean indicating whether this graph is directed."""
		return self.directed

	def is_weighted(self):
		"""Return a boolean indicating whether this graph is weighted."""
		return self.weighted

	def insert_edge(self, u, v, weight=None):
		"""Insert an edge between vertices u and v.

		Arguments:
		u -- index of vertex u
		v -- index of vertex v
		"""
		# Check whether a weight is missing, or whether a weight is given in an unweighted graph.
		if self.weighted:
			if weight is None:
				raise RuntimeError("Inserting unweighted edge (" + str(u) + ", " + str(v) + ") in weighted graph.")
		else:  # unweighted
			if weight is not None:
				raise RuntimeError("Inserting weighted edge (" + str(u) + ", " + str(v) + ") in unweighted graph.")
			weight = 1  # to indicate the presence of the edge

		# An undirected graph cannot have self-loops.
		if not self.directed and u == v:
			raise RuntimeError("Cannot insert self-loop (" + str(u) + ", " + str(v) + ") into undirected graph")

		# Cannot insert multiple edges between two vertices.
		if self.has_edge(u, v):
			raise RuntimeError("An edge (" + str(u) + ", " + str(v) + ") already exists.")
		self.adj_matrix[u, v] = weight
		self.card_E += 1

		# If undirected, insert edge from v to u.
		if not self.directed:
			if self.has_edge(v, u):
				raise RuntimeError("An edge (" + str(v) + ", " + str(u) + ") already exists.")
			self.adj_matrix[v, u] = weight

	def has_edge(self, u, v):
		"""Return True if edge (u, v) is in this graph, False otherwise."""
		return self.adj_matrix[u, v] != self.no_edge

	def delete_edge(self, u, v, delete_undirected=True):
		"""Delete edge (u, v) if it exists.  No error if it does not exist.
			Delete both directions if the graph is undirected and delete_undirected is True."""
		if self.adj_matrix[u, v] != self.no_edge:
			self.adj_matrix[u, v] = self.no_edge
			self.card_E -= 1
		if not self.directed and delete_undirected:
			self.adj_matrix[v, u] = self.no_edge

	def copy(self):
		"""Return a copy of this graph."""
		c = AdjacencyMatrixGraph(self.card_V, self.directed, self.weighted)
		c.adj_matrix = self.adj_matrix.copy()  # deep copy
		c.card_E = self.card_E
		return c

	def get_edge_list(self):
		"""Return a Python list containing the edges of this graph."""
		edge_list = []
		for u in range(self.card_V):
			if self.directed:
				lowest_v = 0
			else:
				lowest_v = u + 1
			for v in range(lowest_v, self.card_V):
				if self.adj_matrix[u, v] != self.no_edge:
					edge_list.append((u, v))
		return edge_list

	def __str__(self):
		"""Return the adjacency matrix."""
		return str(self.adj_matrix)

class AdjacencyListGraph:

	def __init__(self, card_V, directed=True, weighted=False):
		"""Initialize a graph implemented by an adjacency list. Vertices are
		numbered from 0, so that adj_list[i] corresponds to adjacency list of vertex i.

		Arguments:
		card_V -- number of vertices in this graph
		directed -- boolean indicating whether the graph is directed
		weighted -- boolean indicating whether edges are weighted
		"""
		self.directed = directed
		self.weighted = weighted
		self.adj_lists = [None] * card_V
		for i in range(card_V):
			# Each adjacency list is implemented as a linked list.
			self.adj_lists[i] = DLLSentinel(get_key_func=Edge.get_v)  # will be a list of Edge objects
		self.card_V = card_V
		self.card_E = 0

	def get_card_V(self):
		"""Return the number of vertices in this graph."""
		return self.card_V

	def get_card_E(self):
		"""Return the number of edges in this graph."""
		return self.card_E

	def get_adj_lists(self):
		"""Return the adjacency lists of all the vertices in this graph."""
		return self.adj_lists

	def get_adj_list(self, u):
		"""Return an iterator for the adjacency list of vertex u."""
		return self.adj_lists[u].iterator()

	def is_directed(self):
		"""Return a boolean indicating whether this graph is directed."""
		return self.directed

	def is_weighted(self):
		"""Return a boolean indicating whether this graph is weighted."""
		return self.weighted

	def insert_edge(self, u, v, weight=None):
		"""Insert an edge between vertices u and v.

		Arguments:
		u -- index of vertex u
		v -- index of vertex v
		"""
		# Check whether a weight is missing, or whether a weight is given in an unweighted graph.
		if self.weighted:
			if weight is None:
				raise RuntimeError("Inserting unweighted edge (" + str(u) + ", " + str(v) + ") in weighted graph.") 
		else:  # unweighted
			if weight is not None:
				raise RuntimeError("Inserting weighted edge (" + str(u) + ", " + str(v) + ") in unweighted graph.")

		# An undirected graph cannot have self-loops.
		if not self.directed and u == v:
			raise RuntimeError("Cannot insert self-loop (" + str(u) + ", " + str(v) + ") into undirected graph")

		# Cannot insert multiple edges between two vertices.
		if self.has_edge(u, v):
			raise RuntimeError("An edge (" + str(u) + ", " + str(v) + ") already exists.")
		self.adj_lists[u].append(Edge(v, weight))
		self.card_E += 1

		# If this graph is undirected, insert an edge from v to u.
		if not self.directed:
			# Cannot insert multiple edges between two vertices.
			if self.has_edge(v, u):
				raise RuntimeError("An edge (" + str(v) + ", " + str(u) + ") already exists.")
			self.adj_lists[v].append(Edge(u, weight))

	def find_edge(self, u, v):
		"""Return the edge object for edge (u, v) if (u, v) is in this graph, None otherwise."""
		edge = self.adj_lists[u].search(v)
		if edge is None:
			return None
		else:
			return edge.data

	def has_edge(self, u, v):
		"""Return True if edge (u, v) is in this graph, False otherwise."""
		return self.find_edge(u, v) is not None

	def delete_edge(self, u, v, delete_undirected=True):
		"""Delete edge (u, v) if it exists.  No error if it does not exist.
			Delete both directions if the graph is undirected and delete_undirected is True."""
		edge = self.adj_lists[u].search(v)
		if edge is not None:
			self.adj_lists[u].delete(edge)
			self.card_E -= 1

		if not self.directed and delete_undirected:
			edge = self.adj_lists[v].search(u)
			if edge is not None:
				self.adj_lists[v].delete(edge)

	def copy(self):
		"""Return a copy of this graph."""
		copy = AdjacencyListGraph(self.card_V, self.directed, self.weighted)
		copy.card_E = self.card_E
		for u in range(self.card_V):
			copy.adj_lists[u] = self.adj_lists[u].copy()
		return copy

	def get_edge_list(self):
		"""Return a Python list containing the edges of this graph."""
		edge_list = []
		for u in range(self.card_V):
			adj_list = self.get_adj_list(u)
			for edge in adj_list:
				v = edge.get_v()
				if self.directed or u < v:
					edge_list.append((u, v))
		return edge_list

	def transpose(self):
		"""Return the transpose of this graph."""
		xpose = AdjacencyListGraph(self.card_V, self.directed, self.weighted)
		for u in range(self.card_V):
			adj_list = self.get_adj_list(u)
			for edge in adj_list:
				v = edge.get_v()
				if self.weighted:
					weight = edge.get_weight()
				else:
					weight = None
				xpose.insert_edge(v, u, weight)
		return xpose

	def adjacency_matrix(self):
		"""Return the adjacency-matrix representation of this graph."""
		card_V = self.get_card_V()
		matrix = AdjacencyMatrixGraph(card_V, self.directed, self.weighted)
		weight_func = lambda edge: edge.get_weight() if self.weighted else None
		for u in range(card_V):
			adj_list = self.get_adj_list(u)
			for edge in adj_list:
				matrix.insert_edge(u, edge.get_v(), weight_func(edge))
		return matrix

	def __str__(self):
		"""Return the adjacency lists formatted as a string."""
		return self.strmap()

	def strmap(self, mapping_func=None):
		"""Return the adjacency lists formatted as a string, but mapping vertex numbers
		by a mapping function.  If mapping_func is None, then do not map."""
		if mapping_func is None:
			mapping_func = lambda i: i

		result = ""
		for i in range(self.card_V):
			result += str(mapping_func(i)) + ": "
			for edge in self.get_adj_list(i):
				result += edge.strmap(mapping_func) + " "
			result += "\n"
		return result