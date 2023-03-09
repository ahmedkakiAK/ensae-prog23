import sys

class Graph:
    """
    A class representing graphs as adjacency lists and implementing various algorithms on the graphs. Graphs in the class are not oriented. 
    Attributes: 
    -----------
    nodes: NodeType
        A list of nodes. Nodes can be of any immutable type, e.g., integer, float, or string.
        We will usually use a list of integers 1, ..., n.
    graph: dict
        A dictionnary that contains the adjacency list of each node in the form
        graph[node] = [(neighbor1, p1, d1), (neighbor1, p1, d1), ...]
        where p1 is the minimal power on the edge (node, neighbor1) and d1 is the distance on the edge
    nb_nodes: int
        The number of nodes.
    nb_edges: int
        The number of edges. 
    """

    def __init__(self, nodes=[]):
        """
        Initializes the graph with a set of nodes, and no edges. 
        Parameters: 
        -----------
        nodes: list, optional
            A list of nodes. Default is empty.
        """
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
    

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output
    
    def add_edge(self, node1, node2, power_min, dist=1):
        """
        Adds an edge to the graph. Graphs are not oriented, hence an edge is added to the adjacency list of both end nodes. 

        Parameters: 
        -----------
        node1: NodeType
            First end (node) of the edge
        node2: NodeType
            Second end (node) of the edge
        power_min: numeric (int or float)
            Minimum power on this edge
        dist: numeric (int or float), optional
            Distance between node1 and node2 on the edge. Default is 1.
        """
        if node1 not in self.graph:
            self.graph[node1] = []
            self.nb_nodes += 1
            self.nodes.append(node1)
        if node2 not in self.graph:
            self.graph[node2] = []
            self.nb_nodes += 1
            self.nodes.append(node2)

        self.graph[node1].append((node2, power_min, dist))
        self.graph[node2].append((node1, power_min, dist))
        self.nb_edges += 1
    
    def get_neighbours(self, node):
        return [nodes[0] for nodes in self.graph[node]]
    
    def get_power(self, node1, node2):
        for i in range(len(self.graph[node1])):
            if self.graph[node1][0] == node2:
                return self.graph[node1][1]
            
    def get_dist(self, node1, node2):
        for i in range(len(self.graph[node1])):
            if self.graph[node1][0] == node2:
                return self.graph[node1][2]

    
    def get_path_with_power(self, src, dest, power):
        for cc in self.connected_components_set():
            if src in cc and dest not in cc:
                return None 
                break
            elif src and dest in cc:
                def disjkstra(src, dest):
                    visited_nodes = []
                    shortest_dist = {}
                    previous_nodes = {node: None for node in cc}

                    for node in cc:
                        shortest_dist[node] = sys.maxsize
                    shortest_dist[src] = 0
                    
                    while len(visited_nodes) < len(cc):
                        
                        current_min_dist = sys.maxsize
                        current_min_node = None
                        for node, dist in shortest_dist.items():
                            if node not in visited_nodes and dist < current_min_dist:
                                current_min_dist = dist
                                current_min_node = node
                        
                        if current_min_node is None:
                            break
                        
                        visited_nodes.append(current_min_node)
                        
                        for (neighbour, dist, p) in self.graph[current_min_node]:
                            if power >= p:
                                path_distance = shortest_dist[current_min_node] + dist
                                if path_distance < shortest_dist[neighbour]:
                                    shortest_dist[neighbour] = path_distance
                                    previous_nodes[neighbour] = current_min_node
                    
                    if shortest_dist[dest] == sys.maxsize:
                        return None
                    else:
                        path = []
                        current_node = dest
                        while current_min_node !=src:
                            path.append(current_node)
                            current_node = previous_nodes[current_node]
                        return path[::-1]
                return disjkstra(src, dest)
                
                


                                    
                
                
                
        
    

    def connected_components(self):
        list_cc = []
        visited_nodes = {node:False for node in self.nodes}
        
        def dfs(node):
            component = [node]
            visited_nodes[node] = True
            for neighbour in self.graph[node]:
                neighbour = neighbour[0]
                if not visited_nodes[neighbour]:
                    component += dfs(neighbour)
            return component
        
        for node in self.nodes:
            if not visited_nodes[node]:
                list_cc.append(dfs(node))
        
        return list_cc


    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component), 
        For instance, for network01.in: {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        return set(map(frozenset, self.connected_components()))
    
    def min_power(self, src, dest):
        """
        Should return path, min_power. 
        """
        raise NotImplementedError


def graph_from_file(filename):
    """
    Reads a text file and returns the graph as an object of the Graph class.

    The file should have the following format: 
        The first line of the file is 'n m'
        The next m lines have 'node1 node2 power_min dist' or 'node1 node2 power_min' (if dist is missing, it will be set to 1 by default)
        The nodes (node1, node2) should be named 1..n
        All values are integers.

    Parameters: 
    -----------
    filename: str
        The name of the file

    Outputs: 
    -----------
    g: Graph
        An object of the class Graph with the graph from file_name.
    """
    with open(filename, "r") as file:
        n, m = map(int, file.readline().split())
        g = Graph(range(1, n+1))
        for _ in range(m):
            edge = list(map(int, file.readline().split()))
            if len(edge) == 3:
                node1, node2, power_min = edge
                g.add_edge(node1, node2, power_min, 1) # will add dist=1 by default
            elif len(edge) == 4:
                node1, node2, power_min, dist = edge
                g.add_edge(node1, node2, power_min, dist)
            else:
                raise Exception("Format incorrect")
    return g
