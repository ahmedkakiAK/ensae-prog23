import time
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
        self.powers = []
    

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
        visited = []
        queue = [(src, [])]
        while queue:
            node, path = queue.pop(0)
            if node not in visited:
                visited.append(node)
                if node == dest:
                    return path + [node]
                for neighbor, p, dist in self.graph[node]:
                    if p <= power:    
                        if neighbor not in visited:
                            queue.append((neighbor, path + [node]))
        return None

    
    def get_shortest_path_with_power(self, src, dest, power):
        for cc in self.connected_components():
            if src in cc and dest not in cc:
                return None
            elif src and dest in cc:
                temp_graph = self.graph
                for node in cc:
                    for tuple in temp_graph[node]:
                        if tuple[1] > power:
                            temp_graph[node].remove(tuple)

                def disjkstra(g, s):
                    inf = sys.maxsize
                    visited = {s: [0, [s]]}
                    unvisited = {node: [inf, ""] for node in g if node != s}
                    for neighbour, p, dist in g[s]:
                        unvisited[neighbour] = [dist, s]

                    while unvisited and any(unvisited[node][0] < inf for node in unvisited):
                        min_node = min(unvisited, key = unvisited.get)
                        min_dist, previous_node = unvisited[min_node]
                        for neighbour, p, dist in g[min_node]:
                            if neighbour in unvisited:
                                d = min_dist + dist
                                if d < unvisited[neighbour][0]:
                                    unvisited[neighbour] = [d, min_node]
                        visited[min_node] = [min_dist, visited[previous_node][1] + [min_node]]
                        del unvisited[min_node]
                        
                    for node in unvisited:
                        visited[node] = [None, None]

                    return visited
                
                visited = disjkstra(temp_graph, src)
                return visited[dest][1]
            

    
                    
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
        list_powers = sorted(self.powers)
        x = 0
        y = len(list_powers) - 1
        m = (x+y)//2

        while x < y:
            if self.get_path_with_power(src, dest, list_powers[m]) != None:
                y = m
            else:
                x = m + 1
            m = (x+y)//2
        path = self.get_path_with_power(src, dest, list_powers[x])
        if path != None:
            power = list_powers[x]
        else:
            power = None
        return path, power        

                    
                



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
                g.add_edge(node1, node2, power_min, 1)
                g.powers.append(power_min) # will add dist=1 by default
            elif len(edge) == 4:
                node1, node2, power_min, dist = edge
                g.add_edge(node1, node2, power_min, dist)
                g.powers.append(power_min)
            else:
                raise Exception("Format incorrect")
    return g


class UnionFind:
    def __init__(self, vertices):
        self.parents = {v: v for v in vertices}
        self.sizes = {v: 1 for v in vertices}

    def find(self, x):
        if x != self.parents[x]:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False
        if self.sizes[root_x] < self.sizes[root_y]:
            root_x, root_y = root_y, root_x
        self.parents[root_y] = root_x
        self.sizes[root_x] += self.sizes[root_y]
        return True


def kruskal(g):
    edges = []
    for node in g.nodes:
        for neighbor, power, dist in g.graph[node]:
            edges.append((power, node, neighbor))
    edges.sort()
    vertices = set(g.nodes)
    uf = UnionFind(vertices)
    mst = Graph(g.nodes)
    for  power, node, neighbor in edges:
        if uf.union(node, neighbor):
            mst.graph[node].append((neighbor, power, 1))
            mst.graph[neighbor].append((node, power, 1))

            
    return mst
    

def min_power2(g, src, dest):
    mst = kruskal(g)

    visited_nodes = {node: False for node in mst.nodes}
    parent = {}

    def dfs(node):
            visited_nodes[node] = True
            for neighbour in mst.graph[node]:
                neighbour = neighbour[0]
                if not visited_nodes[neighbour]:
                    parent[neighbour] = node
                    dfs(neighbour)

    dfs(src)

    if visited_nodes[dest] == False:
        return None
    
    path = [dest]
    while path[-1] != src:
        path.append(parent[path[-1]])
    path.reverse()
    min_power = sys.maxsize
    for i in range(len(path) - 1):
        for neighbour, power, dist in mst.graph[path[i]]:
            if neighbour == path[i+1] and min_power > power:
                min_power = power
    return path, min_power










