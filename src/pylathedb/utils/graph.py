class Graph:
    __slots__ = ['_graph_dict','_edges_info']
    def __init__(self, graph_dict=None, edges_info=None):
        if graph_dict is None:
            graph_dict = {}
        self._graph_dict = graph_dict

        if edges_info is None:
            edges_info = {}
        self._edges_info = edges_info

    def add_vertex(self, vertex):
        self._graph_dict.setdefault(vertex, (set(),set()) )
        return vertex

    def add_outgoing_neighbor(self,vertex,neighbor):
        self._graph_dict[vertex][0].add(neighbor)

    def add_incoming_neighbor(self,vertex,neighbor):
        self._graph_dict[vertex][1].add(neighbor)

    def outgoing_neighbors(self,vertex):
        yield from self._graph_dict[vertex][0]

    def incoming_neighbors(self,vertex):
        yield from self._graph_dict[vertex][1]

    def neighbors(self,vertex):
        #This method does not use directed_neighbors to avoid generating tuples with directions
        yield from self.outgoing_neighbors(vertex)
        yield from self.incoming_neighbors(vertex)

    def directed_neighbors(self,vertex):
        #This method does not use directed_neighbors to avoid generating tuples with directions
        for outgoing_neighbor in self.outgoing_neighbors(vertex):
            yield ('>',outgoing_neighbor)
        for incoming_neighbor in self.incoming_neighbors(vertex):
            yield ('<',incoming_neighbor)

    def add_edge(self, vertex1, vertex2,edge_info = None, edge_direction='>'):
        if edge_direction=='>':
            self.add_outgoing_neighbor(vertex1,vertex2)
            self.add_incoming_neighbor(vertex2,vertex1)
            if edge_info is not None:
                self._edges_info[(vertex1, vertex2)] = edge_info

        elif edge_direction=='<':
            self.add_edge(vertex2, vertex1,edge_info = edge_info)
        else:
            raise SyntaxError('edge_direction must be > or <')

    def get_edge_info(self,vertex1,vertex2):
        return self._edges_info.get( (vertex1, vertex2), None )

    def vertices(self):
        return self._graph_dict.keys()

    def edges(self):
            for vertex in self.vertices():
                for neighbor in self.outgoing_neighbors(vertex):
                    yield (vertex,)+ (neighbor,)

    def dfs_pair_iter(self, source_iter = None, root_predecessor = False):
        last_vertex_by_level=[]

        if source_iter is None:
            source_iter = self.leveled_dfs_iter()

        for direction,level,vertex in source_iter:
            if level < len(last_vertex_by_level):
                last_vertex_by_level[level] = vertex
            else:
                last_vertex_by_level.append(vertex)

            if level>0:
                prev_vertex = last_vertex_by_level[level-1]
                yield (prev_vertex,direction,vertex)
            elif root_predecessor:
                yield (None,'',vertex)


    def leveled_dfs_iter(self,start_vertex=None,visited = None, level=0, direction='',two_way_transversal=True):
        if len(self)>0:
            if start_vertex is None:
                start_vertex = self.get_starting_vertex()
            if visited is None:
                visited = set()
            visited.add(start_vertex)

            yield( (direction,level,start_vertex) )

            for neighbor in self.outgoing_neighbors(start_vertex):
                if neighbor not in visited:
                    yield from self.leveled_dfs_iter(neighbor,visited,
                                                     level=level+1,
                                                     direction='>',
                                                     two_way_transversal=two_way_transversal)

            # two_way_transversal indicates whether the DFS will expand through incoming neighbors
            if two_way_transversal:
                for neighbor in self.incoming_neighbors(start_vertex):
                    if neighbor not in visited:
                        yield from self.leveled_dfs_iter(neighbor,visited,
                                                         level=level+1,
                                                         direction='<',
                                                         two_way_transversal=two_way_transversal)


    def leaves(self):
        for vertice in self.vertices():
            if self.is_leaf(vertice):
                yield vertice
    
    def is_leaf(self,vertice):
        return sum(1 for neighbor in self.neighbors(vertice)) == 1

    def num_leaves(self):
        return len(list(self.leaves()))

    def get_starting_vertex(self):
        return next(iter(self.vertices()))

    def __repr__(self):
        return repr(self._graph_dict)

    def __len__(self):
         return len(self._graph_dict)

    def str_graph_dict(self):
        return str(self._graph_dict)

    def str_edges_info(self):
        return str(self._edges_info)
