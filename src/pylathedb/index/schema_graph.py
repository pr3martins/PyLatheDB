import pickle
from os import makedirs
from os.path import dirname
from graphviz import Digraph

from pylathedb.utils import Graph

class SchemaGraph(Graph):
    def __init__(self, graph_dict=None,edges_info = None):
        super().__init__(graph_dict,edges_info)

    def add_fk_constraint(self,constraint,cardinality,table,foreign_table,attribute_mappings):
        self.add_vertex(table)
        self.add_vertex(foreign_table)

        edge_info = self._edges_info.setdefault( 
            (table,foreign_table),
            {}
        ) 
        edge_info[constraint] = (cardinality,attribute_mappings)
        self.add_edge(table, foreign_table, edge_info)

    def __repr__(self):
        if len(self)==0:
            return 'EmptyGraph'
        print_string = ['\t'*level+direction+vertex for direction,level,vertex in self.leveled_dfs_iter()]
        return '\n'.join(print_string)

    def persist_to_file(self,filename):
        data = (self._graph_dict,self._edges_info)
        makedirs(dirname(filename), exist_ok=True)
        with open(filename,mode='wb') as f:
            pickle.dump(data,f)

    def load_from_file(self,filename):
        self._graph_dict = {}
        self._edges_info = {}

        with open(filename,mode='rb') as f:
            data = pickle.load(f)
        self._graph_dict,self._edges_info = data

    def show(self):
        g= Digraph(
            graph_attr={'nodesep':'0.2','ranksep':'0.25'},
            node_attr={'fontsize':"9.0",},
            edge_attr={'arrowsize':'0.9',},
        )
        for id in self.vertices():
            g.node(id,label=str(id.upper()))
        for id_a,id_b in self.edges():
            g.edge(id_a,id_b)
        print('Graph:')
        display(g)
    