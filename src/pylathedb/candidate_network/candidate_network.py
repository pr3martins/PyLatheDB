import json
from collections import Counter  #used to check whether two CandidateNetworks are equal
from queue import deque
from sentence_transformers import util
from sklearn.metrics.pairwise import euclidean_distances
from numpy import array, char

from pylathedb.keyword_match import KeywordMatch
from pylathedb.utils import Graph,sort_dataframe_by_token_length,sort_dataframe_by_bow_size

class CandidateNetwork(Graph):

    def __init__(self, graph_dict=None):
        self.score = None
        self.__root = None
        self._df = None

        super().__init__(graph_dict)

        if len(self)>0:
            self.set_root()

    def get_root(self):
        return self.__root

    def set_root(self,vertex=None):
        if len(self) == 0:
            return None

        if vertex is not None:
            keyword_match,_ = vertex
            if not keyword_match.is_free():
                self.__root = vertex
                return vertex
            else:
                return None
        else:
            for candidate in self.vertices():
                keyword_match,_ = candidate
                if not keyword_match.is_free():
                    self.__root = candidate
                    return candidate

        print('The root of a Candidate Network cannot be a Keyword-Free Match.')
        print(self)
        raise ValueError('The root of a Candidate Network cannot be a Keyword-Free Match.')

        # return None

    def get_starting_vertex(self):
        if len(self)==0:
            return None

        if self.__root is None:
            self.set_root()
        return self.__root

    # def get_starting_vertex(self):
    #     vertex = None
    #     for vertex in self.vertices():
    #         keyword_match,alias = vertex
    #         if not keyword_match.is_free():
    #             break
    #     return vertex


    def add_vertex(self, vertex):
        results = super().add_vertex(vertex)
        if self.__root is None:
            self.set_root(vertex)
        return results

    def add_keyword_match(self, heyword_match, **kwargs):
        alias = kwargs.get('alias', 't{}'.format(self.__len__()+1))
        vertex = (heyword_match, alias)
        return self.add_vertex(vertex)

    def add_adjacent_keyword_match(self,parent_vertex,keyword_match,edge_direction='>',**kwargs):
        child_vertex = self.add_keyword_match(keyword_match,**kwargs)
        self.add_edge(parent_vertex,child_vertex,edge_direction=edge_direction)

    def keyword_matches(self):
        # return {keyword_match for keyword_match,alias in self.vertices()}
        for keyword_match,_ in self.vertices():
            yield keyword_match

    def non_free_keyword_matches(self):
        # return {keyword_match for keyword_match,alias in self.vertices() if not keyword_match.is_free()}
        for keyword_match,_ in self.vertices():
            if not keyword_match.is_free():
                yield keyword_match

    def num_free_keyword_matches(self):
        i=0
        for keyword_match,_ in self.vertices():
            if keyword_match.is_free():
                i+=1
        return i

    def reciprocal_neighbors(self,schema_graph,vertex):
        '''
        This function iterates over incoming neighbors which, in practice,
        also behaves like an outgoing neighbor. That is, the foreign key
        is unique (not explicitly), which sugests a cardinality of 1:1
        between the two tables.
        '''
        keyword_match,_ = vertex
        for neighbor_vertex in self.incoming_neighbors(vertex):
            neighbor_km,_ = neighbor_vertex

            edge_info = schema_graph.get_edge_info(
                neighbor_km.table,
                keyword_match.table
            )
            for _,(cardinality,_) in edge_info.items():
                if cardinality=='1:1':
                    yield neighbor_vertex


    def is_sound(self,schema_graph):
        if len(self) < 3:
            return True

        def has_duplicate_tables(keyword_match,neighbors,direction='>'):
            # print(f'keyword_match:\n{keyword_match}\nneighbors:\n{neighbors}\ndirection:{direction}\n')

            if len(neighbors)>=2:
                neighbor_tables = Counter()
                fkeys_tables = Counter()

                for neighbor_km,_ in neighbors:
                    neighbor_tables[neighbor_km.table]+=1

                    if neighbor_km.table not in fkeys_tables:
                        if direction=='>':
                            edge_info = schema_graph.get_edge_info(keyword_match.table, neighbor_km.table)
                        elif direction=='<':
                            edge_info = schema_graph.get_edge_info(neighbor_km.table, keyword_match.table)

                        fkeys_tables[neighbor_km.table]= len(edge_info)

                return len(neighbor_tables-fkeys_tables)>0

            return False

        for vertex in self.vertices():
            km,_ = vertex

            #check if there is a case A->B<-C, when A.table==C.table
            outgoing_neighbors = list(self.outgoing_neighbors(vertex))
            if has_duplicate_tables(km,outgoing_neighbors):
                return False

            #check if there is a case A<-B->C, when A.table==C.table and
            #there is no duplicated foreign key in B in the database instance
            reciprocal_neighbors = list(self.reciprocal_neighbors(schema_graph,vertex))
            if has_duplicate_tables(km,reciprocal_neighbors,direction='<'):
                return False
        return True


    def remove_vertex(self,vertex):
        print('vertex:\n{}\n_graph_dict\n{}'.format(vertex,self._graph_dict))
        _,incoming_neighbors = self._graph_dict[vertex]
        for neighbor in incoming_neighbors:
            self._graph_dict[neighbor][0].remove(vertex)
        self._graph_dict.pop(vertex)

    def is_total(self,query_match):
        return set(self.non_free_keyword_matches())==set(query_match)

    def contains_keyword_free_match_leaf(self):
        for vertex in self.leaves():
            keyword_match,_ = vertex
            if keyword_match.is_free():
                return True
        return False

    def minimal_cover(self,query_match):
        return self.is_total(query_match) and not self.contains_keyword_free_match_leaf()

    def unaliased_edges(self):
        for (keyword_match,_),(neighbor_keyword_match,_) in self.edges():
            yield (keyword_match,neighbor_keyword_match)

    def get_sentences(self,schema_graph,schema_index,database_handler,**kwargs):
        cjn_ranking_method = kwargs.get('cjn_ranking_method',0)
        snapshot_method = kwargs.get('snapshot_method','first')
        snapshot_size = kwargs.get('snapshot_size',512)

        if self._df is None:
            sql = self.get_sql_from_cn(schema_graph,schema_index,
                show_table_alias=True,
                show_only_indexable_attributes=True
            )
            self._df = database_handler.get_dataframe(sql)

        df = self._df.copy()

        if snapshot_method == 'max_length':
            df=sort_dataframe_by_token_length(df)
        elif snapshot_method == 'bow_size':
            df=sort_dataframe_by_bow_size(df)
        elif snapshot_method == 'random':
            df=df.sample(n=min(snapshot_size,len(df)))
        
        df = df.head(snapshot_size)

        sentences = []

        if cjn_ranking_method <= 2:
            for i,(index, row) in enumerate(df.iterrows()):
                attrs_repr = []
                for km,alias in self.vertices():
                    for km_type,table,attr,keywords in km.mappings():
                        if attr!='*':
                            value = row[f'{alias}.{attr}']
                            attrs_repr.append(f'{table}.{attr}: {value}')
                        else:
                            attrs_repr.append(f'{table}: {" ".join(keywords)}')
                            
                sentence = ' | '.join(attrs_repr)
                sentences.append(sentence)

            if cjn_ranking_method==2:
                sentences = [' || '.join(sentences)]

        if cjn_ranking_method == 3:
            attrs_repr = []
            for km,alias in self.vertices():
                for km_type,table,attr,keywords in km.mappings():
                    if attr=='*':
                        attrs_repr.append(f'{table}: {" ".join(keywords)}')
                    else:
                        # numpy.char
                        value = ', '.join(char.mod("%s", df[f'{alias}.{attr}'].values))
                        attrs_repr.append(f'{table}.{attr}: {value}')
            sentence = ' | '.join(attrs_repr)
            sentences.append(sentence)

        return sentences

    def calculate_score(self,query_match,keyword_query,schema_graph,schema_index,database_handler,bert_model,**kwargs):
        cjn_ranking_method = kwargs.get('cjn_ranking_method',0)
        bert_similarity = kwargs.get('bert_similarity','cos')
        snapshot_method = kwargs.get('snapshot_method','first')
        snapshot_size = kwargs.get('snapshot_size',512)        

        if cjn_ranking_method==0:
            self.score = query_match.total_score/len(self)
            return self.score

        sentences_to_compare = [keyword_query]+self.get_sentences(schema_graph,schema_index,database_handler,**kwargs)

        embeddings = bert_model.encode(array(sentences_to_compare))

        query_emb = array([embeddings[0]])
        row_emb = embeddings[1:]
        if bert_similarity == 'cos':
            scores = util.cos_sim(query_emb, row_emb)
        elif bert_similarity == 'dot':
            scores = util.dot_score(query_emb, row_emb)
        else:
            scores = euclidean_distances(query_emb, row_emb)
        self.score = scores.mean().item()
        return self.score

    def get_qm_score(self):
        return self.score*len(self)

    def __eq__(self, other):

        if not isinstance(other,CandidateNetwork):
            return False

        self_root_km,_  = self.__root
        # if other.get_root() is None:
        #     print(f'OTHER ROOT IS NONE')
        #     print(other)
        other_root_km,_= other.get_root()


        other_hash = None

        if self_root_km==other_root_km:
            other_hash = hash(other)
        else:
            for keyword_match,alias in other.vertices():
                if self_root_km == keyword_match:
                    root = (keyword_match,alias)
                    # print(f'Root: {root}')
                    other_hash = other.hash_from_root(root)

        if other_hash is None:
            return False

        # print(f'Equal:{hash(self)==other_hash}\nSelf:\n{self}\nOther:\n{other}\n\n\n')

        return hash(self)==other_hash

    def __hash__(self):
        if len(self)==0:
            return hash(None)
        if self.__root is None:
            self.set_root()
        return self.hash_from_root(self.__root)


    def hash_from_root(self,root):
        hashable = []

        level = 0
        visited = set()

        queue = deque()
        queue.append( (level,root) )

        while queue:
            level,vertex = queue.popleft()
            keyword_match,alias = vertex
            children = Counter()
            visited.add(alias)

            for adj_vertex in self.neighbors(vertex):
                adj_km,adj_alias = adj_vertex
                if adj_alias not in visited:
                    queue.append( (level+1,adj_vertex) )
                    children[adj_km]+=1

            if len(hashable)<level+1:
                hashable.append(set())

            hashable[level].add( (keyword_match,frozenset(children.items())) )

        hashcode = hash(tuple(frozenset(items) for items in hashable))
        return hashcode

    def __repr__(self):
        if len(self)==0:
            return 'EmptyCN'
        print_string = ['\t'*level+direction+str(vertex[0])  for direction,level,vertex in self.leveled_dfs_iter()]
        return '\n'.join(print_string)

    def to_json_serializable(self):
        return [{'keyword_match':keyword_match.to_json_serializable(),
            'alias':alias,
            'outgoing_neighbors':[alias for (km,alias) in outgoing_neighbors],
            'incoming_neighbors':[alias for (km,alias) in incoming_neighbors]}
            for (keyword_match,alias),(outgoing_neighbors,incoming_neighbors) in self._graph_dict.items()]

    def to_json(self):
        return json.dumps(self.to_json_serializable())

    @staticmethod
    def from_json_serializable(json_serializable_cn):
        alias_hash ={}
        edges=[]
        for vertex in json_serializable_cn:
            keyword_match = KeywordMatch.from_json_serializable(vertex['keyword_match'])
            alias_hash[vertex['alias']]=keyword_match

            for outgoing_neighbor in vertex['outgoing_neighbors']:
                edges.append( (vertex['alias'],outgoing_neighbor) )

        candidate_network = CandidateNetwork()
        for alias,keyword_match in alias_hash.items():
            candidate_network.add_vertex( (keyword_match,alias) )
        for alias1, alias2 in edges:
            vertex1 = (alias_hash[alias1],alias1)
            vertex2 = (alias_hash[alias2],alias2)
            candidate_network.add_edge(vertex1,vertex2)

        return candidate_network

    @staticmethod
    def from_json(json_cn):
        return CandidateNetwork.from_json_serializable(json.loads(json_cn))

    def get_sql_from_cn(self,schema_graph,schema_index,**kwargs):
        rows_limit=kwargs.get('rows_limit',1000)
        show_evaluation_fields=kwargs.get('show_evaluation_fields',False)
        tsvector_field_suffix=kwargs.get('tsvector_field_suffix','_tsvector')
        show_table_alias=kwargs.get('show_table_alias',False)
        distinct=kwargs.get('distinct',True)
        show_only_indexable_attributes=kwargs.get('show_only_indexable_attributes',False)


        hashtables = {} # used for disambiguation
        used_fks = {}


        selected_attributes = set()
        filter_conditions = []
        disambiguation_conditions = []
        selected_tables = []

        tables__search_id = []
        relationships__search_id = []


        def get_column_type(table,attribute):
            # TODO: Handle attributes from the "varchar"  domain
            # but do not have to be indexed
            if table == 'title' and attribute == 'production_year':
                return 'integer'
            if table == 'movie' and attribute == 'year':
                return 'integer'
            if table == 'organization' and attribute=='abbreviation':
                return 'varchar'
            if attribute=='stars':
                return 'integer'
            return 'fulltext_indexed'

        for prev_vertex,direction,vertex in self.dfs_pair_iter(root_predecessor=True):
            keyword_match, alias = vertex
            for type_km, table ,attribute,keywords in keyword_match.mappings():
                attributes_to_select = [(alias,attribute)]
                if show_only_indexable_attributes and attribute=='*':
                    attributes_to_select = [(alias,attr) for attr in schema_index[table].keys()]
                for alias_to_select,attribute_to_select in attributes_to_select:
                    selected_attr = f'{alias_to_select}.{attribute_to_select}'
                    if show_table_alias:
                        selected_attr=f'{selected_attr} AS "{alias_to_select}.{attribute_to_select}"'
                    selected_attributes.add(selected_attr)

                sql_keywords = [keyword.replace('\'','\'\'') for keyword in keywords]

                if type_km == 'v':

                    column_type=get_column_type(keyword_match.table,attribute)


                    if column_type == 'fulltext_indexed':
                        condition = f"{alias}.{attribute}{tsvector_field_suffix} @@ to_tsquery(\'{ ' & '.join(sql_keywords) }\')"
                        filter_conditions.append(condition)
                    else:
                        for sql_keyword in sql_keywords:
                            if column_type == 'varchar':
                                condition = f"{alias}.{attribute} ILIKE \'%{sql_keyword}%\'"
                            elif column_type == 'integer':
                                condition = f"{alias}.{attribute} = {sql_keyword}"
                            else:
                                condition = f"CAST({alias}.{attribute} AS VARCHAR) ILIKE \'%{sql_keyword}%\'"
                            filter_conditions.append(condition)

            hashtables.setdefault(keyword_match.table,[]).append(alias)

            if show_evaluation_fields:
                tables__search_id.append(f'{alias}.__search_id')

            if prev_vertex is None:
                selected_tables.append(f'"{keyword_match.table}" {alias}')
            else:
                # After the second table, it starts to use the JOIN syntax
                _ ,prev_alias = prev_vertex
                if direction == '>':
                    constraint_keyword_match,constraint_alias = prev_vertex
                    foreign_keyword_match,foreign_alias = vertex
                else:
                    constraint_keyword_match,constraint_alias = vertex
                    foreign_keyword_match,foreign_alias = prev_vertex

                edge_info = schema_graph.get_edge_info(constraint_keyword_match.table,
                                            foreign_keyword_match.table)

                for constraint in edge_info:
                    if constraint not in used_fks.setdefault(constraint_alias,[]):
                        used_fks[constraint_alias].append(constraint)

                        _,attribute_mappings = edge_info[constraint]

                        join_conditions = []
                        for (constraint_column,foreign_column) in attribute_mappings:
                            join_conditions.append(
                                f'{constraint_alias}.{constraint_column} = {foreign_alias}.{foreign_column}'
                            )
                        txt_join_conditions = '\n\t\tAND '.join(join_conditions)
                        selected_tables.append(f'JOIN "{keyword_match.table}" {alias} ON {txt_join_conditions}')
                        break

                if show_evaluation_fields:
                    relationships__search_id.append(f'({alias}.__search_id, {prev_alias}.__search_id)')

        for _,aliases in hashtables.items():
            for i in range(len(aliases)):
                for j in range(i+1,len(aliases)):
                    disambiguation_conditions.append(f'{aliases[i]}.ctid <> {aliases[j]}.ctid')

        if len(tables__search_id)>0:
            tables__search_id = [f"({', '.join(tables__search_id)}) AS Tuples"]
        if len(relationships__search_id)>0:
            txt_relationships = ', '.join(relationships__search_id)
            relationships__search_id = [f'({txt_relationships}) AS Relationships']

        where_conditions = disambiguation_conditions+filter_conditions
        if len(where_conditions)>0:
            where_clause = 'WHERE\n\t{}'.format(
                '\n\tAND '.join(where_conditions) 
            )
        else:
            where_clause= ''

        sql_text = '\nSELECT{}\n\t{}\nFROM\n\t{}\n{}\nLIMIT {};'.format(
            ' DISTINCT ' if distinct else '',
            ',\n\t'.join( tables__search_id+relationships__search_id+list(selected_attributes) ),
            '\n\t'.join(selected_tables),
            where_clause,
            rows_limit)
        return sql_text
