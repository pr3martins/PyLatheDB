from queue import deque
from copy import deepcopy
from sentence_transformers import SentenceTransformer

from pylathedb.utils import ConfigHandler,get_logger
from pylathedb.keyword_match import KeywordMatch
from pylathedb.database import DatabaseHandler

from .candidate_network import CandidateNetwork

logger = get_logger(__name__)
class CandidateNetworkHandler:
    def __init__(self,database_handler):
        self.database_handler = database_handler
        self.bert_model = SentenceTransformer('all-mpnet-base-v2')


    def generate_cns(self,schema_index,schema_graph,ranked_query_matches,keywords,weight_scheme,keyword_query,**kwargs):
        topk_cns = kwargs.get('topk_cns',20)
        cjn_ranking_method = kwargs.get('cjn_ranking_method',0)

        returned_cns=[]
        num_cns_available = topk_cns

        for query_match in ranked_query_matches:
            if topk_cns!=-1 and num_cns_available<=0:
                break

            cns_per_cur_qm = self.generate_cns_per_qm(schema_index,schema_graph,query_match,keywords,weight_scheme,**kwargs)

            for candidate_network in cns_per_cur_qm:
                if num_cns_available<=0:
                    break

                returned_cns.append(candidate_network)

                candidate_network.calculate_score(
                    query_match,
                    keyword_query,
                    schema_graph,
                    schema_index,
                    self.database_handler,
                    self.bert_model,
                    **kwargs
                )

                num_cns_available -=1

        ranked_cns=sorted(
            returned_cns,
            key=lambda candidate_network: candidate_network.score,
            reverse=True
        )

        return ranked_cns

    def generate_cns_per_qm(self,schema_index,schema_graph,query_match,keywords,weight_scheme,**kwargs):
        max_cn_size = kwargs.get('max_cn_size',5)
        topk_cns_per_qm = kwargs.get('topk_cns_per_qm',1)
        
        instance_based_pruning = kwargs.get('instance_based_pruning',False)
        max_database_accesses  = kwargs.get('max_database_accesses',7)

        directed_neighbor_sorting_function = kwargs.get('directed_neighbor_sorting_function',
                                                        self.factory_sum_norm_attributes(schema_index,weight_scheme))
        schema_prunning = kwargs.get('schema_prunning',True)
        desired_cn = kwargs.get('desired_cn',None)

        returned_cns = []
        empty_cns = []
        pruned_cns = set()

        def meet_pruning_conditions(jnkm):
            return not (
                #corretude rules
                jnkm.is_sound(schema_graph) and
                #checking whether it was already generated
                jnkm not in F and
                jnkm not in pruned_cns and
                jnkm not in returned_cns and
                jnkm not in empty_cns and
                # max cn size pruning
                len(jnkm)<=max_cn_size and
                (
                    not schema_prunning or
                    (
                        # max number of leaves pruning
                        jnkm.num_leaves() <= len(query_match) and
                        # max number of keyword-free matches pruning
                        jnkm.num_free_keyword_matches()+len(query_match) <= max_cn_size
                    )
                )
            )

        # JNKM stands for Joining Network of Keyword Matches
        cur_jnkm = CandidateNetwork()
        cur_jnkm.add_keyword_match(query_match.get_km_from_keyword(keywords[0]))

        if len(query_match)==1:
            returned_cns.append(cur_jnkm)
            return returned_cns

        table_hash={}
        for keyword_match in query_match:
            table_hash.setdefault(keyword_match.table,{KeywordMatch(keyword_match.table)}).add(keyword_match)

        F = deque()
        F.append(cur_jnkm)

        while F:
            cur_jnkm = F.popleft()

            for vertex_u in reversed(list(cur_jnkm.vertices())):
                keyword_match,_ = vertex_u

                sorted_directed_neighbors = sorted(
                    schema_graph.directed_neighbors(keyword_match.table),
                    key=directed_neighbor_sorting_function
                    )

                for direction,adj_table in sorted_directed_neighbors:

                    table_hash.setdefault(adj_table,{KeywordMatch(adj_table)})

                    for adj_keyword_match in table_hash[adj_table]:

                        if (adj_keyword_match not in cur_jnkm.keyword_matches() or
                            adj_keyword_match.is_free()):

                            next_jnkm = deepcopy(cur_jnkm)
                            next_jnkm.add_adjacent_keyword_match(vertex_u,adj_keyword_match,edge_direction=direction)

                            if not meet_pruning_conditions(next_jnkm):
                                # print(f'next_jnkm:\n{next_jnkm}')
                                if next_jnkm.is_total(query_match):
                                    if not next_jnkm.contains_keyword_free_match_leaf():                                      
                                        if not instance_based_pruning or not self.is_cn_empty(schema_graph,schema_index,next_jnkm):
                                            returned_cns.append(next_jnkm)
                                        else:
                                            empty_cns.append(next_jnkm)

                                        if desired_cn is not None and next_jnkm == desired_cn:
                                            return returned_cns
                                    else:
                                        #reduce JNKM
                                        # print(f'next_jnkm(pruned): non minimal\n{next_jnkm}')
                                        pruned_cns.add(next_jnkm)
                                        continue

                                    if (
                                        len(returned_cns)>=topk_cns_per_qm or
                                        (
                                            instance_based_pruning and
                                            len(returned_cns)+len(empty_cns)>=max_database_accesses
                                        )
                                    ):
                                        return returned_cns

                                elif len(next_jnkm)<max_cn_size:
                                    F.append(next_jnkm)
                                    # print(f'next_jnkm:\n{next_jnkm}')
                                else:
                                    pruned_cns.add(next_jnkm)
                            else:
                                pruned_cns.add(next_jnkm)  
        return returned_cns

    def factory_sum_norm_attributes(self,schema_index,weight_scheme):

        def sum_norm_attributes(directed_neighbor):
            _,adj_table = directed_neighbor
            score1 = 0 
            if adj_table in schema_index:
                score1 = sum(
                    schema_index[adj_table][attribute]['norm'][weight_scheme]
                    for attribute in schema_index[adj_table]
                )
            score2 = adj_table
            return (-score1,score2)

        return sum_norm_attributes
    
    def is_cn_empty(self,schema_graph,schema_index,candidate_network):
        sql = candidate_network.get_sql_from_cn(schema_graph,schema_index)
        return self.database_handler.exist_results(sql) == False