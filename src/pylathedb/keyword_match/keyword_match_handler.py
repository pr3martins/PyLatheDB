import itertools
import re

from pylathedb.utils import ConfigHandler,get_logger
from pylathedb.utils import Similarity

from .keyword_match import KeywordMatch

logger = get_logger(__name__)
class KeywordMatchHandler:
    def __init__(self, similarity):
        self.similarities = similarity

    def get_keyword_matches(self, keywords, value_index, schema_index,**kwargs):
        sk_matches = self.schema_keyword_match_generator(keywords, schema_index)
        vk_matches = self.value_keyword_match_generator(keywords, schema_index)
        return sk_matches+vk_matches

    def value_keyword_match_generator(self, Q,value_index,**kwargs):
        ignored_tables = kwargs.get('ignored_tables',[])
        ignored_attributes = kwargs.get('ignored_attributes',[])

        #Input:  A keyword query Q=[k1, k2, . . . , km]
        #Output: Set of non-free and non-empty tuple-sets Rq

        '''
        The tuple-set Rki contains the tuples of Ri that contain all
        terms of K and no other keywords from Q
        '''

        #Part 1: Find sets of tuples containing each keyword
        P = {}
        logger.debug("Processing keyword values")
        for keyword in Q:
            if keyword not in value_index:
                continue

            for table in value_index[keyword]:
                if table in ignored_tables:
                    continue

                for (attribute,ctids) in value_index[keyword][table].items():
                    if (table,attribute) in ignored_attributes:
                        continue

                    vkm = KeywordMatch(table, value_filter={attribute:{keyword}})
                    P[vkm] = set(ctids)

        #Part 2: Find sets of tuples containing larger termsets
        P = self.disjoint_itemsets(P)

        #Part 3: Ignore tuples
        return list(P)

    def disjoint_itemsets(self, pool):
        #Input: A Set of non-empty tuple-sets for each keyword alone P
        #Output: The Set P, but now including larger termsets (process Intersections)

        '''
        Termset is any non-empty subset K of the terms of a query Q
        '''
        
        delayed_removal = {}
        next_stage_pool = {}
        
        for ( vkm_i , vkm_j ) in itertools.combinations(pool,2):
            if vkm_i.table == vkm_j.table:
                joint_tuples = pool[vkm_i] & pool[vkm_j]

                if len(joint_tuples)>0:

                    joint_predicates = {}

                    for attribute, keywords in vkm_i.value_filter:
                        joint_predicates.setdefault(attribute,set()).update(keywords)

                    for attribute, keywords in vkm_j.value_filter:
                        joint_predicates.setdefault(attribute,set()).update(keywords)

                    vkm_ij = KeywordMatch(vkm_i.table,value_filter=joint_predicates)
                    next_stage_pool[vkm_ij] = joint_tuples

                    delayed_removal.setdefault(vkm_i,set()).update(joint_tuples)
                    delayed_removal.setdefault(vkm_j,set()).update(joint_tuples)

        for vkm_k in delayed_removal:
            tuples_to_remove = delayed_removal[vkm_k]
            pool[vkm_k].difference_update(tuples_to_remove)
            if len(pool[vkm_k])==0:
                del pool[vkm_k]

        if len(next_stage_pool)>0:
            pool.update(self.disjoint_itemsets(next_stage_pool))
        return pool

    def schema_keyword_match_generator(self, Q, schema_index,**kwargs):
        ignored_tables = kwargs.get('ignored_tables',[])
        ignored_attributes = kwargs.get('ignored_attributes',[])
        threshold = kwargs.get('threshold',0.99)
        keyword_matches_to_ignore = kwargs.get('keyword_matches_to_ignore',set())

        S = []
        logger.debug("Processing schema matches")
        for keyword in Q:
            for table in schema_index:
                if table in ignored_tables:
                    continue

                for attribute in itertools.chain('*', schema_index[table].keys() ):

                    if (
                        attribute=='id' or
                        (table,attribute) in ignored_attributes or
                        #This line prevents redundant keyword matches
                        attribute==table
                        ):
                        continue

                    sim = self.similarities.word_similarity(keyword,table,attribute)
                    #logger.debug("similiary : {} threshold: {}".format(sim, threshold))
                    if sim >= threshold:
                        # logger.info(f"found a KWmatch for {keyword} in {table}.{attribute} with score: {sim}")
                        skm = KeywordMatch(table,schema_filter={attribute:{keyword}})
                        if skm not in keyword_matches_to_ignore:
                            S.append(skm)
        return S

    def filter_kwmatches_by_compound_keywords(self, vk_matches, compound_keywords):
        '''
        Value-keyword matches which contains only part of a compound_keyword are
        pruned.
        '''
        if len(compound_keywords)==0:
            return vk_matches

        filtered_vk_matches = set()
        for value_keyword_match in vk_matches:
            must_remove = False
            set_a = set(value_keyword_match.keywords())

            for compound_keyword in compound_keywords:
                set_b = set(compound_keyword)

                set_ab = set_a | set_b
                if len(set_ab)>0 or len(set_ab)<len(set_a):
                    must_remove = True

            if not must_remove:
                filtered_vk_matches.add(value_keyword_match)

        return filtered_vk_matches
