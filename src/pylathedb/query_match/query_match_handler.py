import  itertools
from pylathedb.utils import get_logger
from pylathedb.keyword_match import KeywordMatch

from .query_match import QueryMatch

logger = get_logger(__name__)

class QueryMatchHandler:

    def __init__(self):
        return 

    def generate_query_matches(self, keywords,keyword_matches, **kwargs):
        #Input:  A keyword query Q, The set of non-empty non-free tuple-sets Rq
        #Output: The set Mq of query matches for Q
        max_qm_size = kwargs.get('max_qm_size',3)
        max_merged_qm_size = kwargs.get('max_merged_qm_size',3)
        query_matches = []

        for i in range(1,min(len(keywords), max_qm_size)+1):
            for candidate_query_match in itertools.combinations(keyword_matches,i):
                if self.has_minimal_cover(candidate_query_match,keywords):
                    merged_query_match = self.merge_schema_filters(candidate_query_match)
                    query_match = QueryMatch(merged_query_match)
                    if len(merged_query_match) <= max_merged_qm_size:
                        query_matches.append(query_match)

        return query_matches

    def has_minimal_cover(self, candidate_query_match, keywords):
        #Input:  A subset CM (Candidate Query Match) to be checked as total and minimal cover
        #Output: If the match candidate is a TOTAL and MINIMAL cover

        # Check whether it is total

        if len({keyword
            for keyword_match in candidate_query_match
            for keyword in keyword_match.keywords()
            }) != len(set(keywords)):

            return False

        # Check whether it is minimal
        for element in candidate_query_match:
            if len({keyword
                for keyword_match in candidate_query_match
                for keyword in keyword_match.keywords()
                if keyword_match!=element
                }) == len(set(keywords)):
                return False

        return True

    # def has_minimal_cover(self, candidate_query_match, keywords):
    #     #Input:  A subset CM (Candidate Query Match) to be checked as total and minimal cover
    #     #Output: If the match candidate is a TOTAL and MINIMAL cover

    #     total_counter = Counter(keyword
    #                               for keyword_match in candidate_query_match
    #                               for keyword in keyword_match.keywords())

    #     # Check whether it is total
    #     if len(total_counter)!=len(set(keywords)):
    #         return False

    #     # Check whether it is minimal
    #     for keyword_match in candidate_query_match:
    #         local_counter = Counter(keyword_match.keywords())
    #         # subtract operation keeps only positive counts
    #         if len(total_counter-local_counter)==len(keywords):
    #             return False

    #     return True


    # def generate_query_matches2(self, keywords,keyword_matches, **kwargs):
    #     #Input:  A keyword query Q, The set of non-empty non-free tuple-sets Rq
    #     #Output: The set Mq of query matches for Q
    #     max_qm_size = kwargs.get('max_qm_size',3)

    #     keyword_hash = {}
    #     for keyword_match in keyword_matches:
    #         key = frozenset(keyword_match.keywords())
    #         keyword_hash.setdefault(key,[]).append  (keyword_match)

    #     query_matches = []

    #     combinations = []
    #     for i in range(1,min(len(keywords), max_qm_size)+1):
    #         for combination in itertools.combinations(keyword_hash,i):
    #             if self.has_minimal_cover2(combination,keywords):
    #                 combinations.append(combination)


    #     query_matches = [
    #         QueryMatch(self.merge_schema_filters(query_match))
    #         for combination in combinations
    #         for query_match in itertools.product(*
    #             [keyword_hash[key] for key in combination]
    #         )           
    #     ]


    #     return query_matches

    # def has_minimal_cover2(self, combination, keywords):
    #     #Input:  A subset CM (Candidate Query Match) to be checked as total and minimal cover
    #     #Output: If the match candidate is a TOTAL and MINIMAL cover

    #     # Check whether it is total

    #     if len({keyword
    #         for keyword_set in combination
    #         for keyword in keyword_set
    #         }) != len(set(keywords)):

    #         return False

    #     # Check whether it is minimal
    #     for element in combination:
    #         if len({keyword
    #             for keyword_set in combination
    #             for keyword in keyword_set
    #             if keyword_set!=element
    #             }) == len(set(keywords)):
    #             return False

    #     return True

    # def has_minimal_cover2(self,combination, keywords):
    #     #Input:  A subset CM (Candidate Query Match) to be checked as total and minimal cover
    #     #Output: If the match candidate is a TOTAL and MINIMAL cover

    #     total_counter = Counter(keyword
    #                               for keyword_set in combination
    #                               for keyword in keyword_set)
            
    #     # Check whether it is total
    #     if len(total_counter)!=len(set(keywords)):
    #         return False

    #     # Check whether it is minimal
    #     for keyword_set in combination:
    #         local_counter = Counter(keyword_set)
    #         # subtract operation keeps only positive counts
    #         if len(total_counter-local_counter)==len(keywords):
    #             return False

    #     return True    

    def merge_schema_filters(self, query_matches):
        table_hash={}
        for keyword_match in query_matches:
            joint_schema_filter,value_keyword_matches = table_hash.setdefault(keyword_match.table,({},set()))

            for attribute, keywords in keyword_match.schema_filter:
                joint_schema_filter.setdefault(attribute,set()).update(keywords)

            if len(keyword_match.value_filter) > 0:
                value_keyword_matches.add(keyword_match)

        merged_qm = set()
        for table,(joint_schema_filter,value_keyword_matches) in table_hash.items():
            if len(value_keyword_matches) > 0:
                joint_value_filter = {attribute:keywords
                                    for attribute,keywords in value_keyword_matches.pop().value_filter}
            else:
                joint_value_filter={}

            joint_keyword_match = KeywordMatch(table,
                                            value_filter=joint_value_filter,
                                            schema_filter=joint_schema_filter)

            merged_qm.add(joint_keyword_match)
            merged_qm.update(value_keyword_matches)

        return merged_qm

    def rank_query_matches(self,query_matches, value_index, schema_index, similarity,weight_scheme):
        for query_match in query_matches:
            logger.debug("query match: {}".format(query_match))
            query_match.calculate_total_score(
                value_index,
                schema_index,
                similarity,
                weight_scheme,
            )

        ranked_query_matches = sorted(query_matches,key=lambda query_match: query_match.total_score,reverse=True)
    
        return ranked_query_matches
