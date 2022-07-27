import json
from timeit import default_timer as timer

from pylathedb.utils import ConfigHandler, Similarity, get_logger, Tokenizer, next_path
from pylathedb.utils.lathe_result import LatheResult
from pylathedb.index import IndexHandler
from pylathedb.database import DatabaseHandler
from pylathedb.keyword_match import KeywordMatchHandler
from pylathedb.query_match import QueryMatchHandler
from pylathedb.candidate_network import CandidateNetworkHandler
from pylathedb.evaluation import EvaluationHandler


logger = get_logger(__name__)
class Lathe:
    def __init__(self, **kwargs):
        config_directory = kwargs.get('config_directory','../config/')
        self.config = kwargs.get(
            'config',
            ConfigHandler(config_directory=config_directory)
        )

        self.max_qm_size = 3
        self.max_cjn_size = 5
        self.topk_cns = 5
        self.configuration = (5,1,9)

        self.database_handler = DatabaseHandler(self.config)
        self.index_handler = IndexHandler(self.config, self.database_handler)

        self.tokenizer = Tokenizer(tokenize_method = 'simple')
        # self.index_handler.load_indexes()
        # print(f'SCHEMA GRAPH:\n{self.index_handler.schema_graph.str_edges_info()}')

        self.similarity = Similarity(self.index_handler.schema_index)
        self.keyword_match_handler = KeywordMatchHandler(self.similarity)
        self.query_match_handler = QueryMatchHandler()
        self.evaluation_handler = EvaluationHandler(self.config)
        self.candidate_network_handler = CandidateNetworkHandler(self.database_handler)
        self.evaluation_handler.load_golden_standards()


        self._queryset=None

    def load_queryset(self):
        with open(self.config.queryset_filepath,mode='r') as f:
            queryset = json.load(f)
        self._queryset = queryset

    def get_queryset(self,reload=False):
        if self._queryset is None or reload:
            with open(self.config.queryset_filepath,mode='r') as f:
                self._queryset = json.load(f)
        return self._queryset

    def run_queryset(self,**kwargs):
        '''
        results_filename is declared here for sake of readability. But the
        default value is only set later to get a timestamp more accurate.
        '''
        results_filename = kwargs.get('results_filename',None)
        export_results = kwargs.get('export_results',False)
        approach = kwargs.get('approach','standard')
        preprocessed_results = kwargs.get('preprocessed_results',{})

        results =[]

        keywords_to_load = {keyword for item in self.get_queryset() for keyword in set(self.tokenizer.keywords(item['keyword_query']))}

        self.index_handler.load_indexes(keywords = keywords_to_load)

        for item in self.get_queryset():
            keyword_query = item['keyword_query']

            if keyword_query in preprocessed_results:
                print(f'Keyword Query: {keyword_query}')
                print('  Preprocessed results loaded')
                result = preprocessed_results[keyword_query]
            else:
                result = self.keyword_search(keyword_query,**kwargs)
            results.append(result)

        data = {
            "database":self.config.connection['database'],
            "queryset":self.config.queryset_filepath,
            "results":results,
        }

        if export_results:
            if results_filename is None:
                results_filename = next_path(f'{self.config.results_directory}{self.config.queryset_name}-{approach}-%03d.json')

            with open(results_filename,mode='w') as f:
                logger.info(f'Writing results in {results_filename}')
                json.dump(data,f, indent = 4)

        return data

    def evaluate_queryset(self,results, **kwargs):
        self.evaluation_handler.load_golden_standards()
        evaluated_results = self.evaluation_handler.evaluate_results(results, **kwargs)
        return evaluated_results

    def keyword_search(self,keyword_query=None,**kwargs):
        max_qm_size=kwargs.get('max_qm_size', self.max_qm_size)
        max_cjn_size=kwargs.get('max_cjn_size',self.max_cjn_size )
        topk_cns=kwargs.get('topk_cns', self.topk_cns)
        configuration = kwargs.get('configuration', self.configuration)

        
        max_num_query_matches,topk_cns_per_qm,max_database_accesses = configuration  
        kwargs['max_database_accesses']=max_database_accesses
        kwargs['instance_based_pruning'] = (max_database_accesses>0)
        kwargs['max_num_query_matches']=max_num_query_matches
        kwargs['topk_cns_per_qm']=topk_cns_per_qm
        kwargs['weight_scheme'] = 0

        repeat = kwargs.get('repeat',1)
        assume_golden_qms = kwargs.get('assume_golden_qms',False)
        
        input_desired_cn = kwargs.get('input_desired_cn',False)
        skip_cn_generations = kwargs.get('skip_cn_generations',False)
        show_kms_in_result = kwargs.get('show_kms_in_result',True)
        use_result_class = kwargs.get('use_result_class',True)

        weight_scheme = kwargs.get('weight_scheme',0)
        #preventing to send multiple values for weight_scheme
        if 'weight_scheme' in kwargs:
            del kwargs['weight_scheme']

        elapsed_time = {
            'km':[],
            'skm':[],
            'vkm':[],
            'qm':[],
            'cn':[],
            'total':[],
        }

        if keyword_query is None:
            print(f'Please input a keyword query or choose one of the queries below:')
            for i,item in enumerate(self.get_queryset()):
                keyword_query = item['keyword_query']
                print(f'{i+1:02d} - {keyword_query}')
            return None
        if isinstance(keyword_query, int):
            keyword_query=self.get_queryset()[keyword_query-1]['keyword_query']

        print(f'Keyword Query: {keyword_query}')
        keywords =  self.tokenizer.keywords(keyword_query)

        for _ in range(repeat):
            
            if not assume_golden_qms:
                start_skm_time = timer()
                
                sk_matches = self.keyword_match_handler.schema_keyword_match_generator(keywords, self.index_handler.schema_index)
                logger.info('%d SKMs generated: %s',len(sk_matches),sk_matches)
                
                start_vkm_time = timer()
                vk_matches = self.keyword_match_handler.value_keyword_match_generator(keywords, self.index_handler.value_index)
                # vk_matches = self.keyword_match_handler.filter_kwmatches_by_compound_keywords(vk_matches,compound_keywords)
                logger.info('%d VKMs generated: %s',len(vk_matches),vk_matches)

                kw_matches = sk_matches+vk_matches
                start_qm_time = timer()

                query_matches = self.query_match_handler.generate_query_matches(keywords, kw_matches)
            else:
                start_skm_time = timer()
                start_vkm_time = timer()
                start_qm_time = timer()
                kw_matches = []
                query_matches = self.evaluation_handler.golden_standards[keyword_query]['query_matches']

            
            ranked_query_matches = self.query_match_handler.rank_query_matches(query_matches,
                self.index_handler.value_index,
                self.index_handler.schema_index,
                self.similarity,
                weight_scheme,
            )

            ranked_query_matches = ranked_query_matches[:max_num_query_matches]
            logger.info('%d QMs generated: %s',len(ranked_query_matches),ranked_query_matches)

            start_cn_time = timer()

            if input_desired_cn:
                desired_cn = self.evaluation_handler.golden_standards[keyword_query]['candidate_networks'][0]
                kwargs['desired_cn'] = desired_cn
            else:
                kwargs['desired_cn'] = None

            if not skip_cn_generations:
                ranked_cns = self.candidate_network_handler.generate_cns(
                    self.index_handler.schema_index,
                    self.index_handler.schema_graph,
                    ranked_query_matches,
                    keywords,
                    weight_scheme,
                        **kwargs,
                )
            else:
                ranked_cns=[]


            logger.info('%d CNs generated: %s',len(ranked_cns),[(cn.score,cn) for cn in ranked_cns])
            end_cn_time = timer()

            elapsed_time['km'].append(   start_qm_time -start_skm_time)
            elapsed_time['skm'].append(  start_vkm_time-start_skm_time)
            elapsed_time['vkm'].append(  start_qm_time -start_vkm_time)
            elapsed_time['qm'].append(   start_cn_time -start_qm_time)
            elapsed_time['cn'].append(   end_cn_time   -start_cn_time)
            elapsed_time['total'].append(end_cn_time   -start_skm_time)

        aggregated_elapsed_time = {phase:min(times) for phase,times in elapsed_time.items()}

        result = {
            'keyword_query':keyword_query,
            'keywords':list(keywords),
            # 'compound_keywords':list(compound_keywords),
            'query_matches':      [query_match.to_json_serializable()
                                  for query_match in ranked_query_matches],
            'candidate_networks': [candidate_network.to_json_serializable()
                                  for candidate_network in ranked_cns],
            'elapsed_time':       aggregated_elapsed_time,
            'num_keyword_matches':len(kw_matches),
            #consider len of unranked query matches
            'num_query_matches':  len(query_matches),
            'num_candidate_networks':  len(ranked_cns),
        }

        if show_kms_in_result:
            result['value_keyword_matches'] = [vkm.to_json_serializable() for vkm in vk_matches]
            result['schema_keyword_matches']= [skm.to_json_serializable() for skm in sk_matches]

        if use_result_class:
            return LatheResult(self.index_handler,self.database_handler,result)

        return result

    def change_queryset(self,ans=None):
        self._queryset=None
        self.config.change_queryset(ans)
        self.load_indexes()

    def create_indexes(self):
        self.index_handler.create_indexes()

    def load_indexes(self):
        self.index_handler.load_indexes()
