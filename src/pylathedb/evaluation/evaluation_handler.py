import json
from collections import OrderedDict
from os import makedirs
from os.path import dirname
from copy import deepcopy

from pylathedb.utils import ConfigHandler,get_logger,next_path,last_path,LatheResult
from pylathedb.query_match import QueryMatch
from pylathedb.candidate_network import CandidateNetwork

from .evaluation_result import EvaluationResult

logger = get_logger(__name__)

class EvaluationHandler:

    def __init__(self, config):
        self.config = config

    def load_golden_standards(self):
        with open(self.config.queryset_filepath,mode='r') as f:
            data = json.load(f)

        golden_standards = OrderedDict()
        for item in data:
            if 'query_matches' in item:
                item['query_matches'] = [QueryMatch.from_json_serializable(json_serializable_qm)
                                         for json_serializable_qm in item['query_matches']]

            if 'candidate_networks' in item:
                item['candidate_networks'] = [CandidateNetwork.from_json_serializable(json_serializable_cn)
                                         for json_serializable_cn in item['candidate_networks']]

            golden_standards[item['keyword_query']] = item

        self.golden_standards = golden_standards

    def load_results_from_file(self,**kwargs):
        approach = kwargs.get('approach','standard')
        results_filename = kwargs.get(
            'results_filename',
            last_path(f'{self.config.results_directory}{self.config.queryset_name}-{approach}-%03d.json')
        )

        with open(results_filename,mode='r') as f:
            data = json.load(f)
        return data


    def evaluate_results(self,results, **kwargs):
        '''
        results_filename is declared here for sake of readability. But the
        default value is only set later to get a timestamp more accurate.
        '''
        results_filename = kwargs.get('results_filename',None)
        approach = kwargs.get('approach','standard')
        skip_ranking_evaluation = kwargs.get('skip_ranking_evaluation',False)
        write_evaluation_only  = kwargs.get('write_evaluation_only',False)
        
        in_place_results  = kwargs.get('in_place_results',False)
        use_result_class = kwargs.get('use_result_class',True)

        if not in_place_results:
            results = deepcopy(results)
        results['results']=[item.data if isinstance(item,LatheResult) else item for item in results['results']]

        if not skip_ranking_evaluation:
            self.evaluate_query_matches(results)
            self.evaluate_candidate_networks(results)
        self.evaluate_performance(results)
        self.evaluate_num_keyword_matches(results)
        self.evaluate_num_query_matches(results)
        self.evaluate_num_candidate_networks(results)

        if write_evaluation_only:
            del results['results']

        if results_filename is None:
            makedirs(self.config.results_directory, exist_ok=True)
            results_filename = next_path(f'{self.config.results_directory}{self.config.queryset_name}-{approach}-%03d.json')
        else:
            makedirs(dirname(results_filename), exist_ok=True)

        
        with open(results_filename,mode='w') as f:
            logger.info(f'Writing evaluated results in {results_filename}')
            json.dump(results,f, indent = 4)
        
        if use_result_class:
            return EvaluationResult(results)

        return results

    def evaluate_query_matches(self,results,**kwargs):
        max_k = kwargs.get('max_k',10)

        results.setdefault('evaluation',{})
        results['evaluation']['query_matches']={}

        relevant_positions = []
        for item in results['results']:
            if isinstance(item,LatheResult):
                item=item.data
            if 'query_matches' in item:
                golden_qm = self.golden_standards[item['keyword_query']]['query_matches'][0]

                query_matches = [QueryMatch.from_json_serializable(json_serializable_qm)
                                 for json_serializable_qm in item['query_matches']]

                relevant_position = self.get_relevant_position(query_matches,golden_qm)

                relevant_positions.append(relevant_position)

        results['evaluation']['query_matches']['mrr'] = self.get_mean_reciprocal_rank(relevant_positions)

        precision_at_k = {f'p@{k}' : self.get_precision_at_k(k,relevant_positions)
                          for k in range(1,max_k+1)}
        results['evaluation']['query_matches'].update(precision_at_k)

        results['evaluation']['query_matches']['relevant_positions']=relevant_positions

        # print('QM Evaluation {}'.format(results['evaluation']['query_matches']))

    def evaluate_candidate_networks(self,results,**kwargs):
        max_k = kwargs.get('max_k',10)

        results.setdefault('evaluation',{})
        results['evaluation']['candidate_networks']={}

        relevant_positions = []
        for item in results['results']:
            if isinstance(item,LatheResult):
                item=item.data
            if 'candidate_networks' in item:
                golden_cn = self.golden_standards[item['keyword_query']]['candidate_networks'][0]
                # print(f'Golden CN:\n{golden_cn}')

                candidate_networks = [CandidateNetwork.from_json_serializable(json_serializable_cn)
                                 for json_serializable_cn in item['candidate_networks']]

                # for i,candidate_network in enumerate(candidate_networks):
                #     print(f'{i+1} CN:\n{candidate_network}')

                relevant_position = self.get_relevant_position(candidate_networks,golden_cn)

                relevant_positions.append(relevant_position)

        results['evaluation']['candidate_networks']['mrr'] = self.get_mean_reciprocal_rank(relevant_positions)

        precision_at_k = {f'p@{k}' : self.get_precision_at_k(k,relevant_positions)
                          for k in range(1,max_k+1)}
        results['evaluation']['candidate_networks'].update(precision_at_k)

        results['evaluation']['candidate_networks']['relevant_positions']=relevant_positions

        # print('CN Evaluation {}'.format(results['evaluation']['candidate_networks']))


    def evaluate_performance(self,results,**kwargs):
        results.setdefault('evaluation',{})
        results['evaluation']['performance']={}

        for item in results['results']:
            if isinstance(item,LatheResult):
                item=item.data
            if 'elapsed_time' in item:
                for phase in item['elapsed_time']:
                    results['evaluation']['performance'].setdefault(phase,[]).append(item['elapsed_time'][phase])

    def evaluate_num_keyword_matches(self,results,**kwargs):
        results.setdefault('evaluation',{})
        results['evaluation']['num_keyword_matches']=[]

        for item in results['results']:
            if isinstance(item,LatheResult):
                item=item.data
            if 'num_keyword_matches' in item:
                results['evaluation']['num_keyword_matches'].append(item['num_keyword_matches'])

    def evaluate_num_query_matches(self,results,**kwargs):
        results.setdefault('evaluation',{})
        results['evaluation']['num_query_matches']=[]

        for item in results['results']:
            if isinstance(item,LatheResult):
                item=item.data
            if 'num_query_matches' in item:
                results['evaluation']['num_query_matches'].append(item['num_query_matches'])

    def evaluate_num_candidate_networks(self,results,**kwargs):
        results.setdefault('evaluation',{})
        results['evaluation']['num_candidate_networks']=[]
        for item in results['results']:
            if isinstance(item,LatheResult):
                item=item.data
            if 'num_candidate_networks' in item:
                results['evaluation']['num_candidate_networks'].append(item['num_candidate_networks'])

    def get_relevant_position(self,items,ground_truth):
        for i,item in enumerate(items):
            if item==ground_truth:
                return (i+1)
        return -1

    def get_mean_reciprocal_rank(self,relevant_positions):
        if len(relevant_positions)==0:
            return 0
        sum = 0
        for relevant_position in relevant_positions:
            if relevant_position != -1:
                reciprocal_rank = 1/relevant_position
            else:
                reciprocal_rank = 0
            sum += reciprocal_rank

        mrr = sum/len(relevant_positions)
        return mrr

    def get_precision_at_k(self,k,relevant_positions):
        if len(relevant_positions)==0:
            return 0
        sum = 0
        for relevant_position in relevant_positions:
            if relevant_position <= k and relevant_position!=-1:
                sum+=1
        presition_at_k = sum/len(relevant_positions)
        return presition_at_k
