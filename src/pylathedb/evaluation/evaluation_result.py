from IPython.display import display
from pandas import DataFrame,MultiIndex

class EvaluationResult():

    def __init__(self,data):
        self.data = data

    def metrics(self):
        data = {
            'QMs':self.data['evaluation']['query_matches'],
            'CJNs':self.data['evaluation']['candidate_networks'],
        }
        df = DataFrame.from_dict(data, orient='index')
        del df['relevant_positions']
        df.columns = df.columns.str.upper()
        df=df.round(2)
        display(df)
    
    def relevant_positions(self):
        data = {
            'Keyword Query': [item['keyword_query'] for item in self.data['results']],
            'Relevant QM': self.data['evaluation']['query_matches']['relevant_positions'],
            'Relevant CJN':self.data['evaluation']['candidate_networks']['relevant_positions'],
        }
        df = DataFrame.from_dict(data)
        df.index = df.index + 1
        df.columns = MultiIndex(levels=[['','Relevant Position'],['Keyword Query', 'QM','CJN']],
                    codes=[[0,1,1],
                            [0,1,2]])
        display(df)
