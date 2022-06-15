import itertools
from nltk.corpus import wordnet as wn

class Similarity:
    def __init__(self, schema_index, **kwargs):
        self.schema_index = schema_index

        self.use_path_sim=kwargs.get('use_path_sim',True)
        self.use_wup_sim=kwargs.get('use_wup_sim',True)
        self.use_jaccard_sim=kwargs.get('use_jaccard_sim',True)

        # aggregation_method can be 'max' or 'avg'
        self.aggregation_method=kwargs.get('aggregation_method','max')

    def path_similarity(self,word_a,word_b):
        A = set(wn.synsets(word_a))
        B = set(wn.synsets(word_b))

        path_similarities = [0]

        for (sense1,sense2) in itertools.product(A,B):
            path_similarities.append(wn.path_similarity(sense1,sense2) or 0)

        return max(path_similarities)

    def wup_similarity(self,word_a,word_b):
        A = set(wn.synsets(word_a))
        B = set(wn.synsets(word_b))

        wup_similarities = [0]

        for (sense1,sense2) in itertools.product(A,B):
            wup_similarities.append(wn.wup_similarity(sense1,sense2) or 0)

        return max(wup_similarities)

    def jaccard_similarity(self,word_a,word_b):

        A = set(word_a)
        B = set(word_b)

        return (len(A & B) * 1.) / (len(A | B) * 1.)


    def word_similarity(self,word,table,column = '*'):
        sim_list=[]

        if column == '*':
            schema_term = table
        else:
            schema_term = column

        if schema_term==word:
            return 1.0

        if self.use_path_sim:
            sim_list.append( self.path_similarity(schema_term,word) )

        if self.use_wup_sim:
            sim_list.append( self.wup_similarity(schema_term,word) )

        if self.use_jaccard_sim:
            sim_list.append( self.jaccard_similarity(schema_term,word) )

        if len(sim_list) == 0:
            return 0

        if self.aggregation_method == 'max':
            return max(sim_list)*0.99
        elif self.aggregation_method == 'avg':
            return (sum(sim_list) * 1.)/(len(sim_list) * 1.)*0.99
