import shelve
from random import sample
from os import makedirs
from os.path import dirname

from pylathedb.utils import get_logger,calculate_tf,calculate_iaf,calculate_inverse_frequency

from .babel_hash import BabelHash

logger = get_logger(__file__)
class ValueIndex():

    def __init__(self,**kwargs):
        self._dict = {}
        self.persistant_filename = kwargs.get('persistant_filename',None)
        self.update_cache = kwargs.get('update_cache',True)

    def __contains__(self, keyword):
        if keyword in self._dict:
            return True

        if self.persistant_filename:
            with shelve.open(self.persistant_filename,flag='r') as storage:
                if keyword in storage:
                    return True

        return False

    def _set_underlying_item(self, keyword, value):
        self._dict[keyword]=value

    def _get_underlying_item(self, keyword):
        item = None
        if keyword in self._dict:
            item = self._dict[keyword]
        elif self.persistant_filename:
            with shelve.open(self.persistant_filename,flag='r') as storage:
                item = storage.get(keyword,None)
                if self.update_cache:
                    self._set_underlying_item(keyword,item)
        if item is None:
            item = (0, BabelHash())
            # raise KeyError('The keyword {keyword} is not in the ValueIndex.')
        return item

    def __getitem__(self,keyword):
        _inverse_frequency, value = self._get_underlying_item(keyword)
        return value

    def __setitem__(self,keyword,value):
        raise Exception('Invalid operation. ValueIndex items shoud be set using the add_mapping method.')

    def __iter__(self):
        #This method only iterates over the keywords in memory
        yield from self._dict.keys()

    def keys(self):
        yield from self.__iter__()

    def items(self):
        for key in self.__iter__():
            yield key, self.__getitem__(key)

    def get_mappings(self,keyword,table,attribute):
        return self[keyword][table][attribute]

    def add_mapping(self,keyword,table,attribute,ctid):
        # This method does not change the persistant_filename
        self._dict.setdefault( keyword, (0, BabelHash() ) )
        self[keyword].setdefault(table , BabelHash() )
        self[keyword][table].setdefault( attribute , [] ).append(ctid)

    def get_inverse_frequency(self,keyword):
        inverse_frequency, _value = self._get_underlying_item(keyword)
        return inverse_frequency

    def set_inverse_frequency(self,keyword,inverse_frequency):
        # This method does not change the persistant_filename
        _old_inverse_frequency,old_value = self._dict[keyword]
        self._dict[keyword]= (inverse_frequency,old_value)

    def get_frequency(self,keyword,table,attribute):
        return len(self[keyword][table][attribute])

    def frequencies(self):
        for word in self:
            inverse_frequency = self.get_inverse_frequency(word)
            for table in self[word]:
                for attribute, ctids in self[word][table].items():
                    frequency = len(ctids)
                    yield table,attribute,frequency,inverse_frequency

    def get_iaf(self,weight_scheme,keyword):
        return calculate_iaf(weight_scheme, inverse_frequency = self.get_inverse_frequency(keyword))
    
    def get_tf(self,weight_scheme,keyword,table,attribute,max_frequency=None):
        return calculate_tf(weight_scheme,self.get_frequency(keyword,table,attribute), max_frequency = max_frequency)

    def __repr__(self):
        return f'<ValueIndex {repr(self._dict)}>'

    def persist_to_file(self,filename):
        makedirs(dirname(filename), exist_ok=True)
        with shelve.open(filename) as storage:
            storage['__babel__']=BabelHash.babel
            for key,underlying_value in self._dict.items():
                storage[key]=underlying_value

    def load_from_file(self,persistant_filename,**kwargs):
        '''
        The load_method specifies the set of keywords to me loaded, which can be:
        - keywords: A list or set of keywords to be loaded. If the list is empty,
        no keyword is loaded in this method.
        - sample: A sample of keywords from shelve is loaded. The number of keywords
        in the sample is defined by sample_size, which is 15 by default.
        - all_keywords: All the keywords from shelve are loaded. Beware that this
         load methods might be expensive.
        '''
        self._dict = {}

        self.persistant_filename = persistant_filename
        load_method = kwargs.get('load_method','keywords')
        keywords = kwargs.get('keywords',[])
        sample_size = kwargs.get('sample_size',15)

        with shelve.open(self.persistant_filename,flag='r') as storage:

            if load_method != 'keywords':
                keywords = storage.keys()

                if load_method == 'sample':
                    keywords = sample(keywords, k = sample_size)

            if '__babel__' in storage:
                BabelHash.babel.update(storage['__babel__'])

            for keyword in keywords:
                if keyword == '__babel__':
                    continue
                try:
                    self._set_underlying_item(keyword,storage[keyword])
                except KeyError:
                    self._set_underlying_item(keyword,None)
                    # print(f'Keyword {keyword} not present in the ValueIndex.')
                    continue
