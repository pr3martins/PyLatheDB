import gc
import shelve
from contextlib import ExitStack
import json
from pprint import pprint as pp
from math import log
from os.path import exists
from glob import glob
from os import remove, listdir
from re import match

from pylathedb.utils import ConfigHandler,get_logger,memory_size,calculate_tf,calculate_inverse_frequency,calculate_iaf
from pylathedb.database import DatabaseHandler

from .value_index import ValueIndex
from .schema_index import SchemaIndex
from .schema_graph import SchemaGraph
from .babel_hash import BabelHash
import sys
logger = get_logger(__name__)

class IndexHandler:
    def __init__(self, config,database_handler):
        self.config = config
        self.database_handler = database_handler
        
        self.value_index = ValueIndex()
        self.schema_index = SchemaIndex()
        self.schema_graph = SchemaGraph()
        
        self.partial_index_count=0

    def create_indexes(self):
        if not self.config.create_index:
            print(f'Index Creation is disabled.')
            return

        self.create_schema_graph()
        self.create_partial_schema_index()
        self.remove_partial_value_indexes()
        self.create_partial_value_indexes()
        self.merge_partial_value_indexes_and_process_max_frequency()
        self.process_norms()
        self.remove_partial_value_indexes()

    def create_schema_graph(self):
        #TODO considerar custom fk constraints
        fk_constraints = self.database_handler.get_fk_constraints()

        for constraint,values in fk_constraints.items():
            # cardinality,table,foreign_table,attribute_mappings = values
            self.schema_graph.add_fk_constraint(constraint,*values)
        self.schema_graph.persist_to_file(f'{self.config.dataset_directory}/schema_graph')

    def create_partial_schema_index(self):
        tables_attributes = None

        if self.config.attributes_filepath is not None:
            print('Get table_attributes from attributes_filepath')
            with open(self.config.attributes_filepath,'r') as attributes_filepath:
                tables_attributes = [
                    (item['table'],item['attributes'])
                    for item in json.load(attributes_filepath) 
                ]
        else:
            print('Get table_attributes from database')
            tables_attributes = list(self.database_handler.get_tables_and_attributes().items())

        metrics = {'max_frequency':0, 'norm':[0,0,0,0]}
        self.schema_index.create_entries(tables_attributes,metrics)
        print(f'num attr: {self.schema_index.get_num_total_attributes()}')
        print(f'ATTRIBUTES:')
        pp(self.schema_index.tables_attributes())

    def create_partial_value_indexes(self,**kwargs):
        unit = kwargs.get('unit', 2**30)
        max_memory_allowed = kwargs.get('max_memory_allowed', 20)
        max_memory_allowed *= unit
        gb = 1 * unit
        
        def part_index():
            self.partial_index_count+=1
            partial_index_filename = f'{self.config.dataset_directory}/value_index.part{self.partial_index_count:02d}'
            print(partial_index_filename)
            logger.info(f'Storing value_index.part{self.partial_index_count} in {partial_index_filename}')
            self.value_index.persist_to_file(partial_index_filename)
            self.value_index = ValueIndex()
            gc.collect()
            logger.info('Indexing operation resumed.')


        for table,ctid,attribute, word in self.database_handler.iterate_over_keywords(self.schema_index):
            if memory_size() >= max_memory_allowed:
                logger.info(f'Memory usage exceeded the maximum memory allowed (above {max_memory_allowed/gb:.2f}GB).')
                part_index()
            self.value_index.add_mapping(word,table,attribute,ctid)

        part_index()

    def remove_partial_value_indexes(self):
        filenames=glob(f'{self.config.dataset_directory}/value_index.part*')
        for filename in filenames:
            if exists(filename):
                remove(filename)
            else:
                print(f"The file {filename} does not exist.") 
        self.partial_index_count=0


    def merge_partial_value_indexes_and_process_max_frequency(self):
        '''    return '\n'.join(print_string)

        Para garantir que todos os arquivos abertos serão fechados, é bom utilizar
        a cláusula with. Mas como eu quero abrir um numero n de arquivos, eu pensei
        em utilizar essa classe ExitStack.

        Ao todo utilizaremos n arquivos parciais e um arquivo final/completo
         é para poder utilizar vários arquivos
        dentro do with. Dessa forma, eu garanto que todos eles serao fechados.
        '''

        num_total_attributes=self.schema_index.get_num_total_attributes()
        babel = BabelHash.babel

        dataset_directory=self.config.dataset_directory
        # Gets value_index parts ignoring the extensions and possible 
        filenames={f'{dataset_directory}/{m.groups()[0]}' for f in listdir(dataset_directory) if (m:= match(r'(value_index.part\d+).*', f))}
        
        with ExitStack() as stack:
            partial_value_indexes = [stack.enter_context(shelve.open(fname,flag='r')) for fname in filenames]
            final_value_index = stack.enter_context(shelve.open(f'{dataset_directory}/value_index',flag='n'))

            for partial_value_index in partial_value_indexes:
                babel.update(partial_value_index['__babel__'])
            final_value_index['__babel__'] = babel

            for i in range( len(partial_value_indexes)):
                for word in partial_value_indexes[i]:

                    if word in final_value_index or word == '__babel__':
                        continue

                    value_list = [ partial_value_indexes[i][word] ]

                    for j in range(i+1,len(partial_value_indexes)):
                        print(f'  j {j} {filenames[j]}')
                        if word in partial_value_indexes[j]:
                            value_list.append(partial_value_indexes[j][word])

                    merged_babel_hash = BabelHash()
                    if len(value_list)>1:
                        for ( _ ,part_babel_hash) in value_list:
                            for table in part_babel_hash:
                                merged_babel_hash.setdefault(table,BabelHash())
                                for attribute in part_babel_hash[table]:
                                    merged_babel_hash[table].setdefault( attribute , [] )
                                    merged_babel_hash[table][attribute]+= part_babel_hash[table][attribute]
                    else:
                        _ ,merged_babel_hash = value_list[0]

                    for table in merged_babel_hash:
                        for attribute in merged_babel_hash[table]:
                            frequency = len(merged_babel_hash[table][attribute])
                            max_frequency = self.schema_index[table][attribute]['max_frequency']
                            if frequency > max_frequency:
                               self.schema_index[table][attribute]['max_frequency'] = frequency


                    num_attributes_with_this_word = sum([len(merged_babel_hash[table]) for table in merged_babel_hash])

                    inverse_frequency = calculate_inverse_frequency(num_total_attributes,num_attributes_with_this_word)
                    merged_value = (inverse_frequency,merged_babel_hash)
                    final_value_index[word]=merged_value

    def process_norms(self):
        with shelve.open(f'{self.config.dataset_directory}/value_index') as full_index:
            for word in full_index:
                if word == '__babel__':
                    continue

                inverse_frequency, babel_hash = full_index[word]
                for table in babel_hash:
                    for attribute in babel_hash[table]:
                        frequency = len(babel_hash[table][attribute])
                        max_frequency = self.schema_index[table][attribute]['max_frequency']
                        
                        for weight_scheme in range(4):
                            tf  = calculate_tf(weight_scheme,frequency,max_frequency)
                            iaf = calculate_iaf(weight_scheme,inverse_frequency)
                            self.schema_index[table][attribute]['norm'][weight_scheme] += (tf*iaf)**2

        for table in self.schema_index:
            for attribute in self.schema_index[table]:
                for weight_scheme in range(4):
                    self.schema_index[table][attribute]['norm'][weight_scheme] **= 0.5

        self.schema_index.persist_to_file(f'{self.config.dataset_directory}/schema_index')

    def load_indexes(self,**kwargs):
        dataset_directory = self.config.dataset_directory
        self.schema_graph.load_from_file(f'{dataset_directory}/schema_graph')
        self.value_index.load_from_file(f'{dataset_directory}/value_index',**kwargs)
        self.schema_index.load_from_file(f'{dataset_directory}/schema_index')
