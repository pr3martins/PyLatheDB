import json
import logging
import re
from glob import glob


debug_mapping = { \
	'DEBUG' : logging.DEBUG,\
	'INFO': logging.INFO, \
	'ERROR': logging.ERROR, \
	'WARNING': logging.WARNING, \
	'CRITICAL': logging.CRITICAL \
}

class ConfigHandler:

    #this prevents AttributeError when linting the code
    known_entries = {
        'config_directory',
        'connection',
        'logging_mode',
        'create_index',
        'results_directory',
        'plots_directory',
        'queryset_config_filepath',
        'dataset_config_filepath',
        'queryset_filepath',
        'queryset_name',
        'attributes_filepath',
        'schema_index_filepath',
        'value_index_filepath',
        'schema_graph_filepath',
    }

    def __init__(self,**kwargs):

        config_directory = kwargs.get('config_directory','../config/')
        queryset_config_filepath = kwargs.get('queryset_config_filepath',None)
        dataset_config_filepath = kwargs.get('dataset_config_filepath',None)

        self.config = self.__dict__
        
        #General Configs
        general_config_filename = f'{config_directory}config.json'
        general_config = self.load_config(general_config_filename)     
        if 'logging_mode' in general_config:
          general_config['logging_mode'] = debug_mapping[general_config['logging_mode']]           
        self.update_paths(general_config,config_directory)
        self.config.update(general_config)
        self.config_directory =config_directory
        
        # Query Set Configs
        if queryset_config_filepath is not None:
            self.queryset_config_filepath = config_directory+queryset_config_filepath
        
        queryset_config = self.load_config(self.queryset_config_filepath)
        self.update_paths(queryset_config,config_directory)
        self.config.update(queryset_config)

        #Dataset Configs
        if dataset_config_filepath is not None:
            self.dataset_config_filepath = config_directory+dataset_config_filepath     

        dataset_config = self.load_config(self.dataset_config_filepath)
        self.update_paths(dataset_config,config_directory)
        if 'database' in dataset_config:
            self.connection['database']=dataset_config['database']
            del dataset_config['database']
        self.config.update(dataset_config)

    def update_paths(self,config,prefix_directory):
        re_path = re.compile(".+(_filepath|_directory)")
        for key in config:
            #keys that contains path info and are not None
            if config[key] and re_path.match(key):
                config[key]=prefix_directory+config[key]


    def load_config(self,filepath):
        with open(filepath, 'r') as f:
            config = json.load(f)
        return config
    
    def set_logging_level(self,ĺevel):
        self.logging_mode = debug_mapping[ĺevel]
        ConfigHandler.__instance['logging_mode']=self.logging_mode

    def change_queryset(self,ans=None):
        if ans is None:
            print(f'Please input a query set to change:')
            for i,(name,path) in enumerate(self.get_queryset_configs()):
                print(f'{i+1:02d} - {name}')
            return None
        config_directory=self.config_directory
        if isinstance(ans, int):
            name,path = self.get_queryset_configs()[ans-1]
        else:
            for name,path in self.get_queryset_configs():
                if name == ans:
                    break
        print(path)
        previous_database = self.connection['database']
        
        self.queryset_config_filepath = path
        
        queryset_config = self.load_config(path)
        self.update_paths(queryset_config,config_directory)
        self.config.update(queryset_config)
        print(f'Changed to query set {self.queryset_name}.')

        dataset_config = self.load_config(self.dataset_config_filepath)
        self.update_paths(dataset_config,self.config_directory)
        if 'database' in dataset_config:
            new_database = dataset_config['database']
            self.connection['database']=new_database
            if new_database!=previous_database:
                print(f'Changed to dataset {new_database}.')
            del dataset_config['database']
        self.config.update(dataset_config)

    def get_dataset_configs(self):
        subfolder = 'dataset_configs/'
        results = []
        for filepath in glob(f'{self.config_directory}{subfolder}*.json'):
            with open(filepath,'r') as f:
                results.append( (json.load(f)['database'], filepath) )
        return results

    def get_queryset_configs(self,dataset_config_filepath=None):
        subfolder = 'queryset_configs/'
        results = []
        for filepath in glob(f'{self.config_directory}{subfolder}*.json'):
            with open(filepath,'r') as f:
                data = json.load(f)
                if dataset_config_filepath in (None,data['dataset_config_filepath']):
                    results.append( (data['queryset_name'], filepath) )
        return sorted(results)
