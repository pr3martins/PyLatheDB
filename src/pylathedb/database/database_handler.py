import psycopg2
from psycopg2.sql import SQL, Identifier
from prettytable import PrettyTable
from itertools import groupby
import pandas as pd
from sqlalchemy import create_engine

from pylathedb.utils import ConfigHandler,get_logger

from .database_iter import DatabaseIter

logger = get_logger(__name__)

class DatabaseHandler:
    def __init__(self,config):
        self.config = config
        timeout=100000
        self.engine = create_engine('postgresql+psycopg2://{user}:{password}@{host}:5432/{database}'.format(**self.config.connection),
                                connect_args={"options": f"-c statement_timeout={timeout}"})

    def get_tables_and_attributes(self):
        with psycopg2.connect(**self.config.connection) as conn:
            with conn.cursor() as cur:
                sql = '''
                        SELECT table_name,column_name
                        FROM information_schema.columns
                        WHERE table_schema='public'
                        ORDER by 1,2;
                        '''
                cur.execute(sql)
                tables_attributes = {
                    table:[attribute for table,attribute in group]
                    for table,group in groupby(cur.fetchall(),lambda x:x[0])
                }
        return tables_attributes        

    def iterate_over_keywords(self,schema_index,**kwargs):
        database_table_columns=schema_index.tables_attributes()
        return DatabaseIter(self.config,database_table_columns,**kwargs)

    def get_fk_constraints(self):
        with psycopg2.connect(**self.config.connection) as conn:
            with conn.cursor() as cur:
                sql = '''
                    SELECT conname AS constraint_name,
                    conrelid::regclass AS table_name,
                    ta.attname AS column_name,
                    confrelid::regclass AS foreign_table_name,
                    fa.attname AS foreign_column_name
                    FROM (
                        SELECT conname, conrelid, confrelid,
                            unnest(conkey) AS conkey, unnest(confkey) AS confkey
                        FROM pg_constraint
                        WHERE contype = 'f'
                    ) sub
                    JOIN pg_attribute AS ta ON ta.attrelid = conrelid AND ta.attnum = conkey
                    JOIN pg_attribute AS fa ON fa.attrelid = confrelid AND fa.attnum = confkey
                    ORDER BY 1,2,4;
                '''
                cur.execute(sql)

                fk_constraints = {}
                for row in cur.fetchall():
                    constraint,table,attribute,foreign_table,foreign_attribute = (column.strip('\"') for column in row)
                    fk_constraints.setdefault((constraint,table),(table,foreign_table,[]))[2].append( (attribute,foreign_attribute) )
                    
                
                for constraint in fk_constraints:
                    table,foreign_table,attribute_mappings = fk_constraints[constraint]
                    sql = '''
                    SELECT NOT EXISTS (
                        SELECT COUNT(t1.*), {}
                        FROM {} t1
                        GROUP BY {}
                        HAVING COUNT(t1.*)>1
                    )
                    '''
                    attributes_param = SQL(', ').join(
                        Identifier(attribute) 
                        for attribute,foreign_attribute in attribute_mappings
                    )
                    sql= SQL(sql).format(
                        attributes_param,
                        Identifier(table),
                        attributes_param,
                    )
                    cur.execute(sql)
                    cardinality = '1:1' if cur.fetchone()[0] else 'N:1'            
                    fk_constraints[constraint] = (cardinality,table,foreign_table,attribute_mappings)

        return fk_constraints

    def exec_sql (self,sql,**kwargs):
        show_results=kwargs.get('show_results',True)
        table = None
        with psycopg2.connect(**self.config.connection) as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(sql)

                    if cur.description:
                        table = PrettyTable(max_table_width=200)
                        table.field_names = [ f'{i}.{col[0]}' for i,col in enumerate(cur.description)]
                        for row in cur.fetchall():
                            table.add_row(row)
                    if show_results:
                        print(table)
                except Exception as e:
                    print(e)
                    raise
                return table

    def get_dataframe(self,sql,**kwargs):
        timeout = kwargs.get('timeout',100000)
        # engine = create_engine('postgresql+psycopg2://{user}:{password}@{host}:5432/{database}'.format(**self.config.connection),
        #                         connect_args={"options": f"-c statement_timeout={timeout}"})
        df = pd.read_sql_query(sql, self.engine)
        return df

    def exist_results(self,sql):
        sql = f"SELECT EXISTS ({sql.rstrip(';')});"
        with psycopg2.connect(**self.config.connection) as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(sql)
                    return cur.fetchone()[0]
                except Exception as e:
                    print(e)
                    return False
        return None
