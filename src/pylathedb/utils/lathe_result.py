from graphviz import Digraph
from IPython.display import display

from pylathedb.candidate_network import CandidateNetwork
from pylathedb.query_match import QueryMatch
from pylathedb.keyword_match import KeywordMatch
from pylathedb.utils.printmd import printmd
from pylathedb.utils.shift_tab import shift_tab
from pylathedb.utils.ordinal import ordinal


class LatheResult():

    def __init__(self, index_handler,database_handler,data):
        self.index_handler=index_handler
        self.database_handler=database_handler
        self.data = data

    def cjns(self,**kwargs):
        show_str=kwargs.get('show_str',kwargs.get('text',False))
        show_graph=kwargs.get('show_graph',kwargs.get('graph',True))
        show_sql=kwargs.get('show_sql',kwargs.get('sql',False))
        show_df=kwargs.get('show_df',kwargs.get('df',kwargs.get('jnts',True)))
        top_k=kwargs.get('top_k',0)
        head=kwargs.get('head',5)

        for i,json_cn in enumerate(self.data['candidate_networks']):
            cjn=CandidateNetwork.from_json_serializable(json_cn)
            printmd('---')
            printmd(f'**{ordinal(i+1)} CJN**:')

            if show_str:
                printmd('---')
                print(f'Text:\n{shift_tab(cjn)}')
            if show_graph:
                g= Digraph(
                    graph_attr={'nodesep':'0.2','ranksep':'0.25'},
                    node_attr={'fontsize':"9.0",},
                    edge_attr={'arrowsize':'0.9',},
                 )
                for label,id in cjn.vertices():
                    g.node(id,label=str(label))
                for (label_a,id_a),(label_b,id_b) in cjn.edges():
                    g.edge(id_a,id_b)
                printmd('---')
                print('Graph:')
                display(g)
            if show_df or show_sql:
                sql=cjn.get_sql_from_cn(self.index_handler.schema_graph)
                if show_sql:
                    printmd('---')
                    print(f'SQL:\n{shift_tab(sql,sep="  ")}\n')
                if show_df:
                    printmd('---')
                    print(f'JNTs:')
                    df = self.database_handler.get_dataframe(sql)
                    if head>0:
                        df = df.head(head)
                    if df.empty:
                        print('\tThis CJN returns no tuples.')
                    else:
                        df_hide_tsvector = df.loc[:,~df.columns.str.endswith('tsvector')]
                        # dt = data_table.DataTable(df_hide_tsvector, include_index=False, num_rows_per_page=5)
                        # display(dt)
                        display(df_hide_tsvector)
            print()

            if i+1>=top_k and top_k>0:
                break

    def qms(self,**kwargs):
        top_k=kwargs.get('top_k',0)

        for i,json_qm in enumerate(self.data['query_matches']):
            qm=QueryMatch.from_json_serializable(json_qm)

            printmd('---')
            printmd(f'**{ordinal(i+1)} QM**:')
            printmd('---')
            print(qm,end='\n')

            if i+1>=top_k and top_k>0:
                break

    def kms(self,**kwargs):
        show_skms=kwargs.get('schema',True)
        show_vkms=kwargs.get('value',True)
        top_k=kwargs.get('top_k',0)

        self.skms(top_k=top_k)
        self.vkms(top_k=top_k)

    def skms(self,**kwargs):
        top_k=kwargs.get('top_k',0)
        printmd('---')
        printmd(f'**SKMs**:')
        printmd('---')

        if len(self.data['schema_keyword_matches']) == 0:
            print('There is no SKM for this keyword query.')
            return
        for i,json_km in enumerate(self.data['schema_keyword_matches']):
            km=KeywordMatch.from_json_serializable(json_km)
            #print(f'{i+1} KM:')
            print(km)
            if i+1>=top_k and top_k>0:
                break

    def vkms(self,**kwargs):
        top_k=kwargs.get('top_k',0)
        printmd('---')
        printmd(f'**VKMs**:')
        printmd('---')
        if len(self.data['value_keyword_matches']) == 0:
            print('There is no VKM for this keyword query.')
            return
        for i,json_km in enumerate(self.data['value_keyword_matches']):
            km=KeywordMatch.from_json_serializable(json_km)
            print(km)
            if i+1>=top_k and top_k>0:
                break

    def __repr__(self):
        return 'LatheResult'
