import os
import gunicorn
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State,Event
import plotly.graph_objs as go
from datetime import datetime
import numpy as np

import datetime
from collections import defaultdict
from collections import Counter
from operator import itemgetter

import pandas as pd
import networkx as nx
from collections import defaultdict
from data_import import *
from graph_manipulation import *
from time_aware_splits import *
from popularity_based_rec import *
from personalized_prank import *
from pathsim import *
from simrank import *
from accuracy_evaluation import *

# class GraphManipulation:
#
#     def __init__(self, directed=False, G_structure='USA'):
#         if directed == True:
#             self.G = nx.DiGraph()
#         else:
#             self.G = nx.Graph()
#
#         self.G_structure = G_structure
#
#
#     def create_graph(self, df, pk_item = 'pk_article'):
#         '''
#         Create a graph structure on the base of the given dataframe
#         '''
#
#         self.pk_item = pk_item
#
#         # --- Extract unique users, sessions and articles from the dataframe
#         users = df.pk_user.unique()
#         sessions = df.pk_session.unique()
#         articles = df[pk_item].unique()
#
#         # --- Create graph nodes for three entities
#         self.G.add_nodes_from(users, entity='U')
#         self.G.add_nodes_from(articles, entity='A')
#
#         for s in sessions:
#             session_start = df[df['pk_session'] == s]['date-time'].min()
#             self.G.add_node(s, datetime=session_start, entity='S')
#
#         # --- Create edges from users to sessions with the edge type "US"
#         for u in users:
#             s_list = list(df[df['pk_user'] == u]['pk_session'])
#             for s in s_list:
#                 #self.G.add_edges_from([(u, s)], edge_type='US')
#                 self.G.add_edge(u, s, edge_type='US')
#
#         # --- Create edges from sessions to articles with the edge type "SA",
#         # each edge has an additional attributes "reading_datetime" (when article was read)
#         # and "timeview" (for how long the article was being read)
#         for s in sessions:
#             a_list = list(df[df['pk_session'] == s][pk_item])
#             for a in a_list:
#                 date_time = pd.to_datetime(df[(df['pk_session'] == s) &
#                                               (df[pk_item] == a)]['date-time'].values[0])
#                 timeview = pd.to_numeric(df[(df['pk_session'] == s) &
#                                             (df[pk_item] == a)]['timeview'].values[0])
#                 #self.G.add_edges_from([(s, a)], edge_type='SA', reading_datetime=date_time, timeview=timeview)
#                 self.G.add_edge(s, a, edge_type='SA', reading_datetime=date_time, timeview=timeview)
#
#         if self.G_structure == 'USAC':
#             categories = df.pk_category.unique()
#             self.G.add_nodes_from(categories, entity='C')
#             for a in articles:
#                 c = list(df[df[self.pk_item] == a]['pk_category'])[0]
#                 self.G.add_edge(a, c, edge_type='AC')
#
#
#     # def add_categories_data(self, cat_df):
#     #
#     #     self.G_structure = 'USAC'
#     #
#     #     cat_nodes = list(cat_df)
#     #     self.G.add_nodes_from(cat_nodes, entity='C')
#     #
#     #     # Create a dict from the dataframe
#     #     cat_dict = cat_df.to_dict('index')
#     #
#     #     # For each article assign the most probable category
#     #     # (later can be assigned several with different weights)
#     #     article_cat = dict()
#     #     for a in cat_dict:
#     #         most_probable_cat = [k for k, v in cat_dict[a].items() if v == max(cat_dict[a].values())][0]
#     #         article_cat[a] = most_probable_cat
#     #
#     #     list_of_edges = [(a, article_cat[a]) for a in article_cat]
#     #
#     #     # --- Create edges from articles to categories with the edge type "AC"
#     #     articles = [n for n, attr in self.G.nodes(data=True) if attr['entity']=='A']
#     #
#     #     for (a, c) in list_of_edges:
#     #         if a in articles:
#     #             self.G.add_edge(a, c, edge_type='AC')
#     #
#     #     # --- Remove articles without categories
#     #     for a in self.get_articles(self.G):
#     #         if self.map_category(a) == '':
#     #             self.G.remove_node(a)
#     #
#     #     # --- Remove sessions without articles
#     #     for s in self.get_sessions(self.G):
#     #         a_list = [a for a in self.G[s] if self.G[s][a]['edge_type'] == 'SA']
#     #         if len(a_list) == 0:
#     #             self.G.remove_node(s)
#     #
#     #     # --- Remove users without sessions
#     #     for u in self.get_users(self.G):
#     #         s_list = [s for s in self.G[u] if self.G[u][s]['edge_type'] == 'US']
#     #         if len(s_list) == 0:
#     #             self.G.remove_node(u)
#     #
#     #
#     # def add_video_cat_data(self, cat_df):
#     #
#     #     self.G_structure = 'USAC'
#     #
#     #     cat_nodes = cat_df['video_category_id'].unique()
#     #     self.G.add_nodes_from(cat_nodes, entity='C')
#     #
#     #     for a in self.get_articles(self.G):
#     #         if a in cat_df.index:
#     #             c = cat_df.get_value(a, 'video_category_id')
#     #             self.G.add_edge(a, c, edge_type='AC')
#     #         else:
#     #             self.G.remove_node(a)
#     #
#     #     # --- Remove sessions without articles
#     #     for s in self.get_sessions(self.G):
#     #         a_list = [a for a in self.G[s] if self.G[s][a]['edge_type'] == 'SA']
#     #         if len(a_list) == 0:
#     #             self.G.remove_node(s)
#     #
#     #     # --- Remove users without sessions
#     #     for u in self.get_users(self.G):
#     #         s_list = [s for s in self.G[u] if self.G[u][s]['edge_type'] == 'US']
#     #         if len(s_list) == 0:
#     #             self.G.remove_node(u)
#
#
#     def add_locations_data(self, loc_df):
#
#         self.G_structure = 'USACL'
#
#         loc_nodes = loc_df.location.unique()
#         self.G.add_nodes_from(loc_nodes, entity='L')
#
#
#         # For each article in the graph assign its location
#
#         articles = self.get_articles(self.G)
#         for a in articles:
#             l = list(loc_df[loc_df['article'] == a]['location'])[0]
#             self.G.add_edge(a, l, edge_type='AL')
#
#
#     @staticmethod
#     def remove_sessions_with_one_article(test_G):
#
#         # Remove sessions that have only one article
#         short_sessions = [n for n, attr in test_G.nodes(data=True)
#                          if attr['entity']=='S' and test_G.degree(n) == 2] # degree 2 means only one path U-S-A exists
#         test_G.remove_nodes_from(short_sessions)
#
#         # Remove users that don't have sessions anymore
#         single_users = [n for n, attr in test_G.nodes(data=True)
#                         if attr['entity'] == 'U' and nx.degree(test_G, n) == 0]
#         test_G.remove_nodes_from(single_users)
#
#         # Remove articles that don't appear in any session anymore
#         single_articles = [n for n, attr in test_G.nodes(data=True)
#                            if attr['entity'] == 'A'
#                            and all([test_G[n][m]['edge_type'] != 'SA' for m in test_G[n]])]
#         test_G.remove_nodes_from(single_articles)
#
#         return test_G
#
#     @staticmethod
#     def filter_meaningless_sessions(g, timeview):
#
#         sessions = [s for s,attr in g.nodes(data=True) if attr['entity']=='S']
#
#         for session in sessions:
#             timeviews = [attr['timeview'] for s, a, attr in g.edges(data=True)
#                          if attr['edge_type'] == 'SA' and (s==session or a==session)]
#             if all([t<=timeview for t in timeviews]):
#                 # print(session, timeviews)
#                 g.remove_node(session)
#
#         # Remove users that don't have sessions anymore
#         single_users = [n for n, attr in g.nodes(data=True)
#                         if attr['entity'] == 'U' and nx.degree(g, n) == 0]
#         g.remove_nodes_from(single_users)
#
#         # Remove articles that don't appear in any session anymore
#         single_articles = [n for n, attr in g.nodes(data=True)
#                            if attr['entity'] == 'A'
#                            and all([g[n][m]['edge_type'] != 'SA' for m in g[n]])]
#         g.remove_nodes_from(single_articles)
#
#         return g
#
#
#
#
#
#     @staticmethod
#     def filter_sessions(test_G, n_items = 1):
#
#         # Remove sessions that have only one article
#         short_sessions = [n for n, attr in test_G.nodes(data=True)
#                           if attr['entity'] == 'S' and test_G.degree(n) <= (n_items + 1)] # (one edge goes to user)
#         test_G.remove_nodes_from(short_sessions)
#
#         # Remove users that don't have sessions anymore
#         single_users = [n for n, attr in test_G.nodes(data=True)
#                         if attr['entity'] == 'U' and nx.degree(test_G, n) == 0]
#         test_G.remove_nodes_from(single_users)
#
#         # Remove articles that don't appear in any session anymore
#         single_articles = [n for n, attr in test_G.nodes(data=True)
#                            if attr['entity'] == 'A'
#                            and all([test_G[n][m]['edge_type'] != 'SA' for m in test_G[n]])]
#         test_G.remove_nodes_from(single_articles)
#
#         return test_G
#
#     @staticmethod
#     def filter_timeviews(test_G, timeview=2):
#
#         # Remove sessions that have only one article
#         short_timeviews = [(s, a) for s, a, attr in test_G.edges(data=True)
#                            if attr['edge_type'] == 'SA' and attr['timeview'] < timeview]
#         test_G.remove_edges_from(short_timeviews)
#
#         # Remove articles that don't appear in any session anymore
#         single_articles = [n for n, attr in test_G.nodes(data=True)
#                            if attr['entity'] == 'A'
#                            and all([test_G[n][m]['edge_type'] != 'SA' for m in test_G[n]])]
#         test_G.remove_nodes_from(single_articles)
#
#         # Remove users that don't have sessions anymore
#         single_users = [n for n, attr in test_G.nodes(data=True)
#                         if attr['entity'] == 'U' and nx.degree(test_G, n) == 0]
#         test_G.remove_nodes_from(single_users)
#
#         return test_G
#
#
#     @staticmethod
#     def filter_users(test_G, n_sessions = 1):
#
#         # Remove users that have less than specified number of sessions
#         inactive_users = [n for n, attr in test_G.nodes(data=True)
#                           if attr['entity'] == 'U' and test_G.degree(n) < n_sessions]
#         test_G.remove_nodes_from(inactive_users)
#
#         # Remove sessions that don't have user anymore
#         single_sessions = [n for n, attr in test_G.nodes(data=True)
#                            if attr['entity'] == 'S' and nx.degree(test_G, n) == 0]
#         test_G.remove_nodes_from(single_sessions)
#
#         # Remove articles that don't appear in any session anymore
#         single_articles = [n for n, attr in test_G.nodes(data=True)
#                            if attr['entity'] == 'A'
#                            and all([test_G[n][m]['edge_type'] != 'SA' for m in test_G[n]])]
#         test_G.remove_nodes_from(single_articles)
#
#         return test_G
#
#
#     @staticmethod
#     def remove_users_that_werent_active_in_train(test_G, train_G):
#
#         train_users = [n for n, attr in train_G.nodes(data=True) if attr['entity']=='U']
#         test_users = [n for n, attr in test_G.nodes(data=True) if attr['entity']=='U']
#
#         only_test_active_users = [n for n in test_users if n not in train_users]
#
#         sessions_to_remove = []
#         for u in only_test_active_users:
#             u_sessions = [n for n, attr in test_G.nodes(data=True)
#                           if attr['entity']=='S'
#                           and nx.has_path(test_G, u, n) == True
#                           and nx.shortest_path_length(test_G, u, n) == 1]
#
#             sessions_to_remove.extend(u_sessions)
#
#         test_G.remove_nodes_from(only_test_active_users)
#         test_G.remove_nodes_from(sessions_to_remove)
#
#         single_articles = [a for a,attr in test_G.nodes(data=True)
#                            if attr['entity']=='A' and nx.degree(test_G, a) == 0] # Will not work with having categories !
#
#         test_G.remove_nodes_from(single_articles)
#
#         return test_G
#
#
#     @staticmethod
#     def remove_items_that_didnt_exist_in_train(test_G, train_G):
#
#         train_items = [n for n, attr in train_G.nodes(data=True) if attr['entity'] == 'A']
#         test_items = [n for n, attr in test_G.nodes(data=True) if attr['entity'] == 'A']
#
#         only_test_items = [n for n in test_items if n not in train_items]
#
#         test_G.remove_nodes_from(only_test_items)
#
#         return test_G
#
#     # def extract_subgaph_given_time_interval(self, start_time, end_time):
#     #
#     #     sessions = [n for n,attr in self.G.nodes(data=True)
#     #                 if attr['entity'] == 'S'
#     #                 and attr['datetime'] >= start_time
#     #                 and attr['datetime'] < end_time]
#     #
#     #     sub_G = nx.Graph()
#     #     for s in sessions:
#     #         nodes = nx.neighbors(self.G, s)
#     #         nodes.append(s)
#     #         temp_sub_G = self.G.subgraph(nodes)
#     #         sub_G = nx.compose(sub_G, temp_sub_G)
#     #
#     #     return sub_G
#
#     @staticmethod
#     def derive_adjacency_multigraph(G, entity1, entity2, path_len=2):
#         """
#         Derive a "bipartite" graph from a heterogenuos graph
#
#         Parameters
#         -----------
#         entity1 : character
#         entity2 : character
#         path_len : the length of the path needed to go from entity1 to entity2
#
#         Returns
#         -----------
#         G_new : a new multi-graph, consising of only two specified entities and multiple links between them
#         """
#
#         # --- Initialization
#         adj = defaultdict(list)
#         nodes1 = [(n,attr) for n,attr in G.nodes(data=True) if attr['entity'] == entity1]
#         nodes2 = [(n,attr) for n,attr in G.nodes(data=True) if attr['entity'] == entity2]
#
#         # --- Building adjacency matrix
#         for n1, _ in nodes1:
#             adj[n1] = defaultdict(list)
#             for n2, _ in nodes2:
#                 adj[n1][n2] = len([path for path in nx.all_simple_paths(G, n1, n2, path_len)])
#
#         # --- Building a multi-graph on the base of adjacency matrix
#
#         G_new = nx.MultiGraph()
#
#         G_new.add_nodes_from(nodes1)#, entity=entity1)
#         G_new.add_nodes_from(nodes2)#, entity=entity2)
#
#         for n1 in adj:
#             for n2 in adj[n1]:
#                 for i in range(adj[n1][n2]):
#                     G_new.add_edges_from([(n1, n2)])
#
#         return G_new
#
#
#     @staticmethod
#     def get_users(g):
#         return [n for n, attr in g.nodes(data=True) if attr['entity'] == 'U']
#
#     @staticmethod
#     def get_sessions(g):
#         return [n for n, attr in g.nodes(data=True) if attr['entity'] == 'S']
#
#     @staticmethod
#     def get_articles(g):
#         return [n for n, attr in g.nodes(data=True) if attr['entity'] == 'A']
#
#     @staticmethod
#     def get_categories(g):
#         return [n for n, attr in g.nodes(data=True) if attr['entity'] == 'C']
#
#     @staticmethod
#     def get_locations(g):
#         return [n for n, attr in g.nodes(data=True) if attr['entity'] == 'L']
#
#     @staticmethod
#     def get_nodes(g, entity):
#         return [n for n, attr in g.nodes(data=True) if attr['entity'] == entity]
#
#     @staticmethod
#     def get_sessions_per_user(g):
#
#         users = [n for n, attr in g.nodes(data=True) if attr['entity'] == 'U']
#         sessions_per_user = []
#         for u in users:
#             sessions_per_user.append(len(g[u]))
#
#         return sessions_per_user
#
#     @staticmethod
#     def get_articles_per_session(g):
#
#         sessions = [n for n, attr in g.nodes(data=True) if attr['entity'] == 'S']
#
#         articles_per_session = []
#         for s in sessions:
#             articles_per_session.append(nx.degree(g, s)-1) # -1 because one edge goes to the user
#
#         return articles_per_session
#
#     @staticmethod
#     def create_sac_graph(g):
#
#         user_nodes = [n for n, attr in g.nodes(data=True) if attr['entity'] == 'U']
#         g.remove_nodes_from(user_nodes)
#
#         return g
#
#     @staticmethod
#     def create_sa_graph(g):
#
#         new_g = g.copy()
#         user_category_nodes = [n for n, attr in g.nodes(data=True) if attr['entity'] in ['U','C']]
#         new_g.remove_nodes_from(user_category_nodes)
#
#         return new_g
#
#     @staticmethod
#     def create_subgraph_of_adjacent_entities(g, list_of_entities):
#
#         new_g = g.copy()
#         nodes = [n for n, attr in g.nodes(data=True) if attr['entity'] not in list_of_entities]
#         new_g.remove_nodes_from(nodes)
#
#         return new_g
#
#
#
#     def map_category(self, a):
#
#         c = [c for c in self.G[a] if self.G[a][c]['edge_type'] == 'AC']
#         c = c[0] if len(c) > 0 else ''
#
#         return c
#
#     def map_location(self, a):
#
#         l = [l for l in self.G[a] if self.G[a][l]['edge_type'] == 'AL']
#         l = l[0] if len(l) > 0 else ''
#
#         return l
#
#     @staticmethod
#     def map_timeview(g, session, article):
#
#         t = [attr['timeview'] for s, a, attr in g.edges(data=True) if (s==session and a==article) or (s==article and a==session)][0]
#
#         return t
#
#     def map_enity(self, n):
#
#         entity = self.G.node[n]['entity']
#
#         return entity
#
#
#     def count_unique_categories(self, rec):
#
#         c_list = []
#         for a in rec:
#             c = [c for c in self.G[a] if self.G[a][c]['edge_type'] == 'AC']
#             if len(c) > 0:
#                 c_list.append(c[0])
#             #print(a, c)
#             #continue
#             #c_list.append(c)
#
#         return len(list(set(c_list)))
#
#     @staticmethod
#     def get_active_users(g, n_sessions = 1):
#
#         active_users = [n for n, attr in g.nodes(data=True)
#                              if attr['entity'] == 'U' and g.degree(n) >= n_sessions]
#
#         return active_users


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
datasets = ['SDF','Video33','KIM']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.layout = html.Div([
    dcc.Input(id='my-id', value='initial value', type='text'),
    dcc.Interval(
        id='interval-component',
        interval=10 * 1000,  # in milliseconds
        n_intervals=100
    ),
    html.Div([
        dcc.Dropdown(
            id='xaxis-column',
            options=[{'label': i, 'value': i} for i in datasets],
            value='SDF'

        )]),
    html.Div([

        html.Div('Number of Splits'),

        dcc.Input(id='number_of_splits',
                  placeholder='Number Of Splits',
                  type='number',
                  min=1,
                  step=1,
                  value=12,

                  ),


        html.Div('Short Days Window'),

        dcc.Input(id='short_days',
                  placeholder='Short Days Window',
                  type='number',
                  min=1,
                  step=1,
                  value=1,

                  ),

        html.Div('Number of Recommendations'),

        dcc.Input(id='number_recommendations',
                  placeholder='Number Of Recommendations',
                  type='number',
                  min=1,
                  step=1,
                  value=1,
                  ),

        html.Div('Minimum Number of Items per Session'),

        dcc.Input(id='min_items_n',
                  placeholder='Minimum Number of Items per Session',
                  type='number',
                  min=1,
                  step=1,
                  value=1,

                  )

    ],style={'columnCount': 2}),

    html.Br(),
    html.Label('Node types',style={'font-weight':'bold'}),
    html.Div([
    dcc.Checklist(id='nodes',
        options=[
            {'label': 'Users', 'value': 'U'},
            {'label': 'Articles', 'value': 'A'},
            {'label': 'Sessions', 'value': 'S'}
        ],
        values=['S', 'A']
    ),]),
    html.Br(),
    html.Label('Methods Comparison',style={'font-weight':'bold'}),
    html.Div([
    dcc.Checklist(id='methods',
        options=[
            {'label': 'Pop', 'value': 'POP'},
            {'label': 'RWR', 'value': 'RWR'},
            {'label': 'Simrank', 'value': 'Simrank'},
            {'label': 'Pathsim', 'value': 'Pathsim'},
            {'label': 'SKNN', 'value': 'SKNN'}
        ],
        values=['RWR',]
    ),],style={'columnCount': 3}),
    html.Button('Execute',id='execute'),
    html.Div([
        html.Div(id='log_updates'),


        dcc.Graph(id='evaluation'),



    ])
],)


# @app.callback(
#     Output(component_id='my-div', component_property='children'),
#     [Input(component_id='my-id', component_property='value')]
# )
# def update_output_div(input_value):
#     return 'You\'ve entered "{}"'.format(input_value)

@app.callback(Output(component_id='log_updates',component_property='children'),
              events=[Event('interval-component','interval')])
def update_text_area():
    print('Loading')
    return 'Loading '

@app.callback(
    Output(component_id='evaluation', component_property='figure'),
    [Input('execute', 'n_clicks')
     ],state=[  State('xaxis-column','value'),
                State('number_of_splits','value'),
                State('short_days','value'),
                State('number_recommendations','value'),
                State('min_items_n','value'),
                State('nodes','values'),
                State('methods','values'),
              ]
)
def update_div(n_clicks,data_path,number_splits,short_days,number_recommendations,min_items_n,nodes,methods):
    if n_clicks == None:
        print('First Time')
    else:

        DATA_PATH = f'./Data/{data_path} - pk_client, pk_session, pk_article, timeview (s), date, time.txt'
        CAT_DATA_PATH = f'./Data/{data_path}-5topics-doc-topics.txt'
        LOC_DATA_PATH = f'./Data/{data_path} - pk_article, pk_district.txt'
        gm = GraphManipulation()
        di = DataImport()
        di.import_user_click_data(DATA_PATH, adjust_pk_names=True)
        print('Sterguis')
        # --- Reduce dataset to 1 month / 1 week / ...
        # di.reduce_timeframe(dt.datetime(2017,3,1), dt.datetime(2017,3,31)) # if G_Video33_1month is selected
        # di.reduce_timeframe(dt.datetime(2017, 3, 1), dt.datetime(2017, 3, 7)) # if G_Video33_1week is selected

        # --- Remove inactive users (the ones with small number of sessions in total)
        # di.remove_inactive_users(n_sessions=MIN_N_SESSIONS)
        #
        # ---------- Add categories -----------------------------
        print(f'{datetime.datetime.now()} Import Categories')
        di.import_categories_data(CAT_DATA_PATH)
        print(f'{datetime.datetime.now()} Import Categories End')

        print(f'{datetime.datetime.now()} Filter Short Session')
        # ---- Leave only sessions with at least specified number of articles
        di.filter_short_sessions(n_items=min_items_n)
        print(f'{datetime.datetime.now()} Filter Short Session End')


        # ------ Create a graph on the base of the dataframe ----
        print(f'{datetime.datetime.now()} Graph Manipulation')
        gm = GraphManipulation(G_structure='USAC')
        print(f'{datetime.datetime.now()} Graph Manipulation End')

        print(f'{datetime.datetime.now()} Create Graph')
        gm.create_graph(di.user_ses_df)
        print(f'{datetime.datetime.now()} Create Graph End')

        # Filter again, because dataframe filtering leaves sessions where the same article is repeatedly read several times
        # gm.filter_sessions(gm.G, n_items=MIN_ITEMS_N)
        # gm.filter_users(gm.G, n_sessions=MIN_N_SESSIONS)

        # ---------- Add locations ------------------------------
        di.import_locations_data(LOC_DATA_PATH)
        gm.add_locations_data(di.locations_data)
        G = gm.G
        print('Stergios')
        print('\n--- GENERAL STATISTICS ---')
        print('Number of users:', len(gm.get_users(G)))
        print('Number of sessions:', len(gm.get_sessions(G)))
        print('Number of articles:', len(gm.get_articles(G)))
        print('Number of categories:', len(gm.get_categories(G)))
        print('Number of locations:', len(gm.get_locations(G)))

        art_per_session = gm.get_articles_per_session(gm.G)
        print('Avg # of articles per session:', round(np.mean(art_per_session), 2))
        print('Max # of articles per session:', round(np.max(art_per_session), 2))

        ses_per_user = gm.get_sessions_per_user(gm.G)
        print('Avg # of sessions per user:', round(np.mean(ses_per_user), 2))
        print('Max # of sessions per user:', round(np.max(ses_per_user), 2))

        tas = TimeAwareSplits(G)
        tas.create_time_split_graphs(G, num_splits=number_splits)
        tas.create_time_window_graphs(1)
        _dump_process = True
        short_back_timedelta = datetime.timedelta(days=short_days)
        print('--------------------------\nTime span list:\n', tas.time_span_list)
        pop = PopularityBasedRec(G, number_recommendations)

        RWR_SA = PersonalizedPageRankBasedRec(number_recommendations)

        ae = AccuracyEvaluation(G)

        train_set_len = []
        train_len_dict = defaultdict(list)
        n_articles_train = []
        n_recommendation = dict()
        sessions_per_user_in_short_term = []
        avg_ses_len = defaultdict(list)

        for tw_i, tw_iter in enumerate(tas.time_window_graph_list):

            print('\n\n======= Time split', tw_i, '=======')

            n_recommendation[tw_i] = 0

            # long_train_g = tw_iter[0]
            tw_iter[1].frozen = False
            test_g = tw_iter[1].copy()

            # ------ From test_g remove sessions with less or equal number of articles needed for building recommendation
            test_g = gm.filter_sessions(test_g, n_items=min_items_n)
            if len(test_g) == 0:
                continue

            # ------ 1. Create a time-ordered list of user sessions
            test_sessions = sorted(
                [(s, attr['datetime']) for s, attr in test_g.nodes(data=True) if attr['entity'] == 'S'],
                key=lambda x: x[1])

            sessions_knn_dict = defaultdict(tuple)

            # For each step a ranked list of N recommendations is created
            for (s, s_datetime) in test_sessions:

                user = [n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'US'][0]

                test_session_G = nx.Graph()
                test_session_G.add_node(user, entity='U')
                test_session_G.add_node(s, entity='S')
                test_session_G.add_edge(user, s, edge_type='US')

                # -----------------------------------------------------
                articles = sorted(
                    [n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'SA'],
                    key=lambda x: test_g.get_edge_data(s, x)['reading_datetime'])

                avg_ses_len[tw_i].append(len(articles))

                # print('----------\narticles:', articles)
                # print('session:', s, s_datetime)

                for i in range(min_items_n, len(articles)):
                    # ------------ Short-term training set ----
                    short_train_g = tas.create_short_term_train_set(s_datetime, short_back_timedelta,
                                                                    test_session_graph=test_session_G)
                    if len(short_train_g) == 0:
                        continue
                    print(f'user:{user}')
                    active_users = gm.get_users(short_train_g)
                    print(f'active users : {user in active_users}')
                    print(f'Next Article : {articles[i]}')
                    test_session_G.add_nodes_from(articles[:i], entity='A')
                    for a in articles[:i]:
                        test_session_G.add_edge(s, a, edge_type='SA')
                        test_session_G.add_node(gm.map_category(a), entity='C')
                        test_session_G.add_edge(a, gm.map_category(a), edge_type='AC')
                        test_session_G.add_node(gm.map_location(a), entity='L')
                        test_session_G.add_edge(a, gm.map_location(a), edge_type='AL')

                    # ------------ Short Training Set (containing currently analyzed session!) ---------


                    # ----------- Long-term user training set ---
                    users_from_short_train = gm.get_users(short_train_g)
                    user_long_train_g = tas.create_long_term_user_train_set(user, s, s_datetime, articles[:i],
                                                                            users_from_short_train)
                    if len(user_long_train_g) == 0:
                        continue

                    train_set_len.append(len(gm.get_sessions(short_train_g)))
                    train_len_dict[tw_i].append(len(gm.get_sessions(short_train_g)))
                    n_articles_train.append(len(gm.get_articles(short_train_g)))
                    ses_per_user = gm.get_sessions_per_user(short_train_g)
                    sessions_per_user_in_short_term.append(Counter(ses_per_user))

                    # --- Create train graphs
                    sa_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A'])
                    # usa_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A'])
                    # sac_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A', 'C'])
                    sal_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g,
                                                                          list_of_entities=['S', 'A', 'L'])
                    # usac_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A', 'C'])
                    # usal_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A', 'L'])



                    # -------------------------------------------------------------------------------
                    # --------------- SIMILARITIES --------------------------------------------------


                    # -----------------------------------------------------
                    # ------------------- Popularity ----------------------
                    pop.compute_pop(short_train_g)

                    # -----------------------------------------------------
                    # ------------------- SimRank -------------------------

                    # SimRank_SAL.compute_similarity_matrix(sal_train_g, max_iter=10)

                    # -----------------------------------------------------
                    # ------------------- RWR -----------------------------
                    # --- Run models
                    RWR_SA.compute_transition_matrix(sa_train_g)
                    # RWR_USA.compute_transition_matrix(usa_train_g)
                    # RWR_SAC.compute_transition_matrix(sac_train_g)
                    # RWR_SAL.compute_transition_matrix(sal_train_g)
                    # RWR_USAC.compute_transition_matrix(usac_train_g)
                    # RWR_USAL.compute_transition_matrix(usal_train_g)
                    # RWR_USACL.compute_transition_matrix(short_train_g)

                    # --- Extract SS matrices
                    # RWR_SA.create_sessionsession_matrix()
                    # RWR_SA.create_sessionitem_matrix()
                    RWR_SA.create_itemitem_matrix()
                    # RWR_USA.create_sessionsession_matrix()
                    # RWR_USA.create_sessionitem_matrix()
                    # RWR_USA.create_itemitem_matrix()
                    # RWR_SAC.create_sessionsession_matrix()
                    # RWR_SAC.create_sessionitem_matrix()
                    # RWR_SAC.create_itemitem_matrix()
                    # RWR_SAL.create_sessionsession_matrix()
                    # RWR_SAL.create_sessionitem_matrix()
                    # RWR_SAL.create_itemitem_matrix()
                    # RWR_USAC.create_sessionsession_matrix()
                    # RWR_USAC.create_sessionitem_matrix()
                    # RWR_USAC.create_itemitem_matrix()
                    # RWR_USAL.create_sessionsession_matrix()
                    # RWR_USAL.create_sessionitem_matrix()
                    # RWR_USAL.create_itemitem_matrix()
                    # RWR_USACL.create_sessionsession_matrix()
                    # RWR_USACL.create_sessionitem_matrix()
                    # RWR_USACL.create_itemitem_matrix()

                    # -----------------------------------------------------
                    # ------------------ PathSim --------------------------

                    # PathSim_AUA.compute_similarity_matrix(short_train_g, 'A', 'U', 2)
                    # PathSim_ASA.compute_similarity_matrix(short_train_g, 'A', 'S', 1)
                    # PathSim_ACA.compute_similarity_matrix(short_train_g, 'A', 'C', 1)
                    # PathSim_ALA.compute_similarity_matrix(short_train_g, 'A', 'L', 1)

                    # -----------------------------------------------------
                    # ------------------ PathCount --------------------------

                    # PathCount_AUA.compute_similarity_matrix_my(short_train_g, 'A', 'U', 2)
                    # PathCount_ASA.compute_similarity_matrix_my(short_train_g, 'A', 'S', 1)
                    # PathCount_ACA.compute_similarity_matrix_my(short_train_g, 'A', 'C', 1)
                    # PathCount_ALA.compute_similarity_matrix_my(short_train_g, 'A', 'L', 1)


                    # -----------------------------------------------------
                    # ------------------- S-S PathSim ---------------------
                    # SKNN_PathSim_SAS.compute_similarity_matrix(short_train_g, 'S', 'A', 1)
                    # SKNN_PathSim_SACAS.compute_similarity_matrix(sac_train_g, 'S', 'C', 2)
                    # SKNN_PathSim_SALAS.compute_similarity_matrix(sal_train_g, 'S', 'L', 2)


                    # -----------------------------------------------------
                    # ------------------- S-S PathCounts ------------------
                    # SKNN_PathCount_SAS.compute_similarity_matrix_my(short_train_g, 'S', 'A', 1)
                    # SKNN_PathCount_SACAS.compute_similarity_matrix_my(sac_train_g, 'S', 'C', 2)
                    # SKNN_PathCount_SALAS.compute_similarity_matrix_my(sal_train_g, 'S', 'L', 2)


                    # -----------------------------------------------------
                    # ------------------- PathCounts for expl -------------

                    # PathCount_ASA.compute_similarity_matrix(short_train_g, 'A', 'S', 1)
                    # PathCount_ACA.compute_similarity_matrix(short_train_g, 'A', 'C', 1)
                    # PathCount_ALA.compute_similarity_matrix(short_train_g, 'A', 'L', 1)
                    # PathCount_AUA.compute_similarity_matrix(short_train_g, 'A', 'U', 2)
                    #
                    # PathCount_SAS.compute_similarity_matrix(short_train_g, 'S', 'A', 1)
                    # PathCount_SCS.compute_similarity_matrix(short_train_g, 'S', 'C', 2)
                    # PathCount_SLS.compute_similarity_matrix(short_train_g, 'S', 'L', 2)
                    #
                    # PathCount_UAU.compute_similarity_matrix(user_long_train_g, 'U', 'A', 2)
                    # PathCount_UCU.compute_similarity_matrix(user_long_train_g, 'U', 'C', 3)
                    # PathCount_ULU.compute_similarity_matrix(user_long_train_g, 'U', 'L', 3)

                    # -------------------------------------------------------------------------------
                    # --------------- RECOMMENDATIONS -----------------------------------------------

                    session_categories = [gm.map_category(a) for a in articles[:i]]
                    session_timeviews = [gm.map_timeview(test_g, s, a) for a in articles[:i]]

                    # ------- POP --------------------------

                    pop_rec = pop.predict_next(user, articles[:i])
                    if len(pop_rec) == 0:
                        continue

                    # ------- SimRank ----------------------

                    # simrank_sal_s_rec = SimRank_SAL.predict_next(user, articles[:i], method=2)
                    # if len(simrank_sal_s_rec) == 0:
                    #     continue

                    # ------- RWR --------------------------

                    # rwr_ua_s_rec = RWR_UA.predict_next(user, articles[:i], method=2)
                    rwr_sa_s_rec = RWR_SA.predict_next(user, articles[:i], method=2)
                    # if len(rwr_sa_s_rec) == 0:
                    #     continue
                    # rwr_ac_s_rec = RWR_AC.predict_next(user, articles[:i], method=2)
                    # rwr_al_s_rec = RWR_AL.predict_next(user, articles[:i], method=2)
                    #
                    # rwr_usa_s_rec = RWR_USA.predict_next(user, articles[:i], method=2)
                    # if len(rwr_usa_s_rec) == 0:
                    #     continue
                    # rwr_sac_s_rec = RWR_SAC.predict_next(user, articles[:i], method=2)
                    # rwr_sal_s_rec = RWR_SAL.predict_next(user, articles[:i], method=2)
                    # print(f'RWR SAL : {rwr_sal_s_rec}')
                    # if len(rwr_sal_s_rec) == 0:
                    #     continue
                    # rwr_uac_s_rec = RWR_UAC.predict_next(user, articles[:i], method=2)
                    # rwr_ual_s_rec = RWR_UAL.predict_next(user, articles[:i], method=2)
                    # rwr_acl_s_rec = RWR_ACL.predict_next(user, articles[:i], method=2)
                    #
                    # rwr_usac_s_rec = RWR_USAC.predict_next(user, articles[:i], method=2)
                    # rwr_usal_s_rec = RWR_USAL.predict_next(user, articles[:i], method=2)
                    # rwr_sacl_s_rec = RWR_SACL.predict_next(user, articles[:i], method=2)
                    # rwr_uacl_s_rec = RWR_UACL.predict_next(user, articles[:i], method=2)

                    # rwr_usacl_s_rec = RWR_USACL.predict_next(user, articles[:i], method=2)
                    # if len(rwr_usacl_s_rec) == 0:
                    #     continue

                    # ------- PathSim ----------------------

                    # pathsim_aua_s_rec = PathSim_AUA.predict_next(user, articles[:i], method=2)
                    # if len(pathsim_aua_s_rec) == 0:
                    #     continue
                    # pathsim_asa_s_rec = PathSim_ASA.predict_next(user, articles[:i], method=2)
                    # if len(pathsim_asa_s_rec) == 0:
                    #     continue

                    # ------- PathCount --------------------

                    # pathcount_aua_s_rec = PathCount_AUA.predict_next(user, articles[:i], method=2)
                    # if len(pathcount_aua_s_rec) == 0:
                    #     continue
                    # pathcount_asa_s_rec = PathCount_ASA.predict_next(user, articles[:i], method=2)
                    # if len(pathcount_asa_s_rec) == 0:
                    #     continue

                    # ------- Session-kNN ------------------

                    # sknn_rwr_sa_rec = RWR_SA.predict_next_by_sessionKNN(s, kNN_RWR)
                    # sknn_rwr_usa_rec = RWR_USA.predict_next_by_sessionKNN(s, kNN_RWR
                    # sknn_rwr_sac_rec = RWR_SAC.predict_next_by_sessionKNN(s, kNN_RWR)
                    # sknn_rwr_sal_rec = RWR_SAL.predict_next_by_sessionKNN(s, kNN_RWR)
                    # if len(sknn_rwr_sal_rec) == 0:
                    #     continue
                    # sknn_rwr_usac_rec = RWR_USAC.predict_next_by_sessionKNN(s, kNN_RWR)
                    # sknn_rwr_usal_rec = RWR_USAL.predict_next_by_sessionKNN(s, kNN_RWR)
                    # sknn_rwr_usacl_rec = RWR_USACL.predict_next_by_sessionKNN(s, kNN_RWR)
                    # if len(sknn_rwr_usacl_rec) == 0:
                    #     continue

                    # sknn_ps_sas_rec = SKNN_PathSim_SAS.predict_next_by_sessionKNN(s, articles[:i], kNN_PathSim)
                    # if len(sknn_ps_sas_rec) == 0:
                    #     continue
                    # sknn_ps_sacas_rec = PathSim_SACAS.predict_next_by_sessionKNN(s, articles[:i], kNN_PathSim)
                    # sknn_ps_salas_rec = PathSim_SALAS.predict_next_by_sessionKNN(s, articles[:i], kNN_PathSim)

                    # sknn_pc_sas_rec = SKNN_PathCount_SAS.predict_next_by_sessionKNN(s, articles[:i], kNN_PathSim)
                    # if len(sknn_pc_sas_rec) == 0:
                    #     continue
                    # sknn_pc_sacas_rec = SKNN_PathCount_SACAS.predict_next_by_sessionKNN(s, articles[:i], kNN_PathSim)
                    # sknn_pc_salas_rec = SKNN_PathCount_SALAS.predict_next_by_sessionKNN(s, articles[:i], kNN_PathSim)



                    # ----------------- Expl ------------------------------

                    # -------------------------
                    # --------- AB ------------

                    # --- AB(1) - Item-based (only last article matters)
                    # pc_asa_rec_dict_ib = PathCount_ASA.predict_next_by_AB(articles[:i], option='ib', topN=False)
                    # pc_aca_rec_dict_ib = PathCount_ACA.predict_next_by_AB(articles[:i], option='ib', topN=False)
                    # pc_ala_rec_dict_ib = PathCount_ALA.predict_next_by_AB(articles[:i], option='ib', topN=False)
                    # pc_aua_rec_dict_ib = PathCount_AUA.predict_next_by_AB(articles[:i], option='ib', topN=False)
                    #
                    # ab_asa_rec = list(pc_asa_rec_dict_ib.keys())[:N]
                    # ab_aca_rec = list(pc_aca_rec_dict_ib.keys())[:N]
                    # ab_ala_rec = list(pc_ala_rec_dict_ib.keys())[:N]
                    # ab_aua_rec = list(pc_aua_rec_dict_ib.keys())[:N]
                    #
                    # # - Combine
                    # rec_ab_articles = list(set(list(pc_asa_rec_dict_ib.keys()) + list(pc_aca_rec_dict_ib.keys()) +
                    #                            list(pc_ala_rec_dict_ib.keys()) + list(pc_aua_rec_dict_ib.keys())))
                    # rec_ab_df = pd.DataFrame(index=rec_ab_articles, columns=['ASA', 'ACA', 'ALA', 'AUA'])
                    #
                    # for a in rec_ab_df.index:
                    #     rec_ab_df.loc[a, 'ASA'] = pc_asa_rec_dict_ib[a] if a in list(pc_asa_rec_dict_ib.keys()) else 0
                    #     rec_ab_df.loc[a, 'ACA'] = pc_aca_rec_dict_ib[a] if a in list(pc_aca_rec_dict_ib.keys()) else 0
                    #     rec_ab_df.loc[a, 'ALA'] = pc_ala_rec_dict_ib[a] if a in list(pc_ala_rec_dict_ib.keys()) else 0
                    #     rec_ab_df.loc[a, 'AUA'] = pc_aua_rec_dict_ib[a] if a in list(pc_aua_rec_dict_ib.keys()) else 0
                    #
                    # rec_importance_ab_df = rec_ab_df.copy()
                    # for a in rec_importance_ab_df.index:
                    #     rec_importance_ab_df.loc[a, 'AUA'] = round(
                    #         rec_importance_ab_df.loc[a, 'AUA'] / PathCount_AUA.get_avg_n_of_connected_sessions(), 2)
                    #     rec_importance_ab_df.loc[a, 'ASA'] = round(
                    #         rec_importance_ab_df.loc[a, 'ASA'] / PathCount_ASA.get_avg_n_of_connected_sessions(), 2)
                    #     rec_importance_ab_df.loc[a, 'ACA'] = round(
                    #         rec_importance_ab_df.loc[a, 'ACA'] / PathCount_ACA.get_avg_n_of_connected_sessions(), 2)
                    #     rec_importance_ab_df.loc[a, 'ALA'] = round(
                    #         rec_importance_ab_df.loc[a, 'ALA'] / PathCount_ALA.get_avg_n_of_connected_sessions(), 2)
                    #
                    # rec_importance_ab_df['vote_sum'] = rec_importance_ab_df['AUA'] + rec_importance_ab_df['ASA'] + \
                    #                                    rec_importance_ab_df['ACA'] + rec_importance_ab_df['ALA']
                    # ranked_by_vote_sum_rec_importance_ab_df = rec_importance_ab_df.sort_values(by=['vote_sum'],ascending=False).head(N)
                    # ranked_by_relative_importance_ab_df = rec_ab_df.ix[ranked_by_vote_sum_rec_importance_ab_df.index]
                    # ab_comb_rec = ranked_by_relative_importance_ab_df.index.tolist()
                    #
                    #
                    #
                    # # --- AB(All) - Session-based (all articles from the session matter)
                    # pc_asa_rec_dict_sb = PathCount_ASA.predict_next_by_AB(articles[:i], option='sb', topN=False)
                    # pc_aca_rec_dict_sb = PathCount_ACA.predict_next_by_AB(articles[:i], option='sb', topN=False)
                    # pc_ala_rec_dict_sb = PathCount_ALA.predict_next_by_AB(articles[:i], option='sb', topN=False)
                    # pc_aua_rec_dict_sb = PathCount_AUA.predict_next_by_AB(articles[:i], option='sb', topN=False)
                    #
                    # ab_all_asa_rec = list(pc_asa_rec_dict_sb.keys())[:N]
                    # ab_all_aca_rec = list(pc_aca_rec_dict_sb.keys())[:N]
                    # ab_all_ala_rec = list(pc_ala_rec_dict_sb.keys())[:N]
                    # ab_all_aua_rec = list(pc_aua_rec_dict_sb.keys())[:N]
                    #
                    #
                    # # - Combine
                    # rec_ab_all_articles = list(set(list(pc_asa_rec_dict_sb.keys()) + list(pc_aca_rec_dict_sb.keys()) + list(pc_ala_rec_dict_sb.keys()) + list(pc_aua_rec_dict_sb.keys())))
                    # rec_ab_all_df = pd.DataFrame(index=rec_ab_all_articles, columns=['ASA', 'ACA', 'ALA', 'AUA'])
                    #
                    # for a in rec_ab_all_df.index:
                    #     rec_ab_all_df.loc[a, 'ASA'] = pc_asa_rec_dict_sb[a] if a in list(pc_asa_rec_dict_sb.keys()) else 0
                    #     rec_ab_all_df.loc[a, 'ACA'] = pc_aca_rec_dict_sb[a] if a in list(pc_aca_rec_dict_sb.keys()) else 0
                    #     rec_ab_all_df.loc[a, 'ALA'] = pc_ala_rec_dict_sb[a] if a in list(pc_ala_rec_dict_sb.keys()) else 0
                    #     rec_ab_all_df.loc[a, 'AUA'] = pc_aua_rec_dict_sb[a] if a in list(pc_aua_rec_dict_sb.keys()) else 0
                    #
                    # rec_importance_ab_all_df = rec_ab_all_df.copy()
                    # for a in rec_importance_ab_all_df.index:
                    #     rec_importance_ab_all_df.loc[a, 'AUA'] = round(
                    #         rec_importance_ab_all_df.loc[a, 'AUA'] / PathCount_AUA.get_avg_n_of_connected_sessions(), 2)
                    #     rec_importance_ab_all_df.loc[a, 'ASA'] = round(
                    #         rec_importance_ab_all_df.loc[a, 'ASA'] / PathCount_ASA.get_avg_n_of_connected_sessions(), 2)
                    #     rec_importance_ab_all_df.loc[a, 'ACA'] = round(
                    #         rec_importance_ab_all_df.loc[a, 'ACA'] / PathCount_ACA.get_avg_n_of_connected_sessions(), 2)
                    #     rec_importance_ab_all_df.loc[a, 'ALA'] = round(
                    #         rec_importance_ab_all_df.loc[a, 'ALA'] / PathCount_ALA.get_avg_n_of_connected_sessions(), 2)
                    #
                    # rec_importance_ab_all_df['vote_sum'] = rec_importance_ab_all_df['AUA'] + rec_importance_ab_all_df['ASA'] + \
                    #                                        rec_importance_ab_all_df['ACA'] + rec_importance_ab_all_df['ALA']
                    # ranked_by_vote_sum_rec_importance_ab_all_df = rec_importance_ab_all_df.sort_values(by=['vote_sum'], ascending=False).head(N)
                    # ranked_by_relative_importance_ab_all_df = rec_ab_all_df.ix[ranked_by_vote_sum_rec_importance_ab_all_df.index]
                    # ab_all_comb_rec = ranked_by_relative_importance_ab_all_df.index.tolist()
                    #
                    #
                    #
                    # # -------------------------
                    # # --------- SB ------------
                    # pc_sasa_rec_dict = PathCount_SAS.predict_next_by_SB(s, articles[:i], topN=False)
                    # pc_scsa_rec_dict = PathCount_SCS.predict_next_by_SB(s, articles[:i], topN=False)
                    # pc_slsa_rec_dict = PathCount_SLS.predict_next_by_SB(s, articles[:i], topN=False)
                    #
                    # sb_sasa_rec = list(pc_sasa_rec_dict.keys())[:N]
                    # sb_scsa_rec = list(pc_scsa_rec_dict.keys())[:N]
                    # sb_slsa_rec = list(pc_slsa_rec_dict.keys())[:N]
                    #
                    # # -- Combine
                    # rec_articles = list(set(list(pc_sasa_rec_dict.keys()) + list(pc_scsa_rec_dict.keys()) + list(pc_slsa_rec_dict.keys())))
                    # rec_df = pd.DataFrame(index=rec_articles, columns=['SASA', 'SCSA', 'SLSA'])
                    #
                    # for a in rec_df.index:
                    #     rec_df.loc[a, 'SASA'] = pc_sasa_rec_dict[a] if a in list(pc_sasa_rec_dict.keys()) else 0
                    #     rec_df.loc[a, 'SCSA'] = pc_scsa_rec_dict[a] if a in list(pc_scsa_rec_dict.keys()) else 0
                    #     rec_df.loc[a, 'SLSA'] = pc_slsa_rec_dict[a] if a in list(pc_slsa_rec_dict.keys()) else 0
                    #
                    # rec_importance_df = rec_df.copy()
                    # for a in rec_importance_df.index:
                    #     rec_importance_df.loc[a, 'SASA'] = round(
                    #         rec_importance_df.loc[a, 'SASA'] / PathCount_SAS.get_avg_n_of_connected_sessions(), 2)
                    #     rec_importance_df.loc[a, 'SCSA'] = round(
                    #         rec_importance_df.loc[a, 'SCSA'] / PathCount_SCS.get_avg_n_of_connected_sessions(), 2)
                    #     rec_importance_df.loc[a, 'SLSA'] = round(
                    #         rec_importance_df.loc[a, 'SLSA'] / PathCount_SLS.get_avg_n_of_connected_sessions(), 2)
                    #
                    # rec_importance_df['vote_sum'] = rec_importance_df['SASA'] + rec_importance_df['SCSA'] + rec_importance_df['SLSA']
                    # ranked_by_vote_sum_rec_importance_df = rec_importance_df.sort_values(by=['vote_sum'], ascending=False).head(N)
                    # ranked_by_relative_importance_df = rec_df.ix[ranked_by_vote_sum_rec_importance_df.index]
                    # sb_comb_rec = ranked_by_relative_importance_df.index.tolist()
                    #
                    #
                    # # -------------------------
                    # # --------- UB ------------
                    # similar_users_uau = PathCount_UAU.get_similar_users(user, gm.get_users(short_train_g))
                    # similar_users_ucu = PathCount_UCU.get_similar_users(user, gm.get_users(short_train_g), threshold=0.5)
                    # similar_users_ulu = PathCount_ULU.get_similar_users(user, gm.get_users(short_train_g), threshold=0.5)
                    #
                    # uaua_rec_dict = PathCount_UAU.predict_next_by_UB(similar_users_uau, articles[:i], short_train_g, topN=False)
                    # ucua_rec_dict = PathCount_UCU.predict_next_by_UB(similar_users_ucu, articles[:i], short_train_g, topN=False)
                    # ulua_rec_dict = PathCount_ULU.predict_next_by_UB(similar_users_ulu, articles[:i], short_train_g, topN=False)
                    #
                    # ub_uaua_rec = list(uaua_rec_dict.keys())[:N]
                    # ub_ucua_rec = list(ucua_rec_dict.keys())[:N]
                    # ub_ulua_rec = list(ulua_rec_dict.keys())[:N]
                    #
                    # # --- Combine
                    # rec_articles = list(set(list(uaua_rec_dict.keys()) + list(ucua_rec_dict.keys()) + list(ulua_rec_dict.keys())))
                    # rec_df = pd.DataFrame(index=rec_articles, columns=['UAUA', 'UCUA', 'ULUA'])
                    #
                    # for a in rec_df.index:
                    #     rec_df.loc[a, 'UAUA'] = uaua_rec_dict[a] if a in list(uaua_rec_dict.keys()) else 0
                    #     rec_df.loc[a, 'UCUA'] = ucua_rec_dict[a] if a in list(ucua_rec_dict.keys()) else 0
                    #     rec_df.loc[a, 'ULUA'] = ulua_rec_dict[a] if a in list(ulua_rec_dict.keys()) else 0
                    #
                    # rec_importance_df = rec_df.copy()
                    # for a in rec_importance_df.index:
                    #     rec_importance_df.loc[a, 'UAUA'] = round(
                    #         rec_importance_df.loc[a, 'UAUA'] / PathCount_UAU.get_avg_n_of_connected_sessions(), 2)
                    #     rec_importance_df.loc[a, 'UCUA'] = round(
                    #         rec_importance_df.loc[a, 'UCUA'] / PathCount_UCU.get_avg_n_of_connected_sessions(), 2)
                    #     rec_importance_df.loc[a, 'ULUA'] = round(
                    #         rec_importance_df.loc[a, 'ULUA'] / PathCount_ULU.get_avg_n_of_connected_sessions(), 2)
                    #
                    # rec_importance_df['vote_sum'] = rec_importance_df['UAUA'] + rec_importance_df['UCUA'] + rec_importance_df['ULUA']
                    # ranked_by_vote_sum_rec_importance_df = rec_importance_df.sort_values(by=['vote_sum'], ascending=False).head(N)
                    # ranked_by_relative_importance_df = rec_df.ix[ranked_by_vote_sum_rec_importance_df.index]
                    # ub_comb_rec = ranked_by_relative_importance_df.index.tolist()



                    # ------------------------------------------------------------

                    methods = [pop_rec,
                               # rwr_sal_s_rec,
                               rwr_sa_s_rec,
                               # rwr_usacl_s_rec,
                               # sknn_rwr_usacl_rec,
                               # ab_aua_rec, ab_asa_rec, ab_aca_rec, ab_ala_rec, ab_comb_rec,
                               # ab_all_aua_rec, ab_all_asa_rec, ab_all_aca_rec, ab_all_ala_rec, ab_all_comb_rec,
                               # sb_sasa_rec, sb_scsa_rec, sb_slsa_rec, sb_comb_rec,
                               # ub_uaua_rec, ub_ucua_rec, ub_ulua_rec, ub_comb_rec
                               ]
                    # methods = [pop_rec,
                    #            ab_aua_rec, ab_asa_rec, ab_aca_rec, ab_ala_rec, ab_comb_rec,
                    #            ab_all_aua_rec, ab_all_asa_rec, ab_all_aca_rec, ab_all_ala_rec, ab_all_comb_rec,
                    #            sb_sasa_rec, sb_scsa_rec, sb_slsa_rec, sb_comb_rec,
                    #            ub_uaua_rec, ub_ucua_rec, ub_ulua_rec, ub_comb_rec]

                    if any(len(m) == 0 for m in methods):
                        continue

                    n_recommendation[tw_i] += 1

                    # ------- Measuring accuracy ----------------------
                    ae.evaluate_recommendation(rec=pop_rec, truth=articles[i], method='POP', s=s)

                    # ae.evaluate_recommendation(rec=simrank_sal_s_rec, truth=articles[i], method='SimRank_SAL(s)', s=s)

                    ae.evaluate_recommendation(rec=rwr_sa_s_rec, truth=articles[i], method='RWR_SA(s)', s=s)
                    # ae.evaluate_recommendation(rec=rwr_usa_s_rec, truth=articles[i], method='RWR_USA(s)', s=s)
                    # ae.evaluate_recommendation(rec=rwr_sac_s_rec, truth=articles[i], method='RWR_SAC(s)', s=s)
                    # ae.evaluate_recommendation(rec=rwr_sal_s_rec, truth=articles[i], method='RWR_SAL(s)', s=s)
                    # ae.evaluate_recommendation(rec=rwr_usac_s_rec, truth=articles[i], method='RWR_USAC(s)', s=s)
                    # ae.evaluate_recommendation(rec=rwr_usal_s_rec, truth=articles[i], method='RWR_USAL(s)', s=s)
                    # ae.evaluate_recommendation(rec=rwr_usacl_s_rec, truth=articles[i], method='RWR_USACL(s)', s=s)

                    # ae.evaluate_recommendation(rec=pathsim_asa_s_rec, truth=articles[i], method='PathSim_ASA(s)', s=s)
                    # ae.evaluate_recommendation(rec=pathsim_aua_s_rec, truth=articles[i], method='PathSim_AUA(s)', s=s)
                    # ae.evaluate_recommendation(rec=pathcount_asa_s_rec, truth=articles[i], method='PathCount_ASA(s)', s=s)
                    # ae.evaluate_recommendation(rec=pathcount_aua_s_rec, truth=articles[i], method='PathCount_AUA(s)', s=s)

                    # ae.evaluate_recommendation(rec=sknn_rwr_sa_rec, truth=articles[i], method='SKNN_RWR_SA', s=s)
                    # ae.evaluate_recommendation(rec=sknn_rwr_usa_rec, truth=articles[i], method='SKNN_RWR_USA', s=s)
                    # ae.evaluate_recommendation(rec=sknn_rwr_sac_rec, truth=articles[i], method='SKNN_RWR_SAC', s=s)
                    # ae.evaluate_recommendation(rec=sknn_rwr_sal_rec, truth=articles[i], method='SKNN_RWR_SAL', s=s)
                    # ae.evaluate_recommendation(rec=sknn_rwr_usac_rec, truth=articles[i], method='SKNN_RWR_USAC', s=s)
                    # ae.evaluate_recommendation(rec=sknn_rwr_usal_rec, truth=articles[i], method='SKNN_RWR_USAL', s=s)
                    # ae.evaluate_recommendation(rec=sknn_rwr_usacl_rec, truth=articles[i], method='SKNN_RWR_USACL', s=s)

                    # ae.evaluate_recommendation(rec=sknn_ps_sas_rec, truth=articles[i], method='SKNN_PathSim_SAS', s=s)
                    # ae.evaluate_recommendation(rec=sknn_ps_sacas_rec, truth=articles[i], method='SKNN_PathSim_SACAS', s=s)
                    # ae.evaluate_recommendation(rec=sknn_ps_salas_rec, truth=articles[i], method='SKNN_PathSim_SALAS', s=s)

                    # ae.evaluate_recommendation(rec=sknn_pc_sas_rec, truth=articles[i], method='SKNN_PathCount_SAS', s=s)
                    # ae.evaluate_recommendation(rec=sknn_pc_sacas_rec, truth=articles[i], method='SKNN_PathCount_SACAS', s=s)
                    # ae.evaluate_recommendation(rec=sknn_pc_salas_rec, truth=articles[i], method='SKNN_PathCount_SALAS', s=s)

                    # ae.evaluate_recommendation(rec=ab_aua_rec, truth=articles[i], method='AUA', s=s)
                    # ae.evaluate_recommendation(rec=ab_asa_rec, truth=articles[i], method='ASA', s=s)
                    # ae.evaluate_recommendation(rec=ab_aca_rec, truth=articles[i], method='ACA', s=s)
                    # ae.evaluate_recommendation(rec=ab_ala_rec, truth=articles[i], method='ALA', s=s)
                    # ae.evaluate_recommendation(rec=ab_comb_rec, truth=articles[i], method='AB_COMB', s=s)
                    #
                    # ae.evaluate_recommendation(rec=ab_all_aua_rec, truth=articles[i], method='AUA(all)', s=s)
                    # ae.evaluate_recommendation(rec=ab_all_asa_rec, truth=articles[i], method='ASA(all)', s=s)
                    # ae.evaluate_recommendation(rec=ab_all_ala_rec, truth=articles[i], method='ALA(all)', s=s)
                    # ae.evaluate_recommendation(rec=ab_all_aca_rec, truth=articles[i], method='ACA(all)', s=s)
                    # ae.evaluate_recommendation(rec=ab_all_comb_rec, truth=articles[i], method='AB_COMB(all)', s=s)
                    #
                    # ae.evaluate_recommendation(rec=sb_sasa_rec, truth=articles[i], method='SASA', s=s)
                    # ae.evaluate_recommendation(rec=sb_scsa_rec, truth=articles[i], method='SCSA', s=s)
                    # ae.evaluate_recommendation(rec=sb_slsa_rec, truth=articles[i], method='SLSA', s=s)
                    # ae.evaluate_recommendation(rec=sb_comb_rec, truth=articles[i], method='SB_COMB', s=s)
                    #
                    # ae.evaluate_recommendation(rec=ub_uaua_rec, truth=articles[i], method='UAUA', s=s)
                    # ae.evaluate_recommendation(rec=ub_ucua_rec, truth=articles[i], method='UCUA', s=s)
                    # ae.evaluate_recommendation(rec=ub_ulua_rec, truth=articles[i], method='ULUA', s=s)
                    # ae.evaluate_recommendation(rec=ub_comb_rec, truth=articles[i], method='UB_COMB', s=s)

                ae.evaluate_session()

            ae.evaluate_tw()
            # print('- Number of recommendations made:', n_recommendations)

        ae.evaluate_total_performance()

        avg_n_ses_per_train_per_period = [round(np.mean(l)) for l in train_len_dict.values()]
        avg_ses_len_per_period = [round(np.mean(l), 2) for l in avg_ses_len.values()]

        print('\n\n\nNumber of sessions per user per short train period:\n', sessions_per_user_in_short_term)
        print('# of recommendations per time split:', n_recommendation.values())
        print('Total # of recs:', sum(n_recommendation.values()))
        print('Average # sessions per train per period', avg_n_ses_per_train_per_period)
        print('Average # artiles per session per period', avg_ses_len_per_period)
        print('Average # sessions in train:', round(np.mean(train_set_len), 2))
        print('Average # articles in train:', round(np.mean(n_articles_train), 2))

        print('\n---------- METHODS EVALUATION -------------')

        methods = [k for k, v in sorted(ae.precision.items(), key=itemgetter(1), reverse=True)]
        for m in methods:
            print('---', m, ': Precision:', ae.precision[m], 'NDCG:', ae.ndcg[m], 'ILD:', ae.diversity[m],
                  'Explainability:', ae.explainability[m])

        # exit()

        # --- Create period index for plotting
        p_start = tas.time_span_list[1][0]
        p_end = tas.time_span_list[len(tas.time_span_list) - 1][1] + datetime.timedelta(days=1)
        month_range = pd.date_range(p_start, p_end, freq='M')
        p = []
        for period in month_range:
            p.append(datetime.datetime.strftime(period, format='%Y-%m'))
        traces =[]
        for method in methods:
            trace = go.Scatter(x = p,y=ae.tw_precision[method],name=method)
            traces.append(trace)

        return {
            'data':traces,
            'layout': go.Layout(
            xaxis={
                      'title': 'Time_Period',

                  },
                  yaxis = {
                              'title': 'Precision',

                          })

        }

if __name__ == '__main__':
    app.run_server(debug= True,threaded = True)