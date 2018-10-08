import numpy as np
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict
from collections import Counter
from operator import itemgetter

from data_import import *
from graph_manipulation import *
from time_aware_splits import *
from popularity_based_rec import *
from personalized_prank import *
from pathsim import *
#from simrank import *
from accuracy_evaluation import *


# ------ Data paths -------------------------------------
#DATA_PATH = '.\\Data\\Toy example from Gabriele.txt'
DATA_PATH = '.\\Data\\Video33 - pk_client, pk_session, pk_article, timeview (s), date, time.txt'
CAT_DATA_PATH = '.\\Data\\Video33-5topics-doc-topics.txt'
LOC_DATA_PATH = '.\\Data\\Video33 - pk_article, pk_district.txt'

# ------ Parameters -------------------------------------
# 1 month: NUM_SPLITS = 62, WIN_SIZE = 3/4 - Best results for Popularity based

# NUM_SPLITS: To how many time intervals to split the dataset
NUM_SPLITS = 12#62#365
# WIN_SIZE: Window size is the length of the period taken to build the model
WIN_SIZE = 1#30
# N: Number of recommendations to be provided for each user
N = 5
# MIN_ITEMS_N: Minimum number of items to be revealed from test session before building a recommendation
# (e.g. give 2 consequent items in a test session, only then recommendation can be made)
MIN_ITEMS_N = 1
MIN_N_SESSIONS = 1

SHORT_DAYS = 2
MEDIUM_DAYS = 14

print('N:', N)
print('MIN_ITEMS_N:', MIN_ITEMS_N)
print('SHORT_DAYS:', SHORT_DAYS)

# -------------------------------------------------------
# F/P:
# F - take data from data file, create a graph on the base of it, store into gpickle file;
# P - take previously prepared data from gpickle file
file_or_pickle = 'P'

# If file_or_pickle = 'F' specify a file name to store the created graph to.
# Else, specify a pickle file name where to take the graph from.

# -----------------------------------
# pickle_name = 'Articles_AllData'
pickle_name = 'Articles_AllData_loc'
# pickle_name = 'Articles_LongSessions(2)'
# pickle_name = 'Articles_LongSessions(2)_loc'
# pickle_name = 'Articles_LongSessions(3)'
# pickle_name = 'Articles_LongSessions(4)'
#pickle_name = 'Articles_ActiveUsers(3)'
# ------------------------------------

# results_dir = '.\\Data\\Results\\' + pickle_name + '_' + str(SHORT_DAYS) + '_' + str(MEDIUM_DAYS) + '\\'
results_dir = '.\\Data\\Results\\'+pickle_name+'_'+str(SHORT_DAYS)+'_RWR_PS_PC\\'

if file_or_pickle == 'F':

    # ------ Data import ------------------------------------
    di = DataImport()
    di.import_user_click_data(DATA_PATH, adjust_pk_names=True)

    # --- Reduce dataset to 1 month / 1 week / ...
    #di.reduce_timeframe(dt.datetime(2017,3,1), dt.datetime(2017,3,31)) # if G_Video33_1month is selected
    #di.reduce_timeframe(dt.datetime(2017, 3, 1), dt.datetime(2017, 3, 7)) # if G_Video33_1week is selected

    # --- Remove inactive users (the ones with small number of sessions in total)
    # di.remove_inactive_users(n_sessions=MIN_N_SESSIONS)
    #
    # ---------- Add categories -----------------------------
    di.import_categories_data(CAT_DATA_PATH)

    # ---- Leave only sessions with at least specified number of articles
    di.filter_short_sessions(n_items=MIN_ITEMS_N)

    # ------ Create a graph on the base of the dataframe ----
    gm = GraphManipulation(G_structure = 'USAC')
    gm.create_graph(di.user_ses_df)

    # Filter again, because dataframe filtering leaves sessions where the same article is repeatedly read several times
    # gm.filter_sessions(gm.G, n_items=MIN_ITEMS_N)
    # gm.filter_users(gm.G, n_sessions=MIN_N_SESSIONS)

    # ---------- Add locations ------------------------------
    di.import_locations_data(LOC_DATA_PATH)
    gm.add_locations_data(di.locations_data)

    G = gm.G

    nx.write_gpickle(gm.G, '.\\Data\\Pickles\\'+pickle_name+'.gpickle')

else:

    # ------ Extract a saved graph from the pickle ----------
    gm = GraphManipulation()
    G = nx.read_gpickle('.\\Data\\Pickles\\'+pickle_name+'.gpickle')
    gm.G = G

# -----------------------------------------------------------------------------
# ---------------- STATISTICS ABOUT THE DATASET -------------------------------
#
print('--- GENERAL STATISTICS ---')
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

# exit()
#
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

print('\n------ PARAMETERS -------')
print('NUM_SPLITS:', NUM_SPLITS)
print('WIN_SIZE:', WIN_SIZE)

# -----------------------------------------------------------
# ------ Split data to train and test -----------------------
tas = TimeAwareSplits(G)
tas.create_time_split_graphs(G, NUM_SPLITS)
tas.create_time_window_graphs(WIN_SIZE)

print('--------------------------\nTime span list:\n', tas.time_span_list)
#print('--------------\nTime window graph list:\n', tas.time_window_graph_list)
# exit()

# -----------------------------------------------------------
# ------ Building prequential recommendations ---------------

_dump_process = True

short_back_timedelta = datetime.timedelta(days=SHORT_DAYS)
medium_back_timedelta = datetime.timedelta(days=MEDIUM_DAYS)

pop = PopularityBasedRec(G, N)

# ----- PathSim -----
PathSim_SAS = PathSimRec(N)
PathSim_SCS = PathSimRec(N)
PathSim_SLS = PathSimRec(N)


ae = AccuracyEvaluation(G)
explainability = defaultdict(list)

train_set_len = []
train_len_dict = defaultdict(list)
n_articles_train = []
sessions_per_user_in_short_term = []
avg_ses_len = defaultdict(list)


for tw_i, tw_iter in enumerate(tas.time_window_graph_list):

    print('\n\n======= Time split', tw_i, '=======')

    long_train_g = tw_iter[0]
    test_g = tw_iter[1]

    # ------ From test_g remove sessions with less or equal number of articles needed for building recommendation
    test_g = gm.filter_sessions(test_g, n_items=MIN_ITEMS_N)
    if len(test_g) == 0:
        continue

    # ------ 1. Create a time-ordered list of user sessions
    test_sessions = sorted([(s, attr['datetime']) for s, attr in test_g.nodes(data=True) if attr['entity'] == 'S'], key=lambda x: x[1])

    # n_recommendations = 0

    # For each step a ranked list of N recommendations is created
    for (s, s_datetime) in test_sessions:

        user = [n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'US'][0]

        test_session_G = nx.Graph()
        test_session_G.add_node(user, entity='U')
        test_session_G.add_node(s, entity='S')
        test_session_G.add_edge(user, s, edge_type='US')

        # -----------------------------------------------------
        articles = sorted([n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'SA'],
                          key=lambda x: test_g.get_edge_data(s, x)['reading_datetime'])

        for i in range(MIN_ITEMS_N, len(articles)):

            test_session_G.add_nodes_from(articles[:i], entity='A')
            for a in articles[:i]:
                test_session_G.add_edge(s, a, edge_type='SA')
                test_session_G.add_node(gm.map_category(a), entity='C')
                test_session_G.add_edge(a, gm.map_category(a), edge_type='AC')
                test_session_G.add_node(gm.map_location(a), entity='L')
                test_session_G.add_edge(a, gm.map_location(a), edge_type='AL')


            # ------------ Short Training Set (containing currently analyzed session!) ---------

            short_train_g = tas.create_short_term_train_set(s_datetime, short_back_timedelta, test_session_graph=test_session_G)
            if len(short_train_g) == 0:
                continue

            # --- Create train graphs
            sa_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A'])
            sac_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A', 'C'])
            sal_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A', 'L'])


            # -----------------------------------------------------
            # ------------------- PathSim's -----------------------
            PathSim_SAS.compute_similarity_matrix(sa_train_g, 'S', 'A', 1)
            PathSim_SCS.compute_similarity_matrix(sac_train_g, 'S', 'C', 2)
            PathSim_SLS.compute_similarity_matrix(sal_train_g, 'S', 'L', 2)


            # --------- PathSim ------------
            ps_sasa_rec_dict = PathSim_SAS.predict_next_by_SB(s, articles[:i], topN=False)
            ps_scsa_rec_dict = PathSim_SCS.predict_next_by_SB(s, articles[:i], topN=False)
            ps_slsa_rec_dict = PathSim_SLS.predict_next_by_SB(s, articles[:i], topN=False)


            # Create a dataframe with all articles per each meta path together
            rec_articles = list(set(list(ps_sasa_rec_dict.keys()) + list(ps_scsa_rec_dict.keys()) + list(ps_slsa_rec_dict.keys())))
            rec_df = pd.DataFrame(index=rec_articles, columns=['SASA', 'SCSA', 'SLSA'])

            for a in rec_df.index:
                rec_df.loc[a, 'SASA'] = ps_sasa_rec_dict[a] if a in list(ps_sasa_rec_dict.keys()) else 0
                rec_df.loc[a, 'SCSA'] = ps_scsa_rec_dict[a] if a in list(ps_scsa_rec_dict.keys()) else 0
                rec_df.loc[a, 'SLSA'] = ps_slsa_rec_dict[a] if a in list(ps_slsa_rec_dict.keys()) else 0


            # Divide dataframe values by average number of sessions that a single session is connected with through a given path
            rec_importance_df = rec_df.copy()
            for a in rec_importance_df.index:
                rec_importance_df.loc[a, 'SASA'] = round(rec_importance_df.loc[a, 'SASA'] /
                                                         PathSim_SAS.get_avg_n_of_connected_sessions(), 2)
                rec_importance_df.loc[a, 'SCSA'] = round(rec_importance_df.loc[a, 'SCSA'] /
                                                         PathSim_SCS.get_avg_n_of_connected_sessions(), 2)
                rec_importance_df.loc[a, 'SLSA'] = round(rec_importance_df.loc[a, 'SLSA'] /
                                                         PathSim_SLS.get_avg_n_of_connected_sessions(), 2)

            rec_importance_df['vote_sum'] = rec_importance_df['SASA'] + rec_importance_df['SCSA'] + rec_importance_df['SLSA']

            item_expl_score_dict = dict(zip(rec_importance_df.index, rec_importance_df['vote_sum']))
            explainability[s] = item_expl_score_dict


import os
import pickle


results_dir = '.\\Data\\Results\\Explainability\\'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

pickle.dump(explainability, open(results_dir + "SI_Explainability.pickle", "wb")) # Session-Item

# ---

# from_pickle = pickle.load(open(results_dir + 'Explainability.pickle', 'rb'))
# print(pd.DataFrame(from_pickle).fillna(0))