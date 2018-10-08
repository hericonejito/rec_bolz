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
PathSim_UAU = PathSimRec(N)
PathSim_UCU = PathSimRec(N)
PathSim_ULU = PathSimRec(N)


ae = AccuracyEvaluation(G)
explainability = defaultdict(list)

for tw_i, tw_iter in enumerate(tas.time_window_graph_list):

    print('\n\n======= Time split', tw_i, '=======')

    # long_train_g = tw_iter[0]
    test_g = tw_iter[1]

    # ------ From test_g remove sessions with less or equal number of articles needed for building recommendation
    # test_g = gm.filter_sessions(test_g, n_items=MIN_ITEMS_N)
    if len(test_g) == 0:
        continue

    # ------ 1. Create a time-ordered list of user sessions
    test_sessions = sorted([(s, attr['datetime']) for s, attr in test_g.nodes(data=True) if attr['entity'] == 'S'], key=lambda x: x[1])

    # For each step a ranked list of N recommendations is created
    for (s, s_datetime) in test_sessions:

        user = [n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'US'][0]

        # -----------------------------------------------------
        articles = sorted([n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'SA'],
                          key=lambda x: test_g.get_edge_data(s, x)['reading_datetime'])

        for i in range(MIN_ITEMS_N, len(articles)):

            # ------------ Short-term training set -------------
            short_train_g = tas.create_short_term_train_set(s_datetime, short_back_timedelta)
            if len(short_train_g) == 0:
                continue

            users_from_short_train = gm.get_users(short_train_g)

            # ----------- Long-term user training set ------

            user_long_train_g = tas.create_long_term_user_train_set(user, s, s_datetime, articles[:i], users_from_short_train)
            if len(user_long_train_g) == 0:
                continue

            PathSim_UAU.compute_similarity_matrix(user_long_train_g, 'U', 'A', 2)
            PathSim_UCU.compute_similarity_matrix(user_long_train_g, 'U', 'C', 3)
            PathSim_ULU.compute_similarity_matrix(user_long_train_g, 'U', 'L', 3)

            similar_users_uau = PathSim_UAU.get_similar_users(user, gm.get_users(short_train_g), threshold=0.5)
            similar_users_ucu = PathSim_UCU.get_similar_users(user, gm.get_users(short_train_g), threshold=0.5)
            similar_users_ulu = PathSim_ULU.get_similar_users(user, gm.get_users(short_train_g), threshold=0.5)

            uaua_rec_dict = PathSim_UAU.predict_next_by_UB(similar_users_uau, articles[:i], short_train_g, topN=False)
            ucua_rec_dict = PathSim_UCU.predict_next_by_UB(similar_users_ucu, articles[:i], short_train_g, topN=False)
            ulua_rec_dict = PathSim_ULU.predict_next_by_UB(similar_users_ulu, articles[:i], short_train_g, topN=False)


            # Create a dataframe with all recommendations together
            rec_articles = list(set(list(uaua_rec_dict.keys()) + list(ucua_rec_dict.keys()) + list(ulua_rec_dict.keys())))
            rec_df = pd.DataFrame(index=rec_articles, columns=['UAUA', 'UCUA', 'ULUA'])

            for a in rec_df.index:
                rec_df.loc[a, 'UAUA'] = uaua_rec_dict[a] if a in list(uaua_rec_dict.keys()) else 0
                rec_df.loc[a, 'UCUA'] = ucua_rec_dict[a] if a in list(ucua_rec_dict.keys()) else 0
                rec_df.loc[a, 'ULUA'] = ulua_rec_dict[a] if a in list(ulua_rec_dict.keys()) else 0


            # Divide dataframe values by average number of sessions that a single session is connected with through a given path
            rec_importance_df = rec_df.copy()
            for a in rec_importance_df.index:
                rec_importance_df.loc[a, 'UAUA'] = round(
                    rec_importance_df.loc[a, 'UAUA'] / PathSim_UAU.get_avg_n_of_connected_sessions(), 2)
                rec_importance_df.loc[a, 'UCUA'] = round(
                    rec_importance_df.loc[a, 'UCUA'] / PathSim_UCU.get_avg_n_of_connected_sessions(), 2)
                rec_importance_df.loc[a, 'ULUA'] = round(
                    rec_importance_df.loc[a, 'ULUA'] / PathSim_ULU.get_avg_n_of_connected_sessions(), 2)

            rec_importance_df['vote_sum'] = rec_importance_df['UAUA'] + rec_importance_df['UCUA'] + rec_importance_df['ULUA']

            item_expl_score_dict = dict(zip(rec_importance_df.index, rec_importance_df['vote_sum']))
            explainability[user] = item_expl_score_dict

import os
import pickle


results_dir = '.\\Data\\Results\\Explainability\\'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

pickle.dump(explainability, open(results_dir + "UI_Explainability.pickle", "wb")) # User-Itemm

# ---

from_pickle = pickle.load(open(results_dir + 'UI_Explainability.pickle', 'rb'))
print(pd.DataFrame(from_pickle).fillna(0))