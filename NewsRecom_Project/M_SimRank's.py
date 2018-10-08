import numpy as np
import matplotlib.pyplot as plt
import datetime
import networkx as nx
from collections import defaultdict
from operator import itemgetter

from data_import import *
from graph_manipulation import *
from time_aware_splits import *
from popularity_based_rec import *
from personalized_prank import *
from simrank import *
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

SHORT_DAYS = 2#7
MEDIUM_DAYS = 14

# -------------------------------------------------------
# F/P:
# F - take data from data file, create a graph on the base of it, store into gpickle file;
# P - take previously prepared data from gpickle file
file_or_pickle = 'P'

# If file_or_pickle = 'F' specify a file name to store the created graph to.
# Else, specify a pickle file name where to take the graph from.
# ------------------------------------
#pickle_name = 'G_Video33'

#pickle_name = 'G_Video33_cat'
#pickle_name = 'G_Video33_cat_active_users'
#pickle_name = 'G_Video33_cat_active_users_3'
#pickle_name = 'G_Video33_cat_active_users_4'
#pickle_name = 'G_Video33_cat_active_users_5'
#pickle_name = 'G_Video33_1month'
#pickle_name = 'G_Video33_cat_1month'
#pickle_name = 'G_Video33_1week'
#pickle_name = 'G_Video33_cat_1week'
#pickle_name = 'G_Gabriele'

pickle_name = 'Articles_AllData_loc'
# pickle_name = 'Articles_LongSessions(2)_loc'
# ------------------------------------

results_dir = '.\\Data\\Results\\' + pickle_name + '_' + str(SHORT_DAYS) + '_' + str(MEDIUM_DAYS) + '\\'

if file_or_pickle == 'F':

    # ------ Data import ------------------------------------
    di = DataImport()
    di.import_user_click_data(DATA_PATH, adjust_pk_names=True)

    # --- Reduce dataset to 1 month / 1 week / ...
    #di.reduce_timeframe(dt.datetime(2017,3,1), dt.datetime(2017,3,31)) # if G_Video33_1month is selected
    #di.reduce_timeframe(dt.datetime(2017, 3, 1), dt.datetime(2017, 3, 7)) # if G_Video33_1week is selected

    # --- Remove inactive users (the ones with small number of sessions in total)
    di.remove_inactive_users(n_sessions=1)

    # ------ Create a graph on the base of the dataframe ----
    gm = GraphManipulation()
    gm.create_graph(di.user_ses_df)

    # --- Remove short sessions (sessions that contain less or equal than specified number of of articles)
    gm.filter_sessions(gm.G, n_articles=1)

    # ---------- Add categories -----------------------------
    di.import_categories_data(CAT_DATA_PATH)
    gm.add_categories_data(di.categories_data)

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
#
# print('\n--- Sessions per user ---')
sessions_per_user = gm.get_sessions_per_user(G)
print('Avg # of sessions per user:', np.round(np.mean(sessions_per_user), 2))
print('Max # of sessions per user:', np.max(sessions_per_user))
# plt.hist(sessions_per_user, bins=50)
# plt.ylabel('Sessions per user')
# plt.show()
#
# print('\n--- Articles per session ---')
articles_per_session = gm.get_articles_per_session(G)
print('Avg # of articles per session:', np.round(np.mean(articles_per_session), 2))
print('Max # of articles per session:', np.max(articles_per_session))
# plt.hist(articles_per_session, bins=15)
# plt.ylabel('Articles per session')
# plt.show()
#
#exit()
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
#
# SimRank_UA = SimRankRec(N)
# SimRank_SA = SimRankRec(N)
# SimRank_AC = SimRankRec(N)
# SimRank_AL = SimRankRec(N)
#
# SimRank_USA = SimRankRec(N)
# SimRank_UAC = SimRankRec(N)
# SimRank_UAL = SimRankRec(N)
# SimRank_SAC = SimRankRec(N)
# SimRank_SAL = SimRankRec(N)
# SimRank_ACL = SimRankRec(N)
#
# SimRank_USAC = SimRankRec(N)
# SimRank_USAL = SimRankRec(N)
# SimRank_UACL = SimRankRec(N)
# SimRank_SACL = SimRankRec(N)

SimRank_USACL = SimRankRec(N)


ae = AccuracyEvaluation(G)

import pickle
explainability = pickle.load(open('.\\Data\\Results\\Explainability\\SI_Explainability.pickle', 'rb'))
ae.explainability_matrix = explainability



for i, tw_iter in enumerate(tas.time_window_graph_list):

    # if i == 1:
    #     break

    print('\n\n======= Time split', i, '=======')

    # long_train_g = tw_iter[0]
    test_g = tw_iter[1]

    # ------ From test_g remove sessions with less or equal number of articles needed for building recommendation
    # test_g = gm.filter_sessions(test_g, n_articles=MIN_ITEMS_N)
    # nx.draw_networkx(test_g)
    # plt.show()
    # exit()


    # ------ 1. Create a time-ordered list of user sessions
    test_sessions = sorted([(s, attr['datetime']) for s, attr in test_g.nodes(data=True) if attr['entity'] == 'S'], key=lambda x: x[1])

    # For each step a ranked list of N recommendations is created
    for (s, s_datetime) in test_sessions:

        short_train_g = tas.create_short_term_train_set(s_datetime, short_back_timedelta)
        if len(short_train_g) == 0:
            continue

        # medium_train_g = tas.create_short_term_train_set(s_datetime, medium_back_timedelta)

        # -----------------------------------------------------
        # ------------------- Popularity ----------------------
        pop.compute_pop(short_train_g)


        # -----------------------------------------------------
        # ------------------ Random Walks ---------------------

        # -------------------------------
        # ---- Subgraphs of size 2 ------

        # --- Create train graphs
        # ua_train_g = gm.derive_adjacency_multigraph(short_train_g, 'U', 'A', 2)
        # sa_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities = ['S','A'])
        # ac_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities = ['A','C'])
        # al_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities = ['A','L'])
        #
        # # --- Run models
        # SimRank_UA.compute_similarity_matrix(ua_train_g)
        # SimRank_SA.compute_similarity_matrix(sa_train_g)
        # SimRank_AC.compute_similarity_matrix(ac_train_g)
        # SimRank_AL.compute_similarity_matrix(al_train_g)


        # -------------------------------
        # ---- Subgraphs of size 3 ------

        # --- Create train graphs
        # usa_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A'])
        # sac_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A', 'C'])
        # sal_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A', 'L'])
        # acl_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['A', 'C', 'L'])
        # uac_train_g = nx.compose(ua_train_g, nx.MultiGraph(ac_train_g))
        # ual_train_g = nx.compose(ua_train_g, nx.MultiGraph(al_train_g))
        #
        # # --- Run models
        # SimRank_USA.compute_similarity_matrix(usa_train_g)
        # SimRank_SAC.compute_similarity_matrix(sac_train_g)
        # SimRank_SAL.compute_similarity_matrix(sal_train_g)
        # SimRank_ACL.compute_similarity_matrix(acl_train_g)
        # SimRank_UAC.compute_similarity_matrix(uac_train_g)
        # SimRank_UAL.compute_similarity_matrix(ual_train_g)

        # -------------------------------
        # ---- Subgraphs of size 4 ------

        # --- Create train graphs
        # usac_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A', 'C'])
        # usal_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A', 'L'])
        # sacl_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A', 'C', 'L'])
        # uacl_train_g = nx.compose(ua_train_g, nx.MultiGraph(acl_train_g))
        #
        # # --- Run models
        # SimRank_USAC.compute_similarity_matrix(usac_train_g)
        # SimRank_USAL.compute_similarity_matrix(usal_train_g)
        # SimRank_SACL.compute_similarity_matrix(sacl_train_g)
        # SimRank_UACL.compute_similarity_matrix(uacl_train_g)


        # -------------------------------
        # ---- Subgraph of size 5 -------

        # --- Run model
        SimRank_USACL.compute_similarity_matrix(short_train_g, max_iter=10)


        # -----------------------------------------------------------------------------------------------------------
        # ------------- Building Recommendations --------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------

        user = [n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'US'][0]
        articles = sorted([n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'SA'],
                         key=lambda x: test_g.get_edge_data(s, x)['reading_datetime'])

        for i in range(MIN_ITEMS_N, len(articles)):

            session_timeviews = [gm.map_timeview(test_g, s, a) for a in articles[:i]]

            # ------------ POP -------------------------------
            pop_rec = pop.predict_next(user, articles[:i])

            # print('session:', s, 'articles:', articles[:i])
            # print('Actually read next article:', articles[i])

            # --------- Random Walks -------------------------
            # ua_rec = SimRank_UA.predict_next(user, articles[:i])
            # print('ua_rec:', ua_rec)
            # if len(ua_rec) > 0:
            #     continue
            # sa_rec = SimRank_SA.predict_next(user, articles[:i])
            # print('sa_rec:', sa_rec)
            # if len(sa_rec) > 0:
            #     continue
            # ac_rec = SimRank_AC.predict_next(user, articles[:i])
            # print('ac_rec:', ac_rec)
            # if len(ac_rec) > 0:
            #     continue
            # al_rec = SimRank_AL.predict_next(user, articles[:i])
            # print('al_rec:', al_rec)
            # if len(al_rec) > 0:
            #     continue
            #
            # usa_rec = SimRank_USA.predict_next(user, articles[:i])
            # if len(usa_rec) > 0:
            #     continue
            # sac_rec = SimRank_SAC.predict_next(user, articles[:i])
            # if len(sac_rec) > 0:
            #     continue
            # sal_rec = SimRank_SAL.predict_next(user, articles[:i])
            # if len(sal_rec) > 0:
            #     continue
            # uac_rec = SimRank_UAC.predict_next(user, articles[:i])
            # if len(uac_rec) > 0:
            #     continue
            # ual_rec = SimRank_UAL.predict_next(user, articles[:i])
            # if len(ual_rec) > 0:
            #     continue
            # acl_rec = SimRank_ACL.predict_next(user, articles[:i])
            # if len(acl_rec) > 0:
            #     continue
            #
            # usac_rec = SimRank_USAC.predict_next(user, articles[:i])
            # if len(usac_rec) > 0:
            #     continue
            # usal_rec = SimRank_USAL.predict_next(user, articles[:i])
            # if len(usal_rec) > 0:
            #     continue
            # sacl_rec = SimRank_SACL.predict_next(user, articles[:i])
            # if len(sacl_rec) > 0:
            #     continue
            # uacl_rec = SimRank_UACL.predict_next(user, articles[:i])
            # if len(uacl_rec) > 0:
            #     continue

            # usacl_rec_la = SimRank_USACL.predict_next(user, articles[:i], method=0)
            # print(usacl_rec_la)
            usacl_rec_m = SimRank_USACL.predict_next(user, articles[:i], method=1)
            # print(usacl_rec_m)





            # -------------------------------------------------
            # ------- Measuring accuracy ----------------------

            ae.evaluate_recommendation(rec=pop_rec, truth=articles[i], method='POP', s=s)

            # ae.evaluate_recommendation(rec=ua_rec, truth=articles[i], method='SimRank_UA')
            # ae.evaluate_recommendation(rec=sa_rec, truth=articles[i], method='SimRank_SA')
            # ae.evaluate_recommendation(rec=ac_rec, truth=articles[i], method='SimRank_AC')
            # ae.evaluate_recommendation(rec=al_rec, truth=articles[i], method='SimRank_AL')
            #
            # ae.evaluate_recommendation(rec=usa_rec, truth=articles[i], method='SimRank_USA')
            # ae.evaluate_recommendation(rec=sac_rec, truth=articles[i], method='SimRank_SAC')
            # ae.evaluate_recommendation(rec=sal_rec, truth=articles[i], method='SimRank_SAL')
            # ae.evaluate_recommendation(rec=uac_rec, truth=articles[i], method='SimRank_UAC')
            # ae.evaluate_recommendation(rec=ual_rec, truth=articles[i], method='SimRank_UAL')
            # ae.evaluate_recommendation(rec=acl_rec, truth=articles[i], method='SimRank_ACL')
            #
            # ae.evaluate_recommendation(rec=usac_rec, truth=articles[i], method='SimRank_USAC')
            # ae.evaluate_recommendation(rec=usal_rec, truth=articles[i], method='SimRank_USAL')
            # ae.evaluate_recommendation(rec=sacl_rec, truth=articles[i], method='SimRank_SACL')
            # ae.evaluate_recommendation(rec=uacl_rec, truth=articles[i], method='SimRank_UACL')

            # ae.evaluate_recommendation(rec=usacl_rec_la, truth=articles[i], method='SimRank_USACL (la)')
            # print('rec:', ae.rec_precision['SimRank_USACL (la)'])
            ae.evaluate_recommendation(rec=usacl_rec_m, truth=articles[i], method='SimRank_USACL (m)', s=s)


        #     print(ae.rec_precision['RWR_s_SA'] == ae.rec_precision['RWR_s_UA'])
        #     print(ae.rec_precision['RWR_s_SA'] == ae.rec_precision['RWR_s_AC'])
        #     print(ae.rec_precision['RWR_s_SA'] == ae.rec_precision['RWR_s_USACL'])
        #
        # exit()
        ae.evaluate_session()
        # print('session:', ae.session_precision['SimRank_USACL (la)'])

    ae.evaluate_tw()
    # print('tw (la):', ae.tw_precision['SimRank_USACL (la)'])
    print('tw:', ae.tw_precision['SimRank_USACL (m)'])

ae.evaluate_total_performance()
# print('total:', ae.precision['SimRank_USACL (la)'])


print('\n ---------- METHODS EVALUATION -------------\n')

methods = [k for k, v in sorted(ae.precision.items(), key=itemgetter(1), reverse=True)]
for m in methods:
    print('---', m, ': Precision:', ae.precision[m], 'NDCG:', ae.ndcg[m], 'ILD:', ae.diversity[m], 'Explainability:', ae.explainability[m])


exit()

# --- Create period index for plotting
p_start = tas.time_span_list[1][0]
p_end = tas.time_span_list[len(tas.time_span_list)-1][1] + datetime.timedelta(days=1)
month_range = pd.date_range(p_start, p_end, freq='M')
p = []
for period in month_range:
    p.append(datetime.datetime.strftime(period, format='%Y-%m'))


import os
import pickle

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

pickle.dump(p, open(results_dir + "Periods.pickle", "wb"))

pickle.dump(ae.tw_precision, open(results_dir + "TW_Precision.pickle", "wb"))
pickle.dump(ae.tw_ndcg, open(results_dir + "TW_NDCG.pickle", "wb"))
pickle.dump(ae.tw_diversity, open(results_dir + "TW_Diversity.pickle", "wb"))

pickle.dump(ae.precision, open(results_dir + "Precision.pickle", "wb"))
pickle.dump(ae.ndcg, open(results_dir + "NDCG.pickle", "wb"))
pickle.dump(ae.diversity, open(results_dir + "Diversity.pickle", "wb"))
