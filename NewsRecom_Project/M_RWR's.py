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

SHORT_DAYS = 7#7
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

#pickle_name = 'AllData'
#pickle_name = 'ActiveUsers(2)'
#pickle_name = 'LongSessions(2)_USAC'
# pickle_name = 'LongSessions(2)_USACL'
#pickle_name = 'ActiveUsers(2)_LongSessions(2)_USACL'
pickle_name = 'Articles_AllData_loc'
# ------------------------------------

# results_dir = '.\\Data\\Results\\' + pickle_name + '_' + str(SHORT_DAYS) + '_' + str(MEDIUM_DAYS) + '\\'
results_dir = '.\\Data\\Results\\Newest\\RWR_5knn_2methods\\LongSessions(2)\\'

if file_or_pickle == 'F':

    # ------ Data import ------------------------------------
    di = DataImport()
    di.import_user_click_data(DATA_PATH, adjust_pk_names=True)

    # --- Reduce dataset to 1 month / 1 week / ...
    # di.reduce_timeframe(dt.datetime(2017,3,1), dt.datetime(2017,3,31)) # if G_Video33_1month is selected
    # di.reduce_timeframe(dt.datetime(2017, 3, 1), dt.datetime(2017, 3, 7)) # if G_Video33_1week is selected

    # --- Remove inactive users (the ones with small number of sessions in total)
    # di.remove_inactive_users(n_sessions=MIN_N_SESSIONS)

    # ---------- Add categories -----------------------------
    di.import_categories_data(CAT_DATA_PATH)

    # ---- Leave only sessions with at least specified number of articles
    di.filter_short_sessions(n_items=MIN_ITEMS_N)

    # ------ Create a graph on the base of the dataframe ----
    gm = GraphManipulation(G_structure='USAC')
    gm.create_graph(di.user_ses_df)

    # Filter again, because dataframe filtering leaves sessions where the same article is repeatedly read several times
    # gm.filter_sessions(gm.G, n_items=MIN_ITEMS_N)
    # gm.filter_users(gm.G, n_sessions=MIN_N_SESSIONS)

    # ---------- Add locations ------------------------------
    di.import_locations_data(LOC_DATA_PATH)
    gm.add_locations_data(di.locations_data)

    G = gm.G

    nx.write_gpickle(gm.G, '.\\Data\\Pickles\\' + pickle_name + '.gpickle')

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

# RWR_s_UA = PersonalizedPageRankBasedRec(N)
# RWR_s_SA = PersonalizedPageRankBasedRec(N)
# RWR_s_AC = PersonalizedPageRankBasedRec(N)
# RWR_s_AL = PersonalizedPageRankBasedRec(N)
#
# RWR_s_USA = PersonalizedPageRankBasedRec(N)
# RWR_s_UAC = PersonalizedPageRankBasedRec(N)
# RWR_s_UAL = PersonalizedPageRankBasedRec(N)
# RWR_s_SAC = PersonalizedPageRankBasedRec(N)
# RWR_s_SAL = PersonalizedPageRankBasedRec(N)
# RWR_s_ACL = PersonalizedPageRankBasedRec(N)
#
# RWR_s_USAC = PersonalizedPageRankBasedRec(N)
# RWR_s_USAL = PersonalizedPageRankBasedRec(N)
# RWR_s_UACL = PersonalizedPageRankBasedRec(N)
# RWR_s_SACL = PersonalizedPageRankBasedRec(N)
#
RWR_s_USACL = PersonalizedPageRankBasedRec(N)


# pop1 = PopularityBasedRec(G, N)
# pop2 = PopularityBasedRec(G, N)

# RWR_s_USACL05 = PersonalizedPageRankBasedRec(N)
# RWR_s_USACL1 = PersonalizedPageRankBasedRec(N)
# RWR_s_USACL2 = PersonalizedPageRankBasedRec(N)
# RWR_s_USACL3 = PersonalizedPageRankBasedRec(N)
# RWR_s_USACL7 = PersonalizedPageRankBasedRec(N)



ae = AccuracyEvaluation(G)

n_recommendation = dict()

for tw_i, tw_iter in enumerate(tas.time_window_graph_list):

    print('\n\n======= Time split', tw_i, '=======')

    n_recommendation[tw_i] = 0

    long_train_g = tw_iter[0]
    test_g = tw_iter[1]

    # ------ From test_g remove sessions with less or equal number of articles needed for building recommendation
    # test_g = gm.filter_sessions(test_g, n_articles=MIN_ITEMS_N)
    # nx.draw_networkx(test_g)
    # plt.show()
    # exit()

    # ------ Time split statistics -----
    if _dump_process == False:
        print('Number of users:',
              len(gm.get_users(long_train_g)),
              len(gm.get_users(short_train_g)),
              len(gm.get_users(test_g)))
        print('Number of sessions:',
              len(gm.get_sessions(long_train_g)),
              len(gm.get_sessions(short_train_g)),
              len(gm.get_sessions(test_g)))
        print('Number of articles:',
              len(gm.get_articles(long_train_g)),
              len(gm.get_articles(short_train_g)),
              len(gm.get_articles(test_g)))
        print('Number of articles per each category in test set:',
              [(n, nx.degree(short_train_g, n)) for n in gm.get_categories(test_g)])
        print('Avg # of sessions per user:',
              np.round(np.mean(gm.get_sessions_per_user(long_train_g)), 2),
              np.round(np.mean(gm.get_sessions_per_user(short_train_g)), 2),
              np.round(np.mean(gm.get_sessions_per_user(test_g)), 2))
        print('Avg # of articles per session:',
              np.round(np.mean(gm.get_articles_per_session(long_train_g)), 2),
              np.round(np.mean(gm.get_articles_per_session(short_train_g)), 2),
              np.round(np.mean(gm.get_articles_per_session(test_g)), 2))


    # ------ 1. Create a time-ordered list of user sessions
    test_sessions = sorted([(s, attr['datetime']) for s, attr in test_g.nodes(data=True) if attr['entity'] == 'S'], key=lambda x: x[1])

    # For each step a ranked list of N recommendations is created
    for (s, s_datetime) in test_sessions:

        short_train_g = tas.create_short_term_train_set(s_datetime, short_back_timedelta)
        if len(short_train_g) == 0:
            continue
        #
        # short_train_g1 = tas.create_short_term_train_set(s_datetime, datetime.timedelta(days=1))
        # if len(short_train_g1) == 0:
        #     continue
        # short_train_g2 = tas.create_short_term_train_set(s_datetime, datetime.timedelta(days=2))
        # if len(short_train_g2) == 0:
        #     continue
        # short_train_g3 = tas.create_short_term_train_set(s_datetime, datetime.timedelta(days=3))
        # if len(short_train_g3) == 0:
        #     continue
        # short_train_g7 = tas.create_short_term_train_set(s_datetime, datetime.timedelta(days=7))
        # if len(short_train_g7) == 0:
        #     continue

        # medium_train_g = tas.create_short_term_train_set(s_datetime, medium_back_timedelta)

        # -----------------------------------------------------
        # ------------------- Popularity ----------------------
        # pop.compute_pop(short_train_g)

        # pop1.compute_pop(short_train_g1)
        # pop2.compute_pop(short_train_g2)


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
        # RWR_s_UA.compute_transition_matrix(ua_train_g)
        # RWR_s_SA.compute_transition_matrix(sa_train_g)
        # RWR_s_AC.compute_transition_matrix(ac_train_g)
        # RWR_s_AL.compute_transition_matrix(al_train_g)
        #
        # # --- Extract II matrices
        # RWR_s_UA.create_itemitem_matrix()
        # RWR_s_SA.create_itemitem_matrix()
        # RWR_s_AC.create_itemitem_matrix()
        # RWR_s_AL.create_itemitem_matrix()
        #
        #
        #
        # # -------------------------------
        # # ---- Subgraphs of size 3 ------
        #
        # # --- Create train graphs
        # usa_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A'])
        # sac_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A', 'C'])
        # sal_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A', 'L'])
        # acl_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['A', 'C', 'L'])
        # uac_train_g = nx.compose(ua_train_g, nx.MultiGraph(ac_train_g))
        # ual_train_g = nx.compose(ua_train_g, nx.MultiGraph(al_train_g))
        #
        # # --- Run models
        # RWR_s_USA.compute_transition_matrix(usa_train_g)
        # RWR_s_SAC.compute_transition_matrix(sac_train_g)
        # RWR_s_SAL.compute_transition_matrix(sal_train_g)
        # RWR_s_ACL.compute_transition_matrix(acl_train_g)
        # RWR_s_UAC.compute_transition_matrix(uac_train_g)
        # RWR_s_UAL.compute_transition_matrix(ual_train_g)
        #
        # # --- Extract II matrices
        # RWR_s_USA.create_itemitem_matrix()
        # RWR_s_SAC.create_itemitem_matrix()
        # RWR_s_SAL.create_itemitem_matrix()
        # RWR_s_ACL.create_itemitem_matrix()
        # RWR_s_UAC.create_itemitem_matrix()
        # RWR_s_UAL.create_itemitem_matrix()
        #
        #
        # # -------------------------------
        # # ---- Subgraphs of size 4 ------
        #
        # # --- Create train graphs
        # usac_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A', 'C'])
        # usal_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A', 'L'])
        # sacl_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A', 'C', 'L'])
        # uacl_train_g = nx.compose(ua_train_g, nx.MultiGraph(acl_train_g))
        #
        # # --- Run models
        # RWR_s_USAC.compute_transition_matrix(usac_train_g)
        # RWR_s_USAL.compute_transition_matrix(usal_train_g)
        # RWR_s_SACL.compute_transition_matrix(sacl_train_g)
        # RWR_s_UACL.compute_transition_matrix(uacl_train_g)
        #
        # # --- Extract II matrices
        # RWR_s_USAC.create_itemitem_matrix()
        # RWR_s_USAL.create_itemitem_matrix()
        # RWR_s_SACL.create_itemitem_matrix()
        # RWR_s_UACL.create_itemitem_matrix()


        # -------------------------------
        # ---- Subgraph of size 5 -------

        # --- Run model
        RWR_s_USACL.compute_transition_matrix(short_train_g)

        # --- Extract II matrix
        RWR_s_USACL.create_itemitem_matrix()

        # RWR_s_USACL05.compute_transition_matrix(short_train_g05)
        # RWR_s_USACL05.create_itemitem_matrix()
        #
        # RWR_s_USACL1.compute_transition_matrix(short_train_g1)
        # RWR_s_USACL1.create_itemitem_matrix()
        #
        # RWR_s_USACL2.compute_transition_matrix(short_train_g2)
        # RWR_s_USACL2.create_itemitem_matrix()
        #
        # RWR_s_USACL3.compute_transition_matrix(short_train_g3)
        # RWR_s_USACL3.create_itemitem_matrix()
        #
        # RWR_s_USACL7.compute_transition_matrix(short_train_g7)
        # RWR_s_USACL7.create_itemitem_matrix()


        # -----------------------------------------------------------------------------------------------------------
        # ------------- Building Recommendations --------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------

        user = [n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'US'][0]
        articles = sorted([n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'SA'],
                         key=lambda x: test_g.get_edge_data(s, x)['reading_datetime'])

        # articles_also_in_train = [a for a in articles if a in gm.get_articles(short_train_g)]
        # if len(articles_also_in_train) <= MIN_ITEMS_N:
        #     continue
        # articles = articles_also_in_train

        for i in range(MIN_ITEMS_N, len(articles)):

            # ------------ POP -------------------------------
            # pop_rec = pop.predict_next(user, articles[:i])

            # pop_rec1 = pop1.predict_next(user, articles[:i])
            # pop_rec2 = pop2.predict_next(user, articles[:i])
            #
            # print('sessions:', s, 'articles:', articles)
            #
            # --------- Random Walks -------------------------
            # ua_rec = RWR_s_UA.predict_next(user, articles[:i])
            # # print('ua_rec:', ua_rec)
            # # ua_rec = ua_rec if len(ua_rec) > 0 else pop_rec
            # sa_rec = RWR_s_SA.predict_next(user, articles[:i])
            # # print('sa_rec:', sa_rec)
            # # sa_rec = sa_rec if len(sa_rec) > 0 else pop_rec
            # ac_rec = RWR_s_AC.predict_next(user, articles[:i])
            # # print('ac_rec:', ac_rec)
            # # ac_rec = ac_rec if len(ac_rec) > 0 else pop_rec
            # al_rec = RWR_s_AL.predict_next(user, articles[:i])
            # # print('al_rec:', al_rec)
            # # al_rec = al_rec if len(a  l_rec) > 0 else pop_rec
            #
            # usa_rec = RWR_s_USA.predict_next(user, articles[:i])
            # # usa_rec = usa_rec if len(usa_rec) > 0 else pop_rec
            # sac_rec = RWR_s_SAC.predict_next(user, articles[:i])
            # # sac_rec = sac_rec if len(sac_rec) > 0 else pop_rec
            # sal_rec = RWR_s_SAL.predict_next(user, articles[:i])
            # # sal_rec = sal_rec if len(sal_rec) > 0 else pop_rec
            # uac_rec = RWR_s_UAC.predict_next(user, articles[:i])
            # # uac_rec = uac_rec if len(uac_rec) > 0 else pop_rec
            # ual_rec = RWR_s_UAL.predict_next(user, articles[:i])
            # # ual_rec = ual_rec if len(ual_rec) > 0 else pop_rec
            # acl_rec = RWR_s_ACL.predict_next(user, articles[:i])
            # # acl_rec = acl_rec if len(acl_rec) > 0 else pop_rec
            #
            # usac_rec = RWR_s_USAC.predict_next(user, articles[:i])
            # # usac_rec = usac_rec if len(usac_rec) > 0 else pop_rec
            # usal_rec = RWR_s_USAL.predict_next(user, articles[:i])
            # # usal_rec = usal_rec if len(usal_rec) > 0 else pop_rec
            # sacl_rec = RWR_s_SACL.predict_next(user, articles[:i])
            # # sacl_rec = sacl_rec if len(sacl_rec) > 0 else pop_rec
            # uacl_rec = RWR_s_UACL.predict_next(user, articles[:i])
            # # uacl_rec = uacl_rec if len(uacl_rec) > 0 else pop_rec

            usacl_rec = RWR_s_USACL.predict_next(user, articles[:i])
            # usacl_pop_rec = usacl_rec if len(usacl_rec) > 0 else pop_rec



            # methods = [ua_rec, sa_rec, ac_rec, al_rec,
            #            usa_rec, uac_rec, ual_rec, sal_rec, sac_rec, acl_rec,
            #            usac_rec, usal_rec, sacl_rec, uacl_rec,
            #            usacl_rec]
            methods = [usacl_rec]

            if any(len(m) == 0 for m in methods):
                continue

            # usacl_rec1 = RWR_s_USACL1.predict_next(user, articles[:i])
            # usacl_rec1_pop = usacl_rec1 if len(usacl_rec1) > 0 else pop_rec1
            # usacl_rec2 = RWR_s_USACL2.predict_next(user, articles[:i])
            # usacl_rec2_pop = usacl_rec2 if len(usacl_rec2) > 0 else pop_rec2
            # usacl_rec3 = RWR_s_USACL3.predict_next(user, articles[:i])
            # usacl_rec7 = RWR_s_USACL7.predict_next(user, articles[:i])


            # methods = [pop_rec1, pop_rec2, usacl_rec1, usacl_rec2, usacl_rec1_pop, usacl_rec2_pop]
            # methods = [pop_rec1, pop_rec2, usacl_rec1_pop, usacl_rec2_pop]
            # methods = [pop_rec, ua_rec, sa_rec, ac_rec, al_rec,
            #            usa_rec, uac_rec, ual_rec, sal_rec, sac_rec, acl_rec,
            #            usac_rec, usal_rec, sacl_rec, uacl_rec,
            #            usacl_rec]
            # methods = [usacl_rec]

            # if any(len(m)==0 for m in methods):
            #     continue

            n_recommendation[tw_i] += 1


            # -------------------------------------------------
            # ------- Measuring accuracy ----------------------

            # ae.evaluate_recommendation(rec=pop_rec, truth=articles[i], method='POP')

            # ae.evaluate_recommendation(rec=ua_rec, truth=articles[i], method='RWR_s_UA')
            # ae.evaluate_recommendation(rec=sa_rec, truth=articles[i], method='RWR_s_SA')
            # ae.evaluate_recommendation(rec=ac_rec, truth=articles[i], method='RWR_s_AC')
            # ae.evaluate_recommendation(rec=al_rec, truth=articles[i], method='RWR_s_AL')
            #
            # ae.evaluate_recommendation(rec=usa_rec, truth=articles[i], method='RWR_s_USA')
            # ae.evaluate_recommendation(rec=sac_rec, truth=articles[i], method='RWR_s_SAC')
            # ae.evaluate_recommendation(rec=sal_rec, truth=articles[i], method='RWR_s_SAL')
            # ae.evaluate_recommendation(rec=uac_rec, truth=articles[i], method='RWR_s_UAC')
            # ae.evaluate_recommendation(rec=ual_rec, truth=articles[i], method='RWR_s_UAL')
            # ae.evaluate_recommendation(rec=acl_rec, truth=articles[i], method='RWR_s_ACL')
            #
            # ae.evaluate_recommendation(rec=usac_rec, truth=articles[i], method='RWR_s_USAC')
            # ae.evaluate_recommendation(rec=usal_rec, truth=articles[i], method='RWR_s_USAL')
            # ae.evaluate_recommendation(rec=sacl_rec, truth=articles[i], method='RWR_s_SACL')
            # ae.evaluate_recommendation(rec=uacl_rec, truth=articles[i], method='RWR_s_UACL')

            ae.evaluate_recommendation(rec=usacl_rec, truth=articles[i], method='RWR_s_USACL')
            # ae.evaluate_recommendation(rec=usacl_pop_rec, truth=articles[i], method='RWR_POP_s_USACL')

            # ae.evaluate_recommendation(rec=pop_rec1, truth=articles[i], method='POP1')
            # ae.evaluate_recommendation(rec=pop_rec2, truth=articles[i], method='POP2')
            # ae.evaluate_recommendation(rec=usacl_rec1, truth=articles[i], method='RWR_s_USACL1')
            # ae.evaluate_recommendation(rec=usacl_rec2, truth=articles[i], method='RWR_s_USACL2')
            # ae.evaluate_recommendation(rec=usacl_rec1_pop, truth=articles[i], method='RWR_s_USACL1_pop')
            # ae.evaluate_recommendation(rec=usacl_rec2_pop, truth=articles[i], method='RWR_s_USACL2_pop')
            # ae.evaluate_recommendation(rec=usacl_rec3, truth=articles[i], method='RWR_s_USACL3')
            # ae.evaluate_recommendation(rec=usacl_rec7, truth=articles[i], method='RWR_s_USACL7')

        #     print(ae.rec_precision['RWR_s_SA'] == ae.rec_precision['RWR_s_UA'])
        #     print(ae.rec_precision['RWR_s_SA'] == ae.rec_precision['RWR_s_AC'])
        #     print(ae.rec_precision['RWR_s_SA'] == ae.rec_precision['RWR_s_USACL'])
        #
        # exit()
        ae.evaluate_session()

    ae.evaluate_tw()

ae.evaluate_total_performance()


print('\n ---------- METHODS EVALUATION -------------\n')

print('# of recommendations per time split:', n_recommendation.values())
print('Total # of recs:', sum(n_recommendation.values()))

methods = [k for k, v in sorted(ae.precision.items(), key=itemgetter(1), reverse=True)]
for m in methods:
    print('---', m, ': Precision:', ae.precision[m], 'NDCG:', ae.ndcg[m], 'ILD:', ae.diversity[m])
    # print('---', m, ': Precision:', ae.precision[m], 'NDCG:', ae.ndcg[m], 'ILD:', ae.diversity[m], 'Explainability:', ae.explainability[m])





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
