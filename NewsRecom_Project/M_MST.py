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
MIN_ITEMS_N = 2
MIN_N_SESSIONS = 1

SHORT_DAYS = 2
MEDIUM_DAYS = 14

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
# pickle_name = 'Articles_LongSessions(3)'
# pickle_name = 'Articles_LongSessions(4)'
# pickle_name = 'Articles_ActiveUsers(3)'
# ------------------------------------

# results_dir = '.\\Data\\Results\\' + pickle_name + '_' + str(SHORT_DAYS) + '_' + str(MEDIUM_DAYS) + '\\'
results_dir = '.\\Data\\Results\\Newest\\MST\\'

if file_or_pickle == 'F':

    # ------ Data import ------------------------------------
    di = DataImport()
    di.import_user_click_data(DATA_PATH, adjust_pk_names=True)

    # --- Reduce dataset to 1 month / 1 week / ...
    #di.reduce_timeframe(dt.datetime(2017,3,1), dt.datetime(2017,3,31)) # if G_Video33_1month is selected
    #di.reduce_timeframe(dt.datetime(2017, 3, 1), dt.datetime(2017, 3, 7)) # if G_Video33_1week is selected

    # --- Remove inactive users (the ones with small number of sessions in total)
    di.remove_inactive_users(n_sessions=MIN_N_SESSIONS)

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

    G = gm.G

    nx.write_gpickle(gm.G, '.\\Data\\Pickles\\'+pickle_name+'.gpickle')

else:

    # ------ Extract a saved graph from the pickle ----------
    gm = GraphManipulation()
    G = nx.read_gpickle('.\\Data\\Pickles\\'+pickle_name+'.gpickle')

    # G = gm.filter_meaningless_sessions(G, timeview=1)

    gm.G = G

# -----------------------------------------------------------------------------
# ---------------- STATISTICS ABOUT THE DATASET -------------------------------
#
print('--- GENERAL STATISTICS ---')
print('Number of users:', len(gm.get_users(G)))
print('Number of sessions:', len(gm.get_sessions(G)))
print('Number of articles:', len(gm.get_articles(G)))
print('Number of categories:', len(gm.get_categories(G)))

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

# print('--------------------------\nTime span list:\n', tas.time_span_list)
#print('--------------\nTime window graph list:\n', tas.time_window_graph_list)
# exit()

# -----------------------------------------------------------
# ------ Building prequential recommendations ---------------

_dump_process = False
_long_term_user_preferences = False

short_back_timedelta = datetime.timedelta(days=SHORT_DAYS)
medium_back_timedelta = datetime.timedelta(days=MEDIUM_DAYS)

pop = PopularityBasedRec(G, N)

# RWR_SAL = PersonalizedPageRankBasedRec(N)

PathSim_AUA = PathSimRec(N)
PathCount_AUA = PathSimRec(N)


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

    long_train_g = tw_iter[0]
    test_g = tw_iter[1]

    # ------ From test_g remove sessions with less or equal number of articles needed for building recommendation
    test_g = gm.filter_sessions(test_g, n_items=MIN_ITEMS_N)
    if len(test_g) == 0:
        continue

    # print(test_g)

    # -----------------------------------------------------
    # ------- Long-term user preferences ------------------

    # if _long_term_user_preferences:
    #
    #    RWR_l_USACL.compute_transition_matrix(long_train_g)
    #    RWR_l_USACL.create_usercategory_matrix()


    # ------ 1. Create a time-ordered list of user sessions
    test_sessions = sorted([(s, attr['datetime']) for s, attr in test_g.nodes(data=True) if attr['entity'] == 'S'], key=lambda x: x[1])

    # print(test_sessions)

    # For each step a ranked list of N recommendations is created
    for (s, s_datetime) in test_sessions:

        # ------------ Short and Medium Training Sets ---------

        short_train_g = tas.create_short_term_train_set(s_datetime, short_back_timedelta)
        if len(short_train_g) == 0:
            continue

        # sal_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A', 'L'])
        # sal_train_g = short_train_g

        train_set_len.append(len(gm.get_sessions(short_train_g)))
        # print('# sessions:', len(gm.get_sessions(short_train_g)))
        train_len_dict[tw_i].append(len(gm.get_sessions(short_train_g)))
        n_articles_train.append(len(gm.get_articles(short_train_g)))
        # print('# articles:', len(gm.get_articles(short_train_g)))
        ses_per_user = gm.get_sessions_per_user(short_train_g)
        sessions_per_user_in_short_term.append(Counter(ses_per_user))

        # medium_train_g = tas.create_short_term_train_set(s_datetime, medium_back_timedelta)


        # -----------------------------------------------------
        # ------------------- Popularity ----------------------
        pop.compute_pop(short_train_g)

        # -----------------------------------------------------
        # --------------- Short RWR_5knn_2methods's -------------------------
        # RWR_s_USACL.compute_transition_matrix(short_train_g)
        # RWR_s_USACL.create_itemitem_matrix()
        #
        # # -----------------------------------------------------
        # # --------------- Medium RWR_5knn_2methods's ------------------------
        # RWR_m_USACL.compute_transition_matrix(medium_train_g)
        # RWR_m_USACL.create_categorycategory_matrix()


        # RWR_SAL.compute_transition_matrix(sal_train_g)
        # RWR_SAL.create_itemitem_matrix()


        PathSim_AUA.compute_similarity_matrix(short_train_g, 'A', 'U', 2)
        PathCount_AUA.compute_similarity_matrix_my(short_train_g, 'A', 'U', 2)




        user = [n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'US'][0]
        articles = sorted([n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'SA'],
                         key=lambda x: test_g.get_edge_data(s, x)['reading_datetime'])

        articles_also_in_train = [a for a in articles if a in gm.get_articles(short_train_g)]
        if len(articles_also_in_train) <= MIN_ITEMS_N:
            continue
        articles = articles_also_in_train


        avg_ses_len[tw_i].append(len(articles))
        # print('ses len:', len(articles))

        # print('----------\narticles:', articles)
        # print('session:', s, s_datetime)

        for i in range(MIN_ITEMS_N, len(articles)):

            # if articles[i] not in gm.get_articles(sal_train_g):
            #     continue

            # session_categories = [gm.map_category(a) for a in articles[:i]]
            session_timeviews = [gm.map_timeview(test_g, s, a) for a in articles[:i]]

            # if all([t==1 for t in session_timeviews]):
            #     continue

            # ------- POP --------------------------
            pop_rec = pop.predict_next(user, articles[:i])
            # print('--pop_rec:', pop_rec)

            # ------- POP (LC) ---------------------
            # --- Recommend most popular articles from the last viewed item category
            # --- (at first - articles from the last viewed category, then second last, etc. until N recommendations is made)
            # pop_last_cat_rec = pop.predict_next(user, articles[:i], cat_list=session_categories[::-1])
            # pop_last_cat_rec = pop_last_cat_rec if len(pop_last_cat_rec) > 0 else pop_rec
            # # print('--pop_last_cat_rec:', pop_last_cat_rec)
            #
            # # ------- POP (MPC-m) ------------------
            # # --- Recommend most popular articles from the most probable next category
            # # --- (category relevance vector is calculated as the average (mean) transition score of all categories of the session)
            # cat_trans_vector_m = RWR_m_USACL.get_category_relevance_vector(session_categories, method=1)
            # if len(cat_trans_vector_m) > 0:
            #     sorted_cat_trans_vector_m = [k for k, v in sorted(cat_trans_vector_m.items(), key=itemgetter(1), reverse=True)]
            # else:
            #     sorted_cat_trans_vector_m = []
            # pop_mp_m_cat_rec = pop.predict_next(user, articles[:i], cat_list=sorted_cat_trans_vector_m)
            # # print('pop_mp_m_cat_rec:', pop_mp_m_cat_rec)
            #
            # # ------- POP (MPC-s) ------------------
            # # --- Recommend most popular articles from the most probable next category
            # # --- (category relevance vector is calculated with sigmoid weights assignment for transition scores of all categories of the session)
            # cat_trans_vector_s = RWR_m_USACL.get_category_relevance_vector(session_categories, method=2)
            # if len(cat_trans_vector_s) > 0:
            #     sorted_cat_trans_vector_s = [k for k, v in sorted(cat_trans_vector_s.items(), key=itemgetter(1), reverse=True)]
            # else:
            #     sorted_cat_trans_vector_s = []
            # pop_mp_s_cat_rec = pop.predict_next(user, articles[:i], cat_list=sorted_cat_trans_vector_s)
            # # print('pop_mp_s_cat_rec:', pop_mp_s_cat_rec)
            #
            # # ------- POP (MPC-t) ------------------
            # # --- Recommend most popular articles from the most probable next category
            # # --- (category relevance vector is calculated with timeview weights assignment for transition scores of all categories of the session)
            # cat_trans_vector_t = RWR_m_USACL.get_category_relevance_vector(session_categories, method=3, timeviews=session_timeviews)
            # if len(cat_trans_vector_t) > 0:
            #     sorted_cat_trans_vector_t = [k for k, v in sorted(cat_trans_vector_t.items(), key=itemgetter(1), reverse=True)]
            # else:
            #     sorted_cat_trans_vector_t = []
            # pop_mp_t_cat_rec = pop.predict_next(user, articles[:i], cat_list=sorted_cat_trans_vector_t)
            # # print('pop_mp_t_cat_rec:', pop_mp_t_cat_rec)



            # --------- Short ------------
            # s_rec_la = RWR_s_USACL.predict_next(user, articles[:i], method=0)
            # # s_rec_la = s_rec_la if len(s_rec_la) > 0 else pop_rec
            # # print('--s_rec_la:', s_rec_la)
            #
            # s_rec_m = RWR_s_USACL.predict_next(user, articles[:i], method=1)
            # # s_rec_m = s_rec_m if len(s_rec_m) > 0 else pop_rec
            # # print('--s_rec_m:', s_rec_m)
            #
            # s_rec_s = RWR_s_USACL.predict_next(user, articles[:i], method=2)
            # # s_rec_s = s_rec_s if len(s_rec_s) > 0 else pop_rec
            # # print('--s_rec_s:', s_rec_s)
            #
            # s_rec_t = RWR_s_USACL.predict_next(user, articles[:i], method=3, timeviews=session_timeviews)
            # # s_rec_t = s_rec_t if len(s_rec_t) > 0 else pop_rec
            # # print('--s_rec_t:', s_rec_t)


            # s_rec_la = RWR_SAL.predict_next(user, articles[:i], method=0)
            # s_rec_m = RWR_SAL.predict_next(user, articles[:i], method=1)
            # s_rec_s = RWR_SAL.predict_next(user, articles[:i], method=2)
            # s_rec_t = RWR_SAL.predict_next(user, articles[:i], method=3, timeviews=session_timeviews)

            ps_rec_la = PathSim_AUA.predict_next(user, articles[:i], method=0)
            ps_rec_m = PathSim_AUA.predict_next(user, articles[:i], method=1)
            ps_rec_s = PathSim_AUA.predict_next(user, articles[:i], method=2)
            ps_rec_t = PathSim_AUA.predict_next(user, articles[:i], method=3, timeviews=session_timeviews)

            pc_rec_la = PathCount_AUA.predict_next(user, articles[:i], method=0)
            pc_rec_m = PathCount_AUA.predict_next(user, articles[:i], method=1)
            pc_rec_s = PathCount_AUA.predict_next(user, articles[:i], method=2)
            pc_rec_t = PathCount_AUA.predict_next(user, articles[:i], method=3, timeviews=session_timeviews)

            # Only measure accuracy if all predictions could be made (not relying on pop)
            # if len(s_rec_la) == 0 or len(s_rec_m) == 0 or len(s_rec_s) == 0 or len(s_rec_t) == 0:
            #     continue

            # methods = [s_rec_la, s_rec_m, s_rec_s, s_rec_t]
            methods = [ps_rec_la, ps_rec_m, ps_rec_s, ps_rec_t,
                       pc_rec_la, pc_rec_m, pc_rec_s, pc_rec_t]


            if any(len(m) == 0 for m in methods):
                continue

            n_recommendation[tw_i] += 1




            # # ------- Short + Medium -----

            # Only on the base of the last category
            # item_rel_vector = RWR_s_USACL.get_item_relevance_vector(articles[:i], method=3, timeviews=session_timeviews) # (dict)
            #
            # if len(item_rel_vector) > 0:
            #
            #     source_cat = gm.map_category(articles[i-1])
            #     cat_dict = RWR_m_USACL.categorycategory_matrix[source_cat]
            #
            #     cat_vector = [gm.map_category(item) for item in item_rel_vector.keys()]
            #     # print('cat_vector:', cat_vector)
            #
            #     if len(cat_dict) > 0:
            #
            #         sum_item_dict = sum(item_rel_vector.values())
            #         norm_item_relevance_vector = {key: value / sum_item_dict
            #                                       if sum_item_dict != 0 else 0
            #                                       for key, value in item_rel_vector.items()}
            #         # print('norm_item_relevance_vector:', norm_item_relevance_vector)
            #
            #         sum_cat_dict = sum(cat_dict.values())
            #         norm_cat_dict = {key: value / sum_cat_dict for key, value in cat_dict.items()}
            #         # print('norm_cat_dict:', norm_cat_dict)
            #
            #         cat_relevance_vector = [norm_cat_dict[cat]
            #                                 if source_cat in norm_cat_dict else 0
            #                                 for cat in cat_vector]
            #         # print('cat_relevance_vector:', cat_relevance_vector)
            #
            #         mult = [i_rel * c_rel for i_rel, c_rel in zip(norm_item_relevance_vector.values(), cat_relevance_vector)]
            #
            #         final_dict = {key: val for key, val in zip(item_rel_vector.keys(), mult)}
            #         # print(final_dict)
            #         sm_lc_rec = [k for k, v in sorted(final_dict.items(), key=itemgetter(1), reverse=True)][:N]
            #
            #     else:
            #         sm_lc_rec = s_rec_m
            # else:
            #     # Predict the most popular from current category
            #     sm_lc_rec = s_rec_m

            # print('--sm_lc_rec:', sm_lc_rec)


            # On the base of all categories of the session
                        # item_rel_vector = RWR_s_USACL.get_item_relevance_vector(articles[:i], method=1)
            # if len(item_rel_vector) > 0:
            #     item_cat_vector = [gm.map_category(item) for item in item_rel_vector.keys()]
            #     cat_rel_vector = RWR_m_USACL.get_category_relevance_vector(session_categories, method=1)
            #     sm_rec_m = RWR_s_USACL.predict_next_ic(item_rel_vector, item_cat_vector, cat_rel_vector)
            # else:
            #     sm_rec_m = pop_rec
            #
            # item_rel_vector = RWR_s_USACL.get_item_relevance_vector(articles[:i], method=2)
            # if len(item_rel_vector) > 0:
            #     item_cat_vector = [gm.map_category(item) for item in item_rel_vector.keys()]
            #     cat_rel_vector = RWR_m_USACL.get_category_relevance_vector(session_categories, method=2)
            #     sm_rec_s = RWR_s_USACL.predict_next_ic(item_rel_vector, item_cat_vector, cat_rel_vector)
            # else:
            #     sm_rec_s = pop_rec
            #
            # item_rel_vector = RWR_s_USACL.get_item_relevance_vector(articles[:i], method=3, timeviews=session_timeviews)
            # if len(item_rel_vector) > 0:
            #     item_cat_vector = [gm.map_category(item) for item in item_rel_vector.keys()]
            #     cat_rel_vector = RWR_m_USACL.get_category_relevance_vector(session_categories, method=3, timeviews=session_timeviews)
            #     sm_rec_t = RWR_s_USACL.predict_next_ic(item_rel_vector, item_cat_vector, cat_rel_vector)
            # else:
            #     sm_rec_t = pop_rec



            #
            #
            # # ------- Short + Long -----
            #
            # # Currently only category of the last item is considered !
            # if _long_term_user_preferences:
            #     item_rel_vector = RWR_s_USACL.get_item_relevance_vector(articles[:i])  # (dict)
            #     if len(item_rel_vector) > 0:
            #         source_cat = gm.map_category(articles[i - 1])
            #         cat_vector = [gm.map_category(item) for item in item_rel_vector.keys()]
            #
            #         n_items = len(RWR_s_USACL.itemitem_matrix)
            #         norm_item_relevance_vector = [i / n_items for i in item_rel_vector.values()]
            #
            #         if user in RWR_m_USACL.usercategory_matrix:
            #             cat_preference_vector = [RWR_l_USACL.usercategory_matrix[user][cat] for cat in cat_vector]
            #             n_categories = len(RWR_l_USACL.usercategory_matrix[user])
            #             norm_cat_relevance_vector = [c / n_categories for c in cat_relevance_vector]
            #             mult = [i_rel * c_rel for i_rel, c_rel in
            #                     zip(norm_item_relevance_vector, norm_cat_relevance_vector)]
            #         else:
            #             mult = [i_rel for i_rel in norm_item_relevance_vector]
            #
            #         final_dict = {key: val for key, val in zip(item_rel_vector.keys(), mult)}
            #         sl_rec = [k for k, v in sorted(final_dict.items(), key=itemgetter(1), reverse=True)][:N]
            #     else:
            #         # Predict the most popular from current category
            #         source_cat = gm.map_category(articles[i-1])
            #
            #         # Predict the most popular from current category or the next most probable
            #         if user in RWR_l_USACL.usercategory_matrix:
            #             cat_relevance_vector = [k for k, v in
            #                                     sorted(RWR_l_USACL.usercategory_matrix[user].items(),
            #                                            key=itemgetter(1), reverse=True)]
            #             sl_rec = pop.predict_next(user, articles[:i], cat_list=cat_relevance_vector)
            #         else:
            #             sl_rec = pop.predict_next(user, articles[:i])




            # To see in detail which recommendations we build for which sessions
            # if _dump_process == True:
            #     print('User:', user, 'Session:', s, 'Session articles:', articles[:i],
            #           'POP Rec:', pop_rec, 'SB Rec', sb_rec, 'Truth:', articles[i])

            # ------- Measuring accuracy ----------------------
            ae.evaluate_recommendation(rec=pop_rec, truth=articles[i], method='POP')
            # ae.evaluate_recommendation(rec=pop_last_cat_rec, truth=articles[i], method='POP (lc)')
            # ae.evaluate_recommendation(rec=pop_mp_m_cat_rec, truth=articles[i], method='POP (m)')
            # ae.evaluate_recommendation(rec=pop_mp_s_cat_rec, truth=articles[i], method='POP (s)')
            # ae.evaluate_recommendation(rec=pop_mp_t_cat_rec, truth=articles[i], method='POP (t)')

            # ae.evaluate_recommendation(rec=s_rec_la, truth=articles[i], method='AA (la)')
            # ae.evaluate_recommendation(rec=s_rec_m, truth=articles[i], method='AA (m)')
            # ae.evaluate_recommendation(rec=s_rec_s, truth=articles[i], method='AA (s)')
            # ae.evaluate_recommendation(rec=s_rec_t, truth=articles[i], method='AA (t)')

            ae.evaluate_recommendation(rec=ps_rec_la, truth=articles[i], method='PathSim (la)')
            ae.evaluate_recommendation(rec=ps_rec_m, truth=articles[i], method='PathSim (m)')
            ae.evaluate_recommendation(rec=ps_rec_s, truth=articles[i], method='PathSim (s)')
            ae.evaluate_recommendation(rec=ps_rec_t, truth=articles[i], method='PathSim (t)')

            ae.evaluate_recommendation(rec=pc_rec_la, truth=articles[i], method='PathCount (la)')
            ae.evaluate_recommendation(rec=pc_rec_m, truth=articles[i], method='PathCount (m)')
            ae.evaluate_recommendation(rec=pc_rec_s, truth=articles[i], method='PathCount (s)')
            ae.evaluate_recommendation(rec=pc_rec_t, truth=articles[i], method='PathCount (t)')

            # ae.evaluate_recommendation(rec=s_rec_m, truth=articles[i], method='AA + CC (m)')
            # ae.evaluate_recommendation(rec=s_rec_s, truth=articles[i], method='AA + CC (s)')
            # ae.evaluate_recommendation(rec=s_rec_t, truth=articles[i], method='AA + CC (t)')

            # ae.evaluate_recommendation(rec=sm_lc_rec, truth=articles[i], method='AA (m) + CC (LC)')
            # ae.evaluate_recommendation(rec=sm_mpc_rec, truth=articles[i], method='AA (m) + CC (MPC-s)')

            # ae.evaluate_recommendation(rec=s_rec, truth=articles[i], method='AA(s)')
            # ae.evaluate_recommendation(rec=sm_rec, truth=articles[i], method='AA(s) + CC(m)')
            # ae.evaluate_recommendation(rec=sl_rec, truth=articles[i], method='AA(s) + UC(l)')
            # ae.evaluate_recommendation(rec=sml_rec, truth=articles[i], method='AA(s) + CC(m) + UC(l)')

        ae.evaluate_session()

    ae.evaluate_tw()

ae.evaluate_total_performance()

avg_n_ses_per_train_per_period = [round(np.mean(l)) for l in train_len_dict.values()]
avg_ses_len_per_period = [round(np.mean(l),2) for l in avg_ses_len.values()]

print('\n\n\nNumber of sessions per user per short train period:\n', sessions_per_user_in_short_term)
print('# of recommendations per time split:', n_recommendation.values())
print('Average # sessions per train per period', avg_n_ses_per_train_per_period)
print('Average # artiles per session per period', avg_ses_len_per_period)
print('Average # sessions in train:', round(np.mean(train_set_len), 2))
print('Average # articles in train:', round(np.mean(n_articles_train), 2))

print('\n ---------- METHODS EVALUATION -------------')

print('Total # of recs:', sum(n_recommendation.values()))

methods = [k for k, v in sorted(ae.precision.items(), key=itemgetter(1), reverse=True)]
for m in methods:
    print('---', m, ': Precision:', ae.precision[m], 'NDCG:', ae.ndcg[m], 'ILD:', ae.diversity[m])


# exit()


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












exit()

# -------------- PLOT ----------------------

metrics_name = 'Precision'
metrics = ae.precision

plot_title = 'Data: ' + pickle_name + \
        '\nShort term: ' + str(SHORT_DAYS) + ' days' + \
        '\nMin N items: ' + str(MIN_ITEMS_N) + \
        '\nTop N: ' + str(N)

plot_suptitle = 'Comparison of different weight assignment to session articles'

methods = [k for k, v in sorted(metrics.items(), key=itemgetter(1), reverse=True)]
avg = [str(round(metrics[m], 3)) for m in methods]

legend = [str(k)+': '+str(v) for k, v in zip(methods, avg)]

fig = plt.figure(figsize=(15, 7))

for m in methods:
    plt.plot(p, [v for v in ae.tw_precision[m]], marker='o')

plt.suptitle(plot_suptitle, fontsize=12)
plt.title(plot_title, fontsize=10, loc='left')

plt.xticks(rotation=90)
plt.ylabel('% ' + metrics_name)
plt.legend(legend, loc='upper right')
plt.grid()

plt.show()






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

pickle.dump(avg_n_ses_per_train_per_period, open(results_dir + "Avg_N_sessions_per_train", "wb"))



