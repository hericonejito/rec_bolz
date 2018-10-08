import numpy as np
import matplotlib.pyplot as plt
import datetime
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
MIN_N_SESSIONS = 2

SHORT_DAYS = 2
MEDIUM_DAYS = 7
LONG_DAYS = 30

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
#pickle_name = 'Articles_ActiveUsers(3)'
# pickle_name = 'Articles_ActiveUsers(2)'
# ------------------------------------

# results_dir = '.\\Data\\Results\\' + pickle_name + '_' + str(SHORT_DAYS) + '_' + str(MEDIUM_DAYS) + '\\'
# results_dir = '.\\Data\\Results\\'+pickle_name+'_'+str(SHORT_DAYS)+'_'+str(MEDIUM_DAYS)+'_'+str(MIN_ITEMS_N)+'_SML\\'
results_dir = '.\\Data\\Results\\Newest\\SML\\'

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
    # di.filter_short_sessions(n_items=MIN_ITEMS_N)

    # ------ Create a graph on the base of the dataframe ----
    gm = GraphManipulation(G_structure = 'USAC')
    gm.create_graph(di.user_ses_df)

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
_long_term_user_preferences = True

short_back_timedelta = datetime.timedelta(days=SHORT_DAYS)
medium_back_timedelta = datetime.timedelta(days=MEDIUM_DAYS)
long_back_timedelta = datetime.timedelta(days=LONG_DAYS)

pop = PopularityBasedRec(G, N)

RWR_l_USACL = PersonalizedPageRankBasedRec(N)
RWR_m_USACL = PersonalizedPageRankBasedRec(N)
RWR_s_USACL = PersonalizedPageRankBasedRec(N)

ae = AccuracyEvaluation(G)


n_recommendation = dict()


for tw_i, tw_iter in enumerate(tas.time_window_graph_list):

    print('\n\n======= Time split', tw_i, '=======')

    n_recommendation[tw_i] = 0

    # long_train_g = tw_iter[0]
    test_g = tw_iter[1]

    # ------ From test_g remove sessions with less or equal number of articles needed for building recommendation
    test_g = gm.filter_sessions(test_g, n_items=MIN_ITEMS_N)
    if len(test_g) == 0:
        continue

    # -----------------------------------------------------
    # ------- Long-term user preferences ------------------

    # if _long_term_user_preferences:
    #
    #     # active_users = gm.get_active_users(long_train_g, n_sessions=MIN_N_SESSIONS)
    #     active_users = [u for u in gm.get_users(test_g) if u in gm.get_users(long_train_g)]
    #     if len(active_users) == 0:
    #         continue
    #     RWR_l_USACL.compute_transition_matrix(long_train_g)
    #     RWR_l_USACL.create_usercategory_matrix(user_nodes=active_users)



    # ------ 1. Create a time-ordered list of user sessions
    test_sessions = sorted([(s, attr['datetime']) for s, attr in test_g.nodes(data=True) if attr['entity'] == 'S'], key=lambda x: x[1])


    # -------------------------------------------------------------------
    # ------------ CROSS VALIDATION -------------------------------------

    for CV in range(1):

        # n_recommendations = 0

        # For each step a ranked list of N recommendations is created
        for (s, s_datetime) in test_sessions:

            user = [n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'US'][0]
            print('user:', user)

            long_train_g = tas.create_short_term_train_set(s_datetime, long_back_timedelta)
            active_users = gm.get_users(long_train_g)
            print('active users:', user in active_users)
            if user not in active_users:
                continue

            RWR_l_USACL.compute_transition_matrix(long_train_g)
            RWR_l_USACL.create_usercategory_matrix(user_nodes=active_users)


            # if _long_term_user_preferences:
            #     if user not in active_users:
            #         continue

            # n_recommendations += 1

            # ------------ Short and Medium Training Sets ---------

            short_train_g = tas.create_short_term_train_set(s_datetime, short_back_timedelta)
            if len(short_train_g) == 0:
                continue



            medium_train_g = tas.create_short_term_train_set(s_datetime, medium_back_timedelta)


            # -----------------------------------------------------
            # ------------------- Popularity ----------------------
            pop.compute_pop(short_train_g)

            # -----------------------------------------------------
            # --------------- Short RWR_5knn_2methods's -------------------------
            RWR_s_USACL.compute_transition_matrix(short_train_g)
            RWR_s_USACL.create_itemitem_matrix()

            # -----------------------------------------------------
            # --------------- Short RWR_5knn_2methods's -------------------------
            RWR_m_USACL.compute_transition_matrix(medium_train_g)
            RWR_m_USACL.create_categorycategory_matrix()



            articles = sorted([n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'SA'],
                             key=lambda x: test_g.get_edge_data(s, x)['reading_datetime'])

            # print('----------\narticles:', articles)
            # print('session:', s, s_datetime)

            for i in range(MIN_ITEMS_N, len(articles)):

                print('Next article:', articles[i])

                session_categories = [gm.map_category(a) for a in articles[:i]]
                # session_timeviews = [gm.map_timeview(test_g, s, a) for a in articles[:i]]

                # ------- POP --------------------------
                pop_rec = pop.predict_next(user, articles[:i])
                # print('--pop_rec:', pop_rec)

                # ------- POP (LC) ---------------------
                # --- Recommend most popular articles from the last viewed item category
                # --- (at first - articles from the last viewed category, then second last, etc. until N recommendations is made)
                # pop_last_cat_rec = pop.predict_next(user, articles[:i], cat_list=session_categories[::-1])
                # pop_last_cat_rec = pop_last_cat_rec if len(pop_last_cat_rec) > 0 else pop_rec
                # print('--pop_last_cat_rec:', pop_last_cat_rec)

                # ------- POP (MPC-m) ------------------
                # --- Recommend most popular articles from the most probable next category
                # --- (category relevance vector is calculated as the average (mean) transition score of all categories of the session)
                # cat_trans_vector_m = RWR_m_USACL.get_category_relevance_vector(session_categories, method=1)
                # if len(cat_trans_vector_m) > 0:
                #     sorted_cat_trans_vector_m = [k for k, v in sorted(cat_trans_vector_m.items(), key=itemgetter(1), reverse=True)]
                # else:
                #     sorted_cat_trans_vector_m = []
                # pop_mp_m_cat_rec = pop.predict_next(user, articles[:i], cat_list=sorted_cat_trans_vector_m)
                # print('pop_mp_m_cat_rec:', pop_mp_m_cat_rec)

                # ------- POP (MPC-s) ------------------
                # --- Recommend most popular articles from the most probable next category
                # --- (category relevance vector is calculated with sigmoid weights assignment for transition scores of all categories of the session)
                # cat_trans_vector_s = RWR_m_USACL.get_category_relevance_vector(session_categories, method=2)
                # if len(cat_trans_vector_s) > 0:
                #     sorted_cat_trans_vector_s = [k for k, v in sorted(cat_trans_vector_s.items(), key=itemgetter(1), reverse=True)]
                # else:
                #     sorted_cat_trans_vector_s = []
                # pop_mp_s_cat_rec = pop.predict_next(user, articles[:i], cat_list=sorted_cat_trans_vector_s)
                # print('pop_mp_s_cat_rec:', pop_mp_s_cat_rec)

                # ------- POP (MPC-t) ------------------
                # --- Recommend most popular articles from the most probable next category
                # --- (category relevance vector is calculated with timeview weights assignment for transition scores of all categories of the session)
                # cat_trans_vector_t = RWR_m_USACL.get_category_relevance_vector(session_categories, method=3, timeviews=session_timeviews)
                # if len(cat_trans_vector_t) > 0:
                #     sorted_cat_trans_vector_t = [k for k, v in sorted(cat_trans_vector_t.items(), key=itemgetter(1), reverse=True)]
                # else:
                #     sorted_cat_trans_vector_t = []
                # pop_mp_t_cat_rec = pop.predict_next(user, articles[:i], cat_list=sorted_cat_trans_vector_t)
                # print('pop_mp_t_cat_rec:', pop_mp_t_cat_rec)



                # --------- Short ------------
                # s_rec_m = RWR_s_USACL.predict_next(user, articles[:i], method=1)
                # s_rec_m = s_rec_m if len(s_rec_m) > 0 else pop_rec
                # print('--s_rec_m:', s_rec_m)

                s_rec_s = RWR_s_USACL.predict_next(user, articles[:i], method=2)
                # s_rec_s = s_rec_s if len(s_rec_s) > 0 else pop_rec
                print('--s_rec_s:', s_rec_s)

                # s_rec_t = RWR_s_USACL.predict_next(user, articles[:i], method=3, timeviews=session_timeviews)
                # s_rec_t = s_rec_t if len(s_rec_t) > 0 else pop_rec
                # print('--s_rec_t:', s_rec_t)


                # ------ Short + Medium ----------------------
                # On the base of all categories of the session

                # item_rel_vector = RWR_s_USACL.get_item_relevance_vector(articles[:i], method=1)
                # if len(item_rel_vector) > 0:
                #     item_cat_vector = [gm.map_category(item) for item in item_rel_vector.keys()]
                #     cat_rel_vector = RWR_m_USACL.get_category_relevance_vector(session_categories, method=1)
                #     sm_rec_m = RWR_s_USACL.predict_next_ic(item_rel_vector, item_cat_vector, cat_rel_vector)
                #
                #
                #     # print('user:', user)
                #     # print('session_categories:', session_categories)
                #     # print('session_timeviews:', session_timeviews)
                #     # print(pd.DataFrame(RWR_l_USACL.usercategory_matrix))
                #     # print('cat_rel_vector:', cat_rel_vector)
                #     user_cat_rel_vector = RWR_l_USACL.get_user_cat_relevance_vector(user, session_categories, method=1)
                #     # print('user_cat_rel_vector:', user_cat_rel_vector)
                #     sl_rec_m = RWR_s_USACL.predict_next_ic(item_rel_vector, item_cat_vector, user_cat_rel_vector)
                #
                #     sml_rec_m = RWR_s_USACL.predict_next_ic(item_rel_vector, item_cat_vector, cat_rel_vector, user_cat_rel_vector)
                #     # exit()
                # else:
                #     sm_rec_m = pop_rec
                #     sl_rec_m = pop_rec
                #     sml_rec_m = pop_rec



                item_rel_vector = RWR_s_USACL.get_item_relevance_vector(articles[:i], method=2)
                if len(item_rel_vector) > 0:
                    item_cat_vector = [gm.map_category(item) for item in item_rel_vector.keys()]
                    cat_rel_vector = RWR_m_USACL.get_category_relevance_vector(session_categories, method=2)
                    sm_rec_s = RWR_s_USACL.predict_next_ic(item_rel_vector, item_cat_vector, cat_rel_vector=cat_rel_vector)
                    print('---sm_rec_s:', sm_rec_s)

                    if _long_term_user_preferences:
                        user_cat_rel_vector = RWR_l_USACL.get_user_cat_relevance_vector(user, session_categories, method=2)
                        sl_rec_s = RWR_s_USACL.predict_next_ic(item_rel_vector, item_cat_vector, user_cat_rel_vector=user_cat_rel_vector)
                        print('---sl_rec_s:', sl_rec_s)

                        sml_rec_s = RWR_s_USACL.predict_next_ic(item_rel_vector, item_cat_vector, cat_rel_vector, user_cat_rel_vector)
                        print('---sml_rec_s:', sml_rec_s)

                else:
                    # sm_rec_s = pop_rec
                    # if _long_term_user_preferences:
                    #     sl_rec_s = pop_rec
                    #     sml_rec_s = pop_rec
                    continue

                # item_rel_vector = RWR_s_USACL.get_item_relevance_vector(articles[:i], method=3, timeviews=session_timeviews)
                # if len(item_rel_vector) > 0:
                #     item_cat_vector = [gm.map_category(item) for item in item_rel_vector.keys()]
                #     cat_rel_vector = RWR_m_USACL.get_category_relevance_vector(session_categories, method=3, timeviews=session_timeviews)
                #     sm_rec_t = RWR_s_USACL.predict_next_ic(item_rel_vector, item_cat_vector, cat_rel_vector)
                #
                #     user_cat_rel_vector = RWR_l_USACL.get_user_cat_relevance_vector(user, session_categories, method=3, timeviews=session_timeviews)
                #     sl_rec_t = RWR_s_USACL.predict_next_ic(item_rel_vector, item_cat_vector, user_cat_rel_vector)
                #
                #     sml_rec_t = RWR_s_USACL.predict_next_ic(item_rel_vector, item_cat_vector, cat_rel_vector, user_cat_rel_vector)
                # else:
                #     sm_rec_t = pop_rec
                #     sl_rec_t = pop_rec
                #     sml_rec_t = pop_rec




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

                methods = [pop_rec, s_rec_s, sm_rec_s, sl_rec_s, sml_rec_s]

                if any(len(m) == 0 for m in methods):
                    continue

                n_recommendation[tw_i] += 1


                # To see in detail which recommendations we build for which sessions
                # if _dump_process == True:
                #     print('User:', user, 'Session:', s, 'Session articles:', articles[:i],
                #           'POP Rec:', pop_rec, 'SB Rec', sb_rec, 'Truth:', articles[i])

                # ------- Measuring accuracy ----------------------
                ae.evaluate_recommendation(rec=pop_rec, truth=articles[i], method='POP')
                # ae.evaluate_recommendation(rec=pop_last_cat_rec, truth=articles[i], method='POP (lc)')
                # ae.evaluate_recommendation(rec=pop_mp_m_cat_rec, truth=articles[i], method='POP+CC (m)')
                # ae.evaluate_recommendation(rec=pop_mp_s_cat_rec, truth=articles[i], method='POP+CC (s)')
                # ae.evaluate_recommendation(rec=pop_mp_t_cat_rec, truth=articles[i], method='POP+CC (t)')

                # ae.evaluate_recommendation(rec=s_rec_m, truth=articles[i], method='AA (m)')
                # ae.evaluate_recommendation(rec=s_rec_s, truth=articles[i], method='AA (s)')
                # ae.evaluate_recommendation(rec=s_rec_t, truth=articles[i], method='AA (t)')

                # ae.evaluate_recommendation(rec=sm_rec_m, truth=articles[i], method='AA+CC (m)')
                # ae.evaluate_recommendation(rec=sm_rec_s, truth=articles[i], method='AA+CC (s)')
                # ae.evaluate_recommendation(rec=sm_rec_t, truth=articles[i], method='AA+CC (t)')

                # ae.evaluate_recommendation(rec=sl_rec_m, truth=articles[i], method='AA+UC (m)')
                # ae.evaluate_recommendation(rec=sl_rec_s, truth=articles[i], method='AA+UC (s)')
                # ae.evaluate_recommendation(rec=sl_rec_t, truth=articles[i], method='AA+UC (t)')

                # ae.evaluate_recommendation(rec=sml_rec_m, truth=articles[i], method='AA+CC+UC (m)')
                # ae.evaluate_recommendation(rec=sml_rec_s, truth=articles[i], method='AA+CC+UC (s)')
                # ae.evaluate_recommendation(rec=sml_rec_t, truth=articles[i], method='AA+CC+UC (t)')


                ae.evaluate_recommendation(rec=s_rec_s, truth=articles[i], method='S')
                ae.evaluate_recommendation(rec=sm_rec_s, truth=articles[i], method='SM')
                ae.evaluate_recommendation(rec=sl_rec_s, truth=articles[i], method='SL')
                ae.evaluate_recommendation(rec=sml_rec_s, truth=articles[i], method='SML')

            ae.evaluate_session()

    ae.evaluate_tw()
    # print('- Number of recommendations made:', n_recommendations)

ae.evaluate_total_performance()

print('\n---------- METHODS EVALUATION -------------')

print('# of recommendations per time split:', n_recommendation.values())

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

pickle.dump(n_recommendation, open(results_dir + "N_recommendations.pickle", "wb"))

exit()


# ----- PLOT PRECISION
plot_suptile = 'Methods precision comparison'
plot_title = 'Data: ' + pickle_name + '\nShort term: ' + str(SHORT_DAYS) + ' days\nMedium term: ' + str(MEDIUM_DAYS) + ' days'\
             # + \
             # '\n\nPOP: ' + str(round(ae.precision['pop'], 3)) + \
             # '\nPOP + Medium CC: ' + str(round(ae.precision['pop + cc'], 3)) + \
             # '\nShort RWR_SAC + POP: ' + str(round(ae.precision['short_scb_pprank + pop'], 3)) + \
             # '\nShort RWR_SAC + Medium CC: ' + str(round(ae.precision['short_scb_pprank + cc'], 3))

annotation = 'POP: ' + str(round(ae.precision['pop'], 3)) + \
             '\nPOP + Medium CC: ' + str(round(ae.precision['pop + cc'], 3)) + \
             '\nShort RWR_SAC + POP: ' + str(round(ae.precision['short_scb_pprank + pop'], 3)) + \
             '\nShort RWR_SAC + Medium CC: ' + str(round(ae.precision['short_scb_pprank + cc'], 3))


plt.figure(figsize=(17,7))
plt.plot(p, [v for v in ae.tw_precision['pop']], marker = 'o')
plt.plot(p, [v for v in ae.tw_precision['pop + cc']], marker = 'o')
#plt.plot(p, [v for v in ae.tw_precision['short_scb_pprank']], marker = 'o')
plt.plot(p, [v for v in ae.tw_precision['short_scb_pprank + pop']], marker = 'o')
plt.plot(p, [v for v in ae.tw_precision['short_scb_pprank + cc']], marker = 'o')
# short_sb_pprank ?
plt.suptitle(plot_suptile, fontsize=12)
plt.title(plot_title, fontsize=10, loc='left')
# plt.text(0, 0.22, annotation, backgroundcolor='white', multialignment='left')
plt.xlabel('Period')
plt.xticks(rotation=90)
plt.ylabel('% Precision')
plt.legend(['POP', 'POP + Medium CC', 'Short RWR_SAC + POP', 'Short RWR_SAC + Medium CC'], loc='upper right')
plt.grid()
plt.show()

# ------- OBSERVATIONS ---------

# For datafile "G_Video33_cat" (no filter on inactive users):
# Popularity score is very high (35%), Session-based score is quite low (9%)

# For datafile "G_Video33_cat_active_users" (only users with at least 2 sessions are kept):
# Popularity score is lower (27%), but Session-based score is higher (16%)

# Why? Possibly because one time users cannot provide any valuable information for modelling, they randomly press
# what is offered to them (= the most popular items) without a specific reading purpose.
# Users, that come more often and value the website, probably come with a specific purpose to find something,
# thus there are more similar items in their sessions.

# Problem: SB Rec usually builds no recommendation, because doesn't have currently analysed items in the pre-trained model.
# Possible solution: combine sb with pop - if sb cannot recommend - use pop. - Accuracy smaller than plain popularity.


# Problem with popularity - most probable, that the website already uses some basic algorithms to recommend things.
# Most basic and widely used algorithm is popularity. So when we recommend using popularity algorithm we basically
# model the same recommender that is used, so we just "adjust" our model to provide good results, instead of really
# caring about educating users and making them educated from different sides.


# plt.plot(avg_sessions_per_user.keys(), avg_sessions_per_user.values())
# plt.ylabel('Sessions per user')
# plt.show()
#
# plt.plot(avg_articles_per_session.keys(), avg_articles_per_session.values())
# plt.ylabel('Articles per session')
# plt.show()


# VERY LITTLE DIVERSITY IN TRAIN, WE JUST RECOMMEND EVERYTHING FROM THE TRAIN
# POP (basically - random) is both the most accurate and diversified :(

