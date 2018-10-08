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

SHORT_DAYS = 2 #7
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
results_dir = '.\\Data\\Results\\RWR_vs_PathSim' + pickle_name + '_' + str(SHORT_DAYS) + '_' + str(MEDIUM_DAYS) + '\\'

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

# ----- PathSim -----
PathSim_AUA = PathSimRec(N)
PathSim_ASA = PathSimRec(N)
PathSim_ACA = PathSimRec(N)
PathSim_ALA = PathSimRec(N)

# PathSim_SAS = PathSimRec(N)
# PathSim_SCS = PathSimRec(N)
# PathSim_SLS = PathSimRec(N)

# PathSim_UAU = PathSimRec(N)
# PathSim_UCU = PathSimRec(N)
# PathSim_ULU = PathSimRec(N)

PathCount_AUA = PathSimRec(N)
PathCount_ASA = PathSimRec(N)
PathCount_ACA = PathSimRec(N)
PathCount_ALA = PathSimRec(N)


ae = AccuracyEvaluation(G)

n_recommendation = dict()

# methods = ['PathSim_s_ASA', 'PathSim_s_ACA', 'PathSim_s_ALA', 'PathSim_s_ASUSA', 'PathSim_s_ASASA']
# methods = ['PathSim_AUA', 'PathSim_ASA', 'PathSim_ACA', 'PathSim_ALA',
#            'PathSim_SASA', 'PathSim_SCSA', 'PathSim_SLSA']
# n_rec_m = dict()
# for method in methods:
#     n_rec_m[method] = 0
n_rec = 0

for tw_i, tw_iter in enumerate(tas.time_window_graph_list):

    print('\n\n======= Time split', tw_i, '=======')

    n_recommendation[tw_i] = 0

    long_train_g = tw_iter[0]
    test_g = tw_iter[1]

    # ------ From test_g remove sessions with less or equal number of articles needed for building recommendation
    # test_g = gm.filter_sessions(test_g, n_articles=MIN_ITEMS_N)

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
        # ------------------ PathSim --------------------------

        PathSim_AUA.compute_similarity_matrix(short_train_g, 'A', 'U', 2)
        PathSim_ASA.compute_similarity_matrix(short_train_g, 'A', 'S', 1)
        PathSim_ACA.compute_similarity_matrix(short_train_g, 'A', 'C', 1)
        PathSim_ALA.compute_similarity_matrix(short_train_g, 'A', 'L', 1)

        # PathSim_SAS.compute_similarity_matrix(short_train_g, 'S', 'A', 1)
        # PathSim_SCS.compute_similarity_matrix(short_train_g, 'S', 'C', 2)
        # PathSim_SLS.compute_similarity_matrix(short_train_g, 'S', 'L', 2)

        PathCount_AUA.compute_similarity_matrix_my(short_train_g, 'A', 'U', 2)
        PathCount_ASA.compute_similarity_matrix_my(short_train_g, 'A', 'S', 1)
        PathCount_ACA.compute_similarity_matrix_my(short_train_g, 'A', 'C', 1)
        PathCount_ALA.compute_similarity_matrix_my(short_train_g, 'A', 'L', 1)




        # -----------------------------------------------------------------------------------------------------------
        # ------------- Building Recommendations --------------------------------------------------------------------
        # -----------------------------------------------------------------------------------------------------------

        user = [n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'US'][0]
        articles = sorted([n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'SA'],
                         key=lambda x: test_g.get_edge_data(s, x)['reading_datetime'])

        for i in range(MIN_ITEMS_N, len(articles)):

            n_rec += 1
            session_timeviews = [gm.map_timeview(test_g, s, a) for a in articles[:i]]

            # ------------ POP -------------------------------
            pop_rec = pop.predict_next(user, articles[:i])

            # ---------- PathSim -----------------------------

            ps_aua_la_rec = PathSim_AUA.predict_next(user, articles[:i], method=0)
            if len(ps_aua_la_rec) == 0:
                continue
            ps_asa_la_rec = PathSim_ASA.predict_next(user, articles[:i], method=0)
            if len(ps_asa_la_rec) == 0:
                continue
            ps_aca_la_rec = PathSim_ACA.predict_next(user, articles[:i], method=0)
            if len(ps_aca_la_rec) == 0:
                continue
            ps_ala_la_rec = PathSim_ALA.predict_next(user, articles[:i], method=0)
            if len(ps_ala_la_rec) == 0:
                continue

            ps_aua_m_rec = PathSim_AUA.predict_next(user, articles[:i], method=1)
            if len(ps_aua_m_rec) == 0:
                continue
            ps_asa_m_rec = PathSim_ASA.predict_next(user, articles[:i], method=1)
            if len(ps_asa_m_rec) == 0:
                continue
            ps_aca_m_rec = PathSim_ACA.predict_next(user, articles[:i], method=1)
            if len(ps_aca_m_rec) == 0:
                continue
            ps_ala_m_rec = PathSim_ALA.predict_next(user, articles[:i], method=1)
            if len(ps_ala_m_rec) == 0:
                continue

            ps_aua_s_rec = PathSim_AUA.predict_next(user, articles[:i], method=2)
            if len(ps_aua_s_rec) == 0:
                continue
            ps_asa_s_rec = PathSim_ASA.predict_next(user, articles[:i], method=2)
            if len(ps_asa_s_rec) == 0:
                continue
            ps_aca_s_rec = PathSim_ACA.predict_next(user, articles[:i], method=2)
            if len(ps_aca_s_rec) == 0:
                continue
            ps_ala_s_rec = PathSim_ALA.predict_next(user, articles[:i], method=2)
            if len(ps_ala_s_rec) == 0:
                continue

            ps_aua_t_rec = PathSim_AUA.predict_next(user, articles[:i], method=3, timeviews=session_timeviews)
            if len(ps_aua_t_rec) == 0:
                continue
            ps_asa_t_rec = PathSim_ASA.predict_next(user, articles[:i], method=3, timeviews=session_timeviews)
            if len(ps_asa_t_rec) == 0:
                continue
            ps_aca_t_rec = PathSim_ACA.predict_next(user, articles[:i], method=3, timeviews=session_timeviews)
            if len(ps_aca_t_rec) == 0:
                continue
            ps_ala_t_rec = PathSim_ALA.predict_next(user, articles[:i], method=3, timeviews=session_timeviews)
            if len(ps_ala_t_rec) == 0:
                continue


            # -------------- PathCount ------------

            pc_aua_la_rec = PathCount_AUA.predict_next(user, articles[:i], method=0)
            if len(pc_aua_la_rec) == 0:
                continue
            pc_asa_la_rec = PathCount_ASA.predict_next(user, articles[:i], method=0)
            if len(pc_asa_la_rec) == 0:
                continue
            pc_aca_la_rec = PathCount_ACA.predict_next(user, articles[:i], method=0)
            if len(pc_aca_la_rec) == 0:
                continue
            pc_ala_la_rec = PathCount_ALA.predict_next(user, articles[:i], method=0)
            if len(pc_ala_la_rec) == 0:
                continue

            pc_aua_m_rec = PathCount_AUA.predict_next(user, articles[:i], method=1)
            if len(pc_aua_m_rec) == 0:
                continue
            pc_asa_m_rec = PathCount_ASA.predict_next(user, articles[:i], method=1)
            if len(pc_asa_m_rec) == 0:
                continue
            pc_aca_m_rec = PathCount_ACA.predict_next(user, articles[:i], method=1)
            if len(pc_aca_m_rec) == 0:
                continue
            pc_ala_m_rec = PathCount_ALA.predict_next(user, articles[:i], method=1)
            if len(pc_ala_m_rec) == 0:
                continue

            pc_aua_s_rec = PathCount_AUA.predict_next(user, articles[:i], method=2)
            if len(pc_aua_s_rec) == 0:
                continue
            pc_asa_s_rec = PathCount_ASA.predict_next(user, articles[:i], method=2)
            if len(pc_asa_s_rec) == 0:
                continue
            pc_aca_s_rec = PathCount_ACA.predict_next(user, articles[:i], method=2)
            if len(pc_aca_s_rec) == 0:
                continue
            pc_ala_s_rec = PathCount_ALA.predict_next(user, articles[:i], method=2)
            if len(pc_ala_s_rec) == 0:
                continue

            pc_aua_t_rec = PathCount_AUA.predict_next(user, articles[:i], method=3, timeviews=session_timeviews)
            if len(pc_aua_t_rec) == 0:
                continue
            pc_asa_t_rec = PathCount_ASA.predict_next(user, articles[:i], method=3, timeviews=session_timeviews)
            if len(pc_asa_t_rec) == 0:
                continue
            pc_aca_t_rec = PathCount_ACA.predict_next(user, articles[:i], method=3, timeviews=session_timeviews)
            if len(pc_aca_t_rec) == 0:
                continue
            pc_ala_t_rec = PathCount_ALA.predict_next(user, articles[:i], method=3, timeviews=session_timeviews)
            if len(pc_ala_t_rec) == 0:
                continue

            # ------ AB
            # Item-based (only last article matters)
            # ps_asa_rec_ib = list(PathSim_ASA.predict_next_by_AB(articles[:i], option='ib').keys())
            # ps_aca_rec_ib = list(PathSim_ACA.predict_next_by_AB(articles[:i], option='ib').keys())
            # ps_ala_rec_ib = list(PathSim_ALA.predict_next_by_AB(articles[:i], option='ib').keys())
            # ps_aua_rec_ib = list(PathSim_AUA.predict_next_by_AB(articles[:i], option='ib').keys())
            #
            # # Session-based (all articles from the session matter)
            # ps_asa_rec_sb = list(PathSim_ASA.predict_next_by_AB(articles[:i], option='sb').keys())
            # ps_aca_rec_sb = list(PathSim_ACA.predict_next_by_AB(articles[:i], option='sb').keys())
            # ps_ala_rec_sb = list(PathSim_ALA.predict_next_by_AB(articles[:i], option='sb').keys())
            # ps_aua_rec_sb = list(PathSim_AUA.predict_next_by_AB(articles[:i], option='sb').keys())



            # sasa_rec = PathSim_SAS.predict_next_by_sessionKNN(user, s, articles[:i], kNN=5)
            # if len(sasa_rec) == 0:
            #     continue
            # scsa_rec = PathSim_SCS.predict_next_by_sessionKNN(user, s, articles[:i], kNN=5)
            # if len(scsa_rec) == 0:
            #     continue
            # slsa_rec = PathSim_SLS.predict_next_by_sessionKNN(user, s, articles[:i], kNN=5)
            # if len(slsa_rec) == 0:
            #     continue



            # methods = [aua_c_rec, asa_c_rec, aca_c_rec, ala_c_rec,
            #            aua_all_rec, asa_all_rec, aca_all_rec, ala_all_rec]
            # methods = [aua_t_rec, asa_t_rec, aca_t_rec, ala_t_rec]
            # methods = [aua_s_rec, asa_s_rec, aca_s_rec, ala_s_rec,
            #            aua_pc_rec, asa_pc_rec, aca_pc_rec, ala_pc_rec]

            # if any(len(m) == 0 for m in methods):
            #     continue

            n_recommendation[tw_i] += 1

            # -------------------------------------------------
            # ------- Measuring accuracy ----------------------

            ae.evaluate_recommendation(rec=pop_rec, truth=articles[i], method='POP')

            # ------ PathSim

            ae.evaluate_recommendation(rec=ps_aua_la_rec, truth=articles[i], method='PathSim_AUA (LA)')
            ae.evaluate_recommendation(rec=ps_asa_la_rec, truth=articles[i], method='PathSim_ASA (LA)')
            ae.evaluate_recommendation(rec=ps_aca_la_rec, truth=articles[i], method='PathSim_ACA (LA)')
            ae.evaluate_recommendation(rec=ps_ala_la_rec, truth=articles[i], method='PathSim_ALA (LA)')

            ae.evaluate_recommendation(rec=ps_aua_m_rec, truth=articles[i], method='PathSim_AUA (M)')
            ae.evaluate_recommendation(rec=ps_asa_m_rec, truth=articles[i], method='PathSim_ASA (M)')
            ae.evaluate_recommendation(rec=ps_aca_m_rec, truth=articles[i], method='PathSim_ACA (M)')
            ae.evaluate_recommendation(rec=ps_ala_m_rec, truth=articles[i], method='PathSim_ALA (M)')

            ae.evaluate_recommendation(rec=ps_aua_s_rec, truth=articles[i], method='PathSim_AUA (S)')
            ae.evaluate_recommendation(rec=ps_asa_s_rec, truth=articles[i], method='PathSim_ASA (S)')
            ae.evaluate_recommendation(rec=ps_aca_s_rec, truth=articles[i], method='PathSim_ACA (S)')
            ae.evaluate_recommendation(rec=ps_ala_s_rec, truth=articles[i], method='PathSim_ALA (S)')

            ae.evaluate_recommendation(rec=ps_aua_t_rec, truth=articles[i], method='PathSim_AUA (T)')
            ae.evaluate_recommendation(rec=ps_asa_t_rec, truth=articles[i], method='PathSim_ASA (T)')
            ae.evaluate_recommendation(rec=ps_aca_t_rec, truth=articles[i], method='PathSim_ACA (T)')
            ae.evaluate_recommendation(rec=ps_ala_t_rec, truth=articles[i], method='PathSim_ALA (T)')

            # ------ PathCount

            ae.evaluate_recommendation(rec=pc_aua_la_rec, truth=articles[i], method='PathCount_AUA (LA)')
            ae.evaluate_recommendation(rec=pc_asa_la_rec, truth=articles[i], method='PathCount_ASA (LA)')
            ae.evaluate_recommendation(rec=pc_aca_la_rec, truth=articles[i], method='PathCount_ACA (LA)')
            ae.evaluate_recommendation(rec=pc_ala_la_rec, truth=articles[i], method='PathCount_ALA (LA)')

            ae.evaluate_recommendation(rec=pc_aua_m_rec, truth=articles[i], method='PathCount_AUA (M)')
            ae.evaluate_recommendation(rec=pc_asa_m_rec, truth=articles[i], method='PathCount_ASA (M)')
            ae.evaluate_recommendation(rec=pc_aca_m_rec, truth=articles[i], method='PathCount_ACA (M)')
            ae.evaluate_recommendation(rec=pc_ala_m_rec, truth=articles[i], method='PathCount_ALA (M)')

            ae.evaluate_recommendation(rec=pc_aua_s_rec, truth=articles[i], method='PathCount_AUA (S)')
            ae.evaluate_recommendation(rec=pc_asa_s_rec, truth=articles[i], method='PathCount_ASA (S)')
            ae.evaluate_recommendation(rec=pc_aca_s_rec, truth=articles[i], method='PathCount_ACA (S)')
            ae.evaluate_recommendation(rec=pc_ala_s_rec, truth=articles[i], method='PathCount_ALA (S)')

            ae.evaluate_recommendation(rec=pc_aua_t_rec, truth=articles[i], method='PathCount_AUA (T)')
            ae.evaluate_recommendation(rec=pc_asa_t_rec, truth=articles[i], method='PathCount_ASA (T)')
            ae.evaluate_recommendation(rec=pc_aca_t_rec, truth=articles[i], method='PathCount_ACA (T)')
            ae.evaluate_recommendation(rec=pc_ala_t_rec, truth=articles[i], method='PathCount_ALA (T)')


            # ae.evaluate_recommendation(rec=ps_aua_rec_ib, truth=articles[i], method='AB_AUA (Current)')
            # ae.evaluate_recommendation(rec=ps_asa_rec_ib, truth=articles[i], method='AB_ASA (Current)')
            # ae.evaluate_recommendation(rec=ps_aca_rec_ib, truth=articles[i], method='AB_ACA (Current)')
            # ae.evaluate_recommendation(rec=ps_ala_rec_ib, truth=articles[i], method='AB_ALA (Current)')
            #
            # ae.evaluate_recommendation(rec=ps_aua_rec_sb, truth=articles[i], method='AB_AUA (All)')
            # ae.evaluate_recommendation(rec=ps_asa_rec_sb, truth=articles[i], method='AB_ASA (All)')
            # ae.evaluate_recommendation(rec=ps_aca_rec_sb, truth=articles[i], method='AB_ACA (All)')
            # ae.evaluate_recommendation(rec=ps_ala_rec_sb, truth=articles[i], method='AB_ALA (All)')

        ae.evaluate_session()

    ae.evaluate_tw()

ae.evaluate_total_performance()


print('\n ---------- METHODS EVALUATION -------------\n')

print('# of recommendations per time split:', n_recommendation.values())
print('Total # of recs:', sum(n_recommendation.values()))

methods = [k for k, v in sorted(ae.precision.items(), key=itemgetter(1), reverse=True)]
for m in methods:
    print('---', m, ': Precision:', ae.precision[m], 'NDCG:', ae.ndcg[m], 'ILD:', ae.diversity[m])

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
