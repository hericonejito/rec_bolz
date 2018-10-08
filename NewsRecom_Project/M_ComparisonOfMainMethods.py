import numpy as np
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict
from collections import Counter
from operator import itemgetter
import pickle

from data_import import *
from graph_manipulation import *
from time_aware_splits import *
from popularity_based_rec import *
from personalized_prank import *
from pathsim import *
from simrank import *
from accuracy_evaluation import *


# ------ Data paths -------------------------------------
DATA_PATH = '.\\Data\\Video33 - pk_client, pk_session, pk_article, timeview (s), date, time.txt'
CAT_DATA_PATH = '.\\Data\\Video33-5topics-doc-topics.txt'
LOC_DATA_PATH = '.\\Data\\Video33 - pk_article, pk_district.txt'

# ------ Parameters -------------------------------------

NUM_SPLITS = 12 # In how many time intervals to split the dataset
WIN_SIZE = 1 # How many time intervals back are taken (not used anymore for prequential)
N = 5 # Number of recommendations to be provided for each user
# MIN_ITEMS_N: Minimum number of items to be revealed from test session before building a recommendation
MIN_ITEMS_N = 4 # Minimum number of items to be revealed from test session before building a recommendation
MIN_N_SESSIONS = 1 # If we want to filter active users - what should be their minimum number of sessions

SHORT_DAYS = 2
MEDIUM_DAYS = 14
LONG_DAYS = 30

kNN_RWR = 1
kNN_PathSim = 9

print('--------- PARAMETERS -----------')
print('NUM_SPLITS:', NUM_SPLITS)
print('WIN_SIZE:', WIN_SIZE)
print('N:', N)
print('MIN_ITEMS_N:', MIN_ITEMS_N)
print('MIN_SESSIONS_N:', MIN_N_SESSIONS)

print('\nSHORT_DAYS:', SHORT_DAYS)
print('MEDIUM_DAYS:', MEDIUM_DAYS)
print('LONG_DAYS:', LONG_DAYS)

print('\nkNN_RWR:', kNN_RWR)
print('kNN_PathSim:', kNN_PathSim)

# -------------------------------------------------------
# F/P:
# F - take data from data file, create a graph on the base of it, store into gpickle file;
# P - take previously prepared data from gpickle file
file_or_pickle = 'F'

# If file_or_pickle = 'F' specify a file name to store the created graph to.
# Else, specify a pickle file name where to take the graph from.

# -----------------------------------
# pickle_name = 'Articles_AllData'
pickle_name = 'SDF'
# pickle_name = 'SDF_Articles_AllData_loc'
# pickle_name = 'Articles_LongSessions(2)'
# pickle_name = 'Articles_LongSessions(2)_loc'
# pickle_name = 'Articles_LongSessions(3)'
# pickle_name = 'Articles_LongSessions(4)'
#pickle_name = 'Articles_ActiveUsers(3)'
# ------------------------------------

# results_dir = '.\\Data\\Results\\' + pickle_name + '_' + str(SHORT_DAYS) + '_' + str(MEDIUM_DAYS) + '\\'
# results_dir = '.\\Data\\Results\\'+pickle_name+'_'+str(SHORT_DAYS)+'_RWR_PS_PC\\'
results_dir = '.\\Data\\Results\\Newest\\Comparison\\USA\\'

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
    G = nx.read_gpickle('./Data/Pickles/'+pickle_name+'.gpickle')
    gm.G = G

# -----------------------------------------------------------------------------
# ---------------- STATISTICS ABOUT THE DATASET -------------------------------
#
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

# exit()


# -----------------------------------------------------------
# ------ Split data to train and test -----------------------
tas = TimeAwareSplits(G)
tas.create_time_split_graphs(G, NUM_SPLITS)
tas.create_time_window_graphs(WIN_SIZE)

print('--------------------------\nTime span list:\n', tas.time_span_list)

# -----------------------------------------------------------
# ------ Prequential evaluation -----------------------------

_dump_process = True

short_back_timedelta = datetime.timedelta(days=SHORT_DAYS)
medium_back_timedelta = datetime.timedelta(days=MEDIUM_DAYS)
long_back_timedelta = datetime.timedelta(days=LONG_DAYS)

pop = PopularityBasedRec(G, N)

# ------ RWR --------

# RWR_UA = PersonalizedPageRankBasedRec(N)
RWR_SA = PersonalizedPageRankBasedRec(N)
# RWR_AC = PersonalizedPageRankBasedRec(N)
# RWR_AL = PersonalizedPageRankBasedRec(N)
# RWR_USA = PersonalizedPageRankBasedRec(N)
# RWR_UAC = PersonalizedPageRankBasedRec(N)
# RWR_UAL = PersonalizedPageRankBasedRec(N)
# RWR_SAC = PersonalizedPageRankBasedRec(N)
RWR_SAL = PersonalizedPageRankBasedRec(N)
# RWR_ACL = PersonalizedPageRankBasedRec(N)
# RWR_USAC = PersonalizedPageRankBasedRec(N)
# RWR_USAL = PersonalizedPageRankBasedRec(N)
# RWR_UACL = PersonalizedPageRankBasedRec(N)
# RWR_SACL = PersonalizedPageRankBasedRec(N)
RWR_USACL = PersonalizedPageRankBasedRec(N)

# ---- PathSim ------

PathSim_AUA = PathSimRec(N)
PathSim_ASA = PathSimRec(N)
# PathSim_ACA = PathSimRec(N)
# PathSim_ALA = PathSimRec(N)

# ---- PathCount ------

PathCount_AUA = PathSimRec(N)
PathCount_ASA = PathSimRec(N)
# PathCount_ACA = PathSimRec(N)
# PathCount_ALA = PathSimRec(N)

# ---- SimRank ------

SimRank_SAL = SimRankRec(N)


# ------ kNN --------
# SKNN_RWR_SA = PersonalizedPageRankBasedRec(N)
# SKNN_RWR_USA = PersonalizedPageRankBasedRec(N)
# SKNN_RWR_SAC = PersonalizedPageRankBasedRec(N)
SKNN_RWR_SAL = PersonalizedPageRankBasedRec(N)
# SKNN_RWR_USAC = PersonalizedPageRankBasedRec(N)
# SKNN_RWR_USAL = PersonalizedPageRankBasedRec(N)
SKNN_RWR_USACL = PersonalizedPageRankBasedRec(N)

SKNN_PathSim_SAS = PathSimRec(N)
# SKNN_PathSim_SACAS = PathSimRec(N)
# SKNN_PathSim_SALAS = PathSimRec(N)

SKNN_PathCount_SAS = PathSimRec(N)
# SKNN_PathCount_SACAS = PathSimRec(N)
# SKNN_PathCount_SALAS = PathSimRec(N)


# ------ Accurracy and Explainability ---------
ae = AccuracyEvaluation(G)

explainability = pickle.load(open('./Data/Results/Explainability/SI_Explainability.pickle', 'rb'))
ae.explainability_matrix = explainability

train_set_len = []
train_len_dict = defaultdict(list)
n_articles_train = []
n_recommendation = dict()
sessions_per_user_in_short_term = []
avg_ses_len = defaultdict(list)


# For each time window:
for tw_i, tw_iter in enumerate(tas.time_window_graph_list):

    print('\n\n======= Time split', tw_i, '=======')

    n_recommendation[tw_i] = 0

    # long_train_g = tw_iter[0]
    tw_iter[1].frozen = False
    test_g = tw_iter[1].copy()


    # ------ From test_g remove sessions with less or equal number of articles needed for building recommendation
    test_g = gm.filter_sessions(test_g, n_items=MIN_ITEMS_N)
    if len(test_g) == 0:
        continue

    # ------ 1. Create a time-ordered list of user sessions
    test_sessions = sorted([(s, attr['datetime']) for s, attr in test_g.nodes(data=True) if attr['entity'] == 'S'], key=lambda x: x[1])

    sessions_knn_dict = defaultdict(tuple)

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

        avg_ses_len[tw_i].append(len(articles))

        # print('----------\narticles:', articles)
        # print('session:', s, s_datetime)

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

            train_set_len.append(len(gm.get_sessions(short_train_g)))
            train_len_dict[tw_i].append(len(gm.get_sessions(short_train_g)))
            n_articles_train.append(len(gm.get_articles(short_train_g)))
            ses_per_user = gm.get_sessions_per_user(short_train_g)
            sessions_per_user_in_short_term.append(Counter(ses_per_user))

            # --- Create train graphs
            sa_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A'])
            # usa_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A'])
            # sac_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A', 'C'])
            sal_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A', 'L'])
            # usac_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A', 'C'])
            usal_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A', 'L'])



            # -------------------------------------------------------------------------------
            # --------------- SIMILARITIES --------------------------------------------------


            # -----------------------------------------------------
            # ------------------- Popularity ----------------------
            pop.compute_pop(short_train_g)

            # -----------------------------------------------------
            # ------------------- SimRank -------------------------

            SimRank_SAL.compute_similarity_matrix(sal_train_g, max_iter=10)

            # -----------------------------------------------------
            # ------------------- RWR -----------------------------
            # --- Run models
            RWR_SA.compute_transition_matrix(sa_train_g)
            # RWR_USA.compute_transition_matrix(usa_train_g)
            # RWR_SAC.compute_transition_matrix(sac_train_g)
            RWR_SAL.compute_transition_matrix(sal_train_g)
            # RWR_USAC.compute_transition_matrix(usac_train_g)
            # RWR_USAL.compute_transition_matrix(usal_train_g)
            RWR_USACL.compute_transition_matrix(short_train_g)

            # --- Extract SS matrices
            # RWR_SA.create_sessionsession_matrix()
            # RWR_SA.create_sessionitem_matrix()
            # RWR_SA.create_itemitem_matrix()
            # RWR_USA.create_sessionsession_matrix()
            # RWR_USA.create_sessionitem_matrix()
            # RWR_USA.create_itemitem_matrix()
            # RWR_SAC.create_sessionsession_matrix()
            # RWR_SAC.create_sessionitem_matrix()
            # RWR_SAC.create_itemitem_matrix()
            RWR_SAL.create_sessionsession_matrix()
            RWR_SAL.create_sessionitem_matrix()
            RWR_SAL.create_itemitem_matrix()
            # RWR_USAC.create_sessionsession_matrix()
            # RWR_USAC.create_sessionitem_matrix()
            # RWR_USAC.create_itemitem_matrix()
            # RWR_USAL.create_sessionsession_matrix()
            # RWR_USAL.create_sessionitem_matrix()
            # RWR_USAL.create_itemitem_matrix()
            RWR_USACL.create_sessionsession_matrix()
            RWR_USACL.create_sessionitem_matrix()
            RWR_USACL.create_itemitem_matrix()


            # -----------------------------------------------------
            # ------------------ PathSim --------------------------

            PathSim_AUA.compute_similarity_matrix(short_train_g, 'A', 'U', 2)
            PathSim_ASA.compute_similarity_matrix(short_train_g, 'A', 'S', 1)
            # PathSim_ACA.compute_similarity_matrix(short_train_g, 'A', 'C', 1)
            # PathSim_ALA.compute_similarity_matrix(short_train_g, 'A', 'L', 1)

            # -----------------------------------------------------
            # ------------------ PathCount --------------------------

            PathCount_AUA.compute_similarity_matrix_my(short_train_g, 'A', 'U', 2)
            PathCount_ASA.compute_similarity_matrix_my(short_train_g, 'A', 'S', 1)
            # PathCount_ACA.compute_similarity_matrix_my(short_train_g, 'A', 'C', 1)
            # PathCount_ALA.compute_similarity_matrix_my(short_train_g, 'A', 'L', 1)


            # -----------------------------------------------------
            # ------------------- S-S PathSim ---------------------
            SKNN_PathSim_SAS.compute_similarity_matrix(short_train_g, 'S', 'A', 1)
            # SKNN_PathSim_SACAS.compute_similarity_matrix(sac_train_g, 'S', 'C', 2)
            # SKNN_PathSim_SALAS.compute_similarity_matrix(sal_train_g, 'S', 'L', 2)


            # -----------------------------------------------------
            # ------------------- S-S PathCounts ------------------
            SKNN_PathCount_SAS.compute_similarity_matrix_my(short_train_g, 'S', 'A', 1)
            # SKNN_PathCount_SACAS.compute_similarity_matrix_my(sac_train_g, 'S', 'C', 2)
            # SKNN_PathCount_SALAS.compute_similarity_matrix_my(sal_train_g, 'S', 'L', 2)





            # -------------------------------------------------------------------------------
            # --------------- RECOMMENDATIONS -----------------------------------------------

            session_categories = [gm.map_category(a) for a in articles[:i]]
            session_timeviews = [gm.map_timeview(test_g, s, a) for a in articles[:i]]

            # ------- POP --------------------------

            pop_rec = pop.predict_next(user, articles[:i])
            if len(pop_rec) == 0:
                continue

            # ------- SimRank ----------------------

            simrank_sal_s_rec = SimRank_SAL.predict_next(user, articles[:i], method=2)
            if len(simrank_sal_s_rec) == 0:
                continue

            # ------- RWR --------------------------

            # rwr_ua_s_rec = RWR_UA.predict_next(user, articles[:i], method=2)
            # rwr_sa_s_rec = RWR_SA.predict_next(user, articles[:i], method=2)
            # if len(rwr_sa_s_rec) == 0:
            #     continue
            # rwr_ac_s_rec = RWR_AC.predict_next(user, articles[:i], method=2)
            # rwr_al_s_rec = RWR_AL.predict_next(user, articles[:i], method=2)
            #
            # rwr_usa_s_rec = RWR_USA.predict_next(user, articles[:i], method=2)
            # if len(rwr_usa_s_rec) == 0:
            #     continue
            # rwr_sac_s_rec = RWR_SAC.predict_next(user, articles[:i], method=2)
            rwr_sal_s_rec = RWR_SAL.predict_next(user, articles[:i], method=2)
            if len(rwr_sal_s_rec) == 0:
                continue
            # rwr_uac_s_rec = RWR_UAC.predict_next(user, articles[:i], method=2)
            # rwr_ual_s_rec = RWR_UAL.predict_next(user, articles[:i], method=2)
            # rwr_acl_s_rec = RWR_ACL.predict_next(user, articles[:i], method=2)
            #
            # rwr_usac_s_rec = RWR_USAC.predict_next(user, articles[:i], method=2)
            # rwr_usal_s_rec = RWR_USAL.predict_next(user, articles[:i], method=2)
            # rwr_sacl_s_rec = RWR_SACL.predict_next(user, articles[:i], method=2)
            # rwr_uacl_s_rec = RWR_UACL.predict_next(user, articles[:i], method=2)

            rwr_usacl_s_rec = RWR_USACL.predict_next(user, articles[:i], method=2)
            if len(rwr_usacl_s_rec) == 0:
                continue

            # ------- PathSim ----------------------

            pathsim_aua_s_rec = PathSim_AUA.predict_next(user, articles[:i], method=2)
            if len(pathsim_aua_s_rec) == 0:
                continue
            pathsim_asa_s_rec = PathSim_ASA.predict_next(user, articles[:i], method=2)
            if len(pathsim_asa_s_rec) == 0:
                continue

            # ------- PathCount --------------------

            pathcount_aua_s_rec = PathCount_AUA.predict_next(user, articles[:i], method=2)
            if len(pathcount_aua_s_rec) == 0:
                continue
            pathcount_asa_s_rec = PathCount_ASA.predict_next(user, articles[:i], method=2)
            if len(pathcount_asa_s_rec) == 0:
                continue

            # ------- Session-kNN ------------------

            # sknn_rwr_sa_rec = RWR_SA.predict_next_by_sessionKNN(s, kNN_RWR)
            # sknn_rwr_usa_rec = RWR_USA.predict_next_by_sessionKNN(s, kNN_RWR
            # sknn_rwr_sac_rec = RWR_SAC.predict_next_by_sessionKNN(s, kNN_RWR)
            sknn_rwr_sal_rec = RWR_SAL.predict_next_by_sessionKNN(s, kNN_RWR)
            if len(sknn_rwr_sal_rec) == 0:
                continue
            # sknn_rwr_usac_rec = RWR_USAC.predict_next_by_sessionKNN(s, kNN_RWR)
            # sknn_rwr_usal_rec = RWR_USAL.predict_next_by_sessionKNN(s, kNN_RWR)
            sknn_rwr_usacl_rec = RWR_USACL.predict_next_by_sessionKNN(s, kNN_RWR)
            if len(sknn_rwr_usacl_rec) == 0:
                continue

            sknn_ps_sas_rec = SKNN_PathSim_SAS.predict_next_by_sessionKNN(s, articles[:i], kNN_PathSim)
            if len(sknn_ps_sas_rec) == 0:
                continue
            # sknn_ps_sacas_rec = PathSim_SACAS.predict_next_by_sessionKNN(s, articles[:i], kNN_PathSim)
            # sknn_ps_salas_rec = PathSim_SALAS.predict_next_by_sessionKNN(s, articles[:i], kNN_PathSim)

            sknn_pc_sas_rec = SKNN_PathCount_SAS.predict_next_by_sessionKNN(s, articles[:i], kNN_PathSim)
            if len(sknn_pc_sas_rec) == 0:
                continue
            # sknn_pc_sacas_rec = SKNN_PathCount_SACAS.predict_next_by_sessionKNN(s, articles[:i], kNN_PathSim)
            # sknn_pc_salas_rec = SKNN_PathCount_SALAS.predict_next_by_sessionKNN(s, articles[:i], kNN_PathSim)






            # Only measure accuracy if ALL methods provide a recommendation
            methods = [pop_rec, simrank_sal_s_rec, rwr_sal_s_rec, pathsim_aua_s_rec, pathcount_aua_s_rec,
                       sknn_rwr_sal_rec, sknn_ps_sas_rec, sknn_pc_sas_rec]
            # methods = [pop_rec, rwr_sa_s_rec, rwr_usa_s_rec,
            #            pathsim_aua_s_rec, pathsim_asa_s_rec,
            #            pathcount_aua_s_rec, pathcount_asa_s_rec]

            if any(len(m)==0 for m in methods):
                continue

            n_recommendation[tw_i] += 1

            # ------- Measuring accuracy ----------------------
            ae.evaluate_recommendation(rec=pop_rec, truth=articles[i], method='POP', s=s)

            ae.evaluate_recommendation(rec=simrank_sal_s_rec, truth=articles[i], method='SimRank_SAL(s)', s=s)

            # ae.evaluate_recommendation(rec=rwr_sa_s_rec, truth=articles[i], method='RWR_SA(s)', s=s)
            # ae.evaluate_recommendation(rec=rwr_usa_s_rec, truth=articles[i], method='RWR_USA(s)', s=s)
            # ae.evaluate_recommendation(rec=rwr_sac_s_rec, truth=articles[i], method='RWR_SAC(s)', s=s)
            ae.evaluate_recommendation(rec=rwr_sal_s_rec, truth=articles[i], method='RWR_SAL(s)', s=s)
            # ae.evaluate_recommendation(rec=rwr_usac_s_rec, truth=articles[i], method='RWR_USAC(s)', s=s)
            # ae.evaluate_recommendation(rec=rwr_usal_s_rec, truth=articles[i], method='RWR_USAL(s)', s=s)
            ae.evaluate_recommendation(rec=rwr_usacl_s_rec, truth=articles[i], method='RWR_USACL(s)', s=s)

            ae.evaluate_recommendation(rec=pathsim_asa_s_rec, truth=articles[i], method='PathSim_ASA(s)', s=s)
            ae.evaluate_recommendation(rec=pathsim_aua_s_rec, truth=articles[i], method='PathSim_AUA(s)', s=s)
            ae.evaluate_recommendation(rec=pathcount_asa_s_rec, truth=articles[i], method='PathCount_ASA(s)', s=s)
            ae.evaluate_recommendation(rec=pathcount_aua_s_rec, truth=articles[i], method='PathCount_AUA(s)', s=s)

            # ae.evaluate_recommendation(rec=sknn_rwr_sa_rec, truth=articles[i], method='SKNN_RWR_SA', s=s)
            # ae.evaluate_recommendation(rec=sknn_rwr_usa_rec, truth=articles[i], method='SKNN_RWR_USA', s=s)
            # ae.evaluate_recommendation(rec=sknn_rwr_sac_rec, truth=articles[i], method='SKNN_RWR_SAC', s=s)
            ae.evaluate_recommendation(rec=sknn_rwr_sal_rec, truth=articles[i], method='SKNN_RWR_SAL', s=s)
            # ae.evaluate_recommendation(rec=sknn_rwr_usac_rec, truth=articles[i], method='SKNN_RWR_USAC', s=s)
            # ae.evaluate_recommendation(rec=sknn_rwr_usal_rec, truth=articles[i], method='SKNN_RWR_USAL', s=s)
            ae.evaluate_recommendation(rec=sknn_rwr_usacl_rec, truth=articles[i], method='SKNN_RWR_USACL', s=s)

            ae.evaluate_recommendation(rec=sknn_ps_sas_rec, truth=articles[i], method='SKNN_PathSim_SAS', s=s)
            # ae.evaluate_recommendation(rec=sknn_ps_sacas_rec, truth=articles[i], method='SKNN_PathSim_SACAS', s=s)
            # ae.evaluate_recommendation(rec=sknn_ps_salas_rec, truth=articles[i], method='SKNN_PathSim_SALAS', s=s)

            ae.evaluate_recommendation(rec=sknn_pc_sas_rec, truth=articles[i], method='SKNN_PathCount_SAS', s=s)
            # ae.evaluate_recommendation(rec=sknn_pc_sacas_rec, truth=articles[i], method='SKNN_PathCount_SACAS', s=s)
            # ae.evaluate_recommendation(rec=sknn_pc_salas_rec, truth=articles[i], method='SKNN_PathCount_SALAS', s=s)

        ae.evaluate_session()

    ae.evaluate_tw()
    # print('- Number of recommendations made:', n_recommendations)

ae.evaluate_total_performance()

# print(sessions_knn_dict)
# exit()


avg_n_ses_per_train_per_period = [round(np.mean(l)) for l in train_len_dict.values()]
avg_ses_len_per_period = [round(np.mean(l),2) for l in avg_ses_len.values()]

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
    print('---', m, ': Precision:', ae.precision[m], 'NDCG:', ae.ndcg[m], 'ILD:', ae.diversity[m], 'Explainability:', ae.explainability[m])

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

pickle.dump(avg_n_ses_per_train_per_period, open(results_dir + "Avg_N_sessions_per_train.pickle", "wb"))
pickle.dump(n_recommendation, open(results_dir + "N_recommendations_per_test.pickle", "wb"))