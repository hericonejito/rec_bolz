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
PathSim_ASA = PathSimRec(N)
PathSim_ACA = PathSimRec(N)
PathSim_ALA = PathSimRec(N)
PathSim_AUA = PathSimRec(N)


ae = AccuracyEvaluation(G)

import pickle
explainability = pickle.load(open('.\\Data\\Results\\Explainability\\SI_Explainability.pickle', 'rb'))
ae.explainability_matrix = explainability

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

    # ------ 1. Create a time-ordered list of user sessions
    test_sessions = sorted([(s, attr['datetime']) for s, attr in test_g.nodes(data=True) if attr['entity'] == 'S'], key=lambda x: x[1])

    # n_recommendations = 0

    # For each step a ranked list of N recommendations is created
    for (s, s_datetime) in test_sessions:

        user = [n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'US'][0]

        # n_recommendations += 1

        # ------------ Short and Medium Training Sets ---------

        short_train_g = tas.create_short_term_train_set(s_datetime, short_back_timedelta)
        if len(short_train_g) == 0:
            continue



        train_set_len.append(len(gm.get_sessions(short_train_g)))
        train_len_dict[tw_i].append(len(gm.get_sessions(short_train_g)))
        n_articles_train.append(len(gm.get_articles(short_train_g)))
        ses_per_user = gm.get_sessions_per_user(short_train_g)
        sessions_per_user_in_short_term.append(Counter(ses_per_user))

        # --- Create train graphs
        ua_train_g = gm.derive_adjacency_multigraph(short_train_g, 'U', 'A', 2)
        sa_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A'])
        ac_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['A', 'C'])
        al_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['A', 'L'])
        usa_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A'])


        # -----------------------------------------------------
        # ------------------- Popularity ----------------------
        pop.compute_pop(short_train_g)


        # -----------------------------------------------------
        # ------------------- PathSim's -----------------------
        PathSim_ASA.compute_similarity_matrix(sa_train_g, 'A', 'S', 1)
        PathSim_ACA.compute_similarity_matrix(ac_train_g, 'A', 'C', 1)
        PathSim_ALA.compute_similarity_matrix(al_train_g, 'A', 'L', 1)
        PathSim_AUA.compute_similarity_matrix(usa_train_g, 'A', 'U', 2)

        aua_avg_n_con_s = PathSim_AUA.get_avg_n_of_connected_sessions()
        asa_avg_n_con_s = PathSim_ASA.get_avg_n_of_connected_sessions()
        aca_avg_n_con_s = PathSim_ACA.get_avg_n_of_connected_sessions()
        ala_avg_n_con_s = PathSim_ALA.get_avg_n_of_connected_sessions()

        # print('\nAvg # of articles connected with a single article through AUA:',
        #       aua_avg_n_con_s)
        # print('\nAvg # of articles connected with a single article through ASA:',
        #       PathSim_ASA.get_avg_n_of_connected_sessions())
        # print('Avg # of articles connected with a single article through ACA:',
        #       PathSim_ACA.get_avg_n_of_connected_sessions())
        # print('Avg # of articles connected with a single article through ALA:',
        #       PathSim_ALA.get_avg_n_of_connected_sessions())

        # PathCount_ASA.compute_similarity_matrix_my(short_train_g, 'A', 'S', 1)
        # PathCount_ACA.compute_similarity_matrix_my(ac_train_g, 'A', 'C', 1)
        # PathCount_ALA.compute_similarity_matrix_my(al_train_g, 'A', 'L', 1)


        # -----------------------------------------------------
        articles = sorted([n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'SA'],
                         key=lambda x: test_g.get_edge_data(s, x)['reading_datetime'])

        avg_ses_len[tw_i].append(len(articles))

        for i in range(MIN_ITEMS_N, len(articles)):

            # print('---------\nSession:', s, 'Articles:', articles[:i])

            # ------- POP --------------------------
            pop_rec = pop.predict_next(user, articles[:i])

            # ---------- PathSim's Original -----------------------------

            aua_c_rec = PathSim_AUA.predict_next(user, articles[:i], method=0)
            asa_c_rec = PathSim_ASA.predict_next(user, articles[:i], method=0)
            aca_c_rec = PathSim_ACA.predict_next(user, articles[:i], method=0)
            ala_c_rec = PathSim_ALA.predict_next(user, articles[:i], method=0)

            aua_all_rec = PathSim_AUA.predict_next(user, articles[:i], method=1)
            asa_all_rec = PathSim_ASA.predict_next(user, articles[:i], method=1)
            aca_all_rec = PathSim_ACA.predict_next(user, articles[:i], method=1)
            ala_all_rec = PathSim_ALA.predict_next(user, articles[:i], method=1)



            # --------- PathSim's Explainable ------------
            # Item-based (only last article matters)
            ps_asa_rec_dict_ib = PathSim_ASA.predict_next_by_AB(articles[:i], option='ib', topN=False)
            ps_aca_rec_dict_ib = PathSim_ACA.predict_next_by_AB(articles[:i], option='ib', topN=False)
            ps_ala_rec_dict_ib = PathSim_ALA.predict_next_by_AB(articles[:i], option='ib', topN=False)
            ps_aua_rec_dict_ib = PathSim_AUA.predict_next_by_AB(articles[:i], option='ib', topN=False)
            #
            # # Session-based (all articles from the session matter)
            ps_asa_rec_dict_sb = PathSim_ASA.predict_next_by_AB(articles[:i], option='sb', topN=False)
            ps_aca_rec_dict_sb = PathSim_ACA.predict_next_by_AB(articles[:i], option='sb', topN=False)
            ps_ala_rec_dict_sb = PathSim_ALA.predict_next_by_AB(articles[:i], option='sb', topN=False)
            ps_aua_rec_dict_sb = PathSim_AUA.predict_next_by_AB(articles[:i], option='sb', topN=False)

            # print('\nASA articles (IB):', {k: ps_asa_rec_dict_ib[k] for k in list(ps_asa_rec_dict_ib)[:N]})
            # print('ASA articles (SB):', {k: ps_asa_rec_dict_sb[k] for k in list(ps_asa_rec_dict_sb)[:N]})
            # print('\nACA articles (IB):', {k: ps_aca_rec_dict_ib[k] for k in list(ps_aca_rec_dict_ib)[:N]})
            # print('ACA articles (SB):', {k: ps_aca_rec_dict_sb[k] for k in list(ps_aca_rec_dict_sb)[:N]})
            # print('\nALA articles (IB):', {k: ps_ala_rec_dict_ib[k] for k in list(ps_ala_rec_dict_ib)[:N]})
            # print('ALA articles (SB):', {k: ps_ala_rec_dict_sb[k] for k in list(ps_ala_rec_dict_sb)[:N]})
            # print('\nAUA articles (IB):', {k: ps_aua_rec_dict_ib[k] for k in list(ps_aua_rec_dict_ib)[:N]})
            # print('AUA articles (SB):', {k: ps_aua_rec_dict_sb[k] for k in list(ps_aua_rec_dict_sb)[:N]})

            ps_asa_rec_ib = list(ps_asa_rec_dict_ib.keys())[:N]
            ps_aca_rec_ib = list(ps_aca_rec_dict_ib.keys())[:N]
            ps_ala_rec_ib = list(ps_ala_rec_dict_ib.keys())[:N]
            ps_aua_rec_ib = list(ps_aua_rec_dict_ib.keys())[:N]

            ps_asa_rec_sb = list(ps_asa_rec_dict_sb.keys())[:N]
            ps_aca_rec_sb = list(ps_aca_rec_dict_sb.keys())[:N]
            ps_ala_rec_sb = list(ps_ala_rec_dict_sb.keys())[:N]
            ps_aua_rec_sb = list(ps_aua_rec_dict_sb.keys())[:N]



            # Create a dataframe with all recommendations together
            # Item-based
            # rec_ib_articles = list(set(ps_asa_rec_ib + ps_aca_rec_ib + ps_ala_rec_ib + ps_aua_rec_ib))
            rec_ib_articles = list(set(list(ps_asa_rec_dict_ib.keys()) + list(ps_aca_rec_dict_ib.keys()) + list(ps_ala_rec_dict_ib.keys()) + list(ps_aua_rec_dict_ib.keys())))
            rec_ib_df = pd.DataFrame(index=rec_ib_articles, columns=['ASA', 'ACA', 'ALA', 'AUA'])

            for a in rec_ib_df.index:
                rec_ib_df.loc[a, 'ASA'] = ps_asa_rec_dict_ib[a] if a in list(ps_asa_rec_dict_ib.keys()) else 0
                rec_ib_df.loc[a, 'ACA'] = ps_aca_rec_dict_ib[a] if a in list(ps_aca_rec_dict_ib.keys()) else 0
                rec_ib_df.loc[a, 'ALA'] = ps_ala_rec_dict_ib[a] if a in list(ps_ala_rec_dict_ib.keys()) else 0
                rec_ib_df.loc[a, 'AUA'] = ps_aua_rec_dict_ib[a] if a in list(ps_aua_rec_dict_ib.keys()) else 0

            sort_order = ['AUA', 'ASA', 'ACA', 'ALA']

            ranked_rec_ib_df = rec_ib_df.sort_values(by=sort_order, ascending=False).head(N)
            # print('\n', ranked_rec_ib_df)

            rec_ib = ranked_rec_ib_df.index.tolist()
            # rec_ib = rec_ib if len(rec_ib) != 0 else pop_rec

            # Session-based
            # rec_sb_articles = list(set(ps_asa_rec_sb + ps_aca_rec_sb + ps_ala_rec_sb + ps_aua_rec_sb))
            rec_sb_articles = list(set(list(ps_asa_rec_dict_sb.keys()) + list(ps_aca_rec_dict_sb.keys()) + list(ps_ala_rec_dict_sb.keys()) + list(ps_aua_rec_dict_sb.keys())))
            rec_sb_df = pd.DataFrame(index=rec_sb_articles, columns=['ASA', 'ACA', 'ALA', 'AUA'])

            for a in rec_sb_df.index:
                rec_sb_df.loc[a, 'ASA'] = ps_asa_rec_dict_sb[a] if a in list(ps_asa_rec_dict_sb.keys()) else 0
                rec_sb_df.loc[a, 'ACA'] = ps_aca_rec_dict_sb[a] if a in list(ps_aca_rec_dict_sb.keys()) else 0
                rec_sb_df.loc[a, 'ALA'] = ps_ala_rec_dict_sb[a] if a in list(ps_ala_rec_dict_sb.keys()) else 0
                rec_sb_df.loc[a, 'AUA'] = ps_aua_rec_dict_sb[a] if a in list(ps_aua_rec_dict_sb.keys()) else 0

            sort_order = ['AUA', 'ASA', 'ACA', 'ALA']

            ranked_rec_sb_df = rec_sb_df.sort_values(by=sort_order, ascending=False).head(N)
            # print('\n', ranked_rec_sb_df)

            rec_sb = ranked_rec_sb_df.index.tolist()
            # rec_sb = rec_sb if len(rec_sb) != 0 else pop_rec

            # ------------------------------------------
            # ------ Ranked by absolute sum

            # --- IB
            vote_sum_rec_ib_df = rec_ib_df.copy()
            vote_sum_rec_ib_df['vote_sum'] = vote_sum_rec_ib_df['ASA'] + vote_sum_rec_ib_df['AUA'] + vote_sum_rec_ib_df['ACA'] + vote_sum_rec_ib_df['ALA']

            ranked_by_vote_sum_rec_ib_df = vote_sum_rec_ib_df.sort_values(by=['vote_sum'], ascending=False).head(N)
            ranked_by_vote_sum_rec_ib_df = ranked_by_vote_sum_rec_ib_df.drop(['vote_sum'], axis=1)
            # print('\n', ranked_by_vote_sum_rec_ib_df)

            rec_ib_vote_sum = ranked_by_vote_sum_rec_ib_df.index.tolist()
            # rec_ib_vote_sum = rec_ib_vote_sum if len(rec_ib_vote_sum) != 0 else pop_rec

            # --- SB
            vote_sum_rec_sb_df = rec_sb_df.copy()
            vote_sum_rec_sb_df['vote_sum'] = vote_sum_rec_sb_df['ASA'] + vote_sum_rec_sb_df['AUA'] + vote_sum_rec_sb_df[
                'ACA'] + vote_sum_rec_sb_df['ALA']

            ranked_by_vote_sum_rec_sb_df = vote_sum_rec_sb_df.sort_values(by=['vote_sum'], ascending=False).head(N)
            ranked_by_vote_sum_rec_sb_df = ranked_by_vote_sum_rec_sb_df.drop(['vote_sum'], axis=1)
            # print('\n', ranked_by_vote_sum_rec_sb_df)

            rec_sb_vote_sum = ranked_by_vote_sum_rec_sb_df.index.tolist()
            # rec_sb_vote_sum = rec_sb_vote_sum if len(rec_sb_vote_sum) != 0 else pop_rec

            # ------------------------------------------
            # ------ Ranked by connection strength

            # --- IB
            # Divide dataframe values by average number of sessions that a single session is connected with through a given path
            rec_importance_ib_df = rec_ib_df.copy()
            for a in rec_importance_ib_df.index:
                rec_importance_ib_df.loc[a, 'AUA'] = round(
                    rec_importance_ib_df.loc[a, 'AUA'] / aua_avg_n_con_s, 2)
                rec_importance_ib_df.loc[a, 'ASA'] = round(
                    rec_importance_ib_df.loc[a, 'ASA'] / asa_avg_n_con_s, 2)
                rec_importance_ib_df.loc[a, 'ACA'] = round(
                    rec_importance_ib_df.loc[a, 'ACA'] / aca_avg_n_con_s, 2)
                rec_importance_ib_df.loc[a, 'ALA'] = round(
                    rec_importance_ib_df.loc[a, 'ALA'] / ala_avg_n_con_s, 2)

            rec_importance_ib_df['vote_sum'] = rec_importance_ib_df['AUA'] + rec_importance_ib_df['ASA'] + rec_importance_ib_df[
                'ACA'] + rec_importance_ib_df['ALA']
            ranked_by_vote_sum_rec_importance_ib_df = rec_importance_ib_df.sort_values(by=['vote_sum'], ascending=False).head(
                N)

            ranked_by_relative_importance_ib_df = rec_ib_df.ix[ranked_by_vote_sum_rec_importance_ib_df.index]

            # print('\n', ranked_by_relative_importance_ib_df)

            rec_ib_con_strength = ranked_by_relative_importance_ib_df.index.tolist()
            # rec_ib_con_strength = rec_ib_con_strength if len(rec_ib_con_strength) != 0 else pop_rec

            # --- SB
            # Divide dataframe values by average number of sessions that a single session is connected with through a given path
            rec_importance_sb_df = rec_sb_df.copy()
            for a in rec_importance_sb_df.index:
                rec_importance_sb_df.loc[a, 'AUA'] = round(
                    rec_importance_sb_df.loc[a, 'AUA'] / aua_avg_n_con_s, 2)
                rec_importance_sb_df.loc[a, 'ASA'] = round(
                    rec_importance_sb_df.loc[a, 'ASA'] / asa_avg_n_con_s, 2)
                rec_importance_sb_df.loc[a, 'ACA'] = round(
                    rec_importance_sb_df.loc[a, 'ACA'] / aca_avg_n_con_s, 2)
                rec_importance_sb_df.loc[a, 'ALA'] = round(
                    rec_importance_sb_df.loc[a, 'ALA'] / asa_avg_n_con_s, 2)

            rec_importance_sb_df['vote_sum'] = rec_importance_sb_df['AUA'] + rec_importance_sb_df['ASA'] + \
                                               rec_importance_sb_df['ACA'] + rec_importance_sb_df['ALA']
            ranked_by_vote_sum_rec_importance_sb_df = rec_importance_sb_df.sort_values(by=['vote_sum'],
                                                                                       ascending=False).head(N)

            ranked_by_relative_importance_sb_df = rec_sb_df.ix[ranked_by_vote_sum_rec_importance_sb_df.index]

            # print('\n', ranked_by_relative_importance_sb_df)

            rec_sb_con_strength = ranked_by_relative_importance_sb_df.index.tolist()
            # rec_sb_con_strength = rec_sb_con_strength if len(rec_sb_con_strength) != 0 else pop_rec

            # I'M NOT SURE IN THIS STEP ANYMORE FOR OUR CURRENT LOGIC. MAYBE SHOULDN'T BE ALL RECOMMENDATIONS AVAILABLE?
            # OR FOR EACH METHOD MEASURE ACCURACY ONLY IF RECOMMENDATION OF THIS METHOD WAS MADE?
            # Only measure accuracy if all predictions could be made (not relying on pop)
            # methods = [ps_asa_rec_ib, ps_aca_rec_ib, ps_ala_rec_ib, ps_aua_rec_ib, rec_ib, rec_ib_vote_sum, rec_ib_con_strength,
            #            ps_asa_rec_sb, ps_aca_rec_sb, ps_ala_rec_sb, ps_aua_rec_sb, rec_sb, rec_sb_vote_sum, rec_sb_con_strength]
            #
            # for m in methods:
            #     if len(m) == 0:
            #         continue


            n_recommendation[tw_i] += 1

            # ------- Measuring accuracy ----------------------
            ae.evaluate_recommendation(rec=pop_rec, truth=articles[i], method='POP', s=s)

            ae.evaluate_recommendation(rec=asa_c_rec, truth=articles[i], method='PathSim_ASA (IB)', s=s)
            ae.evaluate_recommendation(rec=aca_c_rec, truth=articles[i], method='PathSim_ACA (IB)', s=s)
            ae.evaluate_recommendation(rec=ala_c_rec, truth=articles[i], method='PathSim_ALA (IB)', s=s)
            ae.evaluate_recommendation(rec=aua_c_rec, truth=articles[i], method='PathSim_AUA (IB)', s=s)

            ae.evaluate_recommendation(rec=asa_all_rec, truth=articles[i], method='PathSim_ASA (SB)', s=s)
            ae.evaluate_recommendation(rec=aca_all_rec, truth=articles[i], method='PathSim_ACA (SB)', s=s)
            ae.evaluate_recommendation(rec=ala_all_rec, truth=articles[i], method='PathSim_ALA (SB)', s=s)
            ae.evaluate_recommendation(rec=aua_all_rec, truth=articles[i], method='PathSim_AUA (SB)', s=s)

            ae.evaluate_recommendation(rec=ps_asa_rec_ib, truth=articles[i], method='ASA (IB)', s=s)
            ae.evaluate_recommendation(rec=ps_aca_rec_ib, truth=articles[i], method='ACA (IB)', s=s)
            ae.evaluate_recommendation(rec=ps_ala_rec_ib, truth=articles[i], method='ALA (IB)', s=s)
            ae.evaluate_recommendation(rec=ps_aua_rec_ib, truth=articles[i], method='AUA (IB)', s=s)
            ae.evaluate_recommendation(rec=rec_ib, truth=articles[i], method='Combined (IB) (RANK 1)', s=s)
            ae.evaluate_recommendation(rec=rec_ib_vote_sum, truth=articles[i], method='Combined (IB) (RANK 2)', s=s)
            ae.evaluate_recommendation(rec=rec_ib_con_strength, truth=articles[i], method='Combined (IB) (RANK 3)', s=s)

            ae.evaluate_recommendation(rec=ps_asa_rec_sb, truth=articles[i], method='ASA (SB)', s=s)
            ae.evaluate_recommendation(rec=ps_aca_rec_sb, truth=articles[i], method='ACA (SB)', s=s)
            ae.evaluate_recommendation(rec=ps_ala_rec_sb, truth=articles[i], method='ALA (SB)', s=s)
            ae.evaluate_recommendation(rec=ps_aua_rec_sb, truth=articles[i], method='AUA (SB)', s=s)
            ae.evaluate_recommendation(rec=rec_sb, truth=articles[i], method='Combined (SB) (RANK 1)', s=s)
            ae.evaluate_recommendation(rec=rec_sb_vote_sum, truth=articles[i], method='Combined (SB) (RANK 2)', s=s)
            ae.evaluate_recommendation(rec=rec_sb_con_strength, truth=articles[i], method='Combined (SB) (RANK 3)', s=s)

        ae.evaluate_session()

    ae.evaluate_tw()
    # print('- Number of recommendations made:', n_recommendations)

ae.evaluate_total_performance()


avg_n_ses_per_train_per_period = [round(np.mean(l)) for l in train_len_dict.values()]
avg_ses_len_per_period = [round(np.mean(l),2) for l in avg_ses_len.values()]

print('\n\n\nNumber of sessions per user per short train period:\n', sessions_per_user_in_short_term)
print('# of recommendations per time split:', n_recommendation.values())
print('Average # sessions per train per period', avg_n_ses_per_train_per_period)
print('Average # artiles per session per period', avg_ses_len_per_period)
print('Average # sessions in train:', round(np.mean(train_set_len), 2))
print('Average # articles in train:', round(np.mean(n_articles_train), 2))


print('\n---------- METHODS EVALUATION -------------')

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

pickle.dump(avg_n_ses_per_train_per_period, open(results_dir + "Avg_N_sessions_per_train.pickle", "wb"))
pickle.dump(n_recommendation, open(results_dir + "N_recommendations_per_test.pickle", "wb"))


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

