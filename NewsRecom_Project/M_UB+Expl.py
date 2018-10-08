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

train_set_len = []
train_len_dict = defaultdict(list)
n_articles_train = []
n_recommendation = dict()
sessions_per_user_in_short_term = []
avg_ses_len = defaultdict(list)


import pickle
explainability = pickle.load(open('.\\Data\\Results\\Explainability\\SI_Explainability.pickle', 'rb'))
ae.explainability_matrix = explainability


for tw_i, tw_iter in enumerate(tas.time_window_graph_list):

    print('\n\n======= Time split', tw_i, '=======')

    n_recommendation[tw_i] = 0

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

        avg_ses_len[tw_i].append(len(articles))

        for i in range(MIN_ITEMS_N, len(articles)):

            # print('---------\nUser:', user, 'Session:', s, 'Articles:', articles[:i])

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


            train_set_len.append(len(gm.get_sessions(user_long_train_g)))
            train_len_dict[tw_i].append(len(gm.get_sessions(user_long_train_g)))
            n_articles_train.append(len(gm.get_articles(user_long_train_g)))
            ses_per_user = gm.get_sessions_per_user(user_long_train_g)
            sessions_per_user_in_short_term.append(Counter(ses_per_user))





            # -----------------------------------------------------
            # ------------------- Popularity ----------------------
            pop.compute_pop(short_train_g)

            # -----------------------------------------------------
            # ------------------- PathSim's -----------------------

            similar_users_uau = PathSim_UAU.get_similar_users(user, gm.get_users(short_train_g), threshold=0.5)
            similar_users_ucu = PathSim_UCU.get_similar_users(user, gm.get_users(short_train_g), threshold=0.5)
            similar_users_ulu = PathSim_ULU.get_similar_users(user, gm.get_users(short_train_g), threshold=0.5)

            # Basically, all in any way connected users are concerned here and no strength of connection is considered
            # Later possible to multiply by relevance to enhance more connected ones


            # ------- POP --------------------------
            pop_rec = pop.predict_next(user, articles[:i])

            # --------- PathSim ------------
            uaua_rec_dict = PathSim_UAU.predict_next_by_UB(similar_users_uau, articles[:i], short_train_g, topN=False)
            ucua_rec_dict = PathSim_UCU.predict_next_by_UB(similar_users_ucu, articles[:i], short_train_g, topN=False)
            ulua_rec_dict = PathSim_ULU.predict_next_by_UB(similar_users_ulu, articles[:i], short_train_g, topN=False)

            # print('\nRecommendation (UAUA):', {k: uaua_rec_dict[k] for k in list(uaua_rec_dict)[:N]})
            # print('Recommendation (UCUA):', {k: ucua_rec_dict[k] for k in list(ucua_rec_dict)[:N]})
            # print('Recommendation (ULUA):', {k: ulua_rec_dict[k] for k in list(ulua_rec_dict)[:N]})

            uaua_rec = list(uaua_rec_dict.keys())[:N]
            ucua_rec = list(ucua_rec_dict.keys())[:N]
            ulua_rec = list(ulua_rec_dict.keys())[:N]


            # Create a dataframe with all recommendations together
            # rec_articles = list(set(uaua_rec + ucua_rec + ulua_rec))
            rec_articles = list(set(list(uaua_rec_dict.keys()) + list(ucua_rec_dict.keys()) + list(ulua_rec_dict.keys())))
            rec_df = pd.DataFrame(index=rec_articles, columns=['UAUA', 'UCUA', 'ULUA'])

            for a in rec_df.index:
                rec_df.loc[a, 'UAUA'] = uaua_rec_dict[a] if a in list(uaua_rec_dict.keys()) else 0
                rec_df.loc[a, 'UCUA'] = ucua_rec_dict[a] if a in list(ucua_rec_dict.keys()) else 0
                rec_df.loc[a, 'ULUA'] = ulua_rec_dict[a] if a in list(ulua_rec_dict.keys()) else 0


            # --- Ranked by sort order
            sort_order = ['UAUA', 'UCUA', 'ULUA']

            ranked_by_order_rec_df = rec_df.sort_values(by=sort_order, ascending=False).head(N)
            # print('\n', ranked_by_order_rec_df)

            rec_ub_order = ranked_by_order_rec_df.index.tolist()

            # --- Ranked by absolute path count sum
            vote_sum_rec_df = rec_df.copy()
            vote_sum_rec_df['vote_sum'] = vote_sum_rec_df['UAUA'] + vote_sum_rec_df['UCUA'] + vote_sum_rec_df['ULUA']

            ranked_by_vote_sum_rec_df = vote_sum_rec_df.sort_values(by=['vote_sum'], ascending=False).head(N)
            ranked_by_vote_sum_rec_df = ranked_by_vote_sum_rec_df.drop(['vote_sum'], axis=1)
            # print('\n', ranked_by_vote_sum_rec_df)

            rec_ub_vote_sum = ranked_by_vote_sum_rec_df.index.tolist()

            # --- Ranked by relative path count significance
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
            ranked_by_vote_sum_rec_importance_df = rec_importance_df.sort_values(by=['vote_sum'], ascending=False).head(N)

            ranked_by_relative_importance_df = rec_df.ix[ranked_by_vote_sum_rec_importance_df.index]

            # print('\n', ranked_by_relative_importance_df)

            rec_ub_relative_vote_sum = ranked_by_relative_importance_df.index.tolist()

            # print('\nActually read next article:', articles[i])







            # Only measure accuracy if all predictions could be made (not relying on pop)
            # methods = [rwr_sa_rec2, rwr_usa_rec2, rwr_sac_rec2, rwr_sal_rec2, rwr_usacl_rec2]
            # methods = [uaua_rec, ucua_rec, ulua_rec, rec_ub_order, rec_ub_vote_sum, rec_ub_relative_vote_sum]
            #
            # for m in methods:
            #     if len(m) == 0:
            #         continue

            # n_recommendation[tw_i] += 1

            # ------- Measuring accuracy ----------------------
            expl_base = s
            # expl_base = user

            ae.evaluate_recommendation(rec=pop_rec, truth=articles[i], method='POP', s=expl_base)

            ae.evaluate_recommendation(rec=uaua_rec, truth=articles[i], method='UAUA', s=expl_base)
            ae.evaluate_recommendation(rec=ucua_rec, truth=articles[i], method='UCUA', s=expl_base)
            ae.evaluate_recommendation(rec=ulua_rec, truth=articles[i], method='ULUA', s=expl_base)
            ae.evaluate_recommendation(rec=rec_ub_order, truth=articles[i], method='Combined UB (RANK 1)', s=expl_base)
            ae.evaluate_recommendation(rec=rec_ub_vote_sum, truth=articles[i], method='Combined UB (RANK 2)', s=expl_base)
            ae.evaluate_recommendation(rec=rec_ub_relative_vote_sum, truth=articles[i], method='Combined UB (RANK 3)', s=expl_base)

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

