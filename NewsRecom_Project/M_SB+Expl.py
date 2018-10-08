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


# ----------- NORMALIZING EXPLAINABILITY MATRICES ------------------- #

# import pickle
# results_dir = '.\\Data\\Results\\Explainability\\'
# explainability = pickle.load(open(results_dir + 'SI_Explainability.pickle', 'rb'))
#
# values = []
# for u in explainability:
#     values.extend([explainability[u][i] for i in explainability[u]])
# min_expl = min(values)
# max_expl = max(values)
#
# norm_explainability = explainability.copy()
# for u in explainability:
#     for i in explainability[u]:
#         norm_explainability[u][i] = (explainability[u][i] - min_expl) / (max_expl - min_expl)
#
# pickle.dump(norm_explainability, open(results_dir + "SI_Norm_Explainability.pickle", "wb"))

# -------------------------------------------------------------------------------------------




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

RWR_USA = PersonalizedPageRankBasedRec(N)
RWR_SAC = PersonalizedPageRankBasedRec(N)
RWR_SAL = PersonalizedPageRankBasedRec(N)
RWR_USACL = PersonalizedPageRankBasedRec(N)


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
# print('\nSI NORM EXPLAINABILITY')

for tw_i, tw_iter in enumerate(tas.time_window_graph_list):

    # if tw_i == 1:
    #     break

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

        test_session_G = nx.Graph()
        test_session_G.add_node(user, entity='U')
        test_session_G.add_node(s, entity='S')
        test_session_G.add_edge(user, s, edge_type='US')


        # -----------------------------------------------------
        articles = sorted([n for n in nx.neighbors(test_g, s) if test_g.get_edge_data(s, n)['edge_type'] == 'SA'],
                          key=lambda x: test_g.get_edge_data(s, x)['reading_datetime'])

        avg_ses_len[tw_i].append(len(articles))

        for i in range(MIN_ITEMS_N, len(articles)):

            test_session_G.add_nodes_from(articles[:i], entity='A')
            for a in articles[:i]:
                test_session_G.add_edge(s, a, edge_type='SA')
                test_session_G.add_node(gm.map_category(a), entity='C')
                test_session_G.add_edge(a, gm.map_category(a), edge_type='AC')
                test_session_G.add_node(gm.map_location(a), entity='L')
                test_session_G.add_edge(a, gm.map_location(a), edge_type='AL')

            # print('---------\nSession:', s, 'Articles:', articles[:i])


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
            usa_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['U', 'S', 'A'])
            sac_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A', 'C'])
            sal_train_g = gm.create_subgraph_of_adjacent_entities(short_train_g, list_of_entities=['S', 'A', 'L'])


            # -----------------------------------------------------
            # ------------------- Popularity ----------------------
            pop.compute_pop(short_train_g)

            # -----------------------------------------------------
            # ------------------- Random Walks --------------------
            # RWR_USA.compute_transition_matrix(usa_train_g)
            # RWR_USA.create_itemitem_matrix()
            # rwr_usa_rec = RWR_USA.predict_next(user, articles[:i])
            #
            # RWR_SAC.compute_transition_matrix(sac_train_g)
            # RWR_SAC.create_itemitem_matrix()
            # rwr_sac_rec = RWR_SAC.predict_next(user, articles[:i])
            #
            # RWR_SAL.compute_transition_matrix(sal_train_g)
            # RWR_SAL.create_itemitem_matrix()
            # rwr_sal_rec = RWR_SAL.predict_next(user, articles[:i])
            #
            # RWR_USACL.compute_transition_matrix(short_train_g)
            # RWR_USACL.create_itemitem_matrix()
            # rwr_usacl_rec = RWR_USACL.predict_next(user, articles[:i])


            # -----------------------------------------------------
            # ------------------- Original PathSim's --------------



            # -----------------------------------------------------
            # ------------------- PathSim's -----------------------
            PathSim_SAS.compute_similarity_matrix(sa_train_g, 'S', 'A', 1)
            PathSim_SCS.compute_similarity_matrix(sac_train_g, 'S', 'C', 2)
            PathSim_SLS.compute_similarity_matrix(sal_train_g, 'S', 'L', 2)

            # print('\nAvg # of sessions connected with a single session through SAS:',
            #       PathSim_SAS.get_avg_n_of_connected_sessions())
            # print('Avg # of sessions connected with a single session through SCS:',
            #       PathSim_SCS.get_avg_n_of_connected_sessions())
            # print('Avg # of sessions connected with a single session through SLS:',
            #       PathSim_SLS.get_avg_n_of_connected_sessions())
            #
            #
            # print('\nSimilar sessions (SAS):', PathSim_SAS.get_similar_sessions(s)) # can be a threshold, like not >0, but >0.5
            # print('Similar sessions (SCS):', PathSim_SCS.get_similar_sessions(s))
            # print('Similar sessions (SLS):', PathSim_SLS.get_similar_sessions(s))


            # ------- POP --------------------------
            pop_rec = pop.predict_next(user, articles[:i])

            # --------- PathSim ------------
            ps_sasa_rec_dict = PathSim_SAS.predict_next_by_SB(s, articles[:i], topN=False)
            ps_scsa_rec_dict = PathSim_SCS.predict_next_by_SB(s, articles[:i], topN=False)
            ps_slsa_rec_dict = PathSim_SLS.predict_next_by_SB(s, articles[:i], topN=False)

            # print('\nRecommendation (SASA):', {k: ps_sasa_rec_dict[k] for k in list(ps_sasa_rec_dict)[:N]})
            # print('Recommendation (SCSA):', {k: ps_scsa_rec_dict[k] for k in list(ps_scsa_rec_dict)[:N]})
            # print('Recommendation (SLSA):', {k: ps_slsa_rec_dict[k] for k in list(ps_slsa_rec_dict)[:N]})

            ps_sasa_rec = list(ps_sasa_rec_dict.keys())[:N]
            ps_scsa_rec = list(ps_scsa_rec_dict.keys())[:N]
            ps_slsa_rec = list(ps_slsa_rec_dict.keys())[:N]


            # Create a dataframe with all recommendations together
            # rec_articles = list(set(ps_sasa_rec + ps_scsa_rec + ps_slsa_rec))
            rec_articles = list(set(list(ps_sasa_rec_dict.keys()) + list(ps_scsa_rec_dict.keys()) + list(ps_slsa_rec_dict.keys())))
            rec_df = pd.DataFrame(index=rec_articles, columns=['SASA', 'SCSA', 'SLSA'])

            for a in rec_df.index:
                rec_df.loc[a, 'SASA'] = ps_sasa_rec_dict[a] if a in list(ps_sasa_rec_dict.keys()) else 0
                rec_df.loc[a, 'SCSA'] = ps_scsa_rec_dict[a] if a in list(ps_scsa_rec_dict.keys()) else 0
                rec_df.loc[a, 'SLSA'] = ps_slsa_rec_dict[a] if a in list(ps_slsa_rec_dict.keys()) else 0

            # print('\n', rec_df)

            sort_order = ['SASA', 'SCSA', 'SLSA']

            ranked_by_order_rec_df = rec_df.sort_values(by=sort_order, ascending=False).head(N)
            # print('\n', ranked_by_order_rec_df)

            rec_sb_order = ranked_by_order_rec_df.index.tolist()


            vote_sum_rec_df = rec_df.copy()
            vote_sum_rec_df['vote_sum'] = vote_sum_rec_df['SASA'] + vote_sum_rec_df['SCSA'] + vote_sum_rec_df['SLSA']

            ranked_by_vote_sum_rec_df = vote_sum_rec_df.sort_values(by=['vote_sum'], ascending=False).head(N)
            ranked_by_vote_sum_rec_df = ranked_by_vote_sum_rec_df.drop(['vote_sum'], axis=1)
            # print('\n', ranked_by_vote_sum_rec_df)

            rec_sb_vote_sum = ranked_by_vote_sum_rec_df.index.tolist()



            # Divide dataframe values by average number of sessions that a single session is connected with through a given path
            rec_importance_df = rec_df.copy()
            for a in rec_importance_df.index:
                rec_importance_df.loc[a, 'SASA'] = round(
                    rec_importance_df.loc[a, 'SASA'] / PathSim_SAS.get_avg_n_of_connected_sessions(), 2)
                rec_importance_df.loc[a, 'SCSA'] = round(
                    rec_importance_df.loc[a, 'SCSA'] / PathSim_SCS.get_avg_n_of_connected_sessions(), 2)
                rec_importance_df.loc[a, 'SLSA'] = round(
                    rec_importance_df.loc[a, 'SLSA'] / PathSim_SLS.get_avg_n_of_connected_sessions(), 2)

            rec_importance_df['vote_sum'] = rec_importance_df['SASA'] + rec_importance_df['SCSA'] + rec_importance_df['SLSA']
            ranked_by_vote_sum_rec_importance_df = rec_importance_df.sort_values(by=['vote_sum'], ascending=False).head(N)

            ranked_by_relative_importance_df = rec_df.ix[ranked_by_vote_sum_rec_importance_df.index]

            # print('\n', ranked_by_relative_importance_df)

            rec_sb_relative_vote_sum = ranked_by_relative_importance_df.index.tolist()



            # print('\nActually read next article:', articles[i])






            # # Only measure accuracy if all predictions could be made (not relying on pop)
            # # methods = [rwr_sa_rec2, rwr_usa_rec2, rwr_sac_rec2, rwr_sal_rec2, rwr_usacl_rec2]
            # methods = [ps_sasa_rec, ps_scsa_rec, ps_slsa_rec, rec_sb_order, rec_sb_vote_sum]
            #
            # for m in methods:
            #     if len(m) == 0:
            #         continue

            # methods = [pop_rec,
            #            ps_sasa_rec, ps_scsa_rec, ps_slsa_rec,
            #            rec_sb_order, rec_sb_vote_sum, rec_sb_relative_vote_sum,
            #            rwr_usa_rec, rwr_sac_rec, rwr_sal_rec, rwr_usacl_rec]

            # if any([len(m)==0 for m in methods]):
            #     continue

            expl_base = s
            # expl_base = user

            # ------- Measuring accuracy ----------------------
            ae.evaluate_recommendation(rec=pop_rec, truth=articles[i], method='POP', s=expl_base)

            ae.evaluate_recommendation(rec=ps_sasa_rec, truth=articles[i], method='SASA', s=expl_base)
            ae.evaluate_recommendation(rec=ps_scsa_rec, truth=articles[i], method='SCSA', s=expl_base)
            ae.evaluate_recommendation(rec=ps_slsa_rec, truth=articles[i], method='SLSA', s=expl_base)
            ae.evaluate_recommendation(rec=rec_sb_order, truth=articles[i], method='Combined (RANK 1)', s=expl_base)
            ae.evaluate_recommendation(rec=rec_sb_vote_sum, truth=articles[i], method='Combined (RANK 2)', s=expl_base)
            ae.evaluate_recommendation(rec=rec_sb_relative_vote_sum, truth=articles[i], method='Combined (RANK 3)', s=expl_base)

            # ae.evaluate_recommendation(rec=rwr_usa_rec, truth=articles[i], method='RWR_USA', s=expl_base)
            # ae.evaluate_recommendation(rec=rwr_sac_rec, truth=articles[i], method='RWR_SAC', s=expl_base)
            # ae.evaluate_recommendation(rec=rwr_sal_rec, truth=articles[i], method='RWR_SAL', s=expl_base)
            # ae.evaluate_recommendation(rec=rwr_usacl_rec, truth=articles[i], method='RWR_USACL', s=expl_base)

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

