import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import seaborn as sns



def load_pickle(dir):

    p = pickle.load(open(dir + 'Periods.pickle', 'rb'))

    tw_precision = pickle.load(open(dir + 'TW_Precision.pickle', 'rb'))
    tw_ndcg = pickle.load(open(dir + 'TW_NDCG.pickle', 'rb'))
    tw_diversity = pickle.load(open(dir + 'TW_Diversity.pickle', 'rb'))

    precision = pickle.load(open(dir + 'Precision.pickle', 'rb'))
    ndcg = pickle.load(open(dir + 'NDCG.pickle', 'rb'))
    diversity = pickle.load(open(dir + 'Diversity.pickle', 'rb'))

    return(p, tw_precision, tw_ndcg, tw_diversity, precision, ndcg, diversity)



def print_statistics(precision, ndcg, diversity):

    print('----------- PRECISION ---------------')
    for method, prec in sorted(precision.items(), key=itemgetter(1), reverse=True):
        print('Method:', method, 'Precision:', prec)
    print('\n------------- NDCG ------------------')
    for method, ndcg in sorted(ndcg.items(), key=itemgetter(1), reverse=True):
        print('Method:', method, 'NDCG:', ndcg)
    print('\n-------------- ILD ------------------')
    for method, ild in sorted(diversity.items(), key=itemgetter(1), reverse=True):
        print('Method:', method, 'ILD:', ild)



def get_annotation(metrics):

    methods = [k for k, v in sorted(metrics.items(), key=itemgetter(1), reverse=True)]
    annotation = str()
    for i, m in enumerate(methods):
        if i == 0:
            annotation += m + ': ' + str(round(metrics[m], 3))
        else:
            annotation += '\n' + m + ': ' + str(round(metrics[m], 3))

    return annotation



def plot(metrics, metrics_name, plot_title=None, plot_suptitle=None, annotation=None, annotation_position=None):

    # methods = [k for k, v in sorted(metrics.items(), key=itemgetter(1), reverse=True) if '1' not in k]
    methods = [k for k, v in sorted(metrics.items(), key=itemgetter(1), reverse=True)]
    avg = [str(round(metrics[m], 3)) for m in methods]
    legend = [str(k)+': '+str(v) for k, v in zip(methods, avg)]

    mymethods = [r'${\rm RWR_{SAL}}$', r'Session-kNN${\rm (RWR_{SAL})}$', r'${\rm PathCount_{AUA}}$',
               r'${\rm PathSim_{AUA}}$', r'Session-kNN${\rm (PathCount_{SAS})}$',
               r'Session-kNN${\rm (PathSim_{SAS})}$', r'${\rm POP}$', r'${\rm SimRank_{SAL})}$']
    legend = [str(k)+': '+str(v) for k, v in zip(mymethods, avg)]

    # sns.set_palette("Set1", n_colors=20)
    # sns.color_palette("muted", n_colors = 20)

    fig = plt.figure(figsize=(15, 7))

    for m in methods:
        plt.plot(p, [v for v in tw_precision[m]], marker='o')

    # if plot_suptitle != None:
    #     plt.suptitle(plot_suptitle, fontsize=12)
    # if plot_title != None:
    #     plt.title(plot_title, fontsize=10, loc='left')

    if plot_suptitle != None:
        plt.title(plot_suptitle, fontsize=18)

    # plt.xlabel('Period')
    # plt.xticks(rotation=90)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel(metrics_name, fontsize=18)
    plt.legend(legend, loc='upper right', fontsize=12, frameon=False).draggable()
    # plt.legend([r'$W_{LA}$=0.481', r'$W_{S}$=0.478', r'$W_{M}$=0.457', r'$W_{T}$=0.420'], fontsize=18, frameon=False).draggable()
    # plt.legend([r'$w=0.5$ ($avg=$'+avg[0]+')', r'$w=2$ ($avg=$'+avg[1]+')', r'$w=7$ ($avg=$'+avg[2]+')'], fontsize=18, frameon=False).draggable()
    #, bbox_to_anchor=(1.05, 1), loc=2)#, borderaxespad=0., borderpad=0.5)#.draggable()
    # plt.grid()

    if annotation != None:
        t = plt.text(annotation_position[0], annotation_position[1], annotation)
        t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='black'))

    plt.show()





# pickle_name = 'Articles_AllData_2_14_1_mst'
# pickle_name = 'Articles_LongSessions(2)_2_14_1_SML'
# pickle_name = 'Articles_LongSessions(2)_2_PathSims'
# pickle_name = 'Articles_LongSessions(2)_loc_2_1_ExplPathSims'
# pickle_name = 'Articles_AllData_short2_min2_top5'
# pickle_name = 'Articles_AllData_loc_2_RWR_PS_PC'
pickle_name = 'Video33_2_14'
DATA = 'All Data'
# DATA = 'All data'
SHORT_DAYS = 2
MEDIUM_DAYS = 14
MIN_N_ITEMS = 2
N = 5


dir = './Data/Results/'+pickle_name+'/'
# dir = '.\\Data\\Results\\Newest\\MST\\'
# dir = '.\\Data\\Results\\Newest\\Comparison\\'

p, tw_precision, tw_ndcg, tw_diversity, precision, ndcg, diversity = load_pickle(dir)
print_statistics(precision, ndcg, diversity)

# avg_n_ses_per_train = pickle.load(open(dir + 'Avg_N_sessions_per_train', 'rb'))
# n_rec_per_test = pickle.load(open(dir + 'N_recommendations_per_test', 'rb'))


# annotation = get_annotation(precision)

# title = 'Data: ' + DATA + \
#         '\nShort term: ' + str(SHORT_DAYS) + ' days (AA)' + \
#         '\nMedium term: ' + str(MEDIUM_DAYS) + ' days (CC)' + \
#         '\nLong term: All previous history (UC)' + \
#         '\nPredictions are made for users that had at least 2 sessions in previous history'


# title = 'Data: AllData' + \
#         '\nShort term: ' + str(SHORT_DAYS) + ' days'# + \
#         # '\nMin N items: ' + str(MIN_N_ITEMS) + \
#         # '\nTOP N: ' + str(N)
#

# p = [x.split('-') for x in p]
# p = [x[1] + '.' + x[0] for x in p]

# suptitle = 'Comparing different sliding time window (' + r'$t_p$' + ') sizes'
#
# tw_precision['PathSim_ASA(0.5)'] = [0.27203065134099613, 0.19863013698630136, 0.11757990867579907, 0.29687499999999994, 0.17973856209150327, 0.20390070921985812, 0.12140350877192982, 0.18895833333333328, 0.125, 0.25, 0.30711382113821134]
# precision['PathSim_ASA(0.5)'] = 0.205566421051

suptitle = 'Comparison of models'

plot(metrics=precision, metrics_name='Precision', plot_title=None, plot_suptitle=suptitle)



exit()



fig, ses = plt.subplots(figsize=(15,7))

color = 'tab:blue'
# ses.set_xlabel('Period')
ses.set_ylabel('% Precision', color=color)
ses.plot(p, [v for v in tw_precision['PathCount_ASASA']], marker='o', color=color)
ses.tick_params(axis='y', labelcolor=color)
ses.tick_params(axis='x', rotation=90)
ses.grid()

art = ses.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:orange'
art.set_ylabel('# recommendations made', color=color)
art.plot(p, n_rec_per_test.values(), marker='o', color=color)
art.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()