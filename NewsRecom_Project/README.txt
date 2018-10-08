----------------------------------
Files:
----------------------------------

----------------------------------
-- General data preparations:
data_import - to import data from original txt files
graph_manipulation - to create graphs out of datasets and perform other manipulations with them
time_aware_splits - to split the graph into periods by session time

----------------------------------
-- Modeling and predicting next:
popularity_based_rec - for popularity based recommendations 
personalized_prank - RWR based recommendations
pathsim - PathSim based recommendations
simrank - SimRank based recommendations

----------------------------------
-- Additional files:
accuracy_evaluation - for evaluating recommendation by precision, ndcg, ild and explainability
plots - when results from other files are saved into Data/Results, it is possible to extract them in this file and plot graphs

----------------------------------
-- All files that start from M_ are main files for running different methods in different combinations:
M_RWR's, M_PathSim's, M_SimRank's - comparison of different configurations of one of the methods (+POP)

M_MST - comparison of weight assignment strategies (W_mean, W_sigmoid, W_timeview)
M_SML - comparison of combinations of short, medium, long terms

M_SessionKNN - Session-kNN on different similarity measures (RWR, PathSim)

M_Explainability_SI, M_Explainability_UI - calculating the explainability coverage for session and for user, and saving into the file

M_AB+Expl, M_SB+Expl, M_UB+Expl - AB, SB and UB explained recommendation strategies + measuring their explainability

M_ComparisonOfMainMethods - comparison of RWR, PathSim, SimRank, POP with their different configuration in one run
M_ComparisonOfMethods+Expl - comparison of main methods + AB, SB and UB + measuring explainability of them all
