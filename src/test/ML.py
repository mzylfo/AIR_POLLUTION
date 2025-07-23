import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from src.OrienteeringScenarios.GraphGeneration import *
from src.OrienteeringScenarios.SolutionPermutation import *
#from src.OrienteeringScenarios.ScenarioGeneration import *
from src.OrienteeringScenarios.SampleGeneration import *
#from src.OrienteeringScenarios.TSP_OP_solution import *
from src.OrienteeringScenarios.VisualGraphPath import *

def start_test():
    n_customer = 156
    graph_generate = GraphGeneration(n_customer, rnd_seed=452,nodes_probability="uniform",layout="random_geometric_graph")

    permutation = SolutionPermutation(graph_generate,generation_mode="naive")

    visualgraph = VisualGraphPath(graph_generate)
    visualgraph.drawGraph(node_label="name",edge_label=None)

    scenarios_n = 500
    deadline_coeff = 00.6
    solutions_n = 10000

    save_allscenarios = "last"

    sampleGen = SampleGeneration(graph_generate, permutation, deadline_mode="mst_based",deadline_coeff=deadline_coeff)
    sampleGen.makeSample(solutions_n, scenarios_n, save_onfile="",save_filename="scenario.pkl",save_scenarios="last")
    solution_memory = sampleGen.getIstanceGenrate("dict")
    solution_df = sampleGen.getIstanceGenrate("pandas")


    ###
    ind_sol = 0
    _scenarios = solution_memory[ind_sol]
    visualgraph.drawGraph_SolVisited(labels="order",scenarios=_scenarios,mode="count_visited")    

    _scenarios = solution_memory[ind_sol]
    visualgraph.drawGraph_SolVisited(labels="order",scenarios=_scenarios,mode="count_availability")

    ind_sol = solutions_n-1
    ind_sce = 10
    scens = solution_memory[ind_sol]['permutation_temp']
    visualgraph.drawGraph_Solution(scens, labels="order")

    ind_sol = solutions_n-1
    ind_sce = 10
    scens = solution_memory[ind_sol]['scenarios'].getScenario(ind_sce)
    visualgraph.drawGraph_OP(scens, labels="name")


    ###
    solution_position = list()
    solution_position_withdeadline = list()
    graph_pos = nx.get_node_attributes(graph_generate.getGraph(), "pos")
    p_list = list()
    for index, row in solution_df.iterrows():
        _arr = list()
        _perm = row['solution']
        for node in row['solution']:
            _pos = graph_pos[node]
            _arr.append(float(_pos[0]))
            _arr.append(float(_pos[1]))
        solution_position.append(_arr.copy())
        _arr.append(row['deadline'])
        solution_position_withdeadline.append(_arr)
    solution_df['solution_position'] = pd.Series(solution_position)
    solution_df['solution_position_deadline'] = pd.Series(solution_position_withdeadline)


    ##    
    scaler = MinMaxScaler()

    solution_df['solution_position_scaled'] = solution_df['solution_position']

    train_df=solution_df.sample(frac=0.7,random_state=401) 
    test_df=solution_df.drop(train_df.index)
    print(len(train_df)," : ",len(test_df))
    field2pred = 'time_mean'

    ## SVR

    #Integrate an array of inputs into one
    X_train = train_df['solution_position']
    Y_train = train_df[field2pred]
    #--------------------
    #Integrate an array of inputs into one
    X_test = test_df['solution_position']
    Y_test = test_df[field2pred]

    from sklearn.svm import SVR
    #fitting
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=3,verbose=True)

    mod_rbf = svr_rbf.fit(X_train.values.tolist(), Y_train)
    y_rbf = mod_rbf.predict(X_test.values.tolist())

    mod_lin = svr_lin.fit(X_train.values.tolist(), Y_train)
    y_lin = mod_lin.predict(X_test.values.tolist())

    mod_poly = svr_poly.fit(X_train.values.tolist(), Y_train)
    y_poli = mod_poly.predict(X_test.values.tolist())

    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from sklearn.metrics import accuracy_score

    y_truth = np.array(Y_test)
    #Correlation coefficient calculation
    rbf_corr = np.corrcoef(y_truth, y_rbf)[0, 1]
    lin_corr = np.corrcoef(y_truth, y_lin)[0, 1]
    poly_corr = np.corrcoef(y_truth, y_poli)[0, 1]

    #Calculate RMSE
    rbf_rmse = sqrt(mean_squared_error(y_truth, y_rbf))
    lin_rmse = sqrt(mean_squared_error(y_truth, y_lin))
    poly_rmse = sqrt(mean_squared_error(y_truth, y_poli))

    #Calculate RMSE
    #rbf_acc = accuracy_score(y_truth, y_rbf)
    #lin_acc = accuracy_score(y_truth, y_lin)
    #poly_acc = accuracy_score(y_truth, y_poli)


    print(f"RBF: \t RMSE {rbf_rmse} \t Corr {rbf_corr}")
    print(f"Lnr: \t RMSE {lin_rmse} \t Corr {lin_corr}")
    print(f"Pol: \t RMSE {poly_rmse} \t Corr {poly_corr}")




    #NEURAL NET
