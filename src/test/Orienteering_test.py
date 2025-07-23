from src.OrienteeringScenarios.GraphGeneration import *
from src.OrienteeringScenarios.SolutionPermutation import *
from src.OrienteeringScenarios.ScenarioGeneration import *
from src.OrienteeringScenarios.TSP_OP_solution import *
from src.OrienteeringScenarios.VisualGraphPath import *

def start_test():
    graph_generate = GraphGeneration(n_customer=8, rnd_seed=451,nodes_probability="uniform")
    graph_generate.drawGraph(node_label="name+availability")

    permutation = SolutionPermutation(graph_generate)
    permutation_temp = permutation.naive_permutation()
    s = ScenarioGeneration(permutation_temp,graph_generate)

    tsp = TSP_OP(graph_generate)
    tsp_opt = tsp.TSPOptimal() 
    deadline = tsp_opt['route_time']*0.8

    scenarios_n = 500
    scs = s.scenarios_generation(scenarios_n)
    solutions = dict()
    for i in range(scenarios_n):
        solutions[i] = tsp.OPSolver(scs[i],deadline,back2ware=False)
        solutions[i].idSolutionId = i

    ind = 23
    visualgraph = VisualGraphPath(graph_generate)
    visualgraph.drawGraph_OP(solutions[ind], labels="name")
    #print(solutions[ind].nodes_visited)
    #print(solutions[ind].availability)

    visualgraph = VisualGraphPath(graph_generate)
    visualgraph.drawGraph_OP(solutions[ind], labels="prize")
    #print(solutions[ind].nodes_visited)
    #print(solutions[ind].availability)

    visualgraph = VisualGraphPath(graph_generate)
    visualgraph.drawGraph_OP(solutions[ind], labels="time_visit")
    #print(solutions[ind].nodes_visited)
    #print(solutions[ind].availability)