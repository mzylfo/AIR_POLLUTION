import networkx.algorithms.approximation as nx_app
from src.OrienteeringScenarios.GraphGeneration import *

class TSP_OP():

    def __init__(self,graph):
        if isinstance(graph, GraphGeneration):
            graphdata = graph.getGraphList(info=True)
            self.nodes_data = graphdata["n"]
            self.edge_data = graphdata["e"]
            self.graphNetwork = graph.getGraph()
            
            self.n_nodes = graph.getNodesCount()
        else:
            raise TSP_OP_Exception__GraphGenerationInstance(graph)

    def TSPOptimal(self):
        optimal = nx_app.christofides(self.graphNetwork, weight="distance")
        time = self.TSPOptimal_time(optimal)
        return {"route_optimal":optimal, "route_time":time}
    
    def TSPOptimal_time(self,solution):
        d = 0
        for i in range(len(solution)-1):
            d += self.distanceRoute(solution[i],solution[i+1])            
        return d

    def distanceRoute(self,node_a,node_b):
        if node_a < node_b:
            edge_distance = self.edge_data[(node_a,node_b)]['distance']
        elif node_a > node_b:
            edge_distance = self.edge_data[(node_b,node_a)]['distance']
        else:
            edge_distance = 0
        return edge_distance

    """
    back2ware : back to warehouse before deadline
    """
    def OPSolver(self,scenario,deadline,back2ware=False):
        d = 0 
        p = 0
        nodes_visited = [0]
        route = scenario['route']
        nodes_availability = scenario['availability']
        nodes_visited_time = [None for i in range(self.n_nodes+1)]
        nodes_visited_time[0] = 0
        prev_node = route[0]
        for i in range(1,len(route)-1):
            if nodes_availability[i]:
                dist_prev_act = self.distanceRoute(prev_node,route[i])
                dist_act_ware = self.distanceRoute(route[i],route[-1])
                if back2ware:
                    time_eval = d+dist_prev_act+dist_act_ware
                else:
                    time_eval = d+dist_prev_act
                if time_eval <= deadline: 
                    d += dist_prev_act
                    p += self.nodes_data[route[i]]['prize']
                    nodes_visited.append(route[i])
                    nodes_visited_time[route[i]] = d
                else:
                    dist_prev_ware = self.distanceRoute(prev_node,route[-1])
                    d += dist_prev_ware
                    nodes_visited.append(route[-1])
                    nodes_visited_time[0] = d
                    return OPSolution(route=route,availability=nodes_availability,nodes_visited=nodes_visited,prize=p,time=d,deadline=deadline,nodes_visited_time=nodes_visited_time)
        return OPSolution(route=route,availability=nodes_availability,nodes_visited=nodes_visited,prize=p,time=d,deadline=deadline,nodes_visited_time=nodes_visited_time)

class TSP_OP_Exception__GraphGenerationInstance(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,instance):
        self.instance = instance
          
    def __str__(self):
        return f"TSP_OP module require an instance 'GraphGeneration' but receive an '{str(type(self.instance))}' object."

class OPSolution():
    def __init__(self,availability,deadline,nodes_visited,prize,route,time,nodes_visited_time,idSolutionId=None):
        self.idSolutionId = idSolutionId
        self.availability = availability
        self.deadline = deadline
        self.nodes_visited = nodes_visited
        self.prize = prize
        self.route = route
        self.time = time
        self.nodes_visited_time=nodes_visited_time

    def showpath(self):
        print(self.route) 
        print(self.availability)
        print(self.nodes_visited)
        print(self.nodes_visited_time)
        print(self.time)
        print(self.prize)