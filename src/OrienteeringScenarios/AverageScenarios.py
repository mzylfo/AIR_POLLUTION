import statistics
from src.OrienteeringScenarios.TSP_OP_solution import *

class AverageScenarios():

    def __init__(self,graph):
        self.n_customer = graph.getNodesCount()
        self.scenarios = dict()
        self.time = list()
        self.prizes = list()
        self.count_availability = [0 for i in range(self.n_customer+1)]
        self.count_visited = [0 for i in range(self.n_customer+1)]
    
    def addScenario(self,opsolution,save_scenario=False):
        if isinstance(opsolution,OPSolution):
            if save_scenario:
                index = len(self.scenarios)+1
                self.scenarios[index] = opsolution
            
            self.time.append(opsolution.time)
            self.prizes.append(opsolution.prize)
            for i in range(len(opsolution.route)-1):
                if opsolution.availability[i]:
                    node = opsolution.route[i]
                    self.count_availability[node] += 1
            for j in range(len(opsolution.nodes_visited)):
                node = opsolution.nodes_visited[j]
                self.count_visited[node] += 1
        else:
            raise AverageScenarios_Exception__OPSolutionInstance(OPSolution)

    def getStats(self,param="time"):
        values = dict()
        if param =="time" or param=="t":
            values['mean'] = statistics.mean(self.time)
            values['variance'] = statistics.variance(self.time)
            values['values'] = self.time
        elif param =="prize" or param=="p":
            values['mean'] = statistics.mean(self.prizes)
            values['variance'] = statistics.variance(self.prizes)
            values['values'] = self.prizes
        else:
            raise AverageScenarios_Exception__Param(param)
        return values
    
    def getScenario(self,index):
        return self.scenarios[index]


class AverageScenarios_Exception__OPSolutionInstance(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,instance):
        self.instance = instance
          
    def __str__(self):
        return f"AverageScenarios module require an instance 'OPSolution' but receive an '{str(type(self.instance))}' object."

class AverageScenarios_Exception__Param(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,instance):
        self.instance = instance
          
    def __str__(self):
        return f"{self.instance}is not a param recognized."