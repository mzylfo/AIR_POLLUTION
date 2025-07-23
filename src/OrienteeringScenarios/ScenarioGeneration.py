from src.OrienteeringScenarios.GraphGeneration import *

class ScenarioGeneration():

    def __init__(self,soluction,graph):
        if isinstance(graph, GraphGeneration):
            graphdata = graph.getGraphList(info=True)
            self.nodes_data = graphdata["n"]
            self.edge_data = graphdata["e"]
            self.soluction = soluction
        else:
            raise ScenarioGeneration_Exception__GraphGenerationInstance(graph)
      
    def scenarios_generation(self,n_scenarios=1):
        availability_scenarios = np.random.rand(n_scenarios,len(self.soluction)) # warehouse is a starting and an ending point
        scenarios = dict()
        
        for idx, availability_scenario in enumerate(availability_scenarios):
            route = [0]+self.soluction+[0]
            availability = [True]
            for jdx,node in enumerate(self.soluction):
                node_availability_threshold = self.nodes_data[node]['availability_prob']
                if availability_scenario[jdx] < node_availability_threshold:
                    availability.append(True)
                else:
                    availability.append(False)

            scenario = {"route":route,"availability":availability}
            
            scenarios[idx] = scenario
        return scenarios      
  
class ScenarioGeneration_Exception__GraphGenerationInstance(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,instance):
        self.instance = instance
          
    def __str__(self):
        return f"ScenarioGeneration module require an instance 'GraphGeneration' but receive an '{str(type(self.instance))}' object."