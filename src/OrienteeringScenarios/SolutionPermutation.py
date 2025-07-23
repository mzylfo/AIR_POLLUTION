import random
from src.OrienteeringScenarios.GraphGeneration import *

class SolutionPermutation():

    def __init__(self,graph,generation_mode="naive"):
        if isinstance(graph, GraphGeneration):
            graphdata = graph.getGraphList(info=True)
            self.nodes_data = graphdata["n"]
            self.edge_data = graphdata["e"]
        else:
            raise SolutionPermutation_Excepion__GraphGenerationInstance(graph)
        if generation_mode== None or generation_mode=="naive" or generation_mode=="random":
            self.generation_mode = "naive"
        else:
            raise SolutionPermutation_Excepion__GenerationModality(self.generation_mode)
    
    def naive_permutation(self):
        node_permutation_sample = list(self.nodes_data.keys())[1:]
        return list(random.sample(node_permutation_sample,len(node_permutation_sample)))
  
    def next_permutation(self):
        return self.next()
      
    def next(self):
        if self.generation_mode == "naive":
            return self.naive_permutation()


class SolutionPermutation_Exception__GraphGenerationInstance(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,instance):
        self.instance = instance
          
    def __str__(self):
        return f"Permutation module require an instance 'GraphGeneration' but receive an '{str(type(self.instance_type))}' object."

class SolutionPermutation_Excepion__GenerationModality(Exception):
      """Exception raised for error no training modality recognized"""

      def __init__(self,value):
          self.value = None

      def __str__(self):
          return f"'{self.value}' modality is not recognized."