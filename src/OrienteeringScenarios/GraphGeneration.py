import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
"""
GraphGeneration : generate graph
"""
class GraphGeneration():

    def __init__(self, n_customer,distance=None,nodes_prize=None,rnd_seed=None,nodes_probability=None, layout =None):
        self.n_customer = n_customer
        self.graphNetwork = self.GraphNetworkGeneration(n_customer=self.n_customer,distance=distance,nodes_prize=nodes_prize,rnd_seed=rnd_seed,nodes_probability=nodes_probability,layout=layout)

    # Distance calculation
    def node_distance(self,node_a,node_b):
        return math.hypot(node_a[0] - node_b[0], node_a[1] - node_b[1])

    def getNodesCount(self):
        return self.n_customer
    """
    n_customer : int - number of customers
    rnd_seed : int - random seed
    distance : array? - distance between customers
    nodes_prize = array - prize visiting node
    """
    def GraphNetworkGeneration(self,n_customer,rnd_seed=801,nodes_probability="0.5",distance=None,nodes_prize=None,layout="random_geometric_graph"):
        if layout==None or layout== "random_geometric_graph":
            graph = nx.random_geometric_graph(n_customer+1, radius=1.4, seed=rnd_seed)
        elif layout=="soft_random_geometric_graph":
            graph = nx.soft_random_geometric_graph(n_customer+1, radius=1.4, seed=rnd_seed)
        
        graph_pos = nx.get_node_attributes(graph, "pos")

        #Prize as node attribute
        graph_nodes_prize_attribute = dict()
        if nodes_prize is None:
            graph_nodes_prize = [1 for i in range(n_customer+1)]
        elif len(nodes_prize) != n_customer:
            raise GraphGeneration_Exception__PrizeArray(len(nodes_prize),n_customer)
        else:
            graph_nodes_prize = [1] + nodes_prize
        for node in graph.nodes:
            graph_nodes_prize_attribute[node] = graph_nodes_prize[node]
        nx.set_node_attributes(graph, graph_nodes_prize_attribute, "prize")

        #Availability threshold as node attribute
        graph_nodes_availability_prob_attribute = dict()
        if nodes_probability is None or nodes_probability=="0.5":
            nodes_probability_list = [1]+[0.5 for i in range(n_customer)]
        elif nodes_probability=="all":
            nodes_probability_list = [1]+[1 for i in range(n_customer)]
        elif nodes_probability == "uniform" or nodes_probability=="bernoulli":
            nodes_probability_list = [1]+list(np.random.uniform(0,1,n_customer))
        else:
            raise GraphGeneration_Exception__ProbabilityMode(nodes_probability)
        
        for node in graph.nodes:
            graph_nodes_availability_prob_attribute[node] = nodes_probability_list[node]
        nx.set_node_attributes(graph, graph_nodes_availability_prob_attribute, "availability_prob")

        #Distance as edge attribute
        graph_edges_distance_attribute = dict()
        for edge in graph.edges:
            node_a = edge[0]
            node_b = edge[1]
            distance_nodes = self.node_distance(graph_pos[node_a],graph_pos[node_b])
            graph_edges_distance_attribute[edge] = distance_nodes
        nx.set_edge_attributes(graph, graph_edges_distance_attribute, "distance")   
        return graph

    """
    mode : e = edges_from_line
           n = nodes
           e+n = (nodes,edges)
    """
    def getGraphList(self,mode="e+n",info=True):
        edge_data = dict()
        if mode == "e" or mode =="e+n":
            _edges = self.graphNetwork.edges(data=info)
            for edge in _edges:            
                edge_data[(edge[0],edge[1])] = edge[2]

        node_data = dict()
        if mode == "n" or mode =="e+n":
            _nodes = self.graphNetwork.nodes(data=info)
            for node in _nodes:            
                node_data[node[0]] = node[1]        
        
        if mode == "e":            
            return {"e":edge_data}
        elif mode == "n":
            return {"n":node_data}
        elif mode=="e+n" or mode =="n+e":
            return {"n":node_data,"e":edge_data}
        else:
            raise GraphGeneration_Exception__GetListMode(mode)


    def getGraph(self):
        return self.graphNetwork

    
    
    def drawGraph(self,round_chipher=3,node_label="name+prize", edge_label=None):
        
        graph = self.getGraph()
        graph_pos = nx.get_node_attributes(graph, "pos")
        nx.draw(graph, graph_pos)

        if node_label == "name+prize" or node_label == "prize+name":
            nodes_label = {n[0]:f"{n[0]}({n[1]['prize']})" for n in graph.nodes(data=True)}
        elif node_label == "name+availability" or node_label == "availability+name":
            nodes_label = {n[0]:f"{n[0]}({n[1]['availability_prob']})" for n in graph.nodes(data=True)}
        elif node_label == "availability":
          nodes_label = {n[0]:f"{round(n[1]['availability_prob'],3)}" for n in graph.nodes(data=True)}
        elif node_label == "name":
          nodes_label = {n[0]:f"{n[0]}" for n in graph.nodes(data=True)}
        elif node_label == "prize":
            nodes_label = {n[0]:f"{n[1]['prize']}" for n in graph.nodes(data=True)}
        else:
            raise GraphGeneration_Exception__DrawLabels(node_label)

        nx.draw_networkx_labels(graph, graph_pos, labels = nodes_label)


        
        if edge_label == "distance":
            edge_labels = nx.get_edge_attributes(graph,'distance')
            for edge in edge_labels:
                edge_labels[edge] = round(edge_labels[edge],3)

            nx.draw_networkx_edge_labels(graph, graph_pos, edge_labels  = edge_labels)
        plt.show()


class GraphGeneration_Exception__PrizeArray(Exception):
      """Exception raised for error no training modality recognized"""

      def __init__(self,node_prize_len,n_customer):
          self.node_prize_len = node_prize_len
          self.n_customer = n_customer
          
      def __str__(self):
          return f"The length of prize's array have {self.node_prize_len} items but the nodes are {str(self.n_customer)}."

class GraphGeneration_Exception__DrawLabels(Exception):
      """Exception raised for error no training modality recognized"""

      def __init__(self,value):
          self.value = None

      def __str__(self):
          return f"{self.value} is not a modality for nodes labels."
          
class GraphGeneration_Exception__ProbabilityMode(Exception):
      """Exception raised for error no training modality recognized"""

      def __init__(self,value):
          self.value = None

      def __str__(self):
          return f"{self.value} is not a setting for nodes probability."

class GraphGeneration_Exception__GetListMode(Exception):
      """Exception raised for error no training modality recognized"""

      def __init__(self,value):
          self.value = None

      def __str__(self):
          return f"{self.value} is not recognized."