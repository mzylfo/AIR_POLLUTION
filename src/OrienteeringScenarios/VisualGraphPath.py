import matplotlib.pyplot as plt
import matplotlib as mpl
from src.OrienteeringScenarios.GraphGeneration import *

class VisualGraphPath():

    def __init__(self, graph):
        self.options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
        if isinstance(graph, GraphGeneration):
            self.graph = graph.getGraph()
            self.graph_pos = nx.get_node_attributes(self.graph, "pos")
            self.nodes_label_name = {n[0]:f"{n[0]}" for n in self.graph.nodes(data=True)}
            
            #nodes_label = {n[0]:f"{n[1]['prize']}" for n in self.graph.nodes(data=True)}

            self.nodes_label_prize = {n[0]:n[1]['prize'] for n in self.graph.nodes(data=True)}
            _min = min(self.nodes_label_prize.items(), key=lambda x: x[1])[1]
            _max = max(self.nodes_label_prize.items(), key=lambda x: x[1])[1]
            self.prize_intervall = (_min,_max)

            self.nodes_label_availability = {n[0]:n[1]['availability_prob'] for n in self.graph.nodes(data=True)}
            _min = min(self.nodes_label_availability.items(), key=lambda x: x[1])[1]
            _max = max(self.nodes_label_availability.items(), key=lambda x: x[1])[1]
            self.availability_intervall = (_min,_max)
            
            self.n_nodes = graph.getNodesCount()+1
        else:
            raise VisualGraphPath_Exception__GraphGenerationInstance(graph)

    def drawGraph(self,round_chipher=3,node_label="name+prize", edge_label=None):
        
        f = plt.figure(figsize=(25,18))
        plt.gca().set_aspect('auto')

        if node_label == "name+prize" or node_label == "prize+name":
            nodes_label = {n[0]:f"{n[0]}({n[1]['prize']})" for n in self.graph.nodes(data=True)}
            
            cmap = mpl.cm.cool
            norm = mpl.colors.Normalize(vmin=self.prize_intervall[0], vmax=self.prize_intervall[1])
            size_cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            color_node = [size_cmap.to_rgba(n[1]['prize']) for n in self.graph.nodes(data=True)]

        elif node_label == "name+availability" or node_label == "availability+name":
            nodes_label = {n[0]:f"{n[0]}({n[1]['availability_prob']})" for n in self.graph.nodes(data=True)}

            cmap = mpl.cm.cool
            norm = mpl.colors.Normalize(vmin=self.availability_intervall[0], vmax=self.availability_intervall[1])
            size_cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            color_node = [size_cmap.to_rgba(n[1]['availability_prob']) for n in self.graph.nodes(data=True)]
        elif node_label == "availability":          
            nodes_label = {n[0]:f"{round(n[1]['availability_prob'],3)}" for n in self.graph.nodes(data=True)}

            cmap = mpl.cm.cool
            norm = mpl.colors.Normalize(vmin=self.availability_intervall[0], vmax=self.availability_intervall[1])
            size_cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            color_node = [size_cmap.to_rgba(n[1]['availability_prob']) for n in self.graph.nodes(data=True)]
        elif node_label == "name":
            nodes_label = {n[0]:f"{n[0]}" for n in self.graph.nodes(data=True)}
            
            color_node = ['blue' for n in self.graph.nodes(data=True)]
        elif node_label == "prize":
            nodes_label = {n[0]:f"{n[1]['prize']}" for n in self.graph.nodes(data=True)}

            cmap = mpl.cm.cool
            norm = mpl.colors.Normalize(vmin=self.prize_intervall[0], vmax=self.prize_intervall[1])
            size_cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            color_node = [size_cmap.to_rgba(n[1]['prize']) for n in self.graph.nodes(data=True)]
        else:
            raise GraphGeneration_Exception__DrawLabels(node_label)

        nx.draw_networkx_labels(self.graph, self.graph_pos, labels = nodes_label)
        nx.draw_networkx_edges(self.graph, self.graph_pos, width=0.5, alpha=0.15)
                
        if edge_label == "distance":
            edge_labels = nx.get_edge_attributes(self.graph,'distance')
            for edge in edge_labels:
                edge_labels[edge] = round(edge_labels[edge],3)

            nx.draw_networkx_edge_labels(self.graph, self.graph_pos, edge_labels  = edge_labels)
        else:
            nx_graph_networkx_edges_labels = None

        nx.draw_networkx_nodes(self.graph, self.graph_pos, nodelist=[i for i in range(self.n_nodes)], node_color=color_node, alpha=[0.5 for i in range(self.n_nodes)])
        plt.show()

        
        

    def drawGraph_OP(self, solution,labels="name", round_digits=2):
        f = plt.figure(figsize=(25,18))
        plt.gca().set_aspect('auto')

        edgecolors = ['orange']+['#d7d7d7' for i in range(self.n_nodes-1)]
        node_size = [0]+[0 for i in range(self.n_nodes)]
        color_map = ['tab:orange']+['#d7d7d7' for i in range(self.n_nodes-1)]
        color_map_alfa = ['0.8']+['0.08' for i in range(self.n_nodes-1)]
        node_size = [605]+[650 for i in range(self.n_nodes)] 


        for i in range(self.n_nodes):
            _node = solution.route[i]
            if solution.availability[i]:
                if _node in solution.nodes_visited and _node!=solution.nodes_visited[0]:
                    edgecolors[_node] = '#0c8d00'
                    color_map[_node] = "#14de02"
                    color_map_alfa[_node] = '0.85'
                    node_size[_node] = 800
                elif _node==solution.nodes_visited[0]:
                    edgecolors[_node] = '#d5d21b'
                    color_map[_node] = "#ffa600"
                    color_map_alfa[_node] = '0.85'
                    node_size[_node] = 800
                else:
                    edgecolors[_node] = '#720000'
                    color_map[_node] = "#ff0000"
                    color_map_alfa[_node] = '0.45'
                
        nx.draw_networkx_nodes(self.graph, self.graph_pos, nodelist=[i for i in range(self.n_nodes)], node_color=color_map, edgecolors=edgecolors, alpha=color_map_alfa)
        
        nx.DiGraph()
        if labels=="name":
            nx.draw_networkx_labels(self.graph, self.graph_pos, labels = self.nodes_label_name)
        elif labels=="prize":
            nx.draw_networkx_labels(self.graph, self.graph_pos, labels = self.nodes_label_prize)
        elif labels=="time_visit":
            nodes_label_visited_time_dict = dict()
            for i in range(self.n_nodes):
                _node = solution.route[i]
                if _node == 0:                    
                    nodes_label_visited_time_dict[_node] = f"0 > {round(solution.nodes_visited_time[_node], round_digits)}"
                else:
                    if solution.nodes_visited_time[_node] is None:
                        nodes_label_visited_time_dict[_node] = None
                    else:
                        nodes_label_visited_time_dict[_node] = round(solution.nodes_visited_time[_node], round_digits)
            nx.draw_networkx_labels(self.graph, self.graph_pos, labels = nodes_label_visited_time_dict)
        else:
            raise VisualGraphPath_Exception__GraphGenerationInstance(labels)


        nx.draw_networkx_edges(self.graph, self.graph_pos, width=0.5, alpha=0.15)
        
        prev_node = solution.nodes_visited[0]
        edgelist = []
        for node in solution.nodes_visited[1:]:
            edge = (prev_node,node)
            edgelist.append(edge)
            prev_node = node
        if prev_node !=0:
            edge = (prev_node,solution.nodes_visited[0])
            edgelist.append(edge)
        
        

        nx.draw_networkx_edges(self.graph, self.graph_pos,edgelist=edgelist,width=8,alpha=0.5,edge_color="#559f4f",style='solid',arrows=True,arrowsize=18, arrowstyle='->',node_size=node_size)
        #nx.draw(self.graph, self.graph_pos,node_color=color_map)
        #nx.draw_networkx_edges(self.graph, self.graph_pos, width=0.5, alpha=0.5)
        plt.show()


    def drawGraph_SolVisited(self, scenarios, labels="name", round_digits=2, mode="count_visited"):
        
        if mode == "count_visited":
            scenarios_avg = scenarios['scenarios'].count_visited
        elif mode == "count_availability":
            scenarios_avg = scenarios['scenarios'].count_availability
        else:
            raise VisualGraphPath_Exception__drawGraph_SolVisitedMode(mode)

        f = plt.figure(figsize=(25,18))
        plt.gca().set_aspect('auto')

        edgecolors = ['orange']+['#d7d7d7' for i in range(self.n_nodes-1)]
        node_size = [0]+[1 for i in range(self.n_nodes-1)]
        color_map = ['tab:orange']+['#d7d7d7' for i in range(self.n_nodes-1)]
        color_map_alfa = ['0.8']+['0.2' for i in range(self.n_nodes-1)]
        
        cmap = mpl.cm.cool
        norm = mpl.colors.Normalize(vmin=50, vmax=800)
        size_cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        


        max_size = max(scenarios_avg[1:])
        min_size = min(scenarios_avg[1:])
        for i in range(self.n_nodes):
            node_size[i] = int((scenarios_avg[i] - min_size)*(800-50)/(max_size-min_size) + 50)
            color_map[i] = size_cmap.to_rgba(node_size[i])
            edgecolors[i] = size_cmap.to_rgba(node_size[i])
        nx.draw_networkx_nodes(self.graph, self.graph_pos, nodelist=[i for i in range(self.n_nodes)], node_color=color_map, edgecolors=edgecolors, alpha=color_map_alfa,node_size=[int(v) for v in node_size])
        
        if labels=="name":
            nx.draw_networkx_labels(self.graph, self.graph_pos, labels = self.nodes_label_name)
        elif labels=="prize":
            nx.draw_networkx_labels(self.graph, self.graph_pos, labels = self.nodes_label_prize)        
        elif labels=="order":
            solution = scenarios['permutation_temp']
            order_list = dict()
            order_list[0] = 0
            for _node in range(len(solution)):
                order_list[_node+1] = solution.index(_node+1)+1
            nx.draw_networkx_labels(self.graph, self.graph_pos, labels = order_list) 
        else:
            raise VisualGraphPath_Exception__GraphGenerationInstance(labels)


        nx.draw_networkx_edges(self.graph, self.graph_pos, width=0.5, alpha=0.15)
        
        """
        edgelist = []
        for node in solution.nodes_visited[1:]:
            edge = (prev_node,node)
            edgelist.append(edge)
            prev_node = node
        edge = (prev_node,solution.nodes_visited[0])
        edgelist.append(edge)
        
        

        nx.draw_networkx_edges(self.graph, self.graph_pos,edgelist=edgelist,width=8,alpha=0.5,edge_color="#559f4f",style='solid',arrows=True,arrowsize=18, arrowstyle='->',node_size=node_size)
        #nx.draw(self.graph, self.graph_pos,node_color=color_map)
        #nx.draw_networkx_edges(self.graph, self.graph_pos, width=0.5, alpha=0.5)
        """
        plt.show()


    def drawGraph_Solution(self, solution, labels="order", round_digits=2):
        
        f = plt.figure(figsize=(25,18))
        plt.gca().set_aspect('auto')

        edgecolors = ['orange']+['#d7d7d7' for i in range(self.n_nodes-1)]
        node_size = [0]+[1 for i in range(self.n_nodes-1)]
        color_map = ['tab:orange']+['#14660d' for i in range(self.n_nodes-1)]
        color_map_alfa = ['0.8']+['0.6' for i in range(self.n_nodes-1)]
        
        cmap = mpl.cm.cool
        norm = mpl.colors.Normalize(vmin=0, vmax=len(solution))
        size_cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        edges_color = ['#d7d7d7' for i in range(len(solution)+1)]
            
        nx.draw_networkx_nodes(self.graph, self.graph_pos, nodelist=[i for i in range(self.n_nodes)], node_color=color_map, edgecolors=edgecolors, alpha=color_map_alfa,node_size=[int(v) for v in node_size])
        if labels=="name":
            nx.draw_networkx_labels(self.graph, self.graph_pos, labels = self.nodes_label_name)
        elif labels=="prize":
            nx.draw_networkx_labels(self.graph, self.graph_pos, labels = self.nodes_label_prize)        
        elif labels=="order":
            order_list = dict()
            order_list[0] = 0
            for _node in range(len(solution)):
                order_list[_node+1] = solution.index(_node+1)+1
            nx.draw_networkx_labels(self.graph, self.graph_pos, labels = order_list)        
        else:
            raise VisualGraphPath_Exception__GraphGenerationInstance(labels)
        

        nx.draw_networkx_edges(self.graph, self.graph_pos, width=0.5, alpha=0.15)
        
        
        edgelist = []
        prev_node = 0
        i = 0
        for node in solution:
            edges_color[i] = size_cmap.to_rgba(i)
            i += 1
            edge = (prev_node,node)
            edgelist.append(edge)
            prev_node = node
        edges_color[i] = size_cmap.to_rgba(i)  
        edge = (prev_node,0)
        edgelist.append(edge)
        
        

        nx.draw_networkx_edges(self.graph, self.graph_pos,edgelist=edgelist,width=8,alpha=0.5,edge_color=edges_color,style='solid',arrows=True,arrowsize=18, arrowstyle='->',node_size=node_size)
        #nx.draw(self.graph, self.graph_pos,node_color=color_map)
        #nx.draw_networkx_edges(self.graph, self.graph_pos, width=0.5, alpha=0.5)
        
        plt.show()
        
class VisualGraphPath_Exception__GraphGenerationInstance(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,instance):
        self.instance = instance
          
    def __str__(self):
        return f"VisualGraphPath initialization require an instance 'GraphGeneration' but receive an '{str(type(self.instance))}' object."

class VisualGraphPath_Exception__GraphGenerationInstance(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,instance):
        self.instance = instance
          
    def __str__(self):
        return f"VisualGraphPath module require an instance 'GraphGeneration' but receive an '{str(type(self.instance))}' object."

class VisualGraphPath_Exception__drawGraph_SolVisitedMode(Exception):
      """Exception raised for error no training modality recognized"""

      def __init__(self,value):
          self.value = None

      def __str__(self):
          return f"{self.value} is not recognized."