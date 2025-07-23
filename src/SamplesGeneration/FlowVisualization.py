import osmnx as ox
import xmltodict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import sparse
import networkx as nx
import json
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from colorsys import hls_to_rgb



import warnings
warnings.filterwarnings("ignore")


class FlowVisualization():

    def __init__(self, simulation_name, sampled_dir, sample_id, load_data=True, sampled_geograph=None, edge_stats=None):
        self.simulation_name = simulation_name
        self.sample_id = sample_id
        sample_id_str = f'{sample_id}'
        self.sampled_dir = sampled_dir
        if load_data:
            edge_stats_path = Path(self.sampled_dir,sample_id_str,self.simulation_name+'.random.'+sample_id_str+'.edgestats.json')  
            with open(edge_stats_path) as json_file:
                self.edge_stats = json.load(json_file)
            
            sampled_geograph_path = Path(self.sampled_dir,sample_id_str,self.simulation_name+'.random.'+sample_id_str+'.graph.graphml')
            self.sampled_geograph = ox.load_graphml(sampled_geograph_path)
        else:
            self.sampled_geograph = sampled_geograph
            self.edge_stats = edge_stats
        self.edges_attributes = self.sampled_geograph.edges(keys=True, data=True)

    def draw_sampledgraph(self, attr='travel_time',nbins=10):
        sample_id_str = f'{self.sample_id}'
        if attr=='vehicles_id':
            edge_col = self.edgecolors_attr(self.sampled_geograph, attr=attr, nbins=nbins, categorical_values=True)
        else:
            edge_col = self.edgecolors_attr(self.sampled_geograph, attr=attr, nbins=nbins, categorical_values=False)
            cmap = plt.cm.get_cmap('jet')
            norm=plt.Normalize(vmin=self.edge_stats[attr]['min'], vmax=self.edge_stats[attr]['max'])
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])

        fig, ax = ox.plot_graph(self.sampled_geograph, edge_color=edge_col, node_size=0, edge_linewidth=1.5,  show=False, bgcolor='#FFFFFF', close=False)
        att_label = f'{attr}'
        if attr!='vehicles_id':
            cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal')
            cb.set_label(att_label, fontsize = 20)
        
        sampledgraph_plot = Path(self.sampled_dir,self.simulation_name+'.random.'+sample_id_str+'.'+att_label+'.png')  
        fig.savefig(sampledgraph_plot, dpi=1200)

    def get_colors(self, n, cmap='viridis', start=0., stop=1., alpha=1.):
        colors = [cm.get_cmap(cmap)(x) for x in np.linspace(start, stop, n)]
        colors = [(r, g, b, alpha) for r, g, b, _ in colors]
        return colors

    def get_distinct_colors(self, n):
        colors = []
        for i in np.arange(0., 360., 360. / n):
            h = i / 360.
            l = (50 + np.random.rand() * 10) / 100.
            s = (90 + np.random.rand() * 10) / 100.
            colors.append(hls_to_rgb(h, l, s))
        return colors


    def edgecolors_attr(self, G, attr, nbins, cmap='viridis', start=0, stop=1, na_color='none', categorical_values=False):
        edges_attributes = G.edges(keys=True, data=True)
        att_val_list = [edge[3][attr] for edge in list(edges_attributes)]
        attr_values = pd.DataFrame(att_val_list) # Creates a dataframe with attribute of each node
        if categorical_values:
            attr_values_unique_list = attr_values[0].unique().tolist()
            colors = self.get_distinct_colors(len(attr_values_unique_list))
            attr_values_unique = pd.DataFrame(list(zip(attr_values_unique_list, colors)), columns =['Vehicle', 'colors'])
            for veih in att_val_list:
                if veih not in att_val_list:
                    print(veih)
            edge_colors = [attr_values_unique.loc[attr_values_unique['Vehicle'] == veih]['colors'].values[0] if pd.notnull(veih) else na_color for veih in att_val_list]
        else:            
            categories = pd.qcut(attr_values[0], nbins,  duplicates='drop', labels=False)
            colors = self.get_colors(nbins, cmap, start, stop)  #List of colors of each bin
            edge_colors = [colors[int(cat)] if pd.notnull(cat) else na_color for cat in categories]

        return edge_colors