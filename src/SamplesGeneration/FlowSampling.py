import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import random
import numpy as np
import networkx as nx
import osmnx as ox
import xmltodict
from scipy import sparse
import os
import shutil
import json
import warnings
warnings.filterwarnings("ignore")
import xml.etree.ElementTree as ET
from xml.dom import minidom
from .FlowVisualization import *


class FlowSampling():
    def __init__(self, is_simulation, simulation_name=None, save_path_roadstats=True):
        self.simulation_name = simulation_name
        self.samplingstatsFile = Path('data','sumo_simulation_files',self.simulation_name,self.simulation_name+'.out.samplingstats.pkl')
        self.osmNetworkFile =  Path('data','sumo_simulation_files',self.simulation_name,self.simulation_name+'.geo.osm')
        self.edgeNetworkFile =  Path('data','sumo_simulation_files',self.simulation_name,self.simulation_name+'.netplain.edg.xml')
        self.multiIndex_cols = ["mean", "weighted_mean", "travel_time"]
        
        self.df_samplingstats = pd.read_pickle(self.samplingstatsFile)
        self.mapping_graphData()


    #mapping osm data in sumo data
    def mapping_graphData(self):
        #
        self.geograph = ox.graph.graph_from_xml(self.osmNetworkFile)
        self.edges_attributes = self.geograph.edges(keys=True, data=True)

        with open(self.edgeNetworkFile, 'r', encoding='utf-8') as file:
            netplain_xml = file.read()    
        netplain_dict = xmltodict.parse(netplain_xml)
        
        edges_attributes_dict = dict()
        for i in list(self.edges_attributes):
            if (i[0], i[1]) not in edges_attributes_dict:
                edges_attributes_dict[i[0], i[1]] = dict()
            if i[2] in edges_attributes_dict[i[0], i[1]]:
                raise FlowSampling_Exception__alredyRead(i[2],i[3])
            edges_attributes_dict[i[0], i[1]][i[2]] = i[3]['osmid']
        
        self.edge_dict = dict()
        for edge in netplain_dict['edges']['edge']:
            try:
                _from = int(edge['@from'])
                _to = int(edge['@to'])
                _dir = None    
                dict_dir = edges_attributes_dict[ _from, _to]
                if len(dict_dir) == 1:
                    _dir = next(iter(dict_dir))      
                elif len(dict_dir) > 1:
                    for dir_i in dict_dir:
                        osmid = abs(int(dict_dir[dir_i]))

                        if osmid == int(edge['@id']):
                                _dir = dir_i
                _id = f"{edge['@id']}_0"
                self.edge_dict[_id] = {'from': _from, "to": _to, "dir": _dir}
            except:
                pass

    def generate_samples(self,number_samples,sampled_dir='randomgraph',save_flows=False,save_sample_graphmatrix=True, overwrite_samples=True, draw_graph=False):

        folder_samples = Path('data','sumo_simulation_files',self.simulation_name,sampled_dir)
        if overwrite_samples:
            if os.path.exists(folder_samples):
                shutil.rmtree(folder_samples, ignore_errors=True)
        
        if not os.path.exists(folder_samples):
            os.makedirs(folder_samples)
        
        for i in tqdm(range(number_samples)):
            sample_id = i
            folder_sample = Path(folder_samples, f"{sample_id}")
            if not os.path.exists(folder_sample):
                os.makedirs(folder_sample)
            else:
                if not overwrite_samples:
                    raise FlowSampling_Exception__alredySampleExist(i)

            random_roads_vehicles = self.create_sampled_flows(sample_id=sample_id, sampled_dir=folder_sample, save=save_flows)
            self.create_sampled_graph(random_roads_vehicles=random_roads_vehicles, sampled_dir=folder_sample, sample_id=sample_id, save=save_sample_graphmatrix, draw_graph=draw_graph)

    def create_sampled_flows(self, sampled_dir, sample_id, save=True):
        roads = list(self.df_samplingstats.index.unique(level='roads'))
        vehicles = list(self.df_samplingstats.index.unique(level='vehicles'))
        
        index = pd.MultiIndex.from_tuples(list(), names=["roads", "vehicles"])
        random_roads_vehicles = pd.DataFrame(index=index, columns=self.multiIndex_cols)
        while (len(roads)>0 and len(vehicles)>0):
            #select road r
            random_road = random.choice(roads)
            roads.remove(random_road)
            

            #select vehicle v in the road r
            random_road_vehicles = [x for x in list(self.df_samplingstats.iloc[self.df_samplingstats.index.get_level_values('roads') == random_road].index.unique(level='vehicles')) if x in vehicles]
            if len(random_road_vehicles)>0:
                random_vehicle = random.choice(random_road_vehicles)
                vehicles.remove(random_vehicle)
            
                #select randomly roads rs (without duplicate) visited by vehicle v
                random_roads_vehicle_withDuplicate = self.df_samplingstats.iloc[self.df_samplingstats.index.get_level_values('vehicles') == random_vehicle]
                withDuplicate_perm = np.random.permutation(np.arange(len(random_roads_vehicle_withDuplicate)))
                random_roads_vehicle = random_roads_vehicle_withDuplicate.iloc[withDuplicate_perm][~random_roads_vehicle_withDuplicate.iloc[withDuplicate_perm].index.duplicated(keep='first')]

                #select roads visited by vehicle v BUT not alredy visited by other vehicle
                #random_roads_visited = random_roads_vehicles.iloc[random_roads_vehicles.index.unique(level='roads')]
                random_roads_vehicle_new = random_roads_vehicle[~random_roads_vehicle.index.get_level_values('roads').isin(random_roads_vehicles.index.get_level_values('roads'))]
                #if (random_roads_vehicle_new) >0:
                #append new road data
                random_roads_vehicles = pd.concat([random_roads_vehicles, random_roads_vehicle_new])

                #remove road visited from all road
                visited_road_list = [x for x in list(random_roads_vehicle_new.index.get_level_values('roads')) if x!= random_road]

                for xroad in visited_road_list:
                    roads.remove(xroad)
        if save:
            self.flowsSampled = Path(sampled_dir,self.simulation_name+'.'+str(sample_id)+'.out.flowsSampled.pkl')
            random_roads_vehicles.to_pickle(self.flowsSampled)

            flowsSampled_ExcludedFile = Path(sampled_dir,self.simulation_name+'.'+str(sample_id)+'.out.flowsSampled_excluded.txt')
            excluded_root = ET.Element("Excluded")
            excl_road = ET.SubElement(excluded_root, "Roads")
            for _road in roads:
                r = ET.SubElement(excl_road, "Road") 
                r.set("id",str(_road))
            excl_street = ET.SubElement(excluded_root, "Street") 
            for _veh in vehicles:
                r = ET.SubElement(excl_street, "Vehicle") 
                r.set("id",str(_veh))
            xmlstr = minidom.parseString(ET.tostring(excluded_root)).toprettyxml(indent="   ")
            with open(flowsSampled_ExcludedFile, "w") as f:
                f.write(xmlstr)
            
            
        return random_roads_vehicles


    def create_sampled_graph(self, random_roads_vehicles, sampled_dir, sample_id, save=True, draw_graph=False):
        edge_data = {}
        for idx, data in random_roads_vehicles.groupby(level=0):
            _id_road = list(data.index)[0][0]
            try:
                nodes = self.edge_dict[_id_road]
                _key = (nodes['from'],nodes['to'],nodes['dir'])
                _mean = float(data['mean'].values[0])
                _weighted_mean = float(data['weighted_mean'].values[0])
                _travel_time = float(data['travel_time'].values[0])
                _vehicles_id = list(data.index.get_level_values('vehicles'))[0]
                if _key not in edge_data:
                    edge_data[_key] = { "mean":_mean, "weighted_mean":_weighted_mean, "travel_time":_travel_time, "vehicles_id":_vehicles_id, 'sumo_road:id':_id_road}
            except Exception as e:
                pass

        mean_list = list()
        weighted_mean_list = list()
        travel_time_list = list()
        traved_list = list()

        edge_stats = {'mean':{'min':0.0, 'max':0.0},'weighted_mean':{'min':0.0, 'max':0.0},'travel_time':{'min':0.0, 'max':0.0}}

        attrs = dict()
        for idx,edge in enumerate(list(self.edges_attributes)):
            key = (edge[0],edge[1],edge[2]) 
            
            node_att_dict = dict() 
            
            if key in edge_data:
                data_edge =edge_data[key]
                node_att_dict["speed_mean"]= data_edge["mean"]

                node_att_dict["weighted_mean"]= float(data_edge["weighted_mean"])
                self.stats_edge(edge_stats,'weighted_mean',data_edge["weighted_mean"])
                if float(data_edge["weighted_mean"])<0:
                    print("weighted_mean  ",data_edge["weighted_mean"],"\t\t\t\tTT:\t",node_att_dict["weighted_mean"],data_edge,'\n\n\n\n')
                
                node_att_dict["travel_time"]= float(data_edge["travel_time"])
                if (float(data_edge["travel_time"])<0):
                    print("\t\t\t\tTT:\t",node_att_dict["travel_time"])
                self.stats_edge(edge_stats,'travel_time',float(data_edge["travel_time"]))
                
                node_att_dict["traveled"]= 1
                node_att_dict["vehicles_id"]= float(data_edge["vehicles_id"])
                
                attrs[key]= node_att_dict
            else:
                node_att_dict["speed_mean"]= np.nan
                node_att_dict["weighted_mean"]= np.nan
                node_att_dict["travel_time"]= np.nan
                node_att_dict["traveled"]= 0
                node_att_dict["vehicles_id"]= np.nan
                attrs[key]= node_att_dict

        sampled_geograph = self.geograph.copy()
        nx.set_edge_attributes(sampled_geograph, attrs)

        if save:
            adj_matrix = nx.adjacency_matrix(sampled_geograph)
            att_matrix = dict()
            att_matrix['speed_mean'] = nx.adjacency_matrix(sampled_geograph, nodelist=sorted(sampled_geograph.nodes()), weight="speed_mean")
            att_matrix['weighted_mean'] = nx.adjacency_matrix(sampled_geograph, nodelist=sorted(sampled_geograph.nodes()), weight="weighted_mean")
            att_matrix['travel_time'] = nx.adjacency_matrix(sampled_geograph, nodelist=sorted(sampled_geograph.nodes()), weight="travel_time")
            att_matrix['traveled'] = nx.adjacency_matrix(sampled_geograph, nodelist=sorted(sampled_geograph.nodes()), weight="traveled")
            att_matrix['vehicles_id'] = nx.adjacency_matrix(sampled_geograph, nodelist=sorted(sampled_geograph.nodes()), weight="vehicles_id")
            
            self.sparsematrix_save(adj_matrix, sampled_dir, sample_id, 'matrix.network_adj')
            self.sparsematrix_save(att_matrix['speed_mean'], sampled_dir, sample_id, 'matrix.speed_mean')
            self.sparsematrix_save(att_matrix['weighted_mean'], sampled_dir, sample_id, 'matrix.weighted_mean')
            self.sparsematrix_save(att_matrix['speed_mean'], sampled_dir, sample_id, 'matrix.network')
            self.sparsematrix_save(att_matrix['travel_time'], sampled_dir, sample_id, 'matrix.travel_time')
            self.sparsematrix_save(att_matrix['traveled'], sampled_dir, sample_id, 'matrix.traveled')
            self.sparsematrix_save(att_matrix['vehicles_id'], sampled_dir, sample_id, 'matrix.vehicles_id')
            sample_id_str = f"{sample_id}"
            edge_stats_path = Path(sampled_dir,self.simulation_name+'.random.'+sample_id_str+'.edgestats.json')            
            with open(edge_stats_path, 'w') as outfile:
                json.dump(edge_stats, outfile, indent=4)
            sampled_geograph_path = Path(sampled_dir,self.simulation_name+'.random.'+sample_id_str+'.graph.graphml')
            ox.save_graphml(sampled_geograph, filepath=sampled_geograph_path)
            if draw_graph:
                sampledgraph_visual = FlowVisualization(self.simulation_name, sampled_dir, sample_id, load_data=False, sampled_geograph=sampled_geograph, edge_stats=edge_stats)
                sampledgraph_visual.draw_sampledgraph(attr="travel_time")
                sampledgraph_visual.draw_sampledgraph(attr="weighted_mean")
                sampledgraph_visual.draw_sampledgraph(attr="vehicles_id")


        
    def stats_edge(self, dict_stats, attr, val):
        dict_stats[attr]['max']= max(dict_stats[attr]['max'],val)
        dict_stats[attr]['min']= min(dict_stats[attr]['min'],val)

    def sparsematrix_save(self, matrix, sampled_dir, sample_id, matrix_name):
        sample_id_str = f"{sample_id}"
        matrix_path = Path(sampled_dir,self.simulation_name+'.random.'+sample_id_str+'.'+matrix_name+'.npz')
        sparse.save_npz(matrix_path, matrix)

    def sam_save(self, matrix, sampled_dir, sample_id, matrix_name):
        sample_id_str = f"{sample_id}"
        matrix_path = Path(sampled_dir,self.simulation_name+'.random.'+sample_id_str+'.'+matrix_name+'.npz')
        sparse.save_npz(matrix_path, matrix)
                
class FlowSampling_Exception__alredyRead(Exception):
    """Exception raised when errors?"""
    def __init__(self,a,b):
        self.a = a
        self.b = b
          
    def __str__(self):
        return f"FlowSampling: Errors of error."

class FlowSampling_Exception__alredySampleExist(Exception):
    """Exception raised when errors?"""
    def __init__(self,sample):
        self.sample = sample
          
    def __str__(self):
        return f"FlowSampling: Sample '{self.sample}' alredy exist."