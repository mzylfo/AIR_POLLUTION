import pandas as pd
import numpy as np
import folium
from folium import plugins
import matplotlib
from matplotlib import colormaps
import math
from pathlib import Path
import os

class ScenariosMap():
    
    def __init__(self, data_range, vc_mapping, path_folder_map, path_folder, instance_file, label="x_output"):
        self.case_map = pd.read_csv(path_folder_map)
        self.label = label
        self.vc_mapping = vc_mapping
        points_list = list()
        for xs in self.case_map['points']:
            points = list()
            xxs = (xs[2:-2].split("), ("))
            for item in xxs:
                x = [float(x) for x in item.split(", ")]
                x_tuple = (x[0], x[1])
                points.append(x_tuple)
            points_list.append(points)
        self.case_map['points'] = points_list
        
        self.path_folder = Path(path_folder,"scenarios_map")
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
            
        
        path_folder_istances = Path(path_folder,f"{instance_file}.csv")
        self.scenarios = pd.read_csv(path_folder_istances)
        self.max_val = data_range["max_val"]
        self.min_val = data_range["min_val"]
        self.center = (np.mean([x[0] for xs in self.case_map['points'] for x in xs]), np.mean([x[1] for xs in self.case_map['points'] for x in xs]))
        
    def get_route_color(self, value):
        cmap = colormaps['RdYlGn']
        value_x = (value - self.min_val)/(self.max_val - self.min_val)
        if value_x<=0:
            value_x=0
        elif value_x>=1:
            value_x = 1
        rgba = cmap(value_x)
        
        return matplotlib.colors.rgb2hex(rgba)    
    

    def draw_scenario(self, n_scenario):
        scenario_list = [float(x) for x in self.scenarios[self.label][n_scenario][1:-1].split(", ")]
        scenario = dict()
        for road, value in zip(self.vc_mapping, scenario_list):
            scenario[road] = value
            
        scenario_roads = folium.FeatureGroup(name=f"scenario__{n_scenario}",overlay=False).add_to(self.maps_scenario)
        for index, row in self.case_map.iterrows():
            
            value = scenario[row["name_road"]]
            route_color = self.get_route_color(value)
            
            route = folium.PolyLine(row["points"], color=route_color, weight=5, opacity=0.85)
            route.add_to(scenario_roads)
        
                   
            
    def draw_scenarios(self, list_scenarios=None, save_html=True):
        count = 1
        self.maps_scenario = folium.Map(self.center,  zoom_start=11)
        tile_layer = folium.TileLayer(tiles="https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png", attr='darkmatter', max_zoom=19, name='darkmatter', control=False, opacity=0.7)
        tile_layer.add_to(self.maps_scenario)

        
        if list_scenarios is None:
            list_scenarios = [i for i in range(1, len(self.scenarios))]        
        
        for i in list_scenarios:
            self.draw_scenario(n_scenario=i)
            if count%10 == 0:
                if save_html:
                    self.maps_scenario.add_child(folium.LayerControl(position='topright', collapsed=False, autoZIndex=True))
                    folder_html_file = Path(self.path_folder , f"scenarios_map__{i}.html")
                    self.maps_scenario.save(folder_html_file)
                
                self.maps_scenario = folium.Map(self.center,  zoom_start=11)
                tile_layer = folium.TileLayer(tiles="https://{s}.basemaps.cartocdn.com/rastertiles/dark_all/{z}/{x}/{y}.png", attr='darkmatter', max_zoom=19, name='darkmatter', control=False, opacity=0.7)
                tile_layer.add_to(self.maps_scenario)
                count = 0
            count += 1