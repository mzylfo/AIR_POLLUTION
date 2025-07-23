import pandas as pd
import folium
from folium.plugins import HeatMap
import branca.colormap as cm
import matplotlib
import glob, os
from pathlib import Path
import csv
import vincent
from vincent import Bar
import vincent
from vincent import AxisProperties, PropertySet, ValueRef
import numpy as np
import json
import statistics
from folium import features
import folium
import re

class milanoMap():
    
    def __init__(self):
        self.path_folder = Path("data","dataset","milano_fluctuo","2023","12")
        self.path_file_csv = Path(self.path_folder,"202312_trips.csv")
        self.path_file_map = Path(self.path_folder,"202312_maps.html")
        print("-- load csv trips file: --",self.path_file_csv)
        self.df_trips = pd.read_csv(self.path_file_csv)
    
    def compute_trips(self):
        print("-- compute trips --")
        self.center_coordinates = {"lat":list(), "lon":list()}
        self.trips_data = dict()
        for index, row in self.df_trips.iterrows():
            self.trips_data[index] = {
                "type_vehicle":row['type_vehicle'],
                "local_ts_start":row['local_ts_start'],
                "local_ts_end":row['local_ts_end'],
                "estimated_duration_in_mn":row['estimated_duration_in_mn'],
                "estimated_distance_in_meter":row['estimated_distance_in_meter'],
                "geom_wkt_estimated_route":list()
            }
            multilinestring = row['geom_wkt_estimated_route']
            matches = re.findall(r'\(\(([^)]+)\)\)', multilinestring)
            for match in matches:
                pairs = match.split(',')
                for pair in pairs:
                    lon, lat = map(float, pair.split())
                    self.center_coordinates['lat'].append(lat)
                    self.center_coordinates['lon'].append(lon)
                    self.trips_data[index]["geom_wkt_estimated_route"].append((lat, lon))
                    
    def compute_map(self):
        print("-- compute map: --",self.path_file_map)
        milan__map = folium.Map(location=[statistics.mean(self.center_coordinates["lat"]), statistics.mean(self.center_coordinates["lon"])], zoom_start=15)
        for key in self.trips_data:
            if self.trips_data[key]["type_vehicle"] == "C":    #Car
                cls = "blue"
            elif self.trips_data[key]["type_vehicle"] == "B":  #Bike
                cls = "yellow"
            elif self.trips_data[key]["type_vehicle"] == "M":  #Moped
                cls = "geen"
            elif self.trips_data[key]["type_vehicle"] == "B":  #Scooter
                cls = "red"
            else:
                cls = "purple"
            folium.PolyLine(self.trips_data[key]["geom_wkt_estimated_route"], color=cls, weight=2.5).add_to(milan__map)
        print("-- save map at: --",self.path_file_map)
        milan__map.save(self.path_file_map)

if __name__ == "__main__":
    milano = milanoMap()
    milano.compute_trips()
    milano.compute_map()