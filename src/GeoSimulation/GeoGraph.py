import osmnx as ox
import networkx as nx
import os 
import geopandas
import pandas as pd
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from collections import OrderedDict

class OSMgeometry:
    options = ["nature","landuse","artificial_1","artificial_2","public_1","public_2","highway","military","amenity", "historic", "tourism","all"]

    
    def getOptions():
        option = ["nature","landuse","artificial_1","artificial_2","public_1","public_2","highway","military","amenity", "historic", "tourism","all"]
        return option
    
    def isFilter(name_filter):
        option = OSMgeometry.getOptions()
        filters = OSMgeometry.getFilter()
        if name_filter in option:
            key =  f"{name_filter}"
            value = (key,filters[key])
        else:
            value = None
        return value

    def getFilter():
        poi_filter = {
            "nature":{'natural':True,'geological':True},#'place':True,
            "landuse":{'landuse':True},
            "artificial_1":{'aeroway':True,'aerialway':True,'cycleway':True,'bus_bay':True},
            "artificial_2":{'public_transport':True,'railway':True,'bridge':True,'tunnel':True,'waterway':True},
            "public_1":{'amenity':True,'building':True,'emergency':True,'lifeguard':True,'historic':True},
            "public_2":{'leisure':True,'office':True,'shop':True,'sport':True,'tourism':True},
            "highway":{'highway':True,'route':True},

            "military":{'military':True},  
            "amenity":{'amenity':True},
            "historic": {'historic':True},
            "tourism":{'tourism':True},
        }
        return poi_filter

class GeoGraph:
    def __init__(self, geo_maps_settings, plotGraph=True):
        ox.config(log_console=False, use_cache=True)

        self.maps_name = geo_maps_settings["osm_maps_name"] 

        if "map_folder" in geo_maps_settings:
            if geo_maps_settings["map_folder"] is None:
                self.map_folder = Path('data', 'maps',self.maps_name)
            else:
                self.map_folder = geo_maps_settings["map_folder"]
        else:
            self.map_folder =  Path('data', 'maps',self.maps_name)
        if not os.path.exists(self.map_folder):
            os.makedirs(self.map_folder)
        
        if "poi_folder" in geo_maps_settings:
            if geo_maps_settings["poi_folder"] is None:
                self.poi_folder = Path('data', 'osm_maps')
            else:
                self.poi_folder = geo_maps_settings["poi_folder"]
        else:
            self.poi_folder =  Path('data', 'osm_maps')
        if not os.path.exists(self.poi_folder):
            os.makedirs(self.poi_folder)

        simplification = geo_maps_settings["options"]["simplification"]
        if "road_type" in geo_maps_settings["options"]:
            network_type = geo_maps_settings["options"]["road_type"]
        else:
            network_type = "drive"
        
        if "placetype" in geo_maps_settings["options"]:
            self.placetype = geo_maps_settings["options"]["placetype"]
            if "dist_point" in geo_maps_settings["options"]:
                self.dist_point = geo_maps_settings["options"]["dist_point"]
            else:
                self.dist_point = 500
        if geo_maps_settings["options"]["placetype"] == "cities":
            self.places = geo_maps_settings["options"]["places"]
        elif geo_maps_settings["options"]["placetype"] == "coordinates":
            self.places = self.coordinates2cities(geo_maps_settings["options"]["places"])
            self.placetype = "cities"
        else:
            self.placetype = "cities"
        
        self.geograph = self.request_graph(self.places, simplification, network_type=network_type)
                
        if geo_maps_settings["options"]["poi_geometry"]:
            self.POI_maps= True
            list_filter = geo_maps_settings["options"]["poi_option"]["filter"]
            if list_filter == "all":
                self.filter_poi = OSMgeometry.getFilter()
            else:
                self.filter_poi = dict()
                for filter_name in list_filter:
                    filter_tuple = OSMgeometry.isFilter(filter_name)   
                    self.filter_poi[filter_tuple[0]] = filter_tuple[1]
            self.poigraph = self.geometries_from_place(self.places,save=True)
        else:
            self.POI_maps= False
        if plotGraph:
            self.drawGraph()

    def getGEO(self):
        return self.geograph

    def getPOI(self):
        return self.poigraph

    def request_graph(self,query,simplification=False, network_type="drive"):
        """
        Negotiates a query to OSMNX if no local stored file else loads local file
        :return: result:
        """
        if self.maps_name+ '.geo.graphml' in os.listdir(self.map_folder):
            geograph = ox.load_graphml(Path(self.map_folder, ""+self.maps_name, '.geo.graphml'))
        else:
            geograph = self.graph_from_place(query, simplification, network_type=network_type)
        return geograph



    ##
    ## placetype="place","point"
    ## dist_point in meters
    ##po
    def graph_from_place(self, place,  simplify=True,network_type='drive', simplification=False, save=True, consolidate_intersections=False):
        """
        Request a graph from OSMNX
        network_type : "all_private", "all", "bike", "drive", "drive_service", "walk"
        :return: G: OSMN Graph object
        """
        # query graph from place
        G = None
        try:
            p_bar = tqdm(range(10))
            p_bar.update(1)
            p_bar.refresh()
            if self.placetype=="cities":
                G = ox.graph_from_place(place, simplify=simplify, network_type=network_type)
            elif self.placetype=="point":
                if len(place)==2:
                    G = ox.graph_from_point(place, dist=self.dist_point, simplify=True)
                else:
                    raise GeoGraph_Exception__Place_Coordinates(place)
            else:
                raise GeoGraph_Exception__Placetype_NotRecognize(self.placetype)
            p_bar.update(10)
            p_bar.refresh()
            if simplification:
                G = ox.simplification.simplify_graph(G, strict=True, remove_rings=True, clean_periphery =True)
            if consolidate_intersections:
                G_consolidated = ox.consolidate_intersections(G, rebuild_graph=True, tolerance=15, dead_ends=True)
                #G = ox.project_graph(G_consolidated, to_crs='epsg:4326')
                G = nx.convert_node_labels_to_integers(G_consolidated)
            if save:
                geo_filename = Path(self.map_folder, self.maps_name)
                ox.save_graphml(G, filepath=Path(geo_filename.with_suffix(".geo.geojson")))
                ox.save_graph_xml(G, filepath=Path(geo_filename.with_suffix(".geo.osm") ))
        except Exception:
            raise GeoGraph_Exception__Param(place)
        return G


    def geometries_from_place(self, places, tags={'natural':True,'place':True},save=False):
        POI_list_all = []
        for key in self.filter_poi:
            tags = self.filter_poi[key]
            POI_list = []
            for i in tqdm(range(len(places))):
                _place = places[i]
                place = _place.lower().replace(',', '_').replace(' ', '-')
                poi_filepath = Path(self.poi_folder, (place+"."+key+".poi.geojson"))
                if poi_filepath in os.listdir(self.map_folder):
                    _poifile = geopandas.read_file(poi_filepath)
                    POI_list.append(_poifile)
                    POI_list_all.append(_poifile)
                else:
                    try:
                        ox.config(timeout=10000)
                        _poifile = ox.geometries_from_place(_place,tags=tags)
                        POI_list.append(_poifile)
                        POI_list_all.append(_poifile)
                        if save:
                            with open(poi_filepath, "w") as f:
                                f.write(_poifile.to_json())
                    except Exception:
                        raise GeoGraph_Exception__POI(_place)
            poi_geodf = geopandas.GeoDataFrame(pd.concat(POI_list,ignore_index=True))
            if save:
                global_poi_filepath = Path(self.map_folder, (self.maps_name+".poi."+key))
                with open(global_poi_filepath.with_suffix(global_poi_filepath.suffix+".geojson"), "w") as f:
                    f.write(poi_geodf.to_json())
                
                sumo_cmd = f"ogr2osm {global_poi_filepath}.geojson --output={global_poi_filepath}.osm --force"
                os.system(sumo_cmd)
        if save:
            poi_all_geodf = geopandas.GeoDataFrame(pd.concat(POI_list_all,ignore_index=True))
            global_poi_filepath_all = Path(self.map_folder, (self.maps_name+".poi.all"))
            with open(global_poi_filepath_all.with_suffix(global_poi_filepath_all.suffix+".geojson"), "w") as f:
                f.write(poi_all_geodf.to_json())                
            sumo_cmd = f"ogr2osm {global_poi_filepath_all}.geojson --output={global_poi_filepath_all}.osm --force"
            os.system(sumo_cmd)
        return poi_geodf

    def drawGraph(self):
        fig, ax = plt.subplots(figsize=(25,18))
        if self.POI_maps:
            POI_geo = self.getPOI()
            POI_geo.plot(ax=ax, facecolor='khaki', alpha=0.7)
        GEO_geo = self.getGEO()           
        ox.plot_graph(GEO_geo, ax=ax, node_size=0, edge_linewidth=0.5,show=False)
        mapsploth_filename = Path(self.map_folder, self.maps_name+".geo.png")
        fig.savefig(mapsploth_filename)


    def graph_axis(self,show=False):
        """
        projects graph geometry and plots figure, retrieving an axis
        :return: self.fig, self.axis, ax, graph
        """
        # project and plot
        graph = ox.project_graph(self.geograph)
        fig, ax = ox.plot_graph(graph, node_size=0, edge_linewidth=0.5,
                                show=show,
                                bgcolor='#FFFFFF')
        # set the axis title and grab the dimensions of the figure
        self.fig = fig
        ax.set_title(self.maps_name)
        self.axis = ax.axis()
        return ax, graph

    def coordinates2cities(self, list_coordinates):
        cities_list = []
        for (lat,lon) in list_coordinates:
            par = OrderedDict({'lat':str(lat),'lon':str(lon),"format": "json"})
            nom_req = ox.downloader.nominatim_request(params =par, request_type="reverse")
            city_name = f"{nom_req['address']['city']},{nom_req['address']['country']}"
            if city_name not in cities_list:
                cities_list.append(city_name)
        print(cities_list)
        return cities_list


class GeoGraph_Exception__Param(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,instance):
        self.instance = instance
          
    def __str__(self):
        return f"No graph found for '{self.instance}' location. Please try a geo-codable place from OpenStreetMaps."

class GeoGraph_Exception__POI(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,instance):
        self.instance = instance
          
    def __str__(self):
        return f"No POI-DATA found for '{self.instance}' location. Please try a geo-codable place from OpenStreetMaps."

class GeoGraph_Exception__Placetype_NotRecognize(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,placetype):
        self.placetype = placetype
          
    def __str__(self):
        return f"No placetype found for '{self.placetype}' location. Placetype can be 'place' or 'point'."
    

class GeoGraph_Exception__Place_Coordinates(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,place):
        self.place = place
          
    def __str__(self):
        return f"Place shoul be a geo coordinate (x,y)."
    
