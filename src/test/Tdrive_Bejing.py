from src.GeoSimulation.GeoGraph import *
import matplotlib.pyplot as plt
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
import networkx as nx
import os 
import json
import skmob
import pandas as pd
from skmob.preprocessing import detection
from ast import literal_eval as make_tuple
import pickle

class Tdrive_Bejing():


    def __init__(self, pathfolder, pathinput, filename_save, maps_name="Bejing", users_list=[1], download_maps=False, line_break = None, all_city=True):
        ox.config(log_console=True, use_cache=True)
        self.pathfolder = pathfolder
        self.pathinput = pathinput
        if not os.path.exists(self.pathfolder):
            os.makedirs(self.pathfolder)
        self.pathfolder_plots = Path(self.pathfolder,"plots")
        if not os.path.exists(self.pathfolder_plots):
            os.makedirs(self.pathfolder_plots)
        self.users_list = users_list
        self.maps_name = maps_name
        print(f"We are setted line_break to: ", line_break)
        
        self.filename_save = filename_save
        filename_positions_graph = Path(self.pathfolder,(self.filename_save+"positions_graph.pkl"))
        filename_positions_geo = Path(self.pathfolder,(self.filename_save+"positions_geo.pkl"))
        
        if os.path.exists(filename_positions_graph) and os.path.exists(filename_positions_geo):
            with open(filename_positions_graph, 'rb') as f_pgr:
                self.positions_graph = pickle.load(f_pgr)
            with open(filename_positions_geo, 'rb') as f_pge:
                self.positions_geo = pickle.load(f_pge)
        else:
            users_data = self.get_usersdata(self.users_list, line_break=line_break, save=True, filename=self.filename_save)
            self.positions_geo = users_data[0]
            self.positions_graph = users_data[1]
        
        self.positions_graph_count = dict()

        if download_maps:
            self.from_bejing_files(places_list=self.positions_geo)
        
        self.geograph = self.loadGraph(self.maps_name)


    def from_bejing_files(self, places_list, POI_maps=False, draw_maps=False, all_bejing=True):    
        if all_bejing:
            geo_maps_settings={
                "osm_maps_name":self.maps_name,
                "map_folder":None,
                "poi_folder":None,
                "options":{
                    "places":['Bejing,China'], 
                    "placetype":"cities",
                    "simplification":True,
                    "road_type":"drive",
                    "poi_geometry":False, 
                    "poi_option":{
                        "filter":[]#"nature","landuse","artificial_1","artificial_2","public_1","public_2","highway","military"
                    }
                }
            }
        else:
            geo_maps_settings={
                "osm_maps_name":self.maps_name,
                "map_folder":None,
                "poi_folder":None,
                "options":{
                    "places":places_list, 
                    "placetype":"coordinates",
                    "simplification":True,
                    "road_type":"drive",
                    "poi_geometry":False, 
                    "poi_option":{
                        "filter":[]#"nature","landuse","artificial_1","artificial_2","public_1","public_2","highway","military"
                    }
                }
            }
        geo_settings = GeoGraph(geo_maps_settings=geo_maps_settings)
        if draw_maps:
            if POI_maps:
                POI_geo = geo_settings.getPOI()
            GEO_geo = geo_settings.getGEO()    
            if POI_maps:
                POI_geo.plot(ax=ax, facecolor='khaki', alpha=0.7)
            fig, ax = ox.plot_graph(GEO_geo, ax=ax, node_size=0, edge_linewidth=0.5,show=False, figsize= (150,150), dpi=300)
            mapsploth_filename = Path(self.pathfolder, self.maps_name+".geo.png")
            fig.savefig(mapsploth_filename)


    def get_usersdata(self, userid_list, line_break=None, save=True, filename=None):
        positions_geo = []
        positions_graph = {}
        for user_id in userid_list:
            positions_graph[user_id] = dict()
            inputfile = Path(self.pathinput, f"{user_id}.txt")
            
            df = pd.read_csv(inputfile, sep=',',header=None, names=['user','datetime','lon','lat'])
            tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lon', user_id='user', datetime='datetime')
            
            if len(tdf)>0:
                stdf = detection.stay_locations(tdf, stop_radius_factor=0.5, minutes_for_a_stop=5.0, spatial_radius_km=0.2, leaving_time=True, no_data_for_minutes=10.0)
                stdf = stdf.rename(columns={'datetime':'arriving_time','uid':'user','lng':'lon','leaving_datetime':'newpos_time'})
                if len(stdf)>0:
                    datetime_list =  tdf.sort_values(by=['datetime'])['datetime'].tolist()


                    last_seen = []


                    for index, row in stdf.iterrows():
                        if index>0 and index<len(stdf):
                            lastseen_index = datetime_list.index(row['arriving_time'])-1
                            last_seen.append(datetime_list[lastseen_index])
                        elif index==0:
                            lastseen_index = datetime_list.index(row['arriving_time'])
                            last_seen.append(datetime_list[lastseen_index])

                    stdf['last_seen'] = last_seen


                    for index, row in stdf.iterrows():
                        if line_break is not None and index>line_break:
                            break
                        veh = row['user']
                        arriving_time = datetime.strptime(str(row['arriving_time']), "%Y-%m-%d %H:%M:%S")
                        leaving_time = datetime.strptime(str(row['last_seen']), "%Y-%m-%d %H:%M:%S")
                        lat = row['lat']
                        lon = row['lon']
                            
                        positions_geo.append((lat,lon))

                        positions_graph[user_id][index] = dict()
                        positions_graph[user_id][index]['arriving_time'] = arriving_time
                        positions_graph[user_id][index]['leaving_time'] = leaving_time
                        positions_graph[user_id][index]['lat'] = float(lat)
                        positions_graph[user_id][index]['lon'] = float(lon)
                        positions_graph[user_id][index]['veh'] = veh
        
        print("end")
        if save:
            filename_positions_graph = Path(self.pathfolder,(filename+"positions_graph.pkl"))
            with open(filename_positions_graph, 'wb') as f:
                pickle.dump(positions_graph, f, protocol=pickle.HIGHEST_PROTOCOL)  
                
            
            filename_positions_geo = Path(self.pathfolder,(filename+"positions_geo.pkl"))
            with open(filename, 'wb') as f:
                pickle.dump(positions_geo, f, protocol=pickle.HIGHEST_PROTOCOL)  
                

        return [positions_geo,positions_graph]


    def loadGraph(self, maps_name):
        geo_filename = Path("data","maps", maps_name, maps_name+".geo.geojson")
        return ox.load_graphml(geo_filename)


    def point2Node(self, geograph, y_lat, x_lon, get_nearest_nodes=True, get_nearest_edges=False):
        position_info = dict()
        if get_nearest_nodes:
            (nn, nn_dist) = ox.nearest_nodes(G=geograph, X=x_lon, Y=y_lat, return_dist=True) 
            position_info["nn_id"]= nn
            position_info["nn_dist"] = nn_dist
        else:
            position_info["nn_id"]= None
            position_info["nn_dist"] = None

        if get_nearest_edges:
            ((ne_u, ne_v, ne_key), ne_dist) = ox.nearest_edges(G=geograph, X=x_lon, Y=y_lat, return_dist=True)  
            position_info["ne_u_id"]= ne_u
            position_info["ne_v_id"] = ne_v
            position_info["ne_key"] = ne_key
            position_info["ne_dist"] = ne_dist
        else:
            position_info["ne_u_id"]= None
            position_info["ne_v_id"] = None
            position_info["ne_key"] = None
            position_info["ne_dist"] = None
        
        return position_info


    def Haversine_distance(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        # Radius of earth in kilometers is 6371
        km = 6371* c
        m = km *1000
        return m


    def coordinates2Nodes(self, users_list=None, radius_subgraph=10000):
        print("coordinates2Nodes")
        radius_check_subgraph = radius_subgraph*0.70 #70%
        if users_list is None:
            users_list = self.users_list
        for user_id in users_list:
            print(user_id)
            filename_positions_graph = Path(self.pathfolder,self.maps_name+"."+str(user_id)+".positions_graph.pickle")
            filename_positions_graph_count = Path(self.pathfolder,self.maps_name+"."+str(user_id)+".positions_graph_count.pickle")
            if os.path.exists(filename_positions_graph_count) and os.path.exists(filename_positions_graph_count):
                
                with open(filename_positions_graph, 'rb') as f_pgr:
                    self.positions_graph[user_id] = pickle.load(f_pgr)
                with open(filename_positions_graph_count, 'rb') as f_pge:
                    self.positions_graph_count[user_id] = pickle.load(f_pge)
                    save_pickle = False

            elif len(self.positions_graph[user_id])>0:
                print("id_user: ",user_id)
                
                subedge_lon = self.positions_graph[user_id][0]['lon']
                subedge_lat = self.positions_graph[user_id][0]['lat']
                nn = ox.nearest_nodes(G=self.geograph, X=subedge_lon, Y=subedge_lat, return_dist=False) 

                eg = nx.ego_graph(G=self.geograph, n=nn, radius=radius_subgraph, distance='length')
                lin = 0
                print("positions_graph_count ",user_id)
                self.positions_graph_count[user_id] = lin

                for id_istance in self.positions_graph[user_id]:
                    
                    try:
                        #istance = self.positions_graph[user_id][id_istance]
                        lon = self.positions_graph[user_id][lin]['lon']
                        lat = self.positions_graph[user_id][lin]['lat']
                        dist = self.Haversine_distance(subedge_lon, subedge_lat, lon, lat)
                        
                        #if point over subgraph, make a new subgraph
                        if dist>radius_check_subgraph:
                            subedge_lon = lon
                            subedge_lat = lat
                            nn = ox.nearest_nodes(G=self.geograph, X=subedge_lon, Y=subedge_lat, return_dist=False) 
                            eg = nx.ego_graph(G=self.geograph,  n=nn, radius=10000, distance="length")
                        

                        nodeInfo = self.point2Node(eg, lat, lon, get_nearest_edges=True)
                        self.positions_graph[user_id][lin]["nn_id"] = nodeInfo["nn_id"]
                        self.positions_graph[user_id][lin]["nn_dist"] = nodeInfo["nn_dist"]
                        self.positions_graph[user_id][lin]["ne_u_id"] = nodeInfo["ne_u_id"]
                        self.positions_graph[user_id][lin]["ne_v_id"] = nodeInfo["ne_v_id"]
                        self.positions_graph[user_id][lin]["ne_key"] = nodeInfo["ne_key"]
                        self.positions_graph[user_id][lin]["ne_dist"] = nodeInfo["ne_dist"]
                        lin += 1
                        self.positions_graph_count[user_id] = lin
                        
                    except:                        
                        pass
                save_pickle = True
            else:
                print("positions_graph_count ",user_id)
                self.positions_graph_count[user_id] = 0
                self.positions_graph[user_id] = {}
                save_pickle = True
            
            if save_pickle:                        
                with open(filename_positions_graph, 'wb') as f:
                    pickle.dump(self.positions_graph[user_id], f)  

                with open(filename_positions_graph_count, 'wb') as f:
                    pickle.dump(self.positions_graph_count[user_id], f, protocol=pickle.HIGHEST_PROTOCOL)  




    def compute_roads(self,users_list=None):
        print("compute_roads")
        self.coordinates2Nodes(users_list)
        self.road_speed = dict()
        self.road_traveltime = dict()
        self.road_error = list()
        self.road_info = dict()
        for edge_data, edge_attr in ox.graph_to_gdfs(self.geograph, nodes=False).fillna('').iterrows():            
            edge = (edge_data[0],edge_data[1])
            
            self.road_info[edge] = {"length":edge_attr['length']}
            
            

        if users_list is None:
            users_list = self.users_list
        
        for user_id in users_list:
            num_instance = self.positions_graph_count[user_id]
        
            for id_istance in range(num_instance-1):
                print(id_istance," - ",len(self.positions_graph[user_id]))
                self.positions_graph[user_id][id_istance]['road'] = dict()
                self.positions_graph[user_id][id_istance]['road']['node_orig'] = self.positions_graph[user_id][id_istance]["nn_id"] 
                self.positions_graph[user_id][id_istance]['road']['time_orig'] = self.positions_graph[user_id][id_istance]["leaving_time"] 
                self.positions_graph[user_id][id_istance]['road']['node_dest'] = self.positions_graph[user_id][id_istance+1]["nn_id"] 
                self.positions_graph[user_id][id_istance]['road']['time_dest'] = self.positions_graph[user_id][id_istance+1]["arriving_time"] 
                time_interval = self.positions_graph[user_id][id_istance]["leaving_time"] 
                node_orig = self.positions_graph[user_id][id_istance]['road']['node_orig']
                node_dest = self.positions_graph[user_id][id_istance]['road']['node_dest']
                try:
                    path_route  = nx.shortest_path(G=self.geograph, source=node_orig, target=node_dest)
                    path_length = nx.shortest_path_length(G=self.geograph, source=node_orig, target=node_dest, weight='length')
                    path_edges = list(zip(path_route,path_route[1:]))
                    path_time = (self.positions_graph[user_id][id_istance]['road']['time_dest'] - self.positions_graph[user_id][id_istance]['road']['time_orig']).total_seconds()
                    if path_time!= 0:
                        path_speed = float(path_length/path_time)
                    else:
                        path_speed = 0
                except nx.exception.NetworkXNoPath:
                    error_count = len(self.road_error)/2
                    self.road_error.append((node_orig, ">",error_count))
                    self.road_error.append((node_dest, "<",error_count))
                    path_edges = []
                    path_length = -1
                    path_time = -1
                    path_speed =-1

                self.positions_graph[user_id][id_istance]['road']['length'] = path_length
                self.positions_graph[user_id][id_istance]['road']['edges'] = path_edges
                
                self.positions_graph[user_id][id_istance]['road']['time'] = path_time
                self.positions_graph[user_id][id_istance]['road']['speed'] = path_speed

                for edge in path_edges:
                    
                    if str(edge) in self.road_speed:
                        self.road_speed[str(edge)].append({"speed":str(path_speed),"time":str(time_interval),"userid":str(user_id), 'length':self.road_info[edge]['length']})
                    else:
                        self.road_speed[str(edge)] = [{"speed":str(path_speed),"time":str(time_interval),"userid":str(user_id), 'length':self.road_info[edge]['length']}]
        
            filename_pickle = Path(self.pathfolder,self.maps_name+"."+str(user_id)+".roads_speed.pickle")
            with open(filename_pickle, 'wb') as f:
                pickle.dump(self.road_speed, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        
            dfObj = pd.DataFrame()
            nodeA_list = list()
            nodeB_list = list()
            speed_list = list()
            time_list = list()
            userid_list = list()
            length_list = list()

            for road in self.road_speed:
                edge = make_tuple(road)
                for value in self.road_speed[road]:
                    
                    nodeA_list.append(edge[0])
                    nodeB_list.append(edge[1])
                    speed_list.append(value['speed'])
                    time_list.append(value['time'])
                    userid_list.append(value['userid'])
                    length_list.append(value['length'])

            dfObj['nodeA'] = nodeA_list
            dfObj['nodeB'] = nodeB_list
            dfObj['speed'] = speed_list
            dfObj['time'] = time_list
            dfObj['userid'] = userid_list
            dfObj['length'] = length_list


            filename_pd = Path(self.pathfolder,self.maps_name+"."+str(user_id)+".roads_speed.json")
            dfObj.to_csv(filename_pd, sep='\t', encoding='utf-8')



    def plot_point(self, user_id):
        edges_list = list()
        nodes_list = list()
        nodes_dict = dict()
        #routes_nn_roads = list()
        #routes_nn_roads_list = list()
        num_instance = self.positions_graph_count[user_id]
        
        for id_istance in range(num_instance-1):
            
            istance = self.positions_graph[user_id][id_istance]
            
            edges_list.append((istance["ne_u_id"],istance["ne_v_id"]))
            #edges_list.append((istance["ne_v_id"],istance["ne_u_id"]))
            nodes_list.append(istance["nn_id"])

            nn_actual = istance["nn_id"]
            nn_time = istance['leaving_time'].strftime("%Y-%m-%d %H:%M:%S")
            if nn_actual in nodes_dict:
                _time = nodes_dict[nn_actual]
                
                nodes_dict[nn_actual] = f"{_time}, {nn_time}"
            else:
                nodes_dict[nn_actual] = f"{nn_time}"

        #    routes_nn_roads.append(istance['road']['edges'])

        #for points_roads in routes_nn_roads:
        #    for points_road in points_roads:
        #        routes_nn_roads_list.append(points_road)

        edges_col = [] 
        edges_lnw = []
        for u, v, k in self.geograph.edges(keys=True):
            if (u,v) in edges_list or (v,u) in edges_list:
                edges_col.append('violet')
                edges_lnw.append(2.0)
            elif (u,v) in self.road_speed or (v,u) in self.road_speed:
                edges_col.append('palegreen')
                edges_lnw.append(2.0)
            else:
                edges_col.append('gray')
                edges_lnw.append(0.1)

        nodes_col = []
        nodes_siz = []
        for n in self.geograph.nodes():
            if self.check_itemTuple(n,self.road_error, 0):
                if self.check_itemTuple(">",self.road_error, 1):
                    nodes_col.append('darkorange')
                    nodes_siz.append(20.0)
                else:
                    nodes_col.append('gold')
                    nodes_siz.append(20.0)

            elif n in self.positions_graph[user_id]:
                nodes_col.append('mediumblue')
                nodes_siz.append(2.0)
            else:
                nodes_col.append('darkgray')
                nodes_siz.append(0.5)

        fig, ax = ox.plot_graph(self.geograph, node_size=nodes_siz, node_color=nodes_col, edge_linewidth=edges_lnw, edge_color=edges_col, close=False, show=False, figsize= (150,150), dpi=300)
        mapsploth_filename = Path(self.pathfolder_plots, self.maps_name+f".{user_id}.trips.png")
        fig.savefig(mapsploth_filename)

        for id_istance in range(num_instance):
            
            istance = self.positions_graph[user_id][id_istance]           
            ax.scatter(istance['lon'], istance['lat'], c='orangered', s=2, marker='x')

        mapsploth_filename = Path(self.pathfolder_plots, self.maps_name+f".{user_id}.scatter.png")
        fig.savefig(mapsploth_filename)
       
        for edge_data, edge_attr in ox.graph_to_gdfs(self.geograph, nodes=False).fillna('').iterrows():            
            edge = (edge_data[0],edge_data[1])
            if str(edge) in self.road_speed:
                c = edge_attr['geometry'].centroid                
                c_y_lenght = c.y - 0.0001
                text_edge_length = f"{edge_attr['length']}"
                ax.annotate(text_edge_length, (c.x, c_y_lenght), c='darkgreen')


                text_edge_speed = ', '.join([str(round(float(_speed['speed']), 2)) for _speed in self.road_speed[str(edge)]])
                c_y_speed = c.y +  0.0001
                ax.annotate(text_edge_speed, (c.x, c_y_speed), c='darkcyan')
                
        

        for (node, node_attr) in ox.graph_to_gdfs(self.geograph, edges=False).fillna('').iterrows():            
            if node in nodes_list:
                if self.check_itemTuple(node,self.road_error, 0):
                    text_node = f"{self.road_error[2]}"  
                else:
                    text_node = f"{nodes_dict[node]}"
                c_x = node_attr['x']#+node_attr['x']*0.1
                c_y = node_attr['y']#-node_attr['y']*0.1
                text_node = f"{nodes_dict[node]}"
                ax.annotate(text_node, (c_x, c_y), c='royalblue')
        mapsploth_filename = Path(self.pathfolder_plots, self.maps_name+f".{user_id}.speed.png")
        fig.savefig(mapsploth_filename)


    def check_itemTuple(self, item, in_list, pos):
        for tuple in in_list:
            if tuple[pos] == item:
                return True
        return False

