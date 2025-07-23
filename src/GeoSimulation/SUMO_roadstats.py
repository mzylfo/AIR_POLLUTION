import xml.etree.ElementTree as ET
from xml.dom import minidom
import ast
from dict2xml import dict2xml
import math
import statistics
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd

class SUMO_roadstats():
    def __init__(self, simulation_name, is_osm):
        self.is_osm = is_osm
        self.netFile = Path('data','sumo_simulation_files',simulation_name,simulation_name+'.net.xml')
        self.fcdFile = Path('data','sumo_simulation_files',simulation_name,simulation_name+'.out.fcd.xml')
        self.outFile = Path('data','sumo_simulation_files',simulation_name,simulation_name+'.out.roadstats.xml')
        self.samplingstats = Path('data','sumo_simulation_files',simulation_name,simulation_name+'.out.samplingstats.pkl') 
    
    """
    segmentsMode :
        "fixed" - define a fixed distance, i.e every segment rappresnet 200meters of road if segmentsValue = 200
        "relative" - define a number of segment, i.e. all road have 5 segments if segmentsValue=5
    """
    def compute_roadstats(self, segmentsMode="relative", segmentsValue=5, save_xml=True, save_samplingstats=True,  write_segments=True, write_vehicle=True):
        self.segmentsMode = segmentsMode
        self.segmentsValue = segmentsValue
        print("net2dict")
        self.net2dict(is_osm=self.is_osm)
        print("fcd2dict")
        self.fcd2dict()
        print("saving")
        if save_xml:
            self.saveXML(write_segments=write_segments,write_vehicle=write_vehicle)
        if save_samplingstats:
            self.saveSamplingstats(save=save_samplingstats, path_roadstats= self.samplingstats)

    def net2dict(self, is_osm=False):
        treeNET = ET.parse(self.netFile)
        self.rootNET = treeNET.getroot()
        self.lanes = dict()

        for item in tqdm(self.rootNET):
            if item.tag == "edge":                
                if 'function' not in item.attrib:
                    roadInfo_from = item.attrib['from']
                    roadInfo_to = item.attrib['to']
                    roadInfo_priority = item.attrib['priority']
                    roadInfo_stret = item.attrib['id']
                    for sub_item in item:
                        try:
                            roadInfo = dict()
                            roadInfo['from'] = roadInfo_from
                            roadInfo['to'] = roadInfo_to
                            roadInfo['priority'] = roadInfo_priority
                            if is_osm:
                                roadInfo['street'] = int(roadInfo_stret)*-1
                            else:
                                roadInfo['street'] = roadInfo_stret
                            roadInfo['index'] = sub_item.attrib['index']
                            roadInfo['speed'] = sub_item.attrib['speed']
                            roadInfo['length'] = float(sub_item.attrib['length'])
                            
                            if self.segmentsMode == "relative":
                                segment_length = math.ceil(roadInfo['length']/self.segmentsValue)
                                segment_number = self.segmentsValue
                            elif self.segmentsMode == "fixed":
                                segment_length = self.segmentsValue
                                segmentsValue = math.ceil(roadInfo['length']/self.segmentsValue)
                            else:
                                raise NotImplementedError()

                            roadInfo['segment_length'] = segment_length
                            roadInfo['segment_number'] = segment_number
                            roadInfo['vehicles_trafic'] = dict()
                            for i in range(0, segment_number):
                                roadInfo['vehicles_trafic'][str(i)] = list()
                            lane_id = sub_item.attrib['id']
                            self.lanes[lane_id] = roadInfo
                        
                        except Exception:
                            pass

    def fcd2dict(self):
        treeFCD = ET.parse(self.fcdFile)
        self.rootFCD = treeFCD.getroot()
        
        for item in tqdm(self.rootFCD):    
            if item.tag == "timestep":
                timestep = item.attrib['time']
                for sub_item in item:
                    lane_id = sub_item.attrib['lane']
                    if lane_id in self.lanes:
                        pos_float = float(sub_item.attrib['pos'])
                        if self.lanes[lane_id]['segment_length'] == 0:
                            segment = 0
                        else:
                            segment = math.floor(pos_float/self.lanes[lane_id]['segment_length'])
                        if segment == self.lanes[lane_id]['segment_number']:
                            segment -= 1
                        veh_info = {'vehicle_id':sub_item.attrib['id'], 'speed':sub_item.attrib['speed'], 'position':sub_item.attrib['pos'], 'timestep':timestep}
                        self.lanes[lane_id]['vehicles_trafic'][str(segment)].append(veh_info)

    def saveXML(self, write_segments=True, write_vehicle=True):
        roads_root = ET.Element("roads")
        for lane in self.lanes:
            info_lane = self.lanes[lane]
            road = ET.SubElement(roads_root, "lane") 
            road.set("lane_id",lane)
          
            road.set("from", info_lane['from']) 
            road.set("to", info_lane['to'])
            road.set("priority", info_lane['priority']) 
            road.set("street", str(info_lane['street']))
            road.set("index", info_lane['index']) 
            road.set("speed", info_lane['speed']) 
            road.set("length", str(info_lane['length']))
            if write_segments:
                road.set("segments_length", str(info_lane['segment_length']))
                road.set("segments_number", str(info_lane['segment_number']))
            allveh_road_speed_list = list()
            univeh_road_speed_list = dict()


            for i in range(info_lane['segment_number']):
                if write_segments:
                    traffic_segment = ET.SubElement(road, "traffic", segment=str(i))
                traffic_vehicles = info_lane['vehicles_trafic'][str(i)]
                segment_speed_list = list()
                for traffic_vehicle_point in traffic_vehicles:
                    if write_vehicle:
                        vehicle_point = ET.SubElement(traffic_segment, "vehicle")
                        vehicle_point.set("vehicle_id", str(traffic_vehicle_point['vehicle_id']))
                        vehicle_point.set("speed", str(traffic_vehicle_point['speed']))
                        vehicle_point.set("pos", str(traffic_vehicle_point['position']) )
                        vehicle_point.set("timestep", str(traffic_vehicle_point['timestep']))
                    if write_segments:
                        segment_speed_list.append({'speed': float(traffic_vehicle_point['speed'])})
                    
                    
                    allveh_road_speed_list.append({'speed': float(traffic_vehicle_point['speed'])})

                    traffic_vehicle_point_agreg = {'speed': float(traffic_vehicle_point['speed']),'pos': float(traffic_vehicle_point['position']),'timestep':float(traffic_vehicle_point['timestep'])}
                    if traffic_vehicle_point['vehicle_id'] in univeh_road_speed_list:
                        univeh_road_speed_list[traffic_vehicle_point['vehicle_id']].append(traffic_vehicle_point_agreg)
                    else:
                        univeh_road_speed_list[traffic_vehicle_point['vehicle_id']] = [traffic_vehicle_point_agreg]


                if write_segments:
                    segment_speed_stats = self.computeStats(segment_speed_list)                    
                    traffic_segment.set("allveh_speed_mean", str(segment_speed_stats['mean']))
                    traffic_segment.set("allveh_speed_std", str(segment_speed_stats['std'])) 
                       
            road_speed_stats = self.computeStats(allveh_road_speed_list, is_string=True, digits=5)

            road.set("allveh_speed_mean", str(road_speed_stats['mean']))
            road.set("allveh_speed_std", str(road_speed_stats['std']))

            univeh_road_speed_sum = list()
            for veh_id in univeh_road_speed_list:
                univeh_road_speed_stats_vehicle = self.computeStats(univeh_road_speed_list[veh_id], is_string=False)
                univeh_road_speed_sum.append({'speed':univeh_road_speed_stats_vehicle['mean']})
            
            univeh_road_speed_stats = self.computeStats(univeh_road_speed_sum)
            road.set("unique_veh_speed_mean", str(univeh_road_speed_stats['mean']))
            road.set("unique_veh_speed_std", str(univeh_road_speed_stats['std']))

            if write_vehicle:
                uniqueVehicle_speed_mean = ET.SubElement(road, "agregate_data")
                for veh_id in univeh_road_speed_list:
                    veh_speed_transitions = self.computeStats(univeh_road_speed_list[veh_id], weighted_stats=True)
                    
                    for veh_speed in veh_speed_transitions:
                        
                        vehicle_speed = ET.SubElement(uniqueVehicle_speed_mean, "vehicle")
                        vehicle_speed.set("vehicle_id", str(veh_id) )
                        vehicle_speed.set("mean", str(veh_speed['mean']))
                        vehicle_speed.set("std", str(veh_speed['std']))
                        vehicle_speed.set("weighted_mean", str(veh_speed['weighted_mean']))
                        vehicle_speed.set("travel_time", str(veh_speed['travel_time']))
                        vehicle_speed.set("travel_transitions", str(len(veh_speed_transitions)))

        xmlstr = minidom.parseString(ET.tostring(roads_root)).toprettyxml(indent="   ")
        with open(self.outFile, "w") as f:
            f.write(xmlstr)


    def computeStats(self, listValues, is_string=True, digits=5, values_excluse=[0.00,"Nan"], weighted_stats=False ):
        
        speedValuesList = [item['speed'] for item in listValues]
        for value in values_excluse:
            speedValuesList = list(filter((value).__ne__, speedValuesList))
        if len(speedValuesList) > 0:
            mean_value = statistics.mean(speedValuesList)
            if is_string:
                mean_value = str(mean_value)[:digits]
        else:
            mean_value = 'Nan'    

        if len(speedValuesList) > 1:
            std_value = statistics.stdev(speedValuesList)
            if is_string:
                std_value = str(std_value)[:digits]
        else:                        
            std_value = 'Nan'
        stats = {"mean" : mean_value, "std" : std_value}
        
        if weighted_stats:
            mean_value_glocal = mean_value
            std_value_glocal = std_value
            car_transitions = self.split_car_transitions(listValues.copy(),'timestep')
            stats_list = list()
            for transition in car_transitions:
                #this vehicle on this street on all transitions
                stats_tran = {"mean_glob" : mean_value, "std_glob" : std_value} 
                loc = self.computeStats(transition, is_string, digits, values_excluse, weighted_stats=False)
                stats_tran["mean"] = loc["mean"]
                stats_tran["std"] = loc["std"]
                pos_prev = 0
                weighted_mean = 0
                timestamp_road = 0
                for value in transition:
                    timestamp_road += 1
                    weighted_mean += (value['pos']-pos_prev)*value['speed']
                    pos_prev = value['pos']
                if pos_prev==0:
                    weighted_mean = ''
                else:
                    weighted_mean = weighted_mean/pos_prev
                stats_tran['travel_time'] = timestamp_road
                stats_tran['weighted_mean'] = weighted_mean
                stats_list.append(stats_tran)            
            stats = stats_list
        return stats
    
    def saveSamplingstats(self,save=False, path_roadstats=None):
        cols = ["mean", "weighted_mean", "travel_time"]
        # create MultiIndex dataframe
        index = pd.MultiIndex.from_tuples(list(), names=["roads", "vehicles"])
        self.roadsVehicles = pd.DataFrame(index=index, columns=cols)        
        # fill dataframe
        xtree = ET.parse(self.outFile)
        xroads = xtree.getroot()     
        for xline in tqdm(xroads):
            for xtraffic in xline:
                if xtraffic.tag == "agregate_data":
                    for xvehicle in xtraffic:
                        vehicle_values = pd.Series([xvehicle.attrib['mean'], xvehicle.attrib['weighted_mean'], xvehicle.attrib['travel_time']])
                        self.roadsVehicles.loc[(xline.attrib['lane_id'], xvehicle.attrib['vehicle_id']), cols] = vehicle_values.values
        self.roadsVehicles.sort_index()
        if save:
            self.roadsVehicles.to_pickle(path_roadstats)

    def split_car_transitions(self, list_in, key):
        list_in.sort(key=lambda k : k[key])
        sublist = []

        while list_in:
            v = list_in.pop(0)
            if not sublist or sublist[-1][key] in [v[key], v[key]-1.0]:
                sublist.append(v)
            else:
                yield sublist
                sublist = [v]
        if sublist:
            yield sublist