import xml.etree.ElementTree as ET
from xml.dom import minidom
import ast
from pathlib import Path

#OSM NETWORK STATS
#e.g. type of roads, #roads, speed limits, etc 

class SUMO_mapsstats():

    def __init__(self, maps_name):
        self.filepath = Path('data', 'maps',maps_name, (maps_name+'.geo.osm'))
        self.outFileNodes =  Path('data', 'maps',maps_name, (maps_name+'.out.mapsstat.nodes.osm'))        
        self.outFileWays =  Path('data', 'maps',maps_name, (maps_name+'.out.mapsstat.ways.osm'))        
        self.node_list = dict()
        self.way_list = dict()
        self.node_atts = dict()
        self.way_atts = dict()

    def compute_mapsstats(self, value_unique=True, save=True, show_name=True):
        tree = ET.parse(self.filepath)
        self.root = tree.getroot()
        self.statsFromOSM(value_unique)
        if save:
            self.showStats(show_name)

    def statsFromOSM(self, value_unique=True):
        for item in self.root:
            if item.tag == "node":
                node_key = item.attrib['id']
                if node_key not in self.node_list:
                    self.node_list[node_key] = dict()
                    self.node_list[node_key]['street'] = dict()
                node_selected = self.node_list[node_key]
                for attr_key in item.attrib:                    
                    if attr_key not in node_selected and attr_key!="id":
                        node_selected[attr_key] = item.attrib[attr_key]
                for sub_item in item:
                    sub_item_key = sub_item.attrib['k']
                    sub_item_val = sub_item.attrib['v']
                    if sub_item_key not in node_selected and sub_item_key !="id":
                        node_selected[sub_item_key] = sub_item_val
                    if sub_item_key not in self.node_atts:
                        self.node_atts[sub_item_key] = dict()
                        self.node_atts[sub_item_key][sub_item_val] = 1
                    else:
                        if sub_item_val in self.node_atts[sub_item_key]:
                            self.node_atts[sub_item_key][sub_item_val] += 1
                        else:
                            self.node_atts[sub_item_key][sub_item_val] = 1

            elif item.tag == "way":
                way_key = item.attrib['id']                
                if way_key not in self.way_list:
                    self.way_list[way_key] = dict()
                    self.way_list[way_key]['node_list'] = list()
                way_selected = self.way_list[way_key]
                for attr_key in item.attrib:                    
                    if attr_key not in way_selected and attr_key!="id":
                        way_selected[attr_key] = item.attrib[attr_key]
                if len(item)>0:
                    for sub_item in item:
                        if sub_item.tag == "nd":
                            way_selected['node_list'].append(sub_item.attrib['ref'])
                        elif sub_item.tag == "tag":
                            sub_item_key = sub_item.attrib['k']
                            sub_item_val = sub_item.attrib['v']
                            if value_unique:
                                sub_item_val = sub_item_val.replace("'","").strip('][').split(', ')
                            else:
                                sub_item_val = [sub_item_val]
                            
                            if sub_item_key not in way_selected:
                                way_selected[sub_item_key] = sub_item_val
                            for sub_item_val_item in sub_item_val:
                                sub_item_val_item.replace("'","")
                                if sub_item_key not in self.way_atts:
                                    self.way_atts[sub_item_key] = dict()
                                    self.way_atts[sub_item_key][f'{sub_item_val_item}'] = 1
                                else:
                                    if f'{sub_item_val_item}' in self.way_atts[sub_item_key]:
                                        self.way_atts[sub_item_key][f'{sub_item_val_item}'] += 1
                                    else:
                                        self.way_atts[sub_item_key][f'{sub_item_val_item}'] = 1                            
                        else:
                            print("way att\t\t",sub_item.tag,"\t",sub_item.attrib)
            else:
                print("net tag\t\t",item.tag,"\t",item.attrib)

    def sortDict(self, x_dict, reverse=True):
        return {k: v for k, v in sorted(x_dict.items(), key=lambda item: item[1], reverse=reverse)}

    def showStats(self, show_name=True):
        nodes_root = ET.Element("nodes")
        for key in self.node_atts:
            node = ET.SubElement(nodes_root, "node")
            node.set("att_nn", self.sortDict(self.node_atts[key])) 
        xmlstr = minidom.parseString(ET.tostring(nodes_root)).toprettyxml(indent="   ")
        with open(f"{self.outFileNodes}", "w") as f:
            f.write(xmlstr)
            
        ways_root = ET.Element("nodes")
        for key in self.ways_root:
            ways = ET.SubElement(ways_root, "ways")
            ways.set("key", key) 
            ways.set("att", self.sortDict(self.way_atts[key])) 

        xmlstr = minidom.parseString(ET.tostring(ways_root)).toprettyxml(indent="   ")
        with open(f"{self.outFileWays}", "w") as f:
            f.write(xmlstr)
