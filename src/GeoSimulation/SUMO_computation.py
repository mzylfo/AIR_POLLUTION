import os 
import xml.etree.cElementTree as ET
from .GeoGraph import *
from .SUMO_network import *
from .SUMO_routes import *
from .SUMO_simulation import *
from pathlib import Path

class SUMO_computation():

    def __init__(self, sumo_tool_folder, folder_simulationName, name_simulationFile, network_settings, routes_settings, edgeStats_settings=None, python_cmd="Python3", verbose=True):
        self.sumo_tool_folder = sumo_tool_folder
        if not os.path.exists(self.sumo_tool_folder):
            raise SUMO_INSTALL_Exception__ToolFolder(self.sumo_tool_folder)
        
        self.folder_simulationName = folder_simulationName
        if not os.path.exists(self.folder_simulationName):
            os.makedirs(self.folder_simulationName)
        self.name_simulationFile = name_simulationFile
        self.network_settings = network_settings
        self.routes_settings = routes_settings
        self.edgeStats_settings = edgeStats_settings

        self.net_simulation = SUMO_network(sumo_tool_folder=self.sumo_tool_folder,folder_simulationName = self.folder_simulationName, name_simulationFile=self.name_simulationFile, network_settings=self.network_settings, python_cmd=python_cmd, verbose=verbose)
        
        
    def generate_simulation(self, python_cmd="Python3", verbose=False):
        self.network_file = self.net_simulation.network_generation(verbose=verbose)
        self.geometry_file = self.net_simulation.geometry_generation(verbose=verbose)
        
        self.routes_simulation = SUMO_routes(sumo_tool_folder=self.sumo_tool_folder, folder_simulationName = self.folder_simulationName, name_simulationFile=self.name_simulationFile, routes_settings=self.routes_settings, networkObj=self.net_simulation, python_cmd=python_cmd, verbose=verbose)
        self.generate_routes(verbose=verbose)
        #if demain self.osm_generate_simulation_demand(verbose)
                
    
    def generate_routes(self, verbose=False):
        self.routes_simulation.routes_generation(verbose=verbose)
        
        self.flows_file = self.routes_simulation.get_flowsFile()
        self.routes_file = self.routes_simulation.get_routesFile()
        self.continuos_reroutes_file = self.routes_simulation.get_continuous_reroutingFile()

    def esecute_simulation(self, fcd=False, dump=False, emission=False, lanechange=False, vtk=False, stop=False, tripsInfo=False, verbose=False):
        self.sumo_simulation = SUMO_simulation(sumo_tool_folder=self.sumo_tool_folder, folder_simulationName = self.folder_simulationName, name_simulationFile=self.name_simulationFile, networkObj=self.net_simulation, routesObj=self.routes_simulation, edgeStats_settings=self.edgeStats_settings, verbose=verbose)
        self.sumo_simulation.esecute_simulation(fcd=fcd, dump=dump, emission=emission, lanechange=lanechange, vtk=False, stop=stop, tripsInfo=tripsInfo, verbose=verbose)
        self.config_file = self.sumo_simulation.get_configFile()
        return self.sumo_simulation
    
    def get_networkObj():
        return self.net_simulation
    
    """
    TO DO

    def osm_generate_simulation_demand(self, verbose=False):
        #self.network_file = self.net_simulation.network_generation(verbose=True)
        #self.geometry_file = self.geometry_generation(folder_simulationName=self.folder_simulationName, name_simulationFile=self.name_simulationFile, network_file=self.network_file, geometric_maps_options=['all'], osm_map_name=self.osm_map_name, verbose=verbose)        
        #activitygen-example.stat.xml \
        #vehicles_generation_cityDemand
        self.stats_file = f"{self.osm_map_name}_statistics_files.xml"
        self.citiesdemand_file = self.vehicles_generation_cityDemand(folder_simulationName=self.folder_simulationName, name_simulationFile=self.name_simulationFile, network_file=self.network_file, stats_file=self.stats_file, verbose=verbose)
        
    def vehicles_generation_cityDemand(self, folder_simulationName, name_simulationFile, network_file, stats_file, verbose=False):
        #https://sumo.dlr.de/docs/Tools/Trip.html
        citydemand_file = f"{name_simulationFile}_demand__trips.xml"
        sumo_cmd = f"activitygen --net-file {folder_simulationName}/{network_file} --stat-file {folder_simulationName}/{stats_file} --output-file {folder_simulationName}/{citydemand_file} --random"
        if verbose:
            print("\ndemand generation\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
        return citydemand_file
    
    """



class SUMO_INSTALL_Exception__ToolFolder(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,msg):
        self.msg = msg
          
    def __str__(self):
        return f"SUMO tool folder '{self.msg}' not found."