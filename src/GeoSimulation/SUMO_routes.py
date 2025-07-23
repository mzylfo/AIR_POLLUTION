import os 
from .SUMO_network import *
from pathlib import Path

class SUMO_routes():

    def __init__(self, sumo_tool_folder, folder_simulationName, routes_settings, name_simulationFile, networkObj, python_cmd, verbose=False):#="python"
        self.folder_simulationName = folder_simulationName
        self.name_simulationFile = name_simulationFile
        self.verbose = verbose
        self.sumo_tool_folder = sumo_tool_folder
        self.python_cmd = python_cmd

        self.routes_type = routes_settings['routes_type']
        if self.routes_type not in ['random','demand']:
            raise SUMO_routes_Exception__routesTypeNotFound(self.routes_type)

        if 'begin_time' in routes_settings['options']:
            self.begin_time = routes_settings['options']['begin_time']
        else:
            self.begin_time = 0

        if 'end_time' in routes_settings['options']:
            self.end_time = routes_settings['options']['end_time']
        else:
            self.end_time = 3600

        if 'period' in routes_settings['options']:
            self.period = routes_settings['options']['period']
        else:
            self.period = 1
        
        if 'vehicle' in routes_settings['options']:
            self.vehicle = routes_settings['options']['vehicle']
        else:
            self.vehicle = 200
        print("vehicle\t\t",self.vehicle)

        if 'simulation_opt' in routes_settings:
            if 'continuos_reroutes' in routes_settings['simulation_opt']:
                self.continuos_reroutes_bool = routes_settings['simulation_opt']['continuos_reroutes']
            

        if isinstance(networkObj, SUMO_network):
            self.network_file = networkObj.get_networkFile()
        else:
            raise SampleTrips_Exception__SUMO_networkInstance(networkObj)
    
    def get_routesFile(self):
        if self.routes_file is None:
            raise SUMO_routes_Exception__FileNotInit("Routes")
        return self.routes_file
    
    def get_flowsFile(self):
        if self.flows_file is None:
            raise SUMO_routes_Exception__FileNotInit("Flows")
        return self.flows_file

    def get_continuous_reroutingFile(self):
        if not self.continuos_reroutes_bool:
            return None
        if self.continuos_reroutes_file is None:
            raise SUMO_routes_Exception__FileNotInit("Continuous_rerouting")
        return self.continuos_reroutes_file

    def routes_generation(self,verbose=False):
        if self.routes_type == 'random':
            self.flows_generation_random(verbose=verbose)
            self.routes_generation_jtrrouter(verbose=verbose)
            if self.continuos_reroutes_bool:
                self.continuous_rerouting_generation(verbose=verbose)
        elif self.routes_type == 'demand':
            raise NotImplemented()
            

    def flows_generation_random(self, verbose=False):
        #https://sumo.dlr.de/docs/Tools/Trip.html        
        self.flows_file = Path(self.name_simulationFile+'.flows.xml')
        flows_path = Path(self.folder_simulationName, self.flows_file)
        network_path = Path(self.folder_simulationName, self.network_file)
        path_cmd_sumo_py = Path( self.sumo_tool_folder, 'randomTrips.py')
        sumo_cmd = f'{self.python_cmd} "{path_cmd_sumo_py}" --net-file {network_path} --output-trip-file {flows_path} --begin {self.begin_time} --end {self.end_time} --period {self.period} --flows {self.vehicle}'
        if verbose:
            print("\nflows generation\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
        return self.flows_file

    def routes_generation_jtrrouter(self, verbose=False):
        #https://sumo.dlr.de/docs/jtrrouter.html
        self.routes_file = Path(self.name_simulationFile+'.routes.xml')
        routes_path = Path(self.folder_simulationName, self.routes_file)
        network_path = Path(self.folder_simulationName, self.network_file)
        flows_path = Path(self.folder_simulationName, self.flows_file)

        sumo_cmd = f"jtrrouter --route-files={flows_path} --net-file={network_path} --output-file={routes_path} --begin {self.begin_time}  --end {self.end_time} --accept-all-destinations"        
        if verbose:
            print("\nroutes generation\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
        return self.routes_file
    
    def routes_generation_duarouter(self, citiesdemand_file, verbose=False):
        #https://sumo.dlr.de/docs/Demand/Activity-based_Demand_Generation.html
        self.routes_file = f"{self.name_simulationFile}.routes.xml"
        routes_path = Path(self.folder_simulationName, self.routes_file)
        citiesdemand_path = Path(self.folder_simulationName, citiesdemand_file)
        
        network_path = Path(self.folder_simulationName, self.network_file)
        sumo_cmd = f"duarouter --route-files={citiesdemand_path} --net-file={network_path} --output-file={routes_path} --ignore-errors"        
        if verbose:
            print("\nduarouter_file generation\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
        return self.routes_file

    def continuous_rerouting_generation(self, verbose=False):
        #https://sumo.dlr.de/docs/Tools/Misc.html#generatecontinuousrerouterspy
        self.continuos_reroutes_file = f"{self.name_simulationFile}.rerouting.xml"
        continuos_reroutes_path = Path(self.folder_simulationName, self.continuos_reroutes_file)
        network_path = Path(self.folder_simulationName, self.network_file)

        path_cmd_sumo_py = Path( self.sumo_tool_folder, 'generateContinuousRerouters.py')
        sumo_cmd = f'{self.python_cmd} "{path_cmd_sumo_py}" --net-file {network_path} --output-file {continuos_reroutes_path} --begin {self.begin_time} --end {self.end_time}'
        if verbose:
            print("\ncontinuous rerouting generation\t>>\t",sumo_cmd)
        os.system(sumo_cmd)
        return self.continuos_reroutes_file      
    
class SUMO_routes_Exception__routesTypeNotFound(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self, routes_type):
        self.routes_type = routes_type
    def __str__(self):
        return f"'{self.routes_type}' is not a routes type recognized."    

class SUMO_routes_Exception__SUMO_networkInstance(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,instance):
        self.instance = instance
          
    def __str__(self):
        return f"SUMO_routes module require an instance 'SUMO_network' but receive an '{str(type(self.instance_type))}' object."

class SUMO_routes_Exception__FileNotInit(Exception):
    def __init__(self,key):
        self.key = key

    def __str__(self):
        return f"{key} not inizialized."