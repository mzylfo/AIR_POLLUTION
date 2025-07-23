import os 
import xml.etree.cElementTree as ET
from .GeoGraph import *
from .SUMO_network import *
from .SUMO_routes import *

class SUMO_simulation():

    def __init__(self, sumo_tool_folder, folder_simulationName, name_simulationFile, networkObj, routesObj, edgeStats_settings=None, verbose=False):
        self.folder_simulationName = folder_simulationName
        self.name_simulationFile = name_simulationFile
        self.verbose = verbose
        self.sumo_tool_folder = sumo_tool_folder

        if isinstance(networkObj, SUMO_network):
            self.network_file = networkObj.get_networkFile()
            self.geometry_file = networkObj.get_geometrykFile(raiseExc=False)
        else:
            raise SUMO_simulation_Exception__SUMO_Instance(networkObj,SUMO_network)
        
        if isinstance(routesObj, SUMO_routes):
            self.routes_file = routesObj.get_routesFile()
            self.continuos_reroutes_file = routesObj.get_continuous_reroutingFile()
        else:
            raise SUMO_simulation_Exception__SUMO_Instance(routesObj,SUMO_routes)
        if edgeStats_settings is not None:
            self.write_statsEdgeFile(edgeStats_settings)

    def write_statsEdgeFile(self,edgeStats_settings):
        self.sumoEdgeStats_additionalFile = Path(self.name_simulationFile+'.edgedata.settings.xml')
        additional = ET.Element("additional")
        for file_st in edgeStats_settings["stats"]:
            
            if "type" in file_st:
                file_name = Path(self.name_simulationFile+'.out.edgedata.'+file_st['id']+'.'+file_st['type']+'.xml')
                ET.SubElement(additional, "edgeData", id=f"{file_st['id']}", type=f"{file_st['type']}", file=f"{file_name}")
            else:
                file_name = Path(self.name_simulationFile+'.out.edgedata.'+file_st['id']+'.xml')
                ET.SubElement(additional, "edgeData", id=f"{file_st['id']}", file=f"{file_name}")
        
        tree = ET.ElementTree(additional)
        ET.indent(tree, space="\t", level=0)
        sumoEdgeStats_additional_path = Path(self.folder_simulationName,self.sumoEdgeStats_additionalFile)
        tree.write(sumoEdgeStats_additional_path, encoding="utf-8")

    def get_configFile(self):
        if self.sumoConfig_file is None:
            raise SUMO_simulation_Exception__FileNotInit("sumoConfig")
        return self.sumoConfig_file

    def get_fcdFile(self):
        if self.sumofcd_file is None:
            raise SUMO_simulation_Exception__FileNotInit("sumofcd")
        return self.sumofcd_file

    def get_dumpFile(self):
        if self.sumoDump_file is None:
            raise SUMO_simulation_Exception__FileNotInit("sumoDump")
        return self.sumoDump_file
    
    def get_emissionFile(self):
        if self.sumoEmission_file is None:
            raise SUMO_simulation_Exception__FileNotInit("sumoEmission")
        return self.sumoEmission_file
    
    def get_laneChangeFile(self):
        if self.sumoLaneChange_file is None:
            raise SUMO_simulation_Exception__FileNotInit("sumoLaneChange")
        return self.sumoLaneChange_file
    
    def get_edgeStatsAdditionalFile(self):
        if self.sumoEdgeStats_additionalFile is None:
            raise SUMO_simulation_Exception__FileNotInit("sumoEdgeStats_additionalFile")
        return self.sumoEdgeStats_additionalFile
    
    def get_TripInfoFile(self):
        if self.sumoTripInfo_file is None:
            raise SUMO_simulation_Exception__FileNotInit("sumoTripInfo_file")
        return self.sumoTripInfo_file
    

    def esecute_simulation(self, fcd=False, dump=False, emission=False, lanechange=False, vtk=False, stop=False, tripsInfo=False, verbose=False):
        self.write_sumoconfig(fcd=fcd, dump=dump, emission=emission, stop=stop, tripsInfo=tripsInfo, verbose=verbose)
        self.exe_sumo_simulation(lanechange=lanechange, vtk=vtk, verbose=verbose)
        
    def exe_sumo_simulation(self, lanechange, vtk, verbose=False):
        sumoConfig_path = Path(self.folder_simulationName, self.sumoConfig_file)
        sumo_cmd = f"sumo --configuration-file {sumoConfig_path}"
        
        if lanechange:
            self.sumoLaneChange_file =  Path(self.name_simulationFile+'.out.laneChange.xml')
            sumoLaneChange_path = Path(self.folder_simulationName, self.sumoLaneChange_file)
            sumo_cmd = sumo_cmd+ f" --lanechange-output {sumoLaneChange_path}"
        if vtk:
            self.sumoVTK_file =  Path(self.name_simulationFile+'.out.vtk.xml')
            sumoVTK_path = Path(self.folder_simulationName, self.sumoVTK_file)
            sumo_cmd = sumo_cmd+ f" --vtk-output {sumoVTK_path}"
        
        sumo_cmd = sumo_cmd+ f" --no-warnings"
        if verbose:
            print("\nSUMO SIMULATION generated\t>>\t",sumo_cmd,"")
            #sumo_cmd = sumo_cmd+ f" --verbose"
        #else:
            #sumo_cmd = sumo_cmd+ f" --no-warnings"
        os.system(sumo_cmd)
        

    def write_sumoconfig(self, fcd=False, dump=False, emission=False, stop=False, tripsInfo=False, verbose=False):
        self.sumoConfig_file = Path(self.name_simulationFile+'.sumocfg')
        sumoConfig_path = Path(self.folder_simulationName, self.sumoConfig_file)
        configuration = ET.Element("configuration")
        k_input = ET.SubElement(configuration, "input")
        ET.SubElement(k_input, "net-file", value=f"{self.network_file}")
        ET.SubElement(k_input, "route-files", value=f"{self.routes_file}")
        additional_list = []

        if self.continuos_reroutes_file is not None:
            additional_list.append(str(self.continuos_reroutes_file))
        if self.geometry_file is not None:
            additional_list.append(str(self.geometry_file))
        if self.sumoEdgeStats_additionalFile is not None:
            additional_list.append(str(self.sumoEdgeStats_additionalFile))
        
        if len(additional_list)>0:
            add_filelist = ','.join(additional_list)
            ET.SubElement(k_input, "additional-files", value=f"{add_filelist}")
        

        k_time = ET.SubElement(configuration, "time")
        ET.SubElement(k_time, "begin", value="0")
        ET.SubElement(k_time, "end", value="10000")

        k_output = ET.SubElement(configuration, "output")
        ET.SubElement(k_output, "summary-output", value=str(Path(self.name_simulationFile+'.out.xml')))
        if fcd:
            self.sumofcd_file = Path(self.name_simulationFile+'.out.fcd.xml')
            ET.SubElement(k_output, "fcd-output", value=str(self.sumofcd_file))
        if dump:
            self.sumoDump_file = Path(self.name_simulationFile+'.out.dump.xml')
            ET.SubElement(k_output, "netstate-dump", value=str(self.sumoDump_file))
        if emission:
            self.sumoEmission_file = Path(self.name_simulationFile+'.out.emission.xml')
            ET.SubElement(k_output, "emission-output", value=str(self.sumoEmission_file))
        if stop:
            self.sumoEmission_file = Path(self.name_simulationFile+'.out.stop.xml')
            ET.SubElement(k_output, "stop-output", value=str(self.sumoEmission_file))
        if tripsInfo:
            self.sumoTripInfo_file = Path(self.name_simulationFile+'.out.tripinfo.xml')
            ET.SubElement(k_output, "tripinfo-output", value=str(self.sumoEmission_file))
            
        """
        if edgeStats:            
            self.sumoEdgeStats_file =  Path(self.name_simulationFile+'.out.edgeStats.xml')
            k_out_output = T.SubElement(k_output, "output", value=self.sumoEmission_file)
            k_out_output = T.SubElement(k_out_output, "edgeData", value=self.sumoEmission_file)
        """

        tree = ET.ElementTree(configuration)
        ET.indent(tree, space="\t", level=0)
        tree.write(f"{sumoConfig_path}", encoding="utf-8")


class SUMO_simulation_Exception__SUMO_Instance(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,instance,type_class):
        self.instance = instance
        self.type_class = type_class
          
    def __str__(self):
        return f"SUMO_simulation module require an instance '{type_class}' but receive an '{str(type(self.instance_type))}' object."


class SUMO_simulation_Exception__FileNotInit(Exception):
    def __init__(self,key):
        self.key = key

    def __str__(self):
        return f"{key} not inizialized."