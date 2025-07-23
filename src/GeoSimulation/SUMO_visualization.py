import os 
import xml.etree.cElementTree as ET
from .GeoGraph import *
from .SUMO_network import *
from .SUMO_routes import *
from ..tool.utils_matplot import UtilsMatplot
from pathlib import Path

class SUMO_visualization():

    def __init__(self, sumo_tool_folder, folder_simulationName, name_simulationFile, python_cmd, verbose=False):
        self.folder_simulationName = folder_simulationName
        self.name_simulationFile = name_simulationFile
        self.verbose = verbose
        self.sumo_tool_folder = sumo_tool_folder
        self.python_cmd = python_cmd
            
    def plotAttributes(self):
        #https://sumo.dlr.de/docs/Tools/Visualization.html
        raise NotImplementedError()
    

    """
    attr_code::
        t: Time in s
        d: Distance driven (starts with 0 at the first fcd datapoint for each vehicle). Distance is computed based on speed using Euler-integration. Set option --ballistic for ballistic integration.
        a: Acceleration
        s: Speed (m/s)
        i: Vehicle angle (navigational degrees)
        x: X-Position in m
        y: Y-Position in m
        k: Kilometrage (requires --fcd-output.distance)
        g: gap to leader (requires --fcd-output.max-leader-distance)
    """
    def plotTrajectories(self, filename_output, simulObj, attr_code="ts", routeList=None, edgesList=None, verbose=False):
        fcd_file = simulObj.get_fcdFile()

        for char_code in attr_code:
            if char_code not in ['t', 's', 'd', 'a', 'i', 'x', 'y', 'k']:
                raise NotImplementedError()

        path_cmd_sumo_py = Path( self.sumo_tool_folder, 'plot_trajectories.py')
        output_path = Path(self.folder_simulationName, filename_output)
        fcd_path = Path(self.folder_simulationName, fcd_file)
        sumo_cmd = f'{self.python_cmd} "{path_cmd_sumo_py}" {fcd_path} --trajectory-type {attr_code} --output {output_path}'
        if edgesList is not None:
            sumo_cmd = sumo_cmd +f' --filter-edges {edgesList}'
        elif  routeList is not None:
            sumo_cmd = sumo_cmd +f' --filter-route {routeList}'
        
        if verbose:
            print("\nplot_trajectories\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
    
    def plotNet2Key(self, filename_output, networkFile, fileinput, key_colors, key_widths, color_map, verbose=True):
        if not UtilsMatplot.isColorMap(color_map):
            raise SUMO_visualization_Exception__ColorMapNotExist(color_map)
        dumpinputs = self.checkCountConcat(namePlot="plot_net_dump", type_value="plot", n_required=2, file_list=fileinput, sep=",")
        # dumpinputs : first is used for the edges' color, the second for their widths, in this case are the same

        path_cmd_sumo_py = Path( self.sumo_tool_folder, 'visualization', 'plot_net_dump.py')
        network_path = Path(self.folder_simulationName, networkFile)
        output_path = Path(self.folder_simulationName, filename_output)
        
        sumo_cmd = f'{self.python_cmd} "{path_cmd_sumo_py}" --net {network_path} --dump-inputs {dumpinputs} --measures {key_colors},{key_widths} --colormap {color_map} --min-color-value -.1 --max-color-value .1 --max-width-value .1 --max-width 3 --min-width .5 --output {output_path} --blind'        
        if verbose:
            print("\nplotNet2Key\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)

    def plotNetSpeed(self, filename_output, networkFile, color_map, verbose=True):
        if not UtilsMatplot.isColorMap(color_map):
            raise SUMO_visualization_Exception__ColorMapNotExist(color_map)

        path_cmd_sumo_py = Path( self.sumo_tool_folder, 'visualization', 'plot_net_speeds.py')
        network_path = Path(self.folder_simulationName, networkFile)
        output_path = Path(self.folder_simulationName, filename_output)

        sumo_cmd = f'{self.python_cmd} "{path_cmd_sumo_py}"  --net {network_path}  --edge-width .5 --output {output_path}  --colormap {color_map} --blind'

        if verbose:
            print("\nplotNetSpeed\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)

    def plotTripDistributions(self, filename_output, measure="duration", bins=10, verbose=True):
        path_cmd_sumo_py = Path( self.sumo_tool_folder, 'visualization', 'plot_tripinfo_distributions.py')
        input_path = Path(self.folder_simulationName, fileinput)
        output_path = Path(self.folder_simulationName, filename_output)

        sumo_cmd = f'{self.python_cmd} "{path_cmd_sumo_py}"  --tripinfos-inputs {input_path} --output {output_path} --measure {measure} --bins {bins} --blind'

        if verbose:
            print("\nplotTripDistributions\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
    
    def plotAllTrajectories(self,  measure="duration", bins=10, verbose=True):
        path_cmd_sumo_py = Path( self.sumo_tool_folder, 'visualization', 'plotXMLAttributes.py')
        fcd_path = Path(self.folder_simulationName, self.name_simulationFile+'.out.fcd.xml')

        output_png_path = Path(self.folder_simulationName, self.name_simulationFile+'plot.allTrajectories.png')
        output_csv_path = Path(self.folder_simulationName, self.name_simulationFile+'out.allTrajectories.csv')

        sumo_cmd = f'{self.python_cmd} "{path_cmd_sumo_py}"  -xattr x --yattr y --output {output_png_path} {fcd_path} --scatterplot --csv-output {output_csv_path}'         
        if verbose:
            print("\nplotTripDistributions\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)

    def checkCountConcat(self, namePlot, type_value, n_required, file_list, sep=","):
        n_received = len(file_list)
        if n_required != n_received:
            raise SUMO_visualization_Exception__RequireCount(namePlot=namePlot, type_value=type_value, n_required=n_required, n_received=n_received)
        else:
            dumpinputs_list = []
            for _file in file_list:
                dumpinputs_list.append(str(Path(self.folder_simulationName,_file)))
            dumpinputs = sep.join(dumpinputs_list)
            return dumpinputs




class SUMO_visualization_Exception__ColorMapNotExist(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,msg):
        self.msg = msg
          
    def __str__(self):
        return f"'{self.msg}' is not a color map."


class SUMO_visualization_Exception__RequireCount(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self, namePlot, type_value, n_required,n_received):
        self.namePlot = namePlot
        self.type_value = type_value
        self.n_required = n_required
        self.n_received = n_received
          
    def __str__(self):
        return f"'{self.namePlot}' {self.type_value} require {self.n_required} file/s but received only {self.n_received} file/s."