from src.GeoSimulation.SUMO_computation import *
from src.GeoSimulation.SUMO_visualization import *
from src.GeoSimulation.SUMO_roadstats import *
from pathlib import PurePosixPath
def start_test():
    
    #sumo_tool_folder = 'C:/Program Files (x86)/Eclipse/Sumo/tools'
    #sumo_tool_folder = Path('/','usr','share','sumo','tools')
    sumo_tool_folder = PurePosixPath('/usr','share','sumo','tools') 
    print("Sumo_tool_folder :",sumo_tool_folder)
    folder_name_path = Path('data', 'sumo_simulation_files', '-')
    sumo_obj = SUMO_simulation(sumo_tool_folder= sumo_tool_folder, name_file="-",folder_name=folder_name_path, simulation_network_mode="osm",simulation_route_mode="random",osm_map_name="-")
    sumo_obj.generate_simulation(verbose=True)
    #sumo_obj.download_simulation()

def start_SUMO_simulation(name_simulationFile):
    #https://sumo.dlr.de/docs/netgenerate.html
    sumo_tool_folder = Path('C:/','Program Files (x86)','Eclipse','Sumo','tools')
    #Python3 = "python"
    # sumo_tool_folder = PurePosixPath('/usr','share','sumo','tools') 
    Python3 = "python"
    print("Sumo_tool_folder : ",sumo_tool_folder)
    
    folder_simulationName = Path('data','sumo_simulation_files',name_simulationFile)

    net_grid_settings={
        "network_type":"generate",
        "generate":{
            "geometry":"grid", "grid":{"number":7,"length":200}
        }
    }
    net_spider_settings={
        "network_type":"generate",
        "generate":{
            "geometry":"spider", "spider":{"arm-number":5,"circle-number":4,"space-radius":100,"omit-center":False}
        }
    }
    net_rand_settings={
        "network_type":"generate",
        "generate":{
            "geometry":"rand", "rand":{"iterations":200,"bidi-probability":1,"connectivity":0.95}
        }
    }
    net_maps_settings={
        "network_type":"maps",
        "maps":{
            "osm_maps_name":"cervi", "osm_maps_folder":None, "remove_geometry":True,
            "geometry_settings":['all']
        }
    }

    net_maps_settings_noGeom={
        "network_type":"maps",
        "maps":{
            "osm_maps_name":"cervi", "osm_maps_folder":None, "remove_geometry":False,            
        }
    }

    network_settings=net_maps_settings_noGeom

    routes_settings_random15={
        "routes_type":"random",
        "options":{
            "begin_time":0, "end_time":1000, "period":1000,"vehicle":5000
        },
        "simulation_opt":{
            "continuos_reroutes": True,
        }
    }
    routes_settings = routes_settings_random15

    edgeStats_settings={
        "stats":[
            {"id":"all","type":"emissions"},
            {"id":"all","type":"harmonoise"},
            {"id":"all"},
        ]
    }

    sumo_obj = SUMO_computation(sumo_tool_folder= sumo_tool_folder, folder_simulationName=folder_simulationName, name_simulationFile=name_simulationFile, network_settings=network_settings, routes_settings=routes_settings, edgeStats_settings=edgeStats_settings, python_cmd=Python3)
    sumo_obj.generate_simulation(python_cmd=Python3, verbose=True)
    simulObj = sumo_obj.esecute_simulation(stop=True, fcd=True, tripsInfo=True, verbose=True)
    
    roadstats = SUMO_roadstats(name_simulationFile, is_osm=False)
    roadstats.compute_roadstats()
    
    sumo_viz = SUMO_visualization(sumo_tool_folder= sumo_tool_folder, folder_simulationName=folder_simulationName, name_simulationFile=name_simulationFile, python_cmd=Python3)
    
    
    file_edgeData = [f"{name_simulationFile}.out.edgedata.all.xml",f"{name_simulationFile}.out.edgedata.all.xml"]
    file_emissionEdgeData = [f"{name_simulationFile}.out.edgedata.all.emissions.xml",f"{name_simulationFile}.out.edgedata.all.emissions.xml"]
    file_tripInfo = f"{name_simulationFile}.out.all.tripinfo.xml"
    file_network = sumo_obj.network_file

    

    sumo_viz.plotNet2Key(verbose=True, filename_output="plot_net_speed_density.png", networkFile=file_network, fileinput=file_edgeData, key_colors="speed", key_widths="density", color_map="viridis")
    sumo_viz.plotNet2Key(verbose=True, filename_output="plot_net_NOx_normed_traveltime.png", networkFile=file_network, fileinput=file_emissionEdgeData, key_colors="NOx_normed", key_widths="traveltime", color_map="cividis")
    sumo_viz.plotNetSpeed(verbose=True, filename_output="plot_net_speed.png", networkFile=file_network, color_map="jet")
    sumo_viz.plotTrajectories(verbose=True, filename_output="plot_trajectories_timespeed.png", attr_code="ts", simulObj=simulObj)
    sumo_viz.plotTrajectories(verbose=True, filename_output="plot_trajectories_timeacceleration.png", attr_code="ta", simulObj=simulObj)




def start_SUMO_pointsimulation(name_simulationFile):
    #https://sumo.dlr.de/docs/netgenerate.html
    sumo_tool_folder = Path('C:/','Program Files (x86)','Eclipse','Sumo','tools')
    #Python3 = "python"
    # sumo_tool_folder = PurePosixPath('/usr','share','sumo','tools') 
    Python3 = "python"
    print("Sumo_tool_folder : ",sumo_tool_folder)
    
    folder_simulationName = Path('data','sumo_simulation_files',name_simulationFile)

    
    net_maps_settings_noGeom={
        "network_type":"maps",
        "maps":{
            "osm_maps_name":"point_test", "osm_maps_folder":None, "remove_geometry":False,            
        }
    }

    network_settings=net_maps_settings_noGeom

    routes_settings_random15={
        "routes_type":"random",
        "options":{
            "begin_time":0, "end_time":1000, "period":1000,"vehicle":5000
        },
        "simulation_opt":{
            "continuos_reroutes": True,
        }
    }
    routes_settings = routes_settings_random15

    edgeStats_settings={
        "stats":[
            {"id":"all","type":"emissions"},
            {"id":"all","type":"harmonoise"},
            {"id":"all"},
        ]
    }

    sumo_obj = SUMO_computation(sumo_tool_folder= sumo_tool_folder, folder_simulationName=folder_simulationName, name_simulationFile=name_simulationFile, network_settings=network_settings, routes_settings=routes_settings, edgeStats_settings=edgeStats_settings, python_cmd=Python3)
    sumo_obj.generate_simulation(python_cmd=Python3, verbose=True)
    simulObj = sumo_obj.esecute_simulation(stop=True, fcd=True, tripsInfo=True, verbose=True)
    
    roadstats = SUMO_roadstats(name_simulationFile, is_osm=False)
    roadstats.compute_roadstats()
    
    sumo_viz = SUMO_visualization(sumo_tool_folder= sumo_tool_folder, folder_simulationName=folder_simulationName, name_simulationFile=name_simulationFile, python_cmd=Python3)
    
    
    file_edgeData = [f"{name_simulationFile}.out.edgedata.all.xml",f"{name_simulationFile}.out.edgedata.all.xml"]
    file_emissionEdgeData = [f"{name_simulationFile}.out.edgedata.all.emissions.xml",f"{name_simulationFile}.out.edgedata.all.emissions.xml"]
    file_tripInfo = f"{name_simulationFile}.out.all.tripinfo.xml"
    file_network = sumo_obj.network_file

    

    sumo_viz.plotNet2Key(verbose=True, filename_output="plot_net_speed_density.png", networkFile=file_network, fileinput=file_edgeData, key_colors="speed", key_widths="density", color_map="viridis")
    sumo_viz.plotNet2Key(verbose=True, filename_output="plot_net_NOx_normed_traveltime.png", networkFile=file_network, fileinput=file_emissionEdgeData, key_colors="NOx_normed", key_widths="traveltime", color_map="cividis")
    sumo_viz.plotNetSpeed(verbose=True, filename_output="plot_net_speed.png", networkFile=file_network, color_map="jet")
    sumo_viz.plotTrajectories(verbose=True, filename_output="plot_trajectories_timespeed.png", attr_code="ts", simulObj=simulObj)
    sumo_viz.plotTrajectories(verbose=True, filename_output="plot_trajectories_timeacceleration.png", attr_code="ta", simulObj=simulObj)