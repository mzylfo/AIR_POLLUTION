from src.GeoSimulation.GeoGraph import *
import matplotlib.pyplot as plt
from src.test.Tdrive_Bejing  import *

def downloadmaps_cities(case="perugia", POI_maps=False, draw_maps=False, list_municipalities=None, map_name_file=None):
    if case=="fvg":
        places = ['Cervignano del Friuli,Italy', "Terzo d'Aquileia,Italy",'Aquileia,Italy','Ruda,Italy',"Grado,italy",'Aiello del Friuli,Italy',"San giorgio di Nogaro,Italy",
        "campolongo tapogliano, Italy", "Gonars,Italy", "Visco,Italy", "San Vito al Torre,Italy", "Bagnaria arsa,Italy","Fiumicello Villa Vicentina,Italy", "Torviscosa,Italy",
        "Palmanova,Italy", ]
        maps_name='bassa'
    elif case=="perugia":
        places = list_municipalities
        maps_name = map_name_file
    elif case=="pechino":
        places = ['Beijing,China']
        maps_name='Beijing'
    elif case=="milano":

        places = ['Milano,Italy']
        maps_name='Milan'
    else:
        print("Unknown cities case")
        return 0

    geo_maps_settings={
        "osm_maps_name":maps_name,
        "map_folder":None,
        "poi_folder":None,
        "options":{
            "places":places, 
            "placetype":"cities",
            "simplification":True,
            "poi_geometry":True, 
            "poi_option":{
                "filter":["amenity","historic","tourism"]
            }
        }
    }
    geo_settings = GeoGraph(geo_maps_settings=geo_maps_settings)
    #https://wiki.openstreetmap.org/wiki/Map_features   
    fig, ax = plt.subplots(figsize=(25,18))
    if draw_maps:
        if POI_maps:
            POI_geo = geo_settings.getPOI()
        GEO_geo = geo_settings.getGEO()    
        if POI_maps:
            POI_geo.plot(ax=ax, facecolor='khaki', alpha=0.7)
        ox.plot_graph(GEO_geo, ax=ax, node_size=0, edge_linewidth=0.5,show=True)



def downloadmaps_points(POI_maps=False, draw_maps=False):
    
    
    maps_name='point_test'

    geo_maps_settings={
        "osm_maps_name":maps_name,
        "map_folder":None,
        "poi_folder":None,
        "options":{
            "places":(45.52374, 9.21959),
            "placetype":"point",
            "dist_point": 1200,
            "simplification":True,
            "poi_geometry":False, 
        }
    }
    geo_settings = GeoGraph(geo_maps_settings=geo_maps_settings)
    POI_geo = geo_settings.drawGraph()


def BejingDataset(users_list = range(1,10),line_break=None):
    maps_name="Bejing"
    pathfolder = Path("data","realdataset",maps_name)
    pathinput = Path("data","dataset","taxi_log_2008_by_id")    
    download_maps = False

    tBejing = Tdrive_Bejing(pathfolder=pathfolder, pathinput=pathinput, maps_name=maps_name, users_list=users_list, download_maps=download_maps,line_break=line_break, filename_save="Pechino")
    tBejing.compute_roads(users_list)
    for user_id in users_list:
        print("plot - ", user_id)
        tBejing.plot_point(user_id=user_id)
