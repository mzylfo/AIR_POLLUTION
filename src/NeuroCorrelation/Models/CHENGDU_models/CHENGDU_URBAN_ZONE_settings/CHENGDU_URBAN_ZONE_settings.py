import os
from pathlib import Path
from src.NeuroCorrelation.Models.AutoEncoderModels import *
from src.NeuroCorrelation.Models.GenerativeAdversarialModels import *
from src.NeuroCorrelation.DataLoaders.DataLoader import DataLoader
from src.NeuroCorrelation.ModelTraining.LossFunctions import LossFunction

class CHENGDU_URBAN_ZONE_settings():
    
    def __init__(self, model_case, device, univar_count, lat_dim, dataset_setting, epoch, path_folder, corrCoeff, instaces_size, time_performance, time_slot=None):
        self.model_case = model_case
        self.device = device
        self.dataset_setting = dataset_setting
        self.epoch = epoch
        self.univar_count = univar_count
        self.lat_dim = lat_dim
        self.corrCoeff = corrCoeff
        self.instaces_size = instaces_size
        self.path_folder = path_folder
        self.time_slot = time_slot
        self.model = dict()
        self.model_settings = dict()
        self.time_performance = time_performance
        self.setting_model_case()
    
    def setting_model_case(self):
        if self.model_case == "AE_CHENGDU_URBAN_ZONE_0_graph":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "URBAN_ZONE_0"            
            self.nets = ['AE', 'GAN']
            self.graph_topology = True
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":5e-1, "VARIANCE_LOSS":5e-1, "SPEARMAN_CORRELATION_LOSS":1e-2},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_URBAN_ZONE_settings', 'CHENGDU_URBAN_ZONE_0_graph.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_URBAN_ZONE_settings', 'CHENGDU_URBAN_ZONE_0_graph.json')}
            
        elif self.model_case == "AE_CHENGDU_URBAN_ZONE_1_graph":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "URBAN_ZONE_1"
            self.nets = ['AE', 'GAN']
            self.graph_topology = True
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.5, "SPEARMAN_CORRELATION_LOSS":0.001, "VARIANCE_LOSS":0.5},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_URBAN_ZONE_settings', 'CHENGDU_URBAN_ZONE_1_graph.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_URBAN_ZONE_settings', 'CHENGDU_URBAN_ZONE_1_graph.json')}
        
        elif self.model_case == "AE_CHENGDU_URBAN_ZONE_2_graph":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "URBAN_ZONE_2"
            self.nets = ['AE', 'GAN']
            self.graph_topology = True
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.5, "SPEARMAN_CORRELATION_LOSS":0.001, "VARIANCE_LOSS":0.5},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_URBAN_ZONE_settings', 'CHENGDU_URBAN_ZONE_2_graph.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_URBAN_ZONE_settings', 'CHENGDU_URBAN_ZONE_2_graph.json')}
            
        elif self.model_case == "AE_CHENGDU_URBAN_ZONE_3_graph":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "URBAN_ZONE_3"
            self.nets = ['AE', 'GAN']
            self.graph_topology = True
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.5, "SPEARMAN_CORRELATION_LOSS":0.001, "VARIANCE_LOSS":0.5},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_URBAN_ZONE_settings', 'CHENGDU_URBAN_ZONE_3_graph.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_URBAN_ZONE_settings', 'CHENGDU_URBAN_ZONE_3_graph.json')}
            
        elif self.model_case == "AE_CHENGDU_URBAN_ZONE_1-2_graph":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "URBAN_ZONE_1-2"
            self.nets = ['AE', 'GAN']
            self.graph_topology = True
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.5, "SPEARMAN_CORRELATION_LOSS":0.001, "VARIANCE_LOSS":0.5},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_URBAN_ZONE_settings', 'CHENGDU_SMALLGRAPH_64_graph.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_URBAN_ZONE_settings', 'CHENGDU_SMALLGRAPH_64_graph.json')}   
        
        if self.learning_rate is None:
            self.learning_rate['AE'] = 1e-2   
            self.learning_rate['GAN']['DIS'] = 1e-2   
            self.learning_rate['GAN']['GEN'] = 1e-2   
        self.path_folder_nets = dict()
        for key in self.nets:
            self.path_folder_nets[key] = Path(self.path_folder, key)
            if not os.path.exists(self.path_folder_nets[key]):
                os.makedirs(self.path_folder_nets[key])
    
    def set_edge_index(self,edge_index):
        self.edge_index = edge_index
    
    def deploy_models(self):
        for key in self.model_settings:
            if key == "AE":
                self.model["AE"] = AutoEncoderModels(device=self.device, load_from_file =self.model_settings["AE"]['load_from_file'],
                            json_filepath=self.model_settings["AE"]['json_filepath'],
                            edge_index=self.edge_index)
            elif key=="GAN":
                self.model["GAN"] = GenerativeAdversarialModels(device=self.device, load_from_file =self.model_settings["GAN"]['load_from_file'],
                            json_filepath=self.model_settings["GAN"]['json_filepath'],
                            edge_index=self.edge_index)
            elif key=="WGAN":
                self.model["WGAN"] = GenerativeAdversarialModels(device=self.device, load_from_file =self.model_settings["WGAN"]['load_from_file'],
                            json_filepath=self.model_settings["WGAN"]['json_filepath'],
                            edge_index=self.edge_index)
    
    def get_trainingMode(self):
        return self.trainingMode

    def get_model(self, key):
        return self.model[key]
        
    def get_time_slot(self):
        return self.time_slot
    
    def get_folder_nets(self):
        return self.path_folder_nets
    
    def get_graph_topology(self):
        return self.graph_topology
    
    def get_learning_rate(self):
        return self.learning_rate
    
    def get_DataLoader(self, seed_data):      
        dataloader = DataLoader(mode="graph_roads", seed=seed_data, name_dataset=self.name_dataset, version_dataset=self.version_dataset, time_slot=self.time_slot, device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, time_performance=self.time_performance, path_folder=self.path_folder)
        return dataloader
    
    def get_LossFunction(self):
        loss_obj = dict()
        for key in self.nets:    
            loss_obj[key] = LossFunction(self.loss_dict[key], univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        return loss_obj
