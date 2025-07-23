import os
from pathlib import Path
from src.NeuroCorrelation.Models.AutoEncoderModels import *
from src.NeuroCorrelation.Models.VariationalAutoEncoderModels import *
from src.NeuroCorrelation.Models.GenerativeAdversarialModels import *
from src.NeuroCorrelation.Models.ConditionalVariationalAutoEncoderModels import *
from src.NeuroCorrelation.DataLoaders.DataLoader import DataLoader
from src.NeuroCorrelation.ModelTraining.LossFunctions import LossFunction

class SP500_settings():
    
    def __init__(self, model_case, device, univar_count, lat_dim, dataset_setting, epoch, path_folder, corrCoeff, time_performance, instaces_size,noise_distribution="gaussian", time_slot=None):
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
        self.noise_distribution = noise_distribution
        self.setting_model_case()
        
    
    def setting_model_case(self):
        
        self.learning_rate = dict()
        ####
        #### VAE FULLY CONNECTED
        ####
        
        if self.model_case == "VAE_SP500_T3_linear":
            self.mode = "fin_data"
            self.name_dataset = "SP500"
            self.version_dataset = "SP500_T3"
            self.nets = ['VAE']
            self.graph_topology = True
            self.loss_dict = {
                'VAE':{ "MSE_LOSS":{"type":"fixed","value":1}, "KL_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "VARIANCE_LOSS": {"type": "fixed","value": 1.5}, "JENSEN_SHANNON_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "MEDIAN_LOSS_batch": {"type": "fixed","value": 1}, "PEARSON_CORRELATION_LOSS": {"type": "fixed","value":  1e-1} ,"SPEARMAN_CORRELATION_LOSS": {"type": "fixed","value":  1e-1}},
            }
            self.trainingMode = "VAE"
            self.model_settings['VAE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','SP500_models', 'SP500_T3_vae_linear.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['VAE'] = 1e-2
            
            
            
            
        if self.learning_rate is None:
            self.learning_rate['AE'] = 1e-2  
            self.learning_rate['VAE'] = 1e-2              
            self.learning_rate['CVAE'] = 1e-2 
            self.learning_rate['GAN']['DIS'] = 1e-2   
            self.learning_rate['GAN']['GEN'] = 1e-2
        
        self.path_folder_nets = dict()
        for key in self.nets:
            self.path_folder_nets[key] = Path(self.path_folder, key)
            if not os.path.exists(self.path_folder_nets[key]):
                os.makedirs(self.path_folder_nets[key])
                print(f"Create folder:{ self.path_folder_nets[key]}")
    
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
            elif key=="WGAN":
                self.model["WGAN"] = GenerativeAdversarialModels(device=self.device, load_from_file =self.model_settings["WGAN"]['load_from_file'],
                            json_filepath=self.model_settings["WGAN"]['json_filepath'],
                            edge_index=self.edge_index)
            elif key=="VAE":
                self.model["VAE"] = VariationalAutoEncoderModels(device=self.device, load_from_file =self.model_settings["VAE"]['load_from_file'],
                            json_filepath=self.model_settings["VAE"]['json_filepath'],
                            edge_index=self.edge_index)
            elif key=="CVAE":
                self.model["CVAE"] = ConditionalVariationalAutoEncoderModels(device=self.device, load_from_file =self.model_settings["CVAE"]['load_from_file'],
                            json_filepath=self.model_settings["CVAE"]['json_filepath'],
                            edge_index=self.edge_index)
    
    def get_timeweather_count(self):
        return len(self.timeweather_settings["column_selected"])
    
    def get_learning_rate(self):
        return self.learning_rate
            
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
    
    def get_DataLoader(self, seed_data):      
        print("self.timeweather_settings",self.timeweather_settings)
        print("self.timeweather",self.timeweather)
        print("******************")
        dataloader = DataLoader(mode="graph_roads", seed=seed_data, name_dataset=self.name_dataset, version_dataset=self.version_dataset, time_slot=self.time_slot, device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, time_performance=self.time_performance, path_folder=self.path_folder, timeweather=self.timeweather, timeweather_settings=self.timeweather_settings, noise_distribution=self.noise_distribution)
        return dataloader
    
    def get_LossFunction(self):
        loss_obj = dict()
        for key in self.nets:    
            loss_obj[key] = LossFunction(self.loss_dict[key], univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        return loss_obj
