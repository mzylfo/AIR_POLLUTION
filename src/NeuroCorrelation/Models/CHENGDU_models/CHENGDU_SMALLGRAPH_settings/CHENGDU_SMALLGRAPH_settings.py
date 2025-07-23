import os
from pathlib import Path
from src.NeuroCorrelation.Models.AutoEncoderModels import *
from src.NeuroCorrelation.Models.VariationalAutoEncoderModels import *
from src.NeuroCorrelation.Models.GenerativeAdversarialModels import *
from src.NeuroCorrelation.Models.ConditionalVariationalAutoEncoderModels import *
from src.NeuroCorrelation.DataLoaders.DataLoader import DataLoader
from src.NeuroCorrelation.ModelTraining.LossFunctions import LossFunction

class CHENGDU_SMALLGRAPH_settings():
    
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
        
        if self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_16_A_linear":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_16"            
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":05e-05}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_16_linear.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_16_linear.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 1e-2
        
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_32_A_linear":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_32"
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":05e-05}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_32_linear.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_32_linear.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 1e-2
           
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_48_A_linear":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_48"
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":05e-05}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_48_linear.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_48_linear.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 1e-2
         
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_64_A_linear":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_64"
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {                
                'AE':{'JENSEN_SHANNON_DIVERGENCE_LOSS': {'type': 'fixed', 'value': 1.444962898110864}, 'MEDIAN_LOSS_batch': {'type': 'fixed', 'value': 0.929829415562093}, 'SPEARMAN_CORRELATION_LOSS': {'type': 'fixed', 'value': 1.445002226536778}},
                'GAN': dict()
            }
            #{"JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.005}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_64_linear.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_64_linear.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 1e-2
        
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_96_A_linear":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_96"
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.005}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_96_linear.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_96_linear.json')}   
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}  
            self.learning_rate['AE'] = 1e-2      
            
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_128_A_linear":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_128"
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.005}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_128_linear.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_128_linear.json')}   
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 1e-2
        
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_192_A_linear":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_192"
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.005}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_192_linear.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_192_linear.json')}   
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 1e-2
        
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_256_A_linear":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_256"
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.005}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_256_linear.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_256_linear.json')}   
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 1e-2
            
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_512_A_linear":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_512"
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.005}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_512_linear.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_512_linear.json')}   
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 1e-2
        
        
        ####
        #### VAE GRAPH CONV LAYER
        ####
        
        
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_16_A_graph":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_16"
            self.nets = ['AE', 'GAN']
            self.graph_topology = True
            self.loss_dict = {
                'AE':{'JENSEN_SHANNON_DIVERGENCE_LOSS': {'type': 'fixed', 'value': 0.9843902819550896}, 'MEDIAN_LOSS_batch': {'type': 'fixed', 'value': 0.00945219589436375}, 'SPEARMAN_CORRELATION_LOSS': {'type': 'fixed', 'value': 1.9483751908616564}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_16_graph.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_16_graph.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 1e-2
            
            
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_32_A_graph":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_32"
            self.nets = ['AE', 'GAN']
            self.graph_topology = True
            self.loss_dict = {
                'AE':{'JENSEN_SHANNON_DIVERGENCE_LOSS': {'type': 'fixed', 'value': 1}, 'MEDIAN_LOSS_batch': {'type': 'fixed', 'value': 0.0005}, 'SPEARMAN_CORRELATION_LOSS': {'type': 'fixed', 'value': 1}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_32_graph.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_32_graph.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 0.01

        
            
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_48_A_graph":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_48"
            self.nets = ['AE', 'GAN']
            self.graph_topology = True
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.0005}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_48_graph.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_48_graph.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 1e-2
            
            
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_64_A_graph":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_64"
            self.nets = ['AE', 'GAN']
            self.graph_topology = True
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.0005}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
                'GAN': dict()  
            }      
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_64_graph.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_64_graph.json')}   
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 1e-2


        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_96_A_graph":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_96"
            self.nets = ['AE', 'GAN']
            self.graph_topology = True
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS": {"type": "fixed","value": 1},"MEDIAN_LOSS_batch": {"type": "fixed", "value": 0.0005}, "SPEARMAN_CORRELATION_LOSS": {"type": "fixed", "value": 1}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_96_graph.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_96_graph.json')}   
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}        
            self.learning_rate['AE'] = 1e-2
            
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_128_A_graph":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_128"
            self.nets = ['AE', 'GAN']
            self.graph_topology = True
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1.5}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.005}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_128_graph.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_128_graph.json')}   
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 1e-3
        
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_192_A_graph":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_192"
            self.nets = ['AE', 'GAN']
            self.graph_topology = True
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.005},"VARIANCE_LOSS": { "type": "fixed", "value": 0.005}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}, "CORRELATION_MATRICES_LOSS":{"type":"fixed","value":0.01}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_192_graph.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_192_graph.json')}   
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 1e-3
        
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_256_A_graph":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_256"
            self.nets = ['AE', 'GAN']
            self.graph_topology = True
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch": {"type":"fixed","value":0.005},"SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_256_graph.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_256_graph.json')}   
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 1e-3
            
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_512_A_graph":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_512"
            self.nets = ['AE', 'GAN']
            self.graph_topology = True
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.005},  "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_512_graph.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_512_graph.json')}   
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 1e-3
            
        elif self.model_case == "AE>GAN_CHENGDU_SMALLGRAPH_1024_A_graph":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_1024"
            self.nets = ['AE', 'GAN']
            self.graph_topology = True
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.005}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
                'GAN': dict()
            }
            self.trainingMode = "AE>GAN"
            self.model_settings['AE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_1024_graph.json')}
            self.model_settings['GAN'] = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_1024_graph.json')}   
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['AE'] = 1e-3
        
        #BVAE    
        elif self.model_case == "VAE_CHENGDU_SMALLGRAPH_16_A_graph_kl_fix":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_16"
            self.nets = ['VAE']
            self.graph_topology = True
            self.loss_dict = {
                'VAE':{ "MSE_LOSS":{"type":"fixed","value":1}, "KL_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "VARIANCE_LOSS": {"type": "fixed","value": 0.8}, "JENSEN_SHANNON_DIVERGENCE_LOSS": {"type": "fixed","value": 0}, "MEDIAN_LOSS_batch": {"type": "fixed","value": 0.5}, "MSE_LOSS": {"type": "fixed","value": 1}, "PEARSON_CORRELATION_LOSS": {"type": "fixed","value":  1e-3} ,"SPEARMAN_CORRELATION_LOSS": {"type": "fixed","value":  1e-3}},
            }
            self.trainingMode = "VAE"
            self.model_settings['VAE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_16_vae_graph.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['VAE'] = 1e-2
            
        elif self.model_case == "VAE_CHENGDU_SMALLGRAPH_16_A_graph_kl_lin":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_16"
            self.nets = ['VAE']
            self.graph_topology = True
            self.loss_dict = {
                'VAE':{"MSE_LOSS":{"type":"fixed","value":1}, "KL_DIVERGENCE_LOSS":{"type":"linear","range":{"begin":0, "end":0.5}}, "JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.5}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
            }
            self.trainingMode = "VAE"
            self.model_settings['VAE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_16_vae_graph.json')}
            self.timeweather = False      
            self.timeweather_settings = {"column_selected":[]}         
            self.learning_rate['VAE'] = 1e-2
               
        elif self.model_case == "VAE_CHENGDU_SMALLGRAPH_16_A_graph_kl_cos":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_16"
            self.nets = ['VAE']
            self.graph_topology = True
            self.loss_dict = {
                'VAE':{"MSE_LOSS":{"type":"fixed","value":1}, "KL_DIVERGENCE_LOSS":{"type":"cos","range":{"min":0, "max":1, "period":25}},  "JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.5}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
            }
            self.trainingMode = "VAE"
            self.model_settings['VAE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_16_vae_graph.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['VAE'] = 1e-2
        
        elif self.model_case == "VAE_CHENGDU_SMALLGRAPH_32_A_graph_kl_fix":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_32"
            self.nets = ['VAE']
            self.graph_topology = True
            self.loss_dict = {   
                #'VAE':{"MSE_LOSS": {"type": "fixed","value": 1}, "KL_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "VARIANCE_LOSS": {"type": "fixed","value": 0.5}, "JENSEN_SHANNON_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "MEDIAN_LOSS_batch": {"type": "fixed","value": 0.5}, "SPEARMAN_CORRELATION_LOSS": {"type": "fixed","value": 1}},
                'VAE':{ "MSE_LOSS":{"type":"fixed","value":1}, "KL_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "VARIANCE_LOSS": {"type": "fixed","value": 0.8}, "JENSEN_SHANNON_DIVERGENCE_LOSS": {"type": "fixed","value": 0}, "MEDIAN_LOSS_batch": {"type": "fixed","value": 0.5}, "MSE_LOSS": {"type": "fixed","value": 1}, "PEARSON_CORRELATION_LOSS": {"type": "fixed","value":  1e-3} ,"SPEARMAN_CORRELATION_LOSS": {"type": "fixed","value":  1e-3}},
            }
            self.trainingMode = "VAE"
            self.model_settings['VAE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_32_vae_graph.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['VAE'] = 1e-2
            
        elif self.model_case == "VAE_CHENGDU_SMALLGRAPH_32_A_graph_kl_lin":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_32"
            self.nets = ['VAE']
            self.graph_topology = True
            self.loss_dict = {
                'VAE':{"MSE_LOSS":{"type":"fixed","value":1}, "KL_DIVERGENCE_LOSS":{"type":"linear","range":{"begin":0, "end":1}},  "JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.5}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
            }
            self.trainingMode = "VAE"
            self.model_settings['VAE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_32_vae_graph.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['VAE'] = 1e-2
            
        elif self.model_case == "VAE_CHENGDU_SMALLGRAPH_32_A_graph_kl_cos":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_32"
            self.nets = ['VAE']
            self.graph_topology = True
            self.loss_dict = {
                'VAE':{"MSE_LOSS":{"type":"fixed","value":1}, "KL_DIVERGENCE_LOSS":{"type":"cos","range":{"min":0, "max":1, "period":25}}, "JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.005}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
            }
            self.trainingMode = "VAE"
            self.model_settings['VAE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_32_vae_graph.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['VAE'] = 1e-2
            
        elif self.model_case == "VAE_CHENGDU_SMALLGRAPH_48_A_graph_kl_fix":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_48"
            self.nets = ['VAE']
            self.graph_topology = True
            self.loss_dict = {   
                'VAE':{ "MSE_LOSS":{"type":"fixed","value":1}, "KL_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "VARIANCE_LOSS": {"type": "fixed","value": 0.8}, "JENSEN_SHANNON_DIVERGENCE_LOSS": {"type": "fixed","value": 0}, "MEDIAN_LOSS_batch": {"type": "fixed","value": 0.5}, "MSE_LOSS": {"type": "fixed","value": 1}, "PEARSON_CORRELATION_LOSS": {"type": "fixed","value":  1e-3} ,"SPEARMAN_CORRELATION_LOSS": {"type": "fixed","value":  1e-3}},
            }
            self.trainingMode = "VAE"
            self.model_settings['VAE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_48_vae_graph.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['VAE'] = 1e-2
            
        elif self.model_case == "VAE_CHENGDU_SMALLGRAPH_48_A_graph_kl_lin":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_48"
            self.nets = ['VAE']
            self.graph_topology = True
            self.loss_dict = {
                'VAE':{"MSE_LOSS":{"type":"fixed","value":1}, "KL_DIVERGENCE_LOSS":{"type":"linear","range":{"begin":0, "end":1}},  "VARIANCE_LOSS":{"type":"fixed","value":0.5}, "JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.5}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
            }
            self.trainingMode = "VAE"
            self.model_settings['VAE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_48_vae_graph.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['VAE'] = 1e-2
            
        elif self.model_case == "VAE_CHENGDU_SMALLGRAPH_48_A_graph_kl_cos":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_48"
            self.nets = ['VAE']
            self.graph_topology = True
            self.loss_dict = {
                'VAE':{"MSE_LOSS":{"type":"fixed","value":1}, "KL_DIVERGENCE_LOSS":{"type":"cos","range":{"min":0, "max":1, "period":25}},  "VARIANCE_LOSS":{"type":"fixed","value":0.5}, "JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.5}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
            }
            self.trainingMode = "VAE"
            self.model_settings['VAE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_48_vae_graph.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['VAE'] = 1e-2
        
        elif self.model_case == "VAE_CHENGDU_SMALLGRAPH_64_A_graph_kl_fix":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_64"
            self.nets = ['VAE']
            self.graph_topology = True
            self.loss_dict = {   
                'VAE':{ "MSE_LOSS":{"type":"fixed","value":1}, "KL_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "VARIANCE_LOSS": {"type": "fixed","value": 0.08}, "JENSEN_SHANNON_DIVERGENCE_LOSS": {"type":"linear","range":{"begin":0, "end":0.5}}, "MEDIAN_LOSS_batch": {"type": "fixed","value": 0.05}, "MSE_LOSS": {"type": "fixed","value": 1}, "PEARSON_CORRELATION_LOSS": {"type": "fixed","value":  1e-2} ,"SPEARMAN_CORRELATION_LOSS": {"type": "fixed","value": 1e-2}},
                    #{"MSE_LOSS": {"type": "fixed","value": 1}, "KL_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "VARIANCE_LOSS": {"type": "fixed","value": 0.3}, "JENSEN_SHANNON_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "MEDIAN_LOSS_batch": {"type": "fixed","value": 0.5}, "SPEARMAN_CORRELATION_LOSS": {"type": "fixed","value": 1}},
            }
            self.trainingMode = "VAE"
            self.model_settings['VAE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_64_vae_graph.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['VAE'] = 1e-2    
        
        elif self.model_case == "VAE_CHENGDU_SMALLGRAPH_96_A_graph_kl_fix":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_96"
            self.nets = ['VAE']
            self.graph_topology = True
            self.loss_dict = {   
                'VAE':{ "MSE_LOSS":{"type":"fixed","value":1}, "KL_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "VARIANCE_LOSS": {"type": "fixed","value": 0.8}, "JENSEN_SHANNON_DIVERGENCE_LOSS": {"type": "fixed","value": 0}, "MEDIAN_LOSS_batch": {"type": "fixed","value": 0.5}, "MSE_LOSS": {"type": "fixed","value": 1}, "PEARSON_CORRELATION_LOSS": {"type": "fixed","value":  1e-3} ,"SPEARMAN_CORRELATION_LOSS": {"type": "fixed","value":  1e-3}},
            }
            self.trainingMode = "VAE"
            self.model_settings['VAE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_96_vae_graph.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['VAE'] = 1e-2
        
        elif self.model_case == "VAE_CHENGDU_SMALLGRAPH_128_A_graph_kl_fix":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_128"
            self.nets = ['VAE']
            self.graph_topology = True
            self.loss_dict = {   
                #'VAE':{"MSE_LOSS": {"type": "fixed","value": 1}, "KL_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "VARIANCE_LOSS": {"type": "fixed","value": 0.5}, "JENSEN_SHANNON_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "MEDIAN_LOSS_batch": {"type": "fixed","value": 0.5}, "SPEARMAN_CORRELATION_LOSS": {"type": "fixed","value": 1}},
                'VAE':{ "MSE_LOSS":{"type":"fixed","value":1}, "KL_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "VARIANCE_LOSS": {"type": "fixed","value": 0.8}, "JENSEN_SHANNON_DIVERGENCE_LOSS": {"type": "fixed","value": 0}, "MEDIAN_LOSS_batch": {"type": "fixed","value": 0.5}, "MSE_LOSS": {"type": "fixed","value": 1}, "PEARSON_CORRELATION_LOSS": {"type": "fixed","value":  1e-3} ,"SPEARMAN_CORRELATION_LOSS": {"type": "fixed","value":  1e-3}},
            }
            self.trainingMode = "VAE"
            self.model_settings['VAE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_128_vae_graph.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['VAE'] = 1e-2    
        
        elif self.model_case == "VAE_CHENGDU_SMALLGRAPH_192_A_graph_kl_fix":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_192"
            self.nets = ['VAE']
            self.graph_topology = True
            self.loss_dict = {   
                #'VAE':{"MSE_LOSS": {"type": "fixed","value": 1}, "KL_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "VARIANCE_LOSS": {"type": "fixed","value": 0.5}, "JENSEN_SHANNON_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "MEDIAN_LOSS_batch": {"type": "fixed","value": 0.5}, "SPEARMAN_CORRELATION_LOSS": {"type": "fixed","value": 1}},
                'VAE':{ "MSE_LOSS":{"type":"fixed","value":1}, "KL_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "VARIANCE_LOSS": {"type": "fixed","value": 0.8}, "JENSEN_SHANNON_DIVERGENCE_LOSS": {"type": "fixed","value": 0}, "MEDIAN_LOSS_batch": {"type": "fixed","value": 0.5}, "MSE_LOSS": {"type": "fixed","value": 1}, "PEARSON_CORRELATION_LOSS": {"type": "fixed","value":  1e-3} ,"SPEARMAN_CORRELATION_LOSS": {"type": "fixed","value":  1e-3}},
            }
            self.trainingMode = "VAE"
            self.model_settings['VAE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_192_vae_graph.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['VAE'] = 1e-2    
            
        elif self.model_case == "VAE_CHENGDU_SMALLGRAPH_256_A_graph_kl_fix":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_256"
            self.nets = ['VAE']
            self.graph_topology = True
            self.loss_dict = {   
                'VAE':{"MSE_LOSS": {"type": "fixed","value": 1}, "KL_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "VARIANCE_LOSS": {"type": "fixed","value": 0.5}, "JENSEN_SHANNON_DIVERGENCE_LOSS": {"type": "fixed","value": 1}, "MEDIAN_LOSS_batch": {"type": "fixed","value": 0.5}, "SPEARMAN_CORRELATION_LOSS": {"type": "fixed","value": 1}},
            }
            self.trainingMode = "VAE"
            self.model_settings['VAE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_256_vae_graph.json')}
            self.timeweather = False
            self.timeweather_settings = {"column_selected":[]}
            self.learning_rate['VAE'] = 1e-2    
        
        
        #CVAE            
        elif self.model_case == "CVAE_CHENGDU_SMALLGRAPH_16_A_graph_kl_lin":
            self.mode = "graph_roads"
            self.name_dataset = "CHENGDU"
            self.version_dataset = "SMALLGRAPH_16"
            self.nets = ['CVAE']
            
            self.graph_topology = True
            self.loss_dict = {
                'CVAE':{"CORRELATION_MATRICES_LOSS":{"type":"fixed","value":1}, "MSE_LOSS":{"type":"fixed","value":1}, "KL_DIVERGENCE_LOSS":{"type":"linear","range":{"begin":0, "end":0.5}},  "VARIANCE_LOSS":{"type":"fixed","value":0.5}, "JENSEN_SHANNON_DIVERGENCE_LOSS":{"type":"fixed","value":1}, "MEDIAN_LOSS_batch":{"type":"fixed","value":0.5}, "SPEARMAN_CORRELATION_LOSS":{"type":"fixed","value":1}},
            }
            self.trainingMode = "CVAE"
            self.model_settings['CVAE']  = {"load_from_file":True, "json_filepath":Path('src','NeuroCorrelation','Models','CHENGDU_models', 'CHENGDU_SMALLGRAPH_settings', 'CHENGDU_SMALLGRAPH_16_cvae_graph.json')}
            self.timeweather = True
            self.timeweather_settings = {"column_selected":[
                'period_dayweek_0','period_dayweek_1','period_dayweek_2','period_dayweek_3','period_dayweek_4','period_dayweek_5','period_dayweek_6',
                'period_start_hh_03','period_start_hh_04','period_start_hh_08','period_start_hh_09','period_start_hh_12','period_start_hh_13','period_start_hh_17','period_start_hh_18','period_start_hh_21','period_start_hh_22',
                'period_start_mm_00_08','period_start_mm_10_18','period_start_mm_20_28','period_start_mm_30_38','period_start_mm_40_48','period_start_mm_50_58',
                'Normalized_temp_0','Normalized_temp_1','Normalized_temp_2','Normalized_temp_3','Normalized_temp_4','Normalized_dwpt_0','Normalized_dwpt_1','Normalized_dwpt_2','Normalized_dwpt_3','Normalized_dwpt_4','Normalized_rhum_0','Normalized_rhum_1','Normalized_rhum_2','Normalized_rhum_3','Normalized_rhum_4','Normalized_wspd_0','Normalized_wspd_1','Normalized_wspd_2','Normalized_wspd_4'
                ]}
        
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
