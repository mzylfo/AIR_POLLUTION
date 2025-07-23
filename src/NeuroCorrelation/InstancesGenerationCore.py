from src.NeuroCorrelation.Models.CHENGDU_models.CHENGDU_SMALLGRAPH_settings.CHENGDU_SMALLGRAPH_settings import CHENGDU_SMALLGRAPH_settings
from src.NeuroCorrelation.Models.CHENGDU_models.CHENGDU_URBAN_ZONE_settings.CHENGDU_URBAN_ZONE_settings import CHENGDU_URBAN_ZONE_settings
from src.NeuroCorrelation.Models.PEMS_METR_models.PEMS_METR_settings import PEMS_METR_settings
from src.NeuroCorrelation.Models.SP500_models.SP500_settings import SP500_settings

from src.NeuroCorrelation.Models.ESG_models.ESG_models import *
from src.NeuroCorrelation.DataLoaders.DataSynteticGeneration import DataSyntheticGeneration
from src.NeuroCorrelation.Models.AutoEncoderModels import *
from src.NeuroCorrelation.ModelTraining.LossFunctions import LossFunction
from src.NeuroCorrelation.ModelTraining.ModelTraining import ModelTraining
from src.NeuroCorrelation.Analysis.DataComparison import DataComparison, DataComparison_Advanced, CorrelationComparison
from src.NeuroCorrelation.Analysis.DataStatistics import DataStatistics
from src.NeuroCorrelation.Analysis.ScenariosMap import ScenariosMap
from src.NeuroCorrelation.DataLoaders.DataMapsLoader import DataMapsLoader
from src.NeuroCorrelation.Optimization.Optimization import Optimization
from src.NeuroCorrelation.Models.NetworkDetails import NetworkDetails
from src.NeuroCorrelation.DataLoaders.DataLoader import DataLoader
from src.NeuroCorrelation.ModelPrediction.ModelPrediction import ModelPrediction
from src.NeuroCorrelation.ModelPrediction.PerformePrediction import PerformePrediction
from src.NeuroCorrelation.Analysis.TimeAnalysis import TimeAnalysis

from termcolor import cprint
from colorama import init, Style

import copy
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
from pathlib import Path
import copy
import os


class InstancesGenerationCore():

    def __init__(self, device, path_folder, epoch, case, model_case, dataset_setting, univar_count, lat_dim, instaces_size, input_shape, do_optimization, opt_settings, num_gen_samples, load_copula, seed=0, run_mode="all", time_slot="A", loss_obj=None):
        device = ("cuda:0" if (torch.cuda.is_available()) else "cpu")
        device = "cpu"
        self.noise_distribution ="gaussian"
        self.seed_torch = seed
        self.seed_data = seed
        self.seed_noise = seed
        print("SETTING PHASE: Seed ")
        print("seed torch:\t",self.seed_torch)
        print("seed data:\t",self.seed_data)
        print("seed noise:\t",self.seed_noise)
        torch.manual_seed(self.seed_torch)

        self.device = device
        print("SETTING PHASE: Device selection")
        print("\tdevice:\t",self.device)
        
        self.univar_count = univar_count        
        self.lat_dim = lat_dim
        self.epoch = epoch
        self.dataset_setting = dataset_setting
        self.batch_size = dataset_setting['batch_size']
        self.path_folder = Path(path_folder)
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
        self.instaces_size = instaces_size
        self.input_shape = input_shape
        self.model_trained = None
        self.model = dict()
        self.num_gen_samples = num_gen_samples
        if loss_obj is None:
            self.loss_obj = dict()
        else:
            self.loss_obj = loss_obj
            
        self.instaces_size_noise = (self.instaces_size, self.lat_dim)
        self.corrCoeff = dict()
               
        self.summary_path = Path(self.path_folder,'summary')
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)
        self.time_performance = TimeAnalysis(folder_path=self.summary_path)
        
        if run_mode=="fast":   
            self.performace_cases = {
                "AE":['train'],
                "VAE":['noise_gaussian'],
                "CVAE":['noise_gaussian'],
                "GAN":['noise_gaussian', 'noise_gaussian_reduced'],
                "WGAN":['noise_gaussian', 'noise_gaussian_reduced'],
            }
            self.draw_plot = False
            self.draw_correlationCoeff = False
            self.draw_scenarios = False
        elif run_mode=="train_only":
            self.performace_cases = {
                "AE":[],
                "VAE":[],
                "CVAE":[],
                "GAN":[],
                "WGAN":[]
            }    
            self.draw_plot = True
            self.draw_correlationCoeff = False
            self.draw_scenarios = False
        elif run_mode=="all":
            self.performace_cases = {
                "AE":['train', 'test', 'noise_gaussian', 'noise_gaussian_reduced', 'noise_copula'],
                "VAE":['noise_gaussian'],
                "CVAE":['noise_gaussian'],
                "GAN":['noise_gaussian', 'noise_gaussian_reduced'],
                "WGAN":['noise_gaussian', 'noise_gaussian_reduced']
            }
            self.draw_plot = True
            self.draw_correlationCoeff = True
            self.draw_scenarios = True
        elif run_mode=="ALL_nocorr":
            self.performace_cases = {
                "AE":['train','noise_gaussian','noise_copula'],
                "VAE":['noise_gaussian'],
                "CVAE":['noise_gaussian'],
                "GAN":[],
                "WGAN":['noise_gaussian', 'noise_gaussian_reduced']
            }
            self.draw_plot = True
            self.draw_correlationCoeff = False
            self.draw_scenarios = False
        elif run_mode=="ALL_nocorr_noplot":
            self.performace_cases = {
                "AE":['train','noise_gaussian','noise_copula'],
                "VAE":['noise_gaussian'],
                "CVAE":['noise_gaussian'],
                "GAN":[],
                "WGAN":['noise_gaussian', 'noise_gaussian_reduced']
            }
            self.draw_plot = False
            self.draw_correlationCoeff = False
            self.draw_scenarios = False
        elif run_mode=="ALL_noscen":
            self.performace_cases = {
                "AE":['train', 'test', 'noise_gaussian', 'noise_gaussian_reduced', 'noise_copula'],
                "VAE":['noise_gaussian'],
                "CVAE":['noise_gaussian'],
                "GAN":['noise_gaussian', 'noise_gaussian_reduced'],
                "WGAN":['noise_gaussian', 'noise_gaussian_reduced']
            }
            self.draw_plot = True
            self.draw_correlationCoeff = True
            self.draw_scenarios = False
            
        print("SETTING PHASE: Model creation")
        print("\tmodel_case:\t",model_case)
        self.model_case = model_case
        self.case = case
        
        if self.case == "PEMS_METR" or self.case == "CHENGDU_SMALLGRAPH" or self.case == "CHENGDU_ZONE":
            if self.case == "PEMS_METR":
                self.case_setting =   PEMS_METR_settings(model_case=self.model_case, device=self.device, univar_count=self.univar_count, lat_dim=self.lat_dim, dataset_setting=self.dataset_setting, epoch=self.epoch, path_folder=self.path_folder, corrCoeff=self.corrCoeff, instaces_size=self.instaces_size, time_performance = self.time_performance)
            elif self.case == "CHENGDU_SMALLGRAPH":
                self.time_slot = time_slot
                self.case_setting =   CHENGDU_SMALLGRAPH_settings(model_case=self.model_case, device=self.device, univar_count=self.univar_count, lat_dim=self.lat_dim, dataset_setting=self.dataset_setting, epoch=self.epoch, path_folder=self.path_folder, corrCoeff=self.corrCoeff, instaces_size=self.instaces_size, time_performance = self.time_performance, time_slot=self.time_slot, noise_distribution=self.noise_distribution)
            elif self.case == "CHENGDU_ZONE":
                self.time_slot = time_slot
                self.case_setting =   CHENGDU_URBAN_ZONE_settings(model_case=self.model_case, device=self.device, univar_count=self.univar_count, lat_dim=self.lat_dim, dataset_setting=self.dataset_setting, epoch=self.epoch, path_folder=self.path_folder, corrCoeff=self.corrCoeff, instaces_size=self.instaces_size, time_performance = self.time_performance, time_slot=self.time_slot)
            
            self.trainingMode = self.case_setting.get_trainingMode()
            self.path_folder_nets = self.case_setting.get_folder_nets()
            dataloader = self.case_setting.get_DataLoader(seed_data=self.seed_data)
            self.graph_topology = self.case_setting.get_graph_topology()
            self.learning_rate = self.case_setting.get_learning_rate()
            print("self.learning_rate---",self.learning_rate)
            #se ho ottimizzato la loss, prendo quella
            if loss_obj is None:
                self.loss_obj = self.case_setting.get_LossFunction()
            else:
                self.loss_obj = loss_obj
            self.timeweather_count = self.case_setting.get_timeweather_count()
        
        if self.case == "SP500":
            print("---------SP500")
            if self.case == "SP500":
                self.case_setting =   SP500_settings(model_case=self.model_case, device=self.device, univar_count=self.univar_count, lat_dim=self.lat_dim, dataset_setting=self.dataset_setting, epoch=self.epoch, path_folder=self.path_folder, corrCoeff=self.corrCoeff, instaces_size=self.instaces_size, time_performance = self.time_performance)
            
            self.trainingMode = self.case_setting.get_trainingMode()
            self.path_folder_nets = self.case_setting.get_folder_nets()
            dataloader = self.case_setting.get_DataLoader(seed_data=self.seed_data)
            self.graph_topology = self.case_setting.get_graph_topology()
            self.learning_rate = self.case_setting.get_learning_rate()
            print("self.learning_rate---",self.learning_rate)
            #se ho ottimizzato la loss, prendo quella
            if loss_obj is None:
                self.loss_obj = self.case_setting.get_LossFunction()
            else:
                self.loss_obj = loss_obj
            self.timeweather_count = self.case_setting.get_timeweather_count()    
            
        # TO REFACTORY
        '''
        # ESG - Environmental Social Governance project
        elif self.model_case=="ESG__GAN_linear_pretrained_35":
            self.graph_topology = False
            dataloader = DataLoader(mode="fin_data", seed=self.seed_data, name_dataset="ESG_35", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.05, "VARIANCE_LOSS":0.05, "SPEARMAN_CORRELATION_LOSS":0.001}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        '''
        
        
        
        self.modelTrainedAE = None
        self.data_splitted = dataloader.dataset_load(draw_plots=self.draw_plot, save_summary=True, loss=self.loss_obj, draw_correlationCoeff=self.draw_correlationCoeff)
        
        self.dataGenerator = dataloader.getDataGenerator()
        self.vc_mapping = dataloader.get_vcMapping()
        self.rangeData = dataloader.getRangeData()
        print("rangeData:\t",self.rangeData)
        self.statsData = dataloader.get_statsData()
        self.path_map = dataloader.get_pathMap()
        self.edge_index = dataloader.get_edgeIndex()
        self.case_setting.set_edge_index(self.edge_index)
        
        self.copulaData_filename  = dataloader.get_copulaData_filename()
        self.load_copula = load_copula           
        self.use_copula = True
        if self.load_copula and self.copulaData_filename is not None:
            cprint(Style.BRIGHT +f"| Copula data   : Load data from {self.copulaData_filename}" + Style.RESET_ALL, 'magenta', attrs=["bold"])
        else:
            self.load_copula = False
            cprint(Style.BRIGHT +f"| Copula data   : Generate data" + Style.RESET_ALL, 'magenta', attrs=["bold"])
            
        self.case_setting.deploy_models()
        for key in self.loss_obj:
            self.loss_obj[key].set_stats_data(self.statsData, self.vc_mapping)
        
        
        if do_optimization: 
            self.do_optimization = True
            time_opt_folder = Path(self.summary_path,"time_optimization")
            if not os.path.exists(time_opt_folder):
                os.makedirs(time_opt_folder)
            
            opt_time_analysis = self.time_performance
            self.optimization = Optimization(model=self.model, device=self.device, data_dict=self.data_splitted,
                loss=self.loss_obj, path_folder=self.path_folder, time_performance = opt_time_analysis,
                univar_count=self.univar_count, batch_size=self.batch_size, latent_dim=self.lat_dim, vc_mapping=self.vc_mapping, 
                input_shape=self.input_shape, rangeData=self.rangeData, dataGenerator=self.dataGenerator, learning_rate=self.learning_rate,
                instaces_size_noise=self.instaces_size_noise, direction="maximize", timeout=600,
                graph_topology=self.graph_topology, edge_index=self.edge_index, timeweather_count=self.timeweather_count )
            self.optimization.set_fromDict(opt_settings)
        else:
            self.do_optimization = False
            self.optimization = None

    def start_experiment(self, load_model=False):
        comparison_corr_list = list()
        if self.graph_topology:
            net_details = NetworkDetails(model=self.model, loss=self.loss_obj, path=self.summary_path, edge_index = self.data_splitted['edge_index'])
        else:
            net_details = NetworkDetails(model=self.model, loss=self.loss_obj, path=self.summary_path)
        net_details.saveModelParams()   
        
        
        if self.trainingMode in ["AE>GAN","AE>WGAN"]:
            self.model["AE"] =  self.case_setting.get_model(key="AE")
            self.loss_obj["AE"].set_coefficent(epochs_tot=self.epoch["AE"], path_folder=self.path_folder_nets["AE"])
            if self.do_optimization:
                loss_ob = self.optimization_model(model_type="AE", model=self.model["AE"], loss_obj=self.loss_obj["AE"])
                return loss_ob
            else:
                trained_obj_ae = self.training_model(data_dict=self.data_splitted, model_type="AE", optimizer_trial=self.optimization,  model=self.model["AE"], loss_obj=self.loss_obj["AE"], pre_trained_decoder=False, epoch=self.epoch, model_flatten_in=True, load_model=False, graph_topology = self.graph_topology)
                model_ae_trained = trained_obj_ae[0]
                key_dataout = "ae"
                self.data_metadata = [{"acronym":"ae", "color":(1.0, 0.498, 0.055), "label":key_dataout}]
                self.predict_model(model=model_ae_trained, model_type="AE", data=self.data_splitted, path_folder_pred=self.path_folder_nets["AE"], path_folder_data=self.path_folder, noise_samples=self.num_gen_samples, input_shape="vector", draw_plot=self.draw_plot, draw_scenarios=self.draw_scenarios, draw_correlationCoeff=self.draw_correlationCoeff, key_dataout=key_dataout)   
                model_ae_decoder, model_ae_decoder_size, model_ae_decoder_permutation_forward = model_ae_trained.getModel("decoder", train=True, extra_info=True)
            if self.trainingMode == "AE>GAN":
                model_key = "GAN"
            if self.trainingMode == "AE>WGAN":
                model_key = "WGAN"
            ''' self.model[model_key] = self.case_setting.get_model(key=model_key)
            self.model[model_key].set_partialModel(key="generator", model_net=model_ae_decoder, model_size=model_ae_decoder_size, model_permutation_forward=model_ae_decoder_permutation_forward)
            self.loss_obj[model_key].set_coefficent(epochs_tot=self.epoch["GAN"], path_folder=self.path_folder_nets[model_key])
            trained_obj_gan = self.training_model(self.data_splitted, model_type=model_key, model=self.model[model_key], loss_obj=self.loss_obj[model_key], pre_trained_decoder=True,epoch=self.epoch)
            model_gan_trained = trained_obj_gan[0]
            self.data_metadata = [{"acronym":"ae", "color":(1.0, 0.498, 0.055), "label":"VAE"}]
            self.predict_model(model=model_gan_trained, model_type=model_key, data=self.data_splitted, path_folder_pred=self.path_folder_nets[model_key], path_folder_data=self.path_folder, noise_samples=self.num_gen_samples, input_shape="vector", draw_plot=self.draw_plot, draw_scenarios=self.draw_scenarios, draw_correlationCoeff=self.draw_correlationCoeff)   
            '''
            
        elif self.trainingMode in ["VAE"]:
            
            self.model["VAE"] =  self.case_setting.get_model(key="VAE")
            self.loss_obj["VAE"].set_coefficent(epochs_tot=self.epoch["VAE"], path_folder=self.path_folder_nets["VAE"])
            
            if self.do_optimization:
                loss_ob = self.optimization_model(model_type="VAE", model=self.model["VAE"], loss_obj=self.loss_obj["VAE"])
                return loss_ob
            else:
                trained_obj = self.training_model(data_dict=self.data_splitted, model_type="VAE", optimizer_trial=self.optimization,  model=self.model["VAE"], loss_obj=self.loss_obj["VAE"], pre_trained_decoder=False, epoch=self.epoch, model_flatten_in=True, load_model=False, graph_topology = self.graph_topology)
                model_trained = trained_obj[0]
                
                key_dataout = "vae"
                self.data_metadata = [{"acronym":"vae", "color":(1.0, 0.498, 0.055), "label":"VAE"}]
                self.predict_model(model=model_trained, model_type="VAE", data=self.data_splitted, path_folder_pred=self.path_folder_nets["VAE"], path_folder_data=self.path_folder, noise_samples=self.num_gen_samples, input_shape="vector", draw_plot=self.draw_plot, draw_scenarios=self.draw_scenarios, draw_correlationCoeff=self.draw_correlationCoeff, key_dataout = key_dataout)   
            
        elif self.trainingMode in ["CVAE"]:
            
            self.model["CVAE"] =  self.case_setting.get_model(key="CVAE")
            self.loss_obj["CVAE"].set_coefficent(epochs_tot=self.epoch["CVAE"], path_folder=self.path_folder_nets["CVAE"])
            
            
            
            
            
            trained_obj = self.training_model(data_dict=self.data_splitted, model_type="CVAE", model=self.model["CVAE"], loss_obj=self.loss_obj["CVAE"], epoch=self.epoch, graph_topology = self.graph_topology, optimization=self.do_optimization, optimizer_trial=self.optimization)
            model_trained = trained_obj[0]
            key_dataout = "CVAE"
            self.data_metadata = [{"acronym":"cvae", "color":(1.0, 0.498, 0.055), "label":key_dataout}]
            self.predict_model(model=model_trained, model_type="CVAE", data=self.data_splitted, path_folder_pred=self.path_folder_nets["CVAE"], path_folder_data=self.path_folder, noise_samples=self.num_gen_samples, input_shape="vector", draw_plot=self.draw_plot, draw_scenarios=self.draw_scenarios, draw_correlationCoeff=self.draw_correlationCoeff, key_dataout=key_dataout)   

        ''' 
        #ESG
        elif self.model_case=="ESG__GAN_linear_pretrained_35":
            self.model['AE'] = ESG__GEN_autoEncoder_35
            self.graph_topology = False
            trained_obj_ae = self.training_model(data_dict=self.data_splitted, model_type="AE", model=self.model['AE'], loss_obj=self.loss_obj['AE'], epoch=self.epoch, graph_topology = self.graph_topology, optimization=self.do_optimization, optimizer_trial=self.optimization)
            model_ae_trained = trained_obj_ae[0]
            self.predict_model(model=model_ae_trained, model_type="AE", data=self.data_splitted,path_folder_pred=self.path_folder_nets["AE"], path_folder_data=self.path_folder, noise_samples=self.num_gen_samples, input_shape="vector", draw_plot=self.draw_plot, draw_scenarios=self.draw_scenarios, draw_correlationCoeff=self.draw_correlationCoeff)   
            model_ae_decoder = model_ae_trained.getModel("decoder",train=True)            
            self.model['GAN'] = ESG__GAN_neural_mixed_35(generator=model_ae_decoder)
            trained_obj_gan = self.training_model(self.data_splitted, model_type="GAN", model=self.model['GAN'], loss_obj=self.loss_obj['GAN'], pre_trained_decoder=True,epoch=self.epoch)
            model_gan_trained = trained_obj_gan[0]
            self.predict_model(model=model_gan_trained, model_type="GAN", data=self.data_splitted, path_folder_pred=self.path_folder_nets["GAN"], path_folder_data=self.path_folder, noise_samples=self.num_gen_samples, input_shape="vector", draw_plot=self.draw_plot, draw_scenarios=self.draw_scenarios, draw_correlationCoeff=self.draw_correlationCoeff)   
        ''' 
        
        
        
        corr_comp = CorrelationComparison(self.corrCoeff, self.path_folder)
        corr_comp.compareMatrices(comparison_corr_list)        
        return None
    
    def optimization_model(self, model_type, model=None, loss_obj=None):
        if loss_obj is None:
            loss_obj = self.loss_obj
        if model is None:
            model = self.model
        print("\t\tGRAPH TOPOLOGY:\t",self.graph_topology)
    
        print("\tOPTIMIZATION:\tTrue")
        
        self.optimization.set_modeltype(model_type=model_type)
        self.optimization.optimization()
        self.optimization.setValuesOptimized(loss_obj= loss_obj, model_type=model_type)
        print(model_type,":\t",self.loss_obj[model_type].get_lossTerms())
        return self.loss_obj
            
    def training_model(self, data_dict, model_type, optimizer_trial=None,  model=None, loss_obj=None, pre_trained_decoder=False, epoch=None, model_flatten_in=True, load_model=False, graph_topology=False):
        if loss_obj is None:
            loss_obj = self.loss_obj
        if model is None:
            model = self.model
        if epoch is None:
            epoch=self.epoch
        print("\t\tGRAPH TOPOLOGY:\t",self.graph_topology)
    
        
            
        print("TRAINING PHASE: Training data - ", model_type)
        train_data = data_dict['train_data']   
        test_data = data_dict['test_data'] 
        noise_data = data_dict['noise_data'] 
        edge_index = data_dict['edge_index']
            
        training_obj = ModelTraining(model=model, device=self.device, loss_obj=loss_obj, epoch=epoch, learning_rate=self.learning_rate, train_data=train_data, test_data=test_data, dataGenerator=self.dataGenerator, path_folder=self.path_folder, univar_count_in = self.univar_count, univar_count_out = self.univar_count, latent_dim=self.lat_dim, timeweather_count = self.timeweather_count, model_type=model_type, pre_trained_decoder=pre_trained_decoder, vc_mapping = self.vc_mapping,input_shape=self.input_shape, rangeData=self.rangeData,batch_size=self.batch_size, optimization=False, graph_topology=graph_topology, edge_index=edge_index, time_performance=self.time_performance, noise_data=noise_data)
        if model_type =="AE":
            optim_score = training_obj.training(training_name=f"MAIN_",model_flatten_in=model_flatten_in,load_model=load_model)
        if model_type =="VAE":
            optim_score = training_obj.training(training_name=f"MAIN_",model_flatten_in=model_flatten_in,load_model=load_model)
        if model_type =="CVAE":
            optim_score = training_obj.training(training_name=f"MAIN_",model_flatten_in=model_flatten_in,load_model=load_model)
        elif model_type in ["GAN","WGAN"]:
            optim_score = training_obj.training(training_name=f"MAIN_",noise_size=self.instaces_size_noise, load_model=load_model)
        
        training_obj.eval()
        
        if optimizer_trial is None:
            return training_obj, None
        else:
            return training_obj, optim_score

    def predict_model(self, model, model_type,  data, input_shape, path_folder_data, path_folder_pred, noise_samples, key_dataout, draw_plot=True, draw_scenarios=True, draw_correlationCoeff=True):
        print("load copula--------------",self.load_copula)
        predMod = PerformePrediction(model=model, device=self.device, model_type=model_type, univar_count=self.univar_count, latent_dim=self.lat_dim, data=data, dataGenerator=self.dataGenerator, input_shape = input_shape, rangeData=self.rangeData, vc_mapping=self.vc_mapping, draw_plot=draw_plot, draw_scenarios=draw_scenarios, draw_correlationCoeff= draw_correlationCoeff, noise_samples=noise_samples, path_folder_pred=path_folder_pred, path_folder_data= path_folder_data, path_map=self.path_map, copulaData_filename=self.copulaData_filename, load_copula=self.load_copula, use_copula=self.use_copula, time_performance=self.time_performance, data_metadata=self.data_metadata,key_dataout=key_dataout)
        predMod.predict_model(cases_list = self.performace_cases[model_type])
        self.time_performance.save_time()