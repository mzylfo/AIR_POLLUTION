from src.NeuroCorrelation.DataLoaders.DataSynteticGeneration import DataSyntheticGeneration
from src.NeuroCorrelation.DataLoaders.DataMapsLoader import DataMapsLoader

import math
from pathlib import Path
import os
import torch
from torch import Tensor
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class DataLoader:
    
    def __init__(self, mode, seed,  name_dataset, version_dataset, device, dataset_setting, epoch, univar_count, lat_dim, corrCoeff, instaces_size, path_folder, time_performance, timeweather, timeweather_settings, noise_distribution="gaussian", vc_dict=None, univ_limit=150,  time_slot=None):
        
        self.mode = mode
        self.seed = seed
        self.name_dataset = name_dataset
        self.version_dataset = version_dataset
        self.vc_mapping = None
        self.path_folder = path_folder
        self.instaces_size = instaces_size
        self.device = device
        self.univar_count = univar_count
        self.lat_dim = lat_dim
        self.rangeData = None
        self.epoch = epoch
        self.dataGenerator = None
        self.dataset_setting = dataset_setting
        self.starting_sample = self.checkInDict(self.dataset_setting,"starting_sample",20)
        self.train_percentual = self.checkInDict(self.dataset_setting,"train_percentual",0.70)        
        self.train_samples = self.checkInDict(self.dataset_setting,"train_samples", 50)
        self.test_samples = self.checkInDict(self.dataset_setting,"test_samples", 500)
        self.noise_samples = self.checkInDict(self.dataset_setting,"noise_samples", 1000)
        self.corrCoeff = corrCoeff
        self.noise_distribution = noise_distribution
        self.corrCoeff['data'] = dict()
        self.summary_path = Path(self.path_folder,'summary')
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)
        self.statsData = None
        self.vc_dict = vc_dict
        self.univ_limit = univ_limit
        self.pathMap = None
        self.edge_index = None
        self.timeweather = timeweather
        self.timeweather_settings = timeweather_settings
        
        self.time_slot = time_slot
        self.time_performance = time_performance
        
    def dataset_load(self, draw_plots=True, save_summary=True, loss=None, draw_correlationCoeff=True):
        self.loss = loss
        if self.mode=="random_var" and self.name_dataset=="3var_defined":
            print("DATASET PHASE: Sample generation")
            self.dataGenerator = DataSyntheticGeneration(torch_device=self.device, univar_count=self.univar_count, lat_dim=self.lat_dim, path_folder=self.path_folder)
            
            self.dataGenerator.casualVC_init_3VC(num_of_samples = self.starting_sample, draw_plots=draw_plots)
            train_data, self.corrCoeff['data']['train_data'] = self.dataGenerator.casualVC_generation(name_data="train", num_of_samples = self.train_samples, draw_plots=draw_plots)
            test_data, self.corrCoeff['data']['test'] = self.dataGenerator.casualVC_generation(name_data="test", num_of_samples = self.test_samples,  draw_plots=draw_plots)
            noise_data = self.dataGenerator.get_synthetic_noise_data(name_data="noise", num_of_samples = self.noise_samples, draw_plots=draw_plots)
            self.vc_mapping = ['X', 'Y', 'Z']
            self.pathMap = None
            self.edge_index = None
            timeweather_data = None

        if self.mode=="random_var" and self.name_dataset=="copula":
            print("DATASET PHASE: Sample copula generation")
            self.dataGenerator = DataSyntheticGeneration(torch_device=self.device, univar_count=self.univar_count, lat_dim=self.lat_dim, path_folder=self.path_folder)
            
            if self.vc_dict is None:
                self.vc_dict = {"X":{"dependence":None}, "Y":{"dependence":{"X":1.6}}, "Z":{"dependence":{"X":3}}, "W":{"dependence":None},"K":{"dependence":{"W":0.5}}, "L":{"dependence":{"W":5}}, "M":{"dependence":None}}
            self.vc_mapping = list()
            for key_vc in self.vc_dict:
                self.vc_mapping.append(key_vc)
            
            self.dataGenerator.casualVC_init_multi(num_of_samples = self.starting_sample, vc_dict=self.vc_dict, draw_plots=draw_plots)
            train_data, self.corrCoeff['data']['train'] = self.dataGenerator.casualVC_generation(name_data="train", univar_count=self.univar_count, num_of_samples = self.train_samples, draw_plots=draw_plots, instaces_size=self.instaces_size)
            test_data, self.corrCoeff['data']['test'] = self.dataGenerator.casualVC_generation(name_data="test", univar_count=self.univar_count, num_of_samples = self.test_samples,  draw_plots=draw_plots, instaces_size=self.instaces_size)
            noise_data = self.dataGenerator.get_synthetic_noise_data(name_data="noise", num_of_samples = self.noise_samples, draw_plots=draw_plots, instaces_size=self.instaces_size)
            self.pathMap = None
            self.edge_index = None
            self.timeweather_count = 0
        
        
        
        if self.mode =="graph_roads" or self.mode=="fin_data":
            if self.mode =="graph_roads":
                print("DATASET PHASE: Load maps data")
            elif self.mode=="fin_data":
                print("DATASET PHASE: Load maps data")
            print("draw_plots ",draw_plots)
            
            self.dataGenerator = DataMapsLoader(torch_device=self.device, seed=self.seed, name_dataset=self.name_dataset, version_dataset=self.version_dataset, time_performance=self.time_performance, time_slot=self.time_slot,lat_dim=self.lat_dim, univar_count=self.univar_count, path_folder=self.path_folder, univ_limit=self.univ_limit, timeweather=self.timeweather, timeweather_settings=self.timeweather_settings, noise_distribution=self.noise_distribution)
            self.dataGenerator.mapsVC_load(train_percentual=self.train_percentual, draw_plots=draw_plots)
            
            train_data, self.corrCoeff['data']['train'] = self.dataGenerator.mapsVC_getData(name_data="train", draw_plots=draw_plots, draw_correlationCoeff=draw_correlationCoeff)
            test_data, self.corrCoeff['data']['test'] = self.dataGenerator.mapsVC_getData(name_data="test",  draw_plots=draw_plots, draw_correlationCoeff=draw_correlationCoeff)
            noise_data = self.dataGenerator.get_synthetic_noise_data(name_data="noise", num_of_samples = self.noise_samples, draw_plots=draw_plots)
            self.vc_mapping = self.dataGenerator.get_vc_mapping()
            self.pathMap = self.dataGenerator.get_pathMap()
            self.edge_index = self.dataGenerator.get_edgeIndex()
            self.timeweather_count = self.dataGenerator.getTimeweatherCount()
            self.copulaData_filename = self.dataGenerator.get_copulaData_filename()
            
            
            
                 
        print("----------------------timeweather_count-----------------------:  ",self.timeweather_count)
        self.rangeData = self.dataGenerator.getDataRange()
        self.statsData = self.dataGenerator.getDataStats()
        
        reduced_noise_data = self.generateNoiseReduced(method="percentile", percentile_points=10)
        
        self.export_datasplit(data=train_data, name_split="train_data", key="sample")
        self.export_datasplit(data=train_data, name_split="train_data", key="sample_timeweather")
        
        self.export_datasplit(data=test_data, name_split="test_data", key="sample")
        self.export_datasplit(data=test_data, name_split="test_data", key="sample_timeweather")
        
        data_dict = {"train_data":train_data, "test_data":test_data, "noise_data":noise_data, "reduced_noise_data":reduced_noise_data, "edge_index":self.edge_index}
        
        if save_summary:
            self.saveDataset_setting()
        return data_dict
    
    def get_vcMapping(self):
        return self.vc_mapping
    
    def get_statsData(self):
        if self.statsData is None:
            raise Exception("rangeData not defined.")
        return self.statsData
        
    def getDataGenerator(self):
        if self.dataGenerator is None:
            raise Exception("rangeData not defined.")
        return self.dataGenerator
    
    def get_copulaData_filename(self):
        return self.copulaData_filename
    
    def getRangeData(self):
        if self.rangeData is None:
            raise Exception("rangeData not defined.")
        return self.rangeData
    
    def get_pathMap(self):
        return self.pathMap
        
    def get_edgeIndex(self):
        return self.edge_index
    
    def checkInDict(self, dict_obj, key, value_default):
        if key in dict_obj:
            if dict_obj[key] is not None:
                value = dict_obj[key]
            else:
                value = value_default
        else:
            value = value_default
        return value

    def saveDataset_setting(self):
        settings_list = []
        settings_list.append(f"dataset settings") 
        settings_list.append(f"================") 
        settings_list.append(f"mode_dataset:: {self.mode}") 
        settings_list.append(f"name_dataset:: {self.name_dataset}")
        if self.time_slot is not None:
            settings_list.append(f"time_slot:: {self.time_slot}") 
        settings_list.append(f"mode_dataset:: {self.epoch}") 
        
         
        for key in self.dataset_setting:
            print("saveDataset_setting\t",key)
            data_summary = self.dataset_setting[key]         
            summary_str = f"{key}:: {data_summary}"
            settings_list.append(summary_str)
            
        
        if self.loss is not None:
            settings_list.append(f" ") 
            settings_list.append(f"loss settings") 
            settings_list.append(f"================") 
            for key in self.loss:
                settings_list.append(f"loss part:: {key} -") 
                loss_terms = self.loss[key].get_lossTerms()
                for item in loss_terms:
                    settings_list.append(f"\t\t:: {item} \t\tcoef:: {loss_terms[item]}") 
        
        setting_str = '\n'.join(settings_list)    
        filename = Path(self.summary_path, "summary_dataset.txt")
        with open(filename, 'w') as file:
            file.write(setting_str)
        print("SETTING PHASE: Summary dataset file - DONE")
    
    
    
    
    def find_nearest_kde(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    '''        
    methods :
        'all'     : all possible combination of redux_noise_values
        'percentile' : sampling between 2 different percentile
    '''
    def generateNoiseReduced(self, method, percentile_points = 10,draw_plot = True):
        noise_reduced_path_folder_a = Path(self.path_folder,"maps_analysis_"+self.name_dataset)
        if not os.path.exists(noise_reduced_path_folder_a):
            os.makedirs(noise_reduced_path_folder_a)
        noise_red_path_folder = Path(noise_reduced_path_folder_a,"noise_reduced_data_analysis")
        if not os.path.exists(noise_red_path_folder):
            os.makedirs(noise_red_path_folder)
            
        print("\tNoiseReduced method: ",method)
        noise_redux_samples = list()
        if method=='all':
            redux_noise = list()
            redux_noise_values = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
            redux_noise_values = [-1,  -0.5,  0,  0.5,  1]
            high = self.lat_dim
            c = self.generateNoisePercentile(high, redux_noise_values)
            
            for c_item in c:
                noise_redux_samples.append({'sample': torch.Tensor(c_item).to(device=self.device), 'noise': torch.Tensor(c_item).to(device=self.device)})
            print("\tNoiseReduced samples - all: done")
        elif method== 'percentile':
            mu, sigma = 0, math.sqrt(1) # mean and standard deviation
            s = np.random.normal(mu, sigma, 1000)
            k = 0
            window = 100/(percentile_points)
            values = []
            o = 0
            while o < percentile_points:
                o += 1
                l = np.percentile(s, k)
                if  k+window>=100:
                    r = 100
                else:
                    r = np.percentile(s, k+window)
                c = np.random.uniform(l, r, self.lat_dim)
                noise_redux_samples.append({'sample': torch.Tensor(c).to(device=self.device), 'noise': torch.Tensor(c).to(device=self.device)})
                k += window
        print("\tNoiseReduced samples: done")            
        return noise_redux_samples

    def generateNoisePercentile(self, high, redux_noise_values):
        if high == 0:
            return None
        else:    
            recur_list = list()
            recur_values = self.generateNoisePercentile(high-1, redux_noise_values)
            if recur_values is None:
                for item in redux_noise_values:
                    recur_list.append([item])
                return recur_list
            else:
                recur_list = list()
                for item_list in recur_values:
                    for i in range(len(redux_noise_values)):
                        a = item_list.copy()
                        a.append(redux_noise_values[i])
                        recur_list.append(a)
                return recur_list    
        
    def export_datasplit(self, data, name_split, key='sample'):
        if not torch.any(torch.isnan(data[0][key])):
            list_data = list()
            for inp in range(len(data)):            
                list_data.append(data[inp][key])
                
            data_byVar = dict()
            for univ_id in range(self.univar_count):
                data_byVar[univ_id] = list()
            
            df_export = pd.DataFrame(columns=['x_input']) 
            max_val = self.rangeData["max_val"]
            min_val = self.rangeData["min_val"]
            diff_minmax = max_val - min_val
            for x in list_data:
                x_list = [(i * diff_minmax) + min_val for i in x.detach().cpu().numpy()]
                x_list_clean = [float(f"{val:.8f}") for val in x_list]  # Precisione decimale utile
                x_list_str = "[" + ", ".join(map(str, x_list_clean)) + "]"
                new_row = {'x_input': x_list_str}
                df_export.loc[len(df_export)] = new_row

            datasplit_path = Path(self.path_folder,'datasplit')
            if not os.path.exists(datasplit_path):
                os.makedirs(datasplit_path)
            path_file = Path(datasplit_path,f"datasplit_{name_split}_{key}.csv")
            df_export.to_csv(path_file)