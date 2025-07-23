import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
import json
from matplotlib.ticker import PercentFormatter
import datetime

class ModelPrediction():

    def __init__(self, model, device, dataset, univar_count_in, univar_count_out, latent_dim, path_folder, vc_mapping, time_performance, model_type, data_range=None, input_shape="vector", isnoise_in = False, isnoise_out = False):
        self.model = model
        self.dataset = dataset
        self.univar_count_in = univar_count_in
        self.univar_count_out = univar_count_out
        self.latent_dim = latent_dim
        self.path_folder = path_folder
        self.device = device
        self.vc_mapping = vc_mapping
        self.time_performance = time_performance
        if data_range is not None:
            self.max_val = data_range['max_val']
            self.min_val = data_range['min_val']
            
        self.inpu_data = list()
        self.late_data = list()
        self.pred_data = list()

        self.inpu_byVar = dict()
        self.pred_byVar = dict()
        self.late_byComp = dict()

        self.input_shape = input_shape
        self.isnoise_in = isnoise_in
        self.isnoise_out = isnoise_out
        self.model_type = model_type
        if self.model_type in ["VAE"]:
            self.latent_keys = ["mu","logvar"]
        else:
            self.latent_keys = ["latent"]
        
    
    def predict(self, input_sample, time_key, pred2numpy=True, latent=True, save=True, experiment_name=None, remapping=False):
        self.inpu_data = list()
        self.late_data = dict()
        
        for late_key in self.latent_keys:
            self.late_data[late_key] = list()
                    
        self.pred_data = list()
        pred_time = datetime.timedelta(0)
        
        
        time_key = f"{time_key}_pred"
        for inp in input_sample:            
            with torch.no_grad():
                
                input_2d = [inp["sample"]]
                x_in = torch.Tensor(1, self.univar_count_in).to(device=self.device)
                torch.cat(input_2d, out=x_in) 
                if self.input_shape == "vector":
                    x_in = x_in.view(-1,self.univar_count_in)
                elif self.input_shape == "matrix":
                    x_in = x_in
                x_in.unsqueeze_(1)
                
                with torch.no_grad():
                    self.time_performance.start_time(time_key)

                    out = self.model(x_in)  
                    self.time_performance.stop_time(time_key)
                self.inpu_data.append(out["x_input"]['data'][0][0])
                
                if "x_latent" in out:
                    for late_key in self.latent_keys:
                        self.late_data[late_key].append(out["x_latent"][late_key][0][0])
                
                self.pred_data.append(out["x_output"]['data'][0][0])
                
                
        self.time_performance.compute_time(time_key, fun = "mean") 
        print(f"\tTIME to make {time_key} prediction:\t",self.time_performance.get_time(time_key, fun = "mean"))
        self.predict_sortByUnivar(pred2numpy=pred2numpy)
        if latent:
            self.latent_sortByComponent(pred2numpy=pred2numpy)
        if save:
            self.saveData(experiment_name, latent, remapping)

    def saveData(self, experiment_name, latent, remapping=True):
        if latent:
            columns = ['x_input']
            for late_key in self.latent_keys:
                columns.append(f'x_latent_{late_key}')
            columns.append('x_output')
        else: 
            columns = ['x_input', 'x_output']
        df_export = pd.DataFrame(columns=columns) 

        if remapping:
            diff_minmax = self.max_val - self.min_val
        
        if latent:
            for x,z,y in zip(self.inpu_data, range(len(self.inpu_data)), self.pred_data):
                if remapping is not None:
                    x_list = [(i*diff_minmax)+self.min_val for i in x.detach().cpu().numpy()]
                    y_list = [(i*diff_minmax)+self.min_val for i in y.detach().cpu().numpy()]
                else:
                    x_list = x.detach().cpu().numpy()
                    y_list = y.detach().cpu().numpy()        
                
                z_list = dict()
                
                for late_key in self.latent_keys:
                    z_list[late_key]= self.late_data[late_key][z].detach().cpu().numpy()
                
                new_row = {'x_input': x_list}
                for late_key in self.latent_keys:
                    new_row[f'x_latent_{late_key}'] = z_list[late_key]
                new_row['x_output'] =  y_list
                df_export.loc[len(df_export)] = new_row
        else: 
            for x,y in zip(self.inpu_data, self.pred_data):
                if remapping is not None:
                    x_list = [(i*diff_minmax)+self.min_val for i in x.detach().cpu().numpy()]
                    y_list = [(i*diff_minmax)+self.min_val for i in y.detach().cpu().numpy()]
                else:
                    x_list = x.detach().cpu().numpy()
                    y_list = y.detach().cpu().numpy()        
                new_row = {'x_input': x_list, 'x_output': y_list}
                df_export.loc[len(df_export)] = new_row
                
        path_file = Path(self.path_folder,"prediced_instances_"+experiment_name+'.csv')
        df_export.to_csv(path_file)
        
    def predict_sortByUnivar(self, pred2numpy=True, mode=["input", "output"]):
        
        if "input" in mode:
            self.inpu_byVar = dict() 
            for univ_id in range(self.univar_count_in):
                self.inpu_byVar[univ_id] = list()  
            
            for inp in zip(self.inpu_data):
                if self.input_shape == "vector":
                    for univ_id in range(self.univar_count_in):
                        if pred2numpy:
                            self.inpu_byVar[univ_id].append(inp[0][univ_id].detach().cpu().numpy())
                        else:
                            self.inpu_byVar[univ_id].append(inp[0][univ_id])
                elif self.input_shape == "matrix":
                    for univ_id in range(self.univar_count_in):
                        for in_var_instance in inp[0][univ_id]:
                            if pred2numpy:
                                self.inpu_byVar[univ_id].append(in_var_instance.detach().cpu().numpy())
                            else:
                                self.inpu_byVar[univ_id].append(in_var_instance)
        
        if "output" in mode:     
            self.pred_byVar = dict()
            for univ_id in range(self.univar_count_out):      
                self.pred_byVar[univ_id] = list()
            for out in zip(self.pred_data):
                if self.input_shape == "vector":                    
                    #output
                    for univ_id in range(self.univar_count_out):
                        if pred2numpy:
                            self.pred_byVar[univ_id].append(out[0][univ_id].detach().cpu().numpy())    
                        else:
                            self.pred_byVar[univ_id].append(out[0][univ_id])
                elif self.input_shape == "matrix":
                    for univ_id in range(self.univar_count_out):
                        for out_var_instance in out[0][univ_id]:
                            if pred2numpy:
                                self.pred_byVar[univ_id].append(out_var_instance.detach().numpy())    
                            else:
                                self.pred_byVar[univ_id].append(out_var_instance)
                                
        
    def latent_sortByComponent(self, pred2numpy=True):
        self.late_byComp = dict()
        
        for late_key in self.latent_keys:
            self.late_byComp[late_key] = dict()
            for id_comp in range(self.latent_dim):
                self.late_byComp[late_key][id_comp] = list()
        
        for late_key in self.latent_keys:
            for lat in self.late_data[late_key]:
                if self.input_shape == "vector":
                    for late_key in self.latent_keys:
                        for id_comp in range(self.latent_dim):                
                            if pred2numpy:
                                self.late_byComp[late_key][id_comp].append(lat[id_comp].detach().cpu().numpy())
                            else:
                                self.late_byComp[late_key][id_comp].append(lat[id_comp])
                elif self.input_shape == "matrix":
                
                    for id_comp in range(self.latent_dim):                
                        for lat_varValue_instance in lat[0][id_comp]:
                            if pred2numpy:
                                self.late_byComp[late_key][id_comp].append(lat_varValue_instance.detach().cpu().numpy())
                            else:
                                self.late_byComp[late_key][id_comp].append(lat_varValue_instance)

    def getPred(self):
        by_univar_dict = {"input":self.inpu_data, "latent": self.late_data, "output":self.pred_data}
        return by_univar_dict


    def getPred_byUnivar(self):
        by_univar_dict = {"input":self.inpu_byVar, "output":self.pred_byVar}
        return by_univar_dict

    def getLat_byComponent(self):
        by_comp_dict = {"latent":self.late_byComp}
        return by_comp_dict

    def getLat(self):
        comp_dict = {"latent":self.late_data}
        return comp_dict
    
    def getLatent2data(self):
        data_latent = list()
        for istance in self.late_data["latent"]:
            istance_dict = {'sample': istance}
            data_latent.append(istance_dict)
        comp_dict = {"latent":data_latent}
        return comp_dict
    
    
    def input_byvar(self):
        self.predict_sortByUnivar(mode="input",pred2numpy=True)
        resultDict = dict()
        resultDict["prediction_data_byvar"] = {"input":self.inpu_byVar}
        return resultDict
    
    # latent_dim, path_folder, data_range=None, input_shape
    def compute_prediction(self, experiment_name, time_key, remapping_data=False, force_noLatent=False):
        resultDict = dict()
        #modelPrediction = ModelPrediction(model, device=device, univar_count_in=univar_count_in, univar_count_out=univar_count_out, latent_dim=latent_dim, data_range=data_range, input_shape=input_shape, path_folder=folder)
        if self.latent_dim is not None:
            self.predict(self.dataset, latent=True,  experiment_name=experiment_name, time_key=time_key, remapping=remapping_data)
        else:
            self.predict(self.dataset, latent=False, experiment_name=experiment_name, time_key=time_key, remapping=remapping_data)

        prediction_data = self.getPred()
        prediction_data_byvar = self.getPred_byUnivar() 
        resultDict["prediction_data_byvar"] = prediction_data_byvar
        
        inp_data_vc = pd.DataFrame()
        for id_univar in range(self.univar_count_in):
            if self.isnoise_in:
                var_name = f"{id_univar}"
            else:
                var_name = self.vc_mapping[id_univar]
            inp_data_vc[var_name] = [a.tolist() for a in prediction_data_byvar['input'][id_univar]]
        resultDict["inp_data_vc"] = inp_data_vc

        out_data_vc = pd.DataFrame()
        for id_univar in range(self.univar_count_out):
            if self.isnoise_out:
                var_name = f"{id_univar}"
            else:
                var_name = self.vc_mapping[id_univar]                
            out_data_vc[var_name] = [a.tolist() for a in prediction_data_byvar['output'][id_univar]]
        resultDict["out_data_vc"] = out_data_vc

        if self.latent_dim is not None and not force_noLatent:
            lat_data = self.getLat()
            resultDict["latent_data"] = lat_data
            lat_data_bycomp = self.getLat_byComponent()
            resultDict["latent_data_bycomp"] = lat_data_bycomp['latent']
            lat2dataInput = self.getLatent2data()
            resultDict["latent_data_input"] = lat2dataInput  
        return resultDict