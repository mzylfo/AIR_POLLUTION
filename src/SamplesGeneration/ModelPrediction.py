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

class ModelPrediction():

    def __init__(self, model, count_univar, model_type, name_model=None):
        self.model = model
        self.name_model = name_model
        self.count_univar = count_univar
        self.model_type = model_type
        
        self.inpu_data = list()
        self.late_data = list()
        self.pred_data = list()

        self.inpu_byVar = dict()
        self.pred_byVar = dict()
        #self.late_byVar = dict()
        
    
    def predict(self, input_sample):
        self.inpu_data = list()
        self.late_data = list()
        self.pred_data = list()

        for inp in input_sample:            
            out = self.model(inp)            
            self.inpu_data.append(out["x_input"])
            if self.model_type in ["VAE"]:
                print("---m odelpred")
                x_latent_keys = ["mu","logvar"]
            else:
                x_latent_keys = ["latent"]
            list_latent = list()
            for latent_key in x_latent_keys:
                list_latent.append(out["x_latent"][latent_key])
            self.late_data.append(list_latent)
            self.pred_data.append(out["x_output"])
        self.predict_sortByUnivar()


    def predict_sortByUnivar(self):
        self.inpu_byVar = dict()        
        self.pred_byVar = dict()

        for univ_id in range(count_univar):
            self.inpu_byVar[univ_id] = list()            
            self.pred_byVar[univ_id] = list()

        for inp,out in zip(self.inpu_data, self.pred_data):
            for univ_id in range(count_univar):
                self.inpu_byVar[univ_id].append(inp[univ_id])
                self.pred_byVar[univ_id].append(out[univ_id])
        
    
    def getPred(self):
        by_univar_dict = {"input":self.inpu_data, "latent": self.late_data, "output":self.pred_data}
        return by_univar_dict


    def getPred_byUnivar(self):
        by_univar_dict = {"input":self.inpu_byVar, "output":self.pred_byVar}
        return by_univar_dict