import torchinfo
import torch
from torch import Tensor, zeros
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
import torch_geometric.nn as gm
from pathlib import Path
import os
import json


class GenerativeAdversarialModels(nn.Module):

    def __init__(self, device, layers_list=None, load_from_file =False, json_filepath= None, edge_index=None, **kwargs):
        super().__init__()
        self.models = dict()
        self.device = device
        self.edge_index = edge_index
        self.permutation_forward = dict()
        if load_from_file:
            self.layers_list = self.load_fileJson(json_filepath)
        else:
            self.layers_list = layers_list
        self.models_layers = dict()
        self.models_size = dict()
        self.models_layers['generator'], self.models_size['generator'], self.permutation_forward['generator'] = self.list_to_model(self.layers_list['generator_layers'])
        self.models_layers['discriminator'], self.models_size['discriminator'], self.permutation_forward['discriminator'] = self.list_to_model(self.layers_list['discriminator_layers'])
        self.deploy_nnModel()
    
    def get_size(self, ):
        return self.models_size
    
    def deploy_nnModel(self):
        
        if self.edge_index is not None:
            self.models['discriminator'] = nn_Model(layers= self.models_layers['discriminator'], permutation_forward = self.permutation_forward['discriminator'], edge_index= self.edge_index)
            self.models['generator']     = nn_Model(layers= self.models_layers['generator'],     permutation_forward = self.permutation_forward['generator'], edge_index= self.edge_index)
        else:
            self.models['discriminator'] = nn_Model(layers= self.models_layers['discriminator'])
            self.models['generator']     = nn_Model(layers= self.models_layers['generator'])
        
        self.add_module('discriminator', self.models['discriminator'])
        self.add_module('generator', self.models['generator'])
    
    def set_partialModel(self, key, model_net, model_size, model_permutation_forward):
        self.models[key] = model_net
        self.models_size[key] = model_size
        self.permutation_forward[key] = model_permutation_forward
        self.add_module(key, self.models[key])
        
    def get_generator(self, size=False):
        if size:
            return self.models['generator'], self.models_size['generator']
        else:
            return self.models['generator']
        
    def get_discriminator(self, size=False):
        if size:
            return self.models['discriminator'], self.models_size['discriminator']
        else:
            return self.models['discriminator']

    def get_generator(self, size=False):
        if size:
            return self.models['generator'], self.models_size['generator']
        else:
            return self.models['generator']
        
    
    def summary(self):
        summary = dict()
        summary['discriminator'] = torchinfo.summary(self.models['discriminator'], input_size=(1, self.models_size['discriminator']["input_size"]), device=self.device, batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary['generator'] = torchinfo.summary(self.models['generator'], input_size=(1, self.models_size['generator']["input_size"]), device=self.device, batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
    
    def forward(self, x):
        raise Exception("forward not implemented.")

    def list_to_model(self, layers_list):
        layers = list()
        permutation_forward = dict()
        size = {"input_size":None, "output_size":None}
        for index, layer_item in enumerate(layers_list):
            
            #layer
            if layer_item['layer'] == "Linear":
                layers.append(nn.Linear(in_features=layer_item['in_features'], out_features=layer_item['out_features']))
                if size["input_size"] == None:
                    size["input_size"] = layer_item['in_features']
                size["output_size"] = layer_item['out_features']
            elif layer_item['layer'] == "GCNConv":
                layers.append(gm.GCNConv(in_channels=layer_item['in_channels'], out_channels=layer_item['out_channels']))
            elif layer_item['layer'] == "GCNConv_Permute":
                layers.append(gm.GCNConv(in_channels=layer_item['in_channels'], out_channels=layer_item['out_channels']))
                permutation_forward[index] = {"in_permute":layer_item['in_permute'],"out_permute":layer_item['out_permute']}
            
            #activation function
            elif layer_item['layer'] == "Tanh":
                layers.append(nn.Tanh())
            elif layer_item['layer'] == "LeakyReLU":
                layers.append(nn.LeakyReLU(layer_item['negative_slope']))
            elif layer_item['layer'] == "Sigmoid":
                layers.append(nn.Sigmoid())
            #batch norm
            elif layer_item['layer'] == "BatchNorm1d":
                layers.append(nn.BatchNorm1d(num_features=layer_item['num_features'], affine=layer_item['affine']))
            
            #dropout
            elif layer_item['layer'] == "Dropout":
                layers.append(nn.Dropout(p=layer_item['p']))
        return layers, size, permutation_forward

    def load_fileJson(self, filepath):
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        layers_list = dict()
        layers_list["discriminator_layers"] = config["GAN"]["discriminator_layers"]
        layers_list["generator_layers"] = config["GAN"]["generator_layers"]
        return layers_list

class nn_Model(nn.Module):
    def __init__(self, layers, permutation_forward=None, edge_index=None):
        super().__init__()
        self.permutation_forward = permutation_forward
        self.layers = nn.Sequential(*layers)
        self.edge_index = edge_index
        self.apply(self.weights_init_normal)
        print("Layers initialized:", self.layers)
    
    def weights_init_normal(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        for  index, layer in enumerate(self.layers):
            if isinstance(layer, gm.GCNConv):
                if index in self.permutation_forward:
                    in_permute = self.permutation_forward[index]["in_permute"]
                    x = x.permute(in_permute[0], in_permute[1], in_permute[2])                
                x = layer(x, self.edge_index) 
                if index in self.permutation_forward:
                    out_permute = self.permutation_forward[index]["out_permute"]
                    x = x.permute(out_permute[0], out_permute[1], out_permute[2])                              
            else:
                x = layer(x)
        return {"x_input": {"data":x}, "x_output": {"data":x}}