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


class AutoEncoderModels(nn.Module):

    def __init__(self, device, layers_list=None, load_from_file =False, json_filepath= None, edge_index=None,  **kwargs):
        super().__init__()
        self.device = device
        self.models = dict()
        self.edge_index = edge_index
        self.permutation_forward = dict()
        self.graph_forward = dict()
        if load_from_file:
            self.layers_list = self.load_fileJson(json_filepath)
        else:
            self.layers_list = layers_list
        self.models_layers = dict()
        self.models_size = dict()
        self.models_layers['encoder'], self.models_size['encoder'], self.permutation_forward['encoder'], self.graph_forward['encoder'] = self.list_to_model(self.layers_list['encoder_layers'])
        self.models_layers['decoder'], self.models_size['decoder'], self.permutation_forward['decoder'], self.graph_forward['decoder'] = self.list_to_model(self.layers_list['decoder_layers'])
        self.deploy_nnModel()
        
    def get_size(self, ):
        return self.models_size
        
    def deploy_nnModel(self):
        
        if self.edge_index is not None:
            self.models['encoder'] = nn_Model(layers= self.models_layers['encoder'], permutation_forward = self.permutation_forward['encoder'], edge_index= self.edge_index ,graph_forward=self.graph_forward['encoder'])
            self.models['decoder'] = nn_Model(layers= self.models_layers['decoder'], permutation_forward = self.permutation_forward['decoder'], edge_index= self.edge_index ,graph_forward=self.graph_forward['decoder'])
        else:
            self.models['encoder'] = nn_Model(layers= self.models_layers['encoder'])
            self.models['decoder'] = nn_Model(layers= self.models_layers['decoder'])
        
        self.add_module('encoder', self.models['encoder'])
        self.add_module('decoder', self.models['decoder'])
        
        
    def get_decoder(self, extra_info=False):
        if extra_info:
            return self.models['decoder'], self.models_size['decoder'], self.permutation_forward['decoder']
        else:
            return self.models['decoder']
        

    def get_encoder(self, extra_info=False):
        if extra_info:
            return self.models['encoder'], self.models_size['encoder'], self.permutation_forward['encoder']
        else:
            return self.models['encoder']       
    
    def summary(self):
        summary = dict()
        print("models_size[encoder][input_size]\t\t\t",self.models_size['encoder']["input_size"])
        print("models_size[decoder][input_size]\t\t\t",self.models_size['decoder']["input_size"])
        summary['encoder'] = torchinfo.summary(self.models['encoder'], input_size=(1, self.models_size['encoder']["input_size"]), device=self.device, batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary['decoder'] = torchinfo.summary(self.models['decoder'], input_size=(1, self.models_size['decoder']["input_size"]), device=self.device, batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
    
    def forward(self, x):
        x_latent = self.models['encoder'](x)
        x_hat = self.models['decoder'](x_latent["x_output"]['data'])
        return {"x_input":{"data":x}, "x_latent":{"latent":x_latent["x_output"]['data']}, "x_output":{"data":x_hat["x_output"]['data']}}

    def list_to_model(self, layers_list):
        layers = list()
        permutation_forward = dict()
        graph_forward = dict()
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
                graph_forward[index] = "GCNConv_node"
            elif layer_item['layer'] == "GCNConv_Permute":
                layers.append(gm.GCNConv(in_channels=layer_item['in_channels'], out_channels=layer_item['out_channels']))
                permutation_forward[index] = {"in_permute":layer_item['in_permute'],"out_permute":layer_item['out_permute']}
                graph_forward[index] = "GCNConv_node"
            
            elif layer_item['layer'] == "GCNConv_edges":
                layers.append(gm.GCNConv(in_channels=layer_item['in_channels'], out_channels=layer_item['out_channels']))
                graph_forward[index] = "GCNConv_edge"
            elif layer_item['layer'] == "GCNConv_edges_Permute":
                layers.append(gm.GCNConv(in_channels=layer_item['in_channels'], out_channels=layer_item['out_channels']))
                permutation_forward[index] = {"in_permute":layer_item['in_permute'],"out_permute":layer_item['out_permute']}
                graph_forward[index] = "GCNConv_edge"
            
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
        return layers, size, permutation_forward, graph_forward

    def load_fileJson(self, filepath):
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        layers_list = dict()
        layers_list["encoder_layers"] = config["VAE"]["encoder_layers"]
        layers_list["decoder_layers"] = config["VAE"]["decoder_layers"]
        return layers_list
   
class nn_Model(nn.Module):
    def __init__(self, layers, permutation_forward=None, edge_index=None):
        super().__init__()
        self.permutation_forward = permutation_forward
        self.layers = nn.Sequential(*layers)
        self.edge_index = edge_index
        self.apply(self.weights_init_normal)
        print("Layers initialized:", self.layers)
        
        unique_nodes = torch.unique(edge_index)
        num_nodes = unique_nodes.size(0)
        print(num_nodes)
        self.x_zeros = torch.zeros((num_nodes, 1), dtype=torch.float)
    
    def weights_init_normal(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x_in = x
        for  index, layer in enumerate(self.layers):
            if isinstance(layer, gm.GCNConv):
                if index in self.permutation_forward:
                    in_permute = self.permutation_forward[index]["in_permute"]
                    x = x.permute(in_permute[0], in_permute[1], in_permute[2])                
                if index in self.graph_forward:
                    if self.graph_forward[index] =="GCNConv_node":
                        x = layer(x, self.edge_index) 
                    elif  self.graph_forward[index] =="GCNConv_node":
                    
                        x = layer(self.x_zeros, self.edge_index,x) 
                else:
                    x = layer(x, self.edge_index) 
                
                if index in self.permutation_forward:
                    out_permute = self.permutation_forward[index]["out_permute"]
                    x = x.permute(out_permute[0], out_permute[1], out_permute[2])
            else:
                x = layer(x)
        return {"x_input": {'data':x_in}, "x_output": {'data':x}}