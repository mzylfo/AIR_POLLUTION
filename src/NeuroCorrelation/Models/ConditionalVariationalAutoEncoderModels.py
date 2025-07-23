
import numpy as np
import torch

import torchinfo
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

class ConditionalVariationalAutoEncoderModels(nn.Module):

    def __init__(self, device, layers_list=None, load_from_file=False, json_filepath=None, edge_index=None, **kwargs):
        super().__init__()
        self.device = device
        self.models = dict()
        self.edge_index = edge_index
        self.permutation_forward = dict()
        self.models_layers_parallel = dict()
        self.layers_name = dict()
        if load_from_file:
            self.layers_list = self.load_fileJson(json_filepath)
        else:
            self.layers_list = layers_list
        self.models_layers = dict()
        self.models_size = dict()
        # Encoder
        self.models_layers['encoder'], self.models_size['encoder'], self.permutation_forward['encoder'], self.models_layers_parallel['encoder'], self.layers_name['encoder'] = self.list_to_model(self.layers_list['encoder_layers'])
        
        

        # Decoder
        self.models_layers['decoder'], self.models_size['decoder'], self.permutation_forward['decoder'], self.models_layers_parallel['decoder'], self.layers_name['decoder'], = self.list_to_model(self.layers_list['decoder_layers'])

        
        self.deploy_cvae_model()
        
    def get_size(self):
        return self.models_size
    
    def deploy_cvae_model(self):
        # Latent space layers for mean and logvar (specific to CVAE)
        self.fc_mu = nn.Linear(self.models_size['encoder']["output_size"], self.models_size['encoder']["output_size"])
        self.fc_logvar = nn.Linear(self.models_size['encoder']["output_size"], self.models_size['encoder']["output_size"])
        
        if self.edge_index is not None:
            self.models['encoder'] = nn_Model(layers=self.models_layers['encoder'], permutation_forward=self.permutation_forward['encoder'], edge_index=self.edge_index, parallel_layers=self.models_layers_parallel['encoder'], layers_name =self.layers_name['encoder'])
            self.models['decoder'] = nn_Model(layers=self.models_layers['decoder'], permutation_forward=self.permutation_forward['decoder'], edge_index=self.edge_index, parallel_layers=self.models_layers_parallel['decoder'], layers_name =self.layers_name['decoder'])
        else:
            self.models['encoder'] = nn_Model(layers=self.models_layers['encoder'], parallel_layers=self.models_layers_parallel['encoder'])
            self.models['decoder'] = nn_Model(layers=self.models_layers['decoder'], parallel_layers=self.models_layers_parallel['decoder'])

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
        print("models_size[encoder][input_size]\t\t\t", self.models_size['encoder']["input_size"])
        print("models_size[decoder][input_size]\t\t\t", self.models_size['decoder']["input_size"])
        summary['encoder'] = torchinfo.summary(self.models['encoder'], input_size=(1, self.models_size['encoder']["input_size"]), device=self.device, batch_dim=0, col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose=0)
        summary['decoder'] = torchinfo.summary(self.models['decoder'], input_size=(1, self.models_size['decoder']["input_size"]), device=self.device, batch_dim=0, col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose=0)

    def forward(self, x, condition):
        print("87 beg")
        x_latent = self.models['encoder'](x, condition)        
        print("87 enc")
        mu = x_latent["mu"]
        logvar = x_latent["logvar"]                
        z = self.reparameterize(mu, logvar)
        print("91 dec")
        x_hat = self.models['decoder'](z)        
        print("93 end")
        return {"x_input": {"data":x}, "x_latent":{"mu": mu, "logvar": logvar, "z":z}, "x_output": {"data": x_hat["x_output"]['data']}}

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        reparameterized = mu + eps * std        
        return reparameterized

    def list_to_model(self, layers_list):
        layers = list()
        parallel_layers_flag = False
        parallel_layers = []
        permutation_forward = dict()
        layers_name = dict()
        size = {"input_size": None, "output_size": None}
        print("list_to_model")
        for index, layer_item in enumerate(layers_list):
            parallel_layers_flag = False
            # Layers
            if layer_item['layer'] == "Linear":
                layers.append(nn.Linear(in_features=layer_item['in_features'], out_features=layer_item['out_features']))
                if size["input_size"] is None:
                    size["input_size"] = layer_item['in_features']
                size["output_size"] = layer_item['out_features']
            elif layer_item['layer'] == "GCNConv":
                layers.append(gm.GCNConv(in_channels=layer_item['in_channels'], out_channels=layer_item['out_channels']))
            elif layer_item['layer'] == "GCNConv_Permute":
                layers.append(gm.GCNConv(in_channels=layer_item['in_channels'], out_channels=layer_item['out_channels']))
                permutation_forward[index] = {"in_permute": layer_item['in_permute'], "out_permute": layer_item['out_permute']}

            # Activation functions
            elif layer_item['layer'] == "Tanh":
                layers.append(nn.Tanh())
            elif layer_item['layer'] == "LeakyReLU":
                layers.append(nn.LeakyReLU(layer_item['negative_slope']))
            elif layer_item['layer'] == "Sigmoid":
                layers.append(nn.Sigmoid())
            
            # Batch normalization
            elif layer_item['layer'] == "BatchNorm1d":
                layers.append(nn.BatchNorm1d(num_features=layer_item['num_features'], affine=layer_item['affine']))

            # Dropout
            elif layer_item['layer'] == "Dropout":
                layers.append(nn.Dropout(p=layer_item['p']))
            
            # Parallel layers
            elif layer_item['layer'] == "Parallel":
                parallel_layers_flag = True
                parallel_dict = nn.ModuleDict()
                for sub_layer in layer_item['layers']:
                    sub_layers, sub_size, _, _, sub_name = self.list_to_model(sub_layer['layers'])
                    parallel_dict[sub_layer['name']] = nn.Sequential(*sub_layers)
                layers_name[index] = {"parallel": list(parallel_dict.keys())}
                layers.append(parallel_dict)
            
            # Naming
            if 'name' in layer_item:
                layers_name[index] = layer_item['name']
            elif parallel_layers_flag:
                layers_name[index] = {"parallel": [sub_layer['name'] for sub_layer in layer_item['layers']]}
            else:
                layers_name[index] = f"{layer_item['layer']}_{index}"
        
        return layers, size, permutation_forward, parallel_layers_flag, layers_name

        layers = list()
        parallel_layers_flag = False
        parallel_layers = []
        permutation_forward = dict()
        layers_name = dict()
        size = {"input_size": None, "output_size": None}
        print("list_to_model")
        for index, layer_item in enumerate(layers_list):
            parallel_layers_flag = False
            # Layers
            
            if layer_item['layer'] == "Linear":
                layers.append(nn.Linear(in_features=layer_item['in_features'], out_features=layer_item['out_features']))
                if size["input_size"] is None:
                    size["input_size"] = layer_item['in_features']
                size["output_size"] = layer_item['out_features']
            elif layer_item['layer'] == "GCNConv":
                layers.append(gm.GCNConv(in_channels=layer_item['in_channels'], out_channels=layer_item['out_channels']))
            elif layer_item['layer'] == "GCNConv_Permute":
                layers.append(gm.GCNConv(in_channels=layer_item['in_channels'], out_channels=layer_item['out_channels']))
                permutation_forward[index] = {"in_permute": layer_item['in_permute'], "out_permute": layer_item['out_permute']}

            # Activation functions
            elif layer_item['layer'] == "Tanh":
                layers.append(nn.Tanh())
            elif layer_item['layer'] == "LeakyReLU":
                layers.append(nn.LeakyReLU(layer_item['negative_slope']))
            elif layer_item['layer'] == "Sigmoid":
                layers.append(nn.Sigmoid())
            
            # Batch normalization
            elif layer_item['layer'] == "BatchNorm1d":
                layers.append(nn.BatchNorm1d(num_features=layer_item['num_features'], affine=layer_item['affine']))

            # Dropout
            elif layer_item['layer'] == "Dropout":
                layers.append(nn.Dropout(p=layer_item['p']))
            
            elif layer_item['layer'] == "Parallel":
                parallel_layers_flag = True
                sub_layers = nn.ModuleDict()
                for sub_layer in layer_item['layers']:
                    sub_layer_modules, _, _, _, sub_layer_names = self.list_to_model(sub_layer['layers'])
                    sub_layers.update(sub_layer_modules)
                    layers_name.update(sub_layer_names)
                parallel_layers[layer_item['name']] = sub_layers
                layers.append(parallel_layers[layer_item['name']])

                
            elif layer_item['layer'] == "Parallel":
                parallel_layers_flag = True       
                
                for sub_layer in layer_item['layers']:
                    sub_layers, sub_size, _, _, sub_name = self.list_to_model(sub_layer['layers'])
                    parallel_layers.append((sub_layer['name'], nn.Sequential(*sub_layers)))
            
                layers_name[index] = {"parallel": [name for name, _ in parallel_layers]}
                layers.append(parallel_layers)
                
            if 'name' in layer_item:
                layers_name[index] = layer_item['name']
            elif parallel_layers_flag:
                layers_name[index] = {"parallel": [sub_layer['name'] for sub_layer in layer_item['layers']]}
            else:
                layers_name[index] = f"{layer_item['layer']}_{index}"

        
        return layers, size, permutation_forward, parallel_layers_flag, layers_name

    def load_fileJson(self, filepath):
        with open(filepath, 'r') as f:
            config = json.load(f)

        layers_list = dict()
        layers_list["encoder_layers"] = config["CVAE"]["encoder_layers"]
        layers_list["decoder_layers"] = config["CVAE"]["decoder_layers"]
        return layers_list



class nn_Model(nn.Module):
    def __init__(self, layers, permutation_forward=None, edge_index=None, parallel_layers=False, layers_name=None):
       
        super().__init__()

        self.edge_index = edge_index
        self.permutation_forward = permutation_forward or {}
        self.layers_name = layers_name or {}
        self.parallel_layers_flag = parallel_layers

        # Inizializza i layer sequenziali e paralleli
        self.sequential_layers = nn.Sequential()
        self.parallel_blocks = nn.ModuleDict()
        
        self._initialize_layers(layers)
        self.apply(self.weights_init_normal)
        
        print("Model inizialization:")
        print(f" - Sequential layers:\t {self.sequential_layers}")
        print(f" - Parallel blocks:\t {self.parallel_blocks}")
        
        

    def _initialize_layers(self, layers):
        sequential_layers = []
        for index, layer in enumerate(layers):
            if isinstance(layer, list):
                for name, sub_block in layer:
                    self.parallel_blocks[name] = nn.Sequential(*sub_block)
            else:
                sequential_layers.append(layer)
        self.sequential_layers = nn.Sequential(*sequential_layers)

    def weights_init_normal(self, m):        
        if isinstance(m, nn.Linear):
            init_mode = "xavier_uniform"
            if init_mode == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight, gain=0.01)
            elif init_mode == "normal_":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, condition=None):
        forward_dict = {"x_input": {'data':x, 'condition':condition}}
        for index, layer in enumerate(self.sequential_layers):            
            if isinstance(layer, gm.GCNConv):
                if index in self.permutation_forward:
                    in_permute = self.permutation_forward[index]["in_permute"]
                    x = x.permute(*in_permute)
                x = layer(x, self.edge_index)
                if index in self.permutation_forward:
                    out_permute = self.permutation_forward[index]["out_permute"]
                    x = x.permute(*out_permute)
            else:
                x = layer(x, condition)
        forward_dict["x_output"] = {'data':x}

        if self.parallel_layers_flag:
            for block_name, block in self.parallel_blocks.items():
                forward_dict[block_name] = block(x, condition)
        return forward_dict
    
