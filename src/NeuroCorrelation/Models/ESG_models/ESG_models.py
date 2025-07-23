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


#---------------------------
# MODEL 35
class ESG__GEN_autoEncoder_35(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = ESG__GEN_autoEncoder_Encoder_35()
        self.decoder = ESG__GEN_autoEncoder_Decoder_35()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":{"latent":x_latent["x_output"]}, "x_output":x_hat["x_output"]}

        
    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 35), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 30), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class ESG__GEN_autoEncoder_Encoder_35(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=35, out_features=32)
        self.hidden_layer_2 = nn.Linear(in_features=32, out_features=30)
        
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class ESG__GEN_autoEncoder_Decoder_35(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=28, out_features=30)
        self.hidden_layer_2 = nn.Linear(in_features=30, out_features=32)
        self.hidden_layer_2 = nn.Linear(in_features=32, out_features=34)
        self.hidden_layer_2 = nn.Linear(in_features=34, out_features=35)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#---------------------------


#---------------------------
# GAN 35
class ESG__GAN_LinearDiscriminator_35(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=35, out_features=24)
        self.hidden_layer_2 = nn.Linear(in_features=24, out_features=16)
        self.hidden_layer_3 = nn.Linear(in_features=16, out_features=4)
        self.hidden_layer_4 = nn.Linear(in_features=4, out_features=1)
        

    def forward(self, x):
        layer_nn = x
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        
        x_hat = F.sigmoid(layer_nn)
        return {"x_input":x, "x_output":x_hat}

class ESG__GAN_LinearGenerator_35(nn.Module):
    def __init__(self):    
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=28, out_features=32)
        self.hidden_layer_2 = nn.Linear(in_features=32, out_features=35)
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class ESG__GAN_neural_mixed_35(nn.Module):
    def __init__(self, generator=None, discriminator=None):
        super().__init__()
        if generator is None:
            self.G = ESG__GAN_LinearGenerator_35
        else:
            self.G = generator
        
        if discriminator is None:
            self.D = ESG__GAN_LinearDiscriminator_35
        else:
            self.D = discriminator

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D
    
    def summary(self):
        gen_summary = torchinfo.summary(self.G, input_size=(1, 28), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dis_summary = torchinfo.summary(self.D(), input_size=(1, 35), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary_dict = {"generator": gen_summary, "discriminator": dis_summary}
        return summary_dict
#---------------------------