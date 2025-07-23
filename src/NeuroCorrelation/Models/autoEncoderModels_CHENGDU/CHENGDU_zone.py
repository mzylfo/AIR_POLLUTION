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
# MODEL EXOGENOUS  
class GEN_autoEncoder_Encoder_exogenous_7(nn.Module):
    def __init__(self,):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=7, out_features=5)
        self.hidden_layer_2 = nn.Linear(in_features=5, out_features=3)
        
        self.act_1 = nn.LeakyReLU(0.2)        
        self.dp_1 = nn.Dropout(p=0.2)
       
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
# MODEL zone 0 -- AE
class GEN_autoEncoderGCN_zone0(nn.Module):
    def __init__(self, edge_index, **kwargs):
        super().__init__()
        self.edge_index = edge_index
        self.encoder = GEN_autoEncoderGCN_Encoder_graph_zone0(self.edge_index)
        self.decoder = GEN_autoEncoderGCN_Decoder_zone0(self.edge_index)
        

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        enc_summary = torchinfo.summary(self.encoder, input_size=[(1, 248)], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=[(1, 80)], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoderGCN_Encoder_graph_zone0(nn.Module):
    def __init__(self, edge_index):
       
        super().__init__()
        self.edge_index = edge_index
        
        self.hidden_layer_1 = gm.GCNConv(in_channels=1, out_channels=1)
        self.hidden_layer_2 = nn.Linear(in_features=248, out_features=180)
        
        self.hidden_layer_3 = nn.Linear(in_features=180, out_features=80)
        
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
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_1(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
                
        #== layer 05  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoderGCN_Decoder_zone0(nn.Module):
    
    def __init__(self, edge_index):
       
        super().__init__()
        
        #self.edge_index = edge_index
        self.hidden_layer_1 = nn.Linear(in_features=80, out_features=180)
        self.hidden_layer_2 = nn.Linear(in_features=180, out_features=248)
        self.hidden_layer_3 = gm.GCNConv(in_channels=1, out_channels=1)
        self.edge_index = edge_index
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
                        
        #== layer 04  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_3(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    

# MODEL zone 0 -- GAN
class GAN_LinearDiscriminator_ZONE0(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=248, out_features=64)
        self.hidden_layer_2 = nn.Linear(in_features=64, out_features=32)
        self.hidden_layer_3 = nn.Linear(in_features=32, out_features=16)        
        self.hidden_layer_4 = nn.Linear(in_features=16, out_features=4)
        self.hidden_layer_5 = nn.Linear(in_features=4, out_features=1)
        
        self.act_1 = nn.LeakyReLU(0.2, inplace=True)#nn.Tanh()#
        self.act_2 = nn.LeakyReLU(0.2, inplace=True)#nn.Tanh()#
        self.act_3 = nn.LeakyReLU(0.2, inplace=True)#nn.Tanh()#
        self.act_4 = nn.LeakyReLU(0.2, inplace=True)#nn.Tanh()#

    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = self.act_1(layer_nn)
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)    
        layer_nn = self.act_4(layer_nn)
        
        #== layer 05  ===================
        layer_nn = self.hidden_layer_5(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        x_hat = F.sigmoid(x_out)
        return {"x_input":x, "x_output":x_hat}

class GAN_LinearGenerator_ZONE0(nn.Module):
    def __init__(self):       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=80, out_features=512)
        self.hidden_layer_2 = nn.Linear(in_features=512, out_features=414)
        self.hidden_layer_3 = nn.Linear(in_features=414, out_features=316)        
        self.hidden_layer_4 = nn.Linear(in_features=316, out_features=256)
        self.hidden_layer_5 = nn.Linear(in_features=256, out_features=128)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2, inplace=True)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2, inplace=True)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2, inplace=True)#
        self.act_4 = nn.Tanh()#nn.LeakyReLU(0.2, inplace=True)#

    def forward(self, x):
         #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = self.act_1(layer_nn)
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)    
        layer_nn = self.act_4(layer_nn)
        
        #== layer 05  ===================
        layer_nn = self.hidden_layer_5(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        
        return {"x_input":x, "x_output":x_out}

class GAN_neural_mixed_ZONE0(nn.Module):
    def __init__(self, generator=None, discriminator=None):
        super().__init__()
        if generator is None:
            self.G = GAN_LinearGenerator_ZONE0
        else:
            self.G = generator
        
        if discriminator is None:
            self.D = GAN_LinearDiscriminator_ZONE0
        else:
            self.D = discriminator

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D

    def summary(self):
        gen_summary = torchinfo.summary(self.G, input_size=(1, 80), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dis_summary = torchinfo.summary(self.D(), input_size=(1, 248), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary_dict = {"generator": gen_summary, "discriminator": dis_summary}
        return summary_dict
#---------------------------


#---------------------------

#---------------------------
# MODEL zone 1
class GEN_autoEncoderGCN_zone1(nn.Module):
    def __init__(self, edge_index, **kwargs):
        super().__init__()
        self.edge_index = edge_index
        self.encoder = GEN_autoEncoderGCN_Encoder_graph_zone1(edge_index)
        self.decoder = GEN_autoEncoderGCN_Decoder_zone1(edge_index)
        

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=[(1, 240)], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=[(1, 85)], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoderGCN_Encoder_graph_zone1(nn.Module):
    def __init__(self, edge_index):
       
        super().__init__()
        #self.edge_index = edge_index
        
        
        self.hidden_layer_1 = gm.GCNConv(in_channels=1, out_channels=1)
        self.hidden_layer_2 = nn.Linear(in_features=240, out_features=180)
        
        self.hidden_layer_3 = nn.Linear(in_features=180, out_features=85)
        self.edge_index = edge_index
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
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_1(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
                
        #== layer 05  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoderGCN_Decoder_zone1(nn.Module):
    
    def __init__(self, edge_index):
       
        super().__init__()
        
        #self.edge_index = edge_index
        self.hidden_layer_1 = nn.Linear(in_features=85, out_features=180)
        self.hidden_layer_2 = nn.Linear(in_features=180, out_features=240)
        self.hidden_layer_3 = gm.GCNConv(in_channels=1, out_channels=1)
        self.edge_index = edge_index
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
                        
        #== layer 04  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_3(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    

#---------------------------

#---------------------------
# MODEL zone 2
class GEN_autoEncoderGCN_zone2(nn.Module):
    def __init__(self, edge_index, **kwargs):
        super().__init__()
        self.edge_index = edge_index
        self.encoder = GEN_autoEncoderGCN_Encoder_graph_zone2(edge_index)
        self.decoder = GEN_autoEncoderGCN_Decoder_zone2(edge_index)
        

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=[(1, 197)], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=[(1, 90)], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoderGCN_Encoder_graph_zone2(nn.Module):
    def __init__(self, edge_index):
       
        super().__init__()
        #self.edge_index = edge_index
        
        
        self.hidden_layer_1 = gm.GCNConv(in_channels=1, out_channels=1)
        self.hidden_layer_2 = nn.Linear(in_features=197, out_features=180)
        
        self.hidden_layer_3 = nn.Linear(in_features=180, out_features=90)
        self.edge_index = edge_index
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
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_1(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
                
        #== layer 05  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoderGCN_Decoder_zone2(nn.Module):
    
    def __init__(self, edge_index):
       
        super().__init__()
        
        #self.edge_index = edge_index
        self.hidden_layer_1 = nn.Linear(in_features=90, out_features=180)
        self.hidden_layer_2 = nn.Linear(in_features=180, out_features=197)
        self.hidden_layer_3 = gm.GCNConv(in_channels=1, out_channels=1)
        self.edge_index = edge_index
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
                        
        #== layer 04  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_3(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    

#---------------------------

#---------------------------
# MODEL zone 1 + zone 2
class GEN_autoEncoderGCN_zones_1_2(nn.Module):
    def __init__(self, edge_index, **kwargs):
        super().__init__()
        self.edge_index = edge_index
        self.encoder = GEN_autoEncoderGCN_Encoder_graph_zones_1_2(edge_index)
        self.decoder = GEN_autoEncoderGCN_Decoder_zones_1_2(edge_index)
        

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=[(1, 437)], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=[(1, 95)], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoderGCN_Encoder_graph_zones_1_2(nn.Module):
    def __init__(self, edge_index):
       
        super().__init__()
        #self.edge_index = edge_index
        
        
        self.hidden_layer_1 = gm.GCNConv(in_channels=1, out_channels=1)
        self.hidden_layer_2 = nn.Linear(in_features=437, out_features=280)
        
        self.hidden_layer_3 = nn.Linear(in_features=280, out_features=95)
        self.edge_index = edge_index
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
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_1(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
                
        #== layer 05  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoderGCN_Decoder_zones_1_2(nn.Module):
    
    def __init__(self, edge_index):
       
        super().__init__()
        
        #self.edge_index = edge_index
        self.hidden_layer_1 = nn.Linear(in_features=95, out_features=280)
        self.hidden_layer_2 = nn.Linear(in_features=280, out_features=437)
        self.hidden_layer_3 = gm.GCNConv(in_channels=1, out_channels=1)
        self.edge_index = edge_index
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
                        
        #== layer 04  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_3(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    

#---------------------------