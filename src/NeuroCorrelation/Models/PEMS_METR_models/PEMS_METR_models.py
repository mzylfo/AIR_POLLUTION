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
# MODEL 16  
class PEMS_METR_AE_16(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = PEMS_METR_AE_Encoder_16()
        self.decoder = PEMS_METR_AE_Decoder_16()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 16), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 12), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class PEMS_METR_AE_Encoder_16(nn.Module):
    def __init__(self):

        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=16, out_features=14)
        self.hidden_layer_2 = nn.Linear(in_features=14, out_features=12)
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)
        
        
        self.dp_1 = nn.Dropout(p=0.2)
        
        
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

class PEMS_METR_AE_Decoder_16(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=12, out_features=14)
        self.hidden_layer_2 = nn.Linear(in_features=14, out_features=16)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)
        
        self.dp_1 = nn.Dropout(p=0.2)        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        #layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    

class PEMS_METR_GAN_LinearDiscriminator_16(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=16, out_features=8)
        self.hidden_layer_2 = nn.Linear(in_features=8, out_features=4)
        self.hidden_layer_3 = nn.Linear(in_features=4, out_features=2)        
        self.hidden_layer_4 = nn.Linear(in_features=2, out_features=1)
        
        self.act_1 = nn.LeakyReLU(0.2, inplace=True)#nn.Tanh()#
        self.act_2 = nn.LeakyReLU(0.2, inplace=True)#nn.Tanh()#
        self.act_3 = nn.LeakyReLU(0.2, inplace=True)#nn.Tanh()#
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = self.act_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)    
        
         #== layer OUT ===================
        x_out = layer_nn
        x_hat = F.sigmoid(x_out)
        return {"x_input":x, "x_output":x_hat}

class PEMS_METR_GAN_LinearGenerator_16(nn.Module):
    def __init__(self):       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=12, out_features=14)
        self.hidden_layer_2 = nn.Linear(in_features=14, out_features=16)
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
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

class PEMS_METR_GAN_16(nn.Module):
    def __init__(self, generator=None, discriminator=None):
        super().__init__()
        if generator is None:
            self.G = PEMS_METR_GAN_LinearGenerator_16
        else:
            self.G = generator
        
        if discriminator is None:
            self.D = PEMS_METR_GAN_LinearDiscriminator_16
        else:
            self.D = discriminator

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D

    def summary(self):
        gen_summary = torchinfo.summary(self.G, input_size=(1, 12), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dis_summary = torchinfo.summary(self.D(), input_size=(1, 16), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary_dict = {"generator": gen_summary, "discriminator": dis_summary}
        return summary_dict
#---------------------------

#---------------------------
# MODEL 32
class PEMS_METR_AE_32(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = PEMS_METR_AE_Encoder_32()
        self.decoder = PEMS_METR_AE_Decoder_32()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 32), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 30), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class PEMS_METR_AE_Encoder_32(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=32, out_features=28)
        self.hidden_layer_2 = nn.Linear(in_features=28, out_features=22)
        
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
       
        
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

class PEMS_METR_AE_Decoder_32(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=22, out_features=28)
        self.hidden_layer_2 = nn.Linear(in_features=28, out_features=32)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
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

class PEMS_METR_GAN_LinearDiscriminator_32(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=32, out_features=16)
        self.hidden_layer_2 = nn.Linear(in_features=16, out_features=8)
        self.hidden_layer_3 = nn.Linear(in_features=8, out_features=4)        
        self.hidden_layer_4 = nn.Linear(in_features=4, out_features=1)
        
        self.act_1 = nn.LeakyReLU(0.2, inplace=True)#nn.Tanh()#
        self.act_2 = nn.LeakyReLU(0.2, inplace=True)#nn.Tanh()#
        self.act_3 = nn.LeakyReLU(0.2, inplace=True)#nn.Tanh()#
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = self.act_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)    
        
         #== layer OUT ===================
        x_out = layer_nn
        x_hat = F.sigmoid(x_out)
        return {"x_input":x, "x_output":x_hat}

class PEMS_METR_GAN_LinearGenerator_32(nn.Module):
    def __init__(self):       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=22, out_features=28)
        self.hidden_layer_2 = nn.Linear(in_features=28, out_features=30)
        self.hidden_layer_3 = nn.Linear(in_features=30, out_features=36)
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
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class PEMS_METR_GAN_32(nn.Module):
    def __init__(self, generator=None, discriminator=None):
        super().__init__()
        if generator is None:
            self.G = PEMS_METR_GAN_LinearGenerator_32
        else:
            self.G = generator
        
        if discriminator is None:
            self.D = PEMS_METR_GAN_LinearDiscriminator_32
        else:
            self.D = discriminator

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D

    def summary(self):
        gen_summary = torchinfo.summary(self.G, input_size=(1, 22), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dis_summary = torchinfo.summary(self.D(), input_size=(1, 32), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary_dict = {"generator": gen_summary, "discriminator": dis_summary}
        return summary_dict
#---------------------------

#---------------------------
# MODEL 48  
class PEMS_METR_AE_48(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = PEMS_METR_AE_Encoder_48()
        self.decoder = PEMS_METR_AE_Decoder_48()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 48), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 36), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class PEMS_METR_AE_Encoder_48(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=48, out_features=42)
        self.hidden_layer_2 = nn.Linear(in_features=42, out_features=36)
        
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
       
        
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

class PEMS_METR_AE_Decoder_48(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=36, out_features=42)
        self.hidden_layer_2 = nn.Linear(in_features=42, out_features=48)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
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
 
class PEMS_METR_GAN_LinearDiscriminator_48(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=48, out_features=16)
        self.hidden_layer_2 = nn.Linear(in_features=16, out_features=8)
        self.hidden_layer_3 = nn.Linear(in_features=8, out_features=4)        
        self.hidden_layer_4 = nn.Linear(in_features=4, out_features=1)
        
        self.act_1 = nn.LeakyReLU(0.2, inplace=True)#nn.Tanh()#
        self.act_2 = nn.LeakyReLU(0.2, inplace=True)#nn.Tanh()#
        self.act_3 = nn.LeakyReLU(0.2, inplace=True)#nn.Tanh()#
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = self.act_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)    
        
         #== layer OUT ===================
        x_out = layer_nn
        x_hat = F.sigmoid(x_out)
        return {"x_input":x, "x_output":x_hat}

class PEMS_METR_GAN_LinearGenerator_48(nn.Module):
    def __init__(self):       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=36, out_features=38)
        self.hidden_layer_2 = nn.Linear(in_features=38, out_features=42)
        self.hidden_layer_3 = nn.Linear(in_features=42, out_features=48)
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
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class PEMS_METR_GAN_48(nn.Module):
    def __init__(self, generator=None, discriminator=None):
        super().__init__()
        if generator is None:
            self.G = PEMS_METR_GAN_LinearGenerator_48
        else:
            self.G = generator
        
        if discriminator is None:
            self.D = PEMS_METR_GAN_LinearDiscriminator_48
        else:
            self.D = discriminator

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D

    def summary(self):
        gen_summary = torchinfo.summary(self.G, input_size=(1, 36), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dis_summary = torchinfo.summary(self.D(), input_size=(1, 48), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary_dict = {"generator": gen_summary, "discriminator": dis_summary}
        return summary_dict
#---------------------------

#---------------------------
# MODEL 64  
class PEMS_METR_AE_64(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = PEMS_METR_AE_Encoder_64()
        self.decoder = PEMS_METR_AE_Decoder_64()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 64), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 48), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class PEMS_METR_AE_Encoder_64(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=64, out_features=54)
        self.hidden_layer_2 = nn.Linear(in_features=54, out_features=48)
        
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
       
        
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

class PEMS_METR_AE_Decoder_64(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=48, out_features=54)
        self.hidden_layer_2 = nn.Linear(in_features=54, out_features=64)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
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
 
class PEMS_METR_GAN_LinearDiscriminator_64(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=64, out_features=32)
        self.hidden_layer_2 = nn.Linear(in_features=32, out_features=16)
        self.hidden_layer_3 = nn.Linear(in_features=16, out_features=4)        
        self.hidden_layer_4 = nn.Linear(in_features=4, out_features=1)
        
        self.act_1 = nn.LeakyReLU(0.2, inplace=True)#nn.Tanh()#
        self.act_2 = nn.LeakyReLU(0.2, inplace=True)#nn.Tanh()#
        self.act_3 = nn.LeakyReLU(0.2, inplace=True)#nn.Tanh()#
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = self.act_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)    
        
         #== layer OUT ===================
        x_out = layer_nn
        x_hat = F.sigmoid(x_out)
        return {"x_input":x, "x_output":x_hat}

class PEMS_METR_GAN_LinearGenerator_64(nn.Module):
    def __init__(self):       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=48, out_features=52)
        self.hidden_layer_2 = nn.Linear(in_features=52, out_features=58)
        self.hidden_layer_3 = nn.Linear(in_features=58, out_features=64)
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
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class PEMS_METR_GAN_64(nn.Module):
    def __init__(self, generator=None, discriminator=None):
        super().__init__()
        if generator is None:
            self.G = PEMS_METR_GAN_LinearGenerator_64
        else:
            self.G = generator
        
        if discriminator is None:
            self.D = PEMS_METR_GAN_LinearDiscriminator_64
        else:
            self.D = discriminator

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D

    def summary(self):
        gen_summary = torchinfo.summary(self.G, input_size=(1, 48), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dis_summary = torchinfo.summary(self.D(), input_size=(1, 64), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary_dict = {"generator": gen_summary, "discriminator": dis_summary}
        return summary_dict
#---------------------------