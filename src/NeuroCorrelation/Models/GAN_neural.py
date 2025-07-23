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
from pathlib import Path
import os
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
import torch.nn as nn
import torch
from torch.functional import F
from torchmetrics.functional.regression import kendall_rank_corrcoef
from torchmetrics import SpearmanCorrCoef
import torchinfo 


#---------------------------

#---------------------------
# MODEL 7
class GAN_LinearNeural_7(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.D = GAN_LinearDiscriminator_7
        self.G = GAN_LinearGenerator_7

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D
    
class GAN_LinearDiscriminator_7(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=7, out_features=10)
        self.hidden_layer_2 = nn.Linear(in_features=10, out_features=6)
        self.hidden_layer_3 = nn.Linear(in_features=6, out_features=4)
        self.hidden_layer_4 = nn.Linear(in_features=4, out_features=1)
        

    def forward(self, x):
        x_flat = x.view(-1)
        layer_nn = self.hidden_layer_1(x_flat)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        x_hat = F.sigmoid(layer_nn)
        return {"x_input":x, "x_output":x_hat}

class GAN_LinearGenerator_7(nn.Module):
    def __init__(self):       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=4, out_features=12)
        self.hidden_layer_2 = nn.Linear(in_features=12, out_features=18)
        self.hidden_layer_3 = nn.Linear(in_features=18, out_features=24)
        self.hidden_layer_4 = nn.Linear(in_features=24, out_features=7)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_4(layer_nn)
        
        return {"x_input":x, "x_output":x_out}

class GAN_neural_mixed_7(nn.Module):
    def __init__(self, generator=None, discriminator=None):
        super().__init__()
        if generator is None:
            self.G = GAN_LinearGenerator_7
        else:
            self.G = generator
        
        if discriminator is None:
            self.D = GAN_LinearDiscriminator_7
        else:
            self.D = discriminator

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D


class GAN_Conv_neural_7(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.D = GAN_ConvDiscriminator_7
        self.G = GAN_ConvGenerator_7

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D

class GAN_ConvDiscriminator_7(nn.Module):
    # in  1, 1, 7, 32
    # out 1, 1, 1, 1
    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=3, padding=2)
        self.hidden_layer_2 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=2)
        self.hidden_layer_3 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=3, padding=2)
        self.hidden_layer_4 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.hidden_layer_5 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        x_hat = F.sigmoid(layer_nn)
        return {"x_input":x, "x_output":x_hat}

class GAN_ConvGenerator_7(nn.Module):
    # in  1, 1, 2, 6
    # out 1, 1, 7, 32
    def __init__(self):       
        super().__init__()
        self.hidden_layer_1 = nn.ConvTranspose2d(in_channels=1, out_channels=5, kernel_size=2, stride=2, padding=0)
        self.hidden_layer_2 = nn.ConvTranspose2d(in_channels=5, out_channels=3, kernel_size=2, stride=2, padding=(0,3))
        self.hidden_layer_3 = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=2, stride=(1,2), padding=(1,2))

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_3(layer_nn)
        return {"x_input":x, "x_output":x_out}
#---------------------------

#---------------------------
# MODEL 16
class GAN_neural_mixed_16(nn.Module):
    def __init__(self, generator=None, discriminator=None):
        super().__init__()
        if generator is None:
            self.G = GAN_LinearGenerator_16
        else:
            self.G = generator
        
        if discriminator is None:
            self.D = GAN_LinearDiscriminator_16
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
        
class GAN_Conv_neural_16(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.D = GAN_ConvDiscriminator_16
        self.G = GAN_ConvGenerator_16

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D

class GAN_ConvDiscriminator_16(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=2, padding=0)
        self.hidden_layer_2 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2, stride=2, padding=0)
        self.hidden_layer_3 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=2, stride=2, padding=0)
        self.hidden_layer_4 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=2, stride=4, padding=0)
        self.hidden_layer_5 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        x_hat = F.sigmoid(layer_nn)
        return {"x_input":x, "x_output":x_hat}

class GAN_ConvGenerator_16(nn.Module):
    def __init__(self):       
        # in  1, 1, 2, 8
        # out 1, 1, 2, 64
        super().__init__()
        self.hidden_layer_1 = nn.ConvTranspose2d(in_channels=1, out_channels=5, kernel_size=2, stride=2, padding=0)
        self.hidden_layer_2 = nn.ConvTranspose2d(in_channels=1, out_channels=5, kernel_size=2, stride=2, padding=0)
        self.hidden_layer_3 = nn.ConvTranspose2d(in_channels=5, out_channels=1, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_3(layer_nn)
        
        
        return {"x_input":x, "x_output":x_out}


class GAN_Linear_neural_16(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.D = GAN_LinearDiscriminator_16
        self.G = GAN_LinearGenerator_16

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D

class GAN_LinearDiscriminator_16(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=16, out_features=12)
        self.hidden_layer_2 = nn.Linear(in_features=12, out_features=8)
        self.hidden_layer_3 = nn.Linear(in_features=8, out_features=4)
        self.hidden_layer_4 = nn.Linear(in_features=4, out_features=1)
        
        #self.activation_function_1 = nn.LeakyReLU(0.2, inplace=True)
        #self.activation_function_2 = nn.LeakyReLU(0.2, inplace=True)
        #self.activation_function_3 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = nn.LeakyReLU(0.2, inplace=True)(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = nn.LeakyReLU(0.2, inplace=True)(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = nn.LeakyReLU(0.2, inplace=True)(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        x_hat = F.sigmoid(layer_nn)
        return {"x_input":x, "x_output":x_hat}

class GAN_LinearGenerator_16(nn.Module):
    def __init__(self):       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=12, out_features=14)
        self.hidden_layer_2 = nn.Linear(in_features=14, out_features=16)
        self.hidden_layer_3 = nn.Linear(in_features=16, out_features=32)
        self.hidden_layer_4 = nn.Linear(in_features=32, out_features=48)
        self.hidden_layer_5 = nn.Linear(in_features=48, out_features=64)
        self.hidden_layer_6 = nn.Linear(in_features=64, out_features=48)
        self.hidden_layer_7 = nn.Linear(in_features=48, out_features=16)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_7(layer_nn)
        return {"x_input":x, "x_output":x_out}

#---------------------------

#---------------------------
# MODEL 32
class GAN_LinearDiscriminator_32(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=32, out_features=24)
        self.hidden_layer_2 = nn.Linear(in_features=24, out_features=16)
        self.hidden_layer_3 = nn.Linear(in_features=16, out_features=8)        
        self.hidden_layer_4 = nn.Linear(in_features=8, out_features=4)
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

class GAN_LinearGenerator_32(nn.Module):
    def __init__(self):       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=30, out_features=48)
        self.hidden_layer_2 = nn.Linear(in_features=48, out_features=40)
        self.hidden_layer_3 = nn.Linear(in_features=40, out_features=32)
        #self.hidden_layer_4 = nn.Linear(in_features=32, out_features=48)
        #self.hidden_layer_5 = nn.Linear(in_features=48, out_features=64)
        #self.hidden_layer_6 = nn.Linear(in_features=64, out_features=48)
        #self.hidden_layer_7 = nn.Linear(in_features=48, out_features=32)

    def forward(self, x):
        layer_nn = x
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        #layer_nn = F.tanh(layer_nn)
        #layer_nn = self.hidden_layer_4(layer_nn)
        #layer_nn = F.tanh(layer_nn)
        #layer_nn = self.hidden_layer_5(layer_nn)
        #layer_nn = F.tanh(layer_nn)
        #layer_nn = self.hidden_layer_6(layer_nn)
        #layer_nn = F.tanh(layer_nn)
        #layer_nn = self.hidden_layer_7(layer_nn)
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GAN_neural_mixed_32(nn.Module):
    def __init__(self, generator=None, discriminator=None):
        super().__init__()
        if generator is None:
            self.G = GAN_LinearGenerator_32
        else:
            self.G = generator
        
        if discriminator is None:
            self.D = GAN_LinearDiscriminator_32
        else:
            self.D = discriminator

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D

    def summary(self):
        gen_summary = torchinfo.summary(self.G, input_size=(1, 30), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dis_summary = torchinfo.summary(self.D(), input_size=(1, 32), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary_dict = {"generator": gen_summary, "discriminator": dis_summary}
        return summary_dict
#---------------------------

#---------------------------
# MODEL 48    
class GAN_LinearDiscriminator_48(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=48, out_features=64)
        self.hidden_layer_2 = nn.Linear(in_features=64, out_features=192)
        self.hidden_layer_3 = nn.Linear(in_features=192, out_features=128)
        self.hidden_layer_4 = nn.Linear(in_features=128, out_features=86)
        self.hidden_layer_5 = nn.Linear(in_features=86, out_features=64)
        self.hidden_layer_6 = nn.Linear(in_features=64, out_features=48)
        self.hidden_layer_7 = nn.Linear(in_features=48, out_features=32)
        self.hidden_layer_8 = nn.Linear(in_features=32, out_features=16)
        self.hidden_layer_9 = nn.Linear(in_features=16, out_features=8)
        self.hidden_layer_10 = nn.Linear(in_features=8, out_features=4)
        self.hidden_layer_11 = nn.Linear(in_features=4, out_features=2)
        self.hidden_layer_12 = nn.Linear(in_features=2, out_features=1)
        
    def forward(self, x):
        layer_nn = x
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_7(layer_nn)        
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_8(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_9(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_10(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_11(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_12(layer_nn)
        
        x_hat = F.sigmoid(layer_nn)
        return {"x_input":x, "x_output":x_hat}

class GAN_LinearGenerator_48(nn.Module):
    def __init__(self):    
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=44, out_features=48)
        self.hidden_layer_2 = nn.Linear(in_features=48, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=196)
        self.hidden_layer_4 = nn.Linear(in_features=196, out_features=256)
        self.hidden_layer_5 = nn.Linear(in_features=256, out_features=512)        
        self.hidden_layer_6 = nn.Linear(in_features=512, out_features=1024)
        self.hidden_layer_7 = nn.Linear(in_features=1024, out_features=512)
        self.hidden_layer_8 = nn.Linear(in_features=512, out_features=256)
        self.hidden_layer_9 = nn.Linear(in_features=256, out_features=128)
        self.hidden_layer_10 = nn.Linear(in_features=128, out_features=64)
        self.hidden_layer_11 = nn.Linear(in_features=64, out_features=48)
        
    def forward(self, x):
        #layer_nn = self.bn_1(x)
        layer_nn = x
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_7(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_8(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_9(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_10(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_11(layer_nn)
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GAN_neural_mixed_48(nn.Module):
    def __init__(self, generator=None, discriminator=None):
        super().__init__()
        if generator is None:
            self.G = GAN_LinearGenerator_48
        else:
            self.G = generator
        
        if discriminator is None:
            self.D = GAN_LinearDiscriminator_48
        else:
            self.D = discriminator

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D
    
    def summary(self):
        gen_summary = torchinfo.summary(self.G, input_size=(1, 44), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dis_summary = torchinfo.summary(self.D(), input_size=(1, 48), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary_dict = {"generator": gen_summary, "discriminator": dis_summary}
        return summary_dict


    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=60, out_features=96)
        self.hidden_layer_2 = nn.Linear(in_features=96, out_features=128)
        self.hidden_layer_3 = nn.Linear(in_features=128, out_features=192)
        self.hidden_layer_4 = nn.Linear(in_features=192, out_features=224)
        self.hidden_layer_5 = nn.Linear(in_features=224, out_features=256)
        self.hidden_layer_6 = nn.Linear(in_features=256, out_features=128)
        self.hidden_layer_7 = nn.Linear(in_features=128, out_features=96)
        self.hidden_layer_8 = nn.Linear(in_features=96, out_features=60)
#---------------------------

#---------------------------
# MODEL 64
class GAN_LinearDiscriminator_64(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=64, out_features=96)
        self.hidden_layer_2 = nn.Linear(in_features=96, out_features=192)
        self.hidden_layer_3 = nn.Linear(in_features=192, out_features=128)
        self.hidden_layer_4 = nn.Linear(in_features=128, out_features=86)
        self.hidden_layer_5 = nn.Linear(in_features=86, out_features=64)
        self.hidden_layer_6 = nn.Linear(in_features=64, out_features=48)
        self.hidden_layer_7 = nn.Linear(in_features=48, out_features=32)
        self.hidden_layer_8 = nn.Linear(in_features=32, out_features=16)
        self.hidden_layer_9 = nn.Linear(in_features=16, out_features=8)
        self.hidden_layer_10 = nn.Linear(in_features=8, out_features=4)
        self.hidden_layer_11 = nn.Linear(in_features=4, out_features=2)
        self.hidden_layer_12 = nn.Linear(in_features=2, out_features=1)
        
        

    def forward(self, x):
        layer_nn = x
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_7(layer_nn)        
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_8(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_9(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_10(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_11(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_12(layer_nn)
        
        x_hat = F.sigmoid(layer_nn)
        return {"x_input":x, "x_output":x_hat}

class GAN_LinearGenerator_64(nn.Module):
    def __init__(self):    
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=50, out_features=64)
        self.hidden_layer_2 = nn.Linear(in_features=64, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=196)
        self.hidden_layer_4 = nn.Linear(in_features=196, out_features=256)
        self.hidden_layer_5 = nn.Linear(in_features=256, out_features=512)        
        self.hidden_layer_6 = nn.Linear(in_features=512, out_features=1024)
        self.hidden_layer_7 = nn.Linear(in_features=1024, out_features=512)
        self.hidden_layer_8 = nn.Linear(in_features=512, out_features=256)
        self.hidden_layer_9 = nn.Linear(in_features=256, out_features=128)
        self.hidden_layer_10 = nn.Linear(in_features=128, out_features=96)
        self.hidden_layer_11 = nn.Linear(in_features=96, out_features=64)
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_7(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_8(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_9(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_10(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_11(layer_nn)
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GAN_neural_mixed_64(nn.Module):
    def __init__(self, generator=None, discriminator=None):
        super().__init__()
        if generator is None:
            self.G = GAN_LinearGenerator_64
        else:
            self.G = generator
        
        if discriminator is None:
            self.D = GAN_LinearDiscriminator_64
        else:
            self.D = discriminator

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D
    
    def summary(self):
        gen_summary = torchinfo.summary(self.G, input_size=(1, 50), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dis_summary = torchinfo.summary(self.D(), input_size=(1, 64), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary_dict = {"generator": gen_summary, "discriminator": dis_summary}
        return summary_dict
#---------------------------


#---------------------------
# MODEL 128
class GAN_LinearDiscriminator_128(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=128, out_features=64)
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

class GAN_LinearGenerator_128(nn.Module):
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

class GAN_neural_mixed_128(nn.Module):
    def __init__(self, generator=None, discriminator=None):
        super().__init__()
        if generator is None:
            self.G = GAN_LinearGenerator_128
        else:
            self.G = generator
        
        if discriminator is None:
            self.D = GAN_LinearDiscriminator_128
        else:
            self.D = discriminator

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D

    def summary(self):
        gen_summary = torchinfo.summary(self.G, input_size=(1, 80), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dis_summary = torchinfo.summary(self.D(), input_size=(1, 128), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary_dict = {"generator": gen_summary, "discriminator": dis_summary}
        return summary_dict
#---------------------------

#---------------------------
# MODEL 256
class GAN_LinearDiscriminator_256(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=256, out_features=128)
        self.hidden_layer_2 = nn.Linear(in_features=128, out_features=64)
        self.hidden_layer_3 = nn.Linear(in_features=64, out_features=32)
        self.hidden_layer_4 = nn.Linear(in_features=32, out_features=16)        
        self.hidden_layer_5 = nn.Linear(in_features=16, out_features=4)
        self.hidden_layer_6 = nn.Linear(in_features=4, out_features=1)
        
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
        layer_nn = self.act_5(layer_nn)
        
        #== layer 06  ===================
        layer_nn = self.hidden_layer_6(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        x_hat = F.sigmoid(x_out)
        return {"x_input":x, "x_output":x_hat}

class GAN_LinearGenerator_256(nn.Module):
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

class GAN_neural_mixed_256(nn.Module):
    def __init__(self, generator=None, discriminator=None):
        super().__init__()
        if generator is None:
            self.G = GAN_LinearGenerator_256
        else:
            self.G = generator
        
        if discriminator is None:
            self.D = GAN_LinearDiscriminator_256
        else:
            self.D = discriminator

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D

    def summary(self):
        gen_summary = torchinfo.summary(self.G, input_size=(1, 80), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dis_summary = torchinfo.summary(self.D(), input_size=(1, 256), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary_dict = {"generator": gen_summary, "discriminator": dis_summary}
        return summary_dict
#---------------------------

#---------------------------
# MODEL 5943
class GAN_LinearDiscriminator_5943(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=5943, out_features=4096)
        self.hidden_layer_2 = nn.Linear(in_features=4096, out_features=2048)
        self.hidden_layer_3 = nn.Linear(in_features=2048, out_features=1024)
        self.hidden_layer_4 = nn.Linear(in_features=1024, out_features=512)
        self.hidden_layer_5 = nn.Linear(in_features=512, out_features=256)
        self.hidden_layer_6 = nn.Linear(in_features=256, out_features=128)
        self.hidden_layer_7 = nn.Linear(in_features=128, out_features=64)
        self.hidden_layer_8 = nn.Linear(in_features=64, out_features=32)
        self.hidden_layer_9 = nn.Linear(in_features=32, out_features=16)
        self.hidden_layer_10 = nn.Linear(in_features=16, out_features=8)
        self.hidden_layer_11 = nn.Linear(in_features=8, out_features=4)
        self.hidden_layer_12 = nn.Linear(in_features=4, out_features=2)
        self.hidden_layer_13 = nn.Linear(in_features=2, out_features=1)
        
        

    def forward(self, x):
        x_flat = x.view(-1)
        layer_nn = self.hidden_layer_1(x_flat)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_7(layer_nn)        
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_8(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_9(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_10(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_11(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_12(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_13(layer_nn)
        
        x_hat = F.sigmoid(layer_nn)
        return {"x_input":x, "x_output":x_hat}

class GAN_LinearGenerator_5943(nn.Module):
    def __init__(self):    
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=4896, out_features=2359)
        self.hidden_layer_2 = nn.Linear(in_features=2359, out_features=2871)
        self.hidden_layer_3 = nn.Linear(in_features=2871, out_features=3383)
        self.hidden_layer_4 = nn.Linear(in_features=3383, out_features=3895)
        self.hidden_layer_5 = nn.Linear(in_features=3895, out_features=4407)        
        self.hidden_layer_6 = nn.Linear(in_features=4407, out_features=4919)
        self.hidden_layer_7 = nn.Linear(in_features=4919, out_features=5431)
        self.hidden_layer_8 = nn.Linear(in_features=5431, out_features=5943)
        
    def forward(self, x):
        #layer_nn = self.bn_1(x)
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_7(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_8(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_9(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_10(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_11(layer_nn)
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GAN_neural_mixed_5943(nn.Module):
    def __init__(self, generator=None, discriminator=None):
        super().__init__()
        if generator is None:
            self.G = GAN_LinearGenerator_5943
        else:
            self.G = generator
        
        if discriminator is None:
            self.D = GAN_LinearDiscriminator_5943
        else:
            self.D = discriminator

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D
    
    def summary(self):
        gen_summary = torchinfo.summary(self.G, input_size=(1, 2048), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dis_summary = torchinfo.summary(self.D(), input_size=(1, 5943), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary_dict = {"generator": gen_summary, "discriminator": dis_summary}
        return summary_dict

437

#---------------------------
# MODEL 0437
class GAN_LinearDiscriminator_0437(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=437, out_features=340)
        self.hidden_layer_2 = nn.Linear(in_features=340, out_features=256)
        self.hidden_layer_3 = nn.Linear(in_features=256, out_features=128)
        self.hidden_layer_4 = nn.Linear(in_features=128, out_features=86)
        self.hidden_layer_5 = nn.Linear(in_features=86, out_features=64)
        self.hidden_layer_6 = nn.Linear(in_features=64, out_features=48)
        self.hidden_layer_7 = nn.Linear(in_features=48, out_features=32)
        self.hidden_layer_8 = nn.Linear(in_features=32, out_features=16)
        self.hidden_layer_9 = nn.Linear(in_features=16, out_features=8)
        self.hidden_layer_10 = nn.Linear(in_features=8, out_features=4)
        self.hidden_layer_11 = nn.Linear(in_features=4, out_features=2)
        self.hidden_layer_12 = nn.Linear(in_features=2, out_features=1)
        
        

    def forward(self, x):
        layer_nn = x
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_7(layer_nn)        
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_8(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_9(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_10(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_11(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_12(layer_nn)
        
        x_hat = F.sigmoid(layer_nn)
        return {"x_input":x, "x_output":x_hat}

class GAN_LinearGenerator_0437(nn.Module):
    def __init__(self):    
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=90, out_features=64)
        self.hidden_layer_2 = nn.Linear(in_features=64, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=196)
        self.hidden_layer_4 = nn.Linear(in_features=196, out_features=256)
        self.hidden_layer_5 = nn.Linear(in_features=256, out_features=512)        
        self.hidden_layer_6 = nn.Linear(in_features=512, out_features=1024)
        self.hidden_layer_7 = nn.Linear(in_features=1024, out_features=512)
        self.hidden_layer_8 = nn.Linear(in_features=512, out_features=256)
        self.hidden_layer_9 = nn.Linear(in_features=256, out_features=128)
        self.hidden_layer_10 = nn.Linear(in_features=128, out_features=96)
        self.hidden_layer_11 = nn.Linear(in_features=96, out_features=437)
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_7(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_8(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_9(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_10(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_11(layer_nn)
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GAN_neural_mixed_0437(nn.Module):
    def __init__(self, generator=None, discriminator=None):
        super().__init__()
        if generator is None:
            self.G = GAN_LinearGenerator_0437
        else:
            self.G = generator
        
        if discriminator is None:
            self.D = GAN_LinearDiscriminator_0437
        else:
            self.D = discriminator

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D
    
    def summary(self):
        gen_summary = torchinfo.summary(self.G, input_size=(1, 90), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dis_summary = torchinfo.summary(self.D(), input_size=(1, 437), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary_dict = {"generator": gen_summary, "discriminator": dis_summary}
        return summary_dict
#---------------------------