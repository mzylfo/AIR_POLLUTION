from torch import Tensor, zeros
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
import torch.nn as nn
import torch.nn.functional as F

class NeuralModels():

    def __init__(self, model_case):
        self.model_case = model_case
    
    def get_model(self):
        if self.model_case=="fullyRectangle":
            return self.fullyRectangle()
        else:
            return None


    def fullyRectangle(self):
        model = GEN_fl()
        return model

class GEN_fl(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=78, out_features=78)
        self.hidden_layer_2 = nn.Linear(in_features=78, out_features=78)
        self.hidden_layer_3 = nn.Linear(in_features=78, out_features=78)
        self.hidden_layer_4 = nn.Linear(in_features=78, out_features=78)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        x_hat = F.tanh(layer_nn)
        return {"x_input":x, "x_latent":{"latent":None}, "x_output":x_hat}

class GEN_autoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder()
        self.decoder = GEN_autoEncoder_Decoder()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent)
        return {"x_input":x, "x_latent":{"latent":x_latent}, "x_output":x_hat}

    def get_decoder(self):
        return self.decoder

class GEN_autoEncoder_Encoder(nn.Module):
    def __init__(self):
       
        super().__init__()
        #self.bn_1 = nn.BatchNorm1d(40)
        self.hidden_layer_1 = nn.Linear(in_features=78, out_features=60)
        self.hidden_layer_2 = nn.Linear(in_features=60, out_features=40)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        #layer_nn = self.bn_1(layer_nn)
        return layer_nn

class GEN_autoEncoder_Decoder(nn.Module):
    def __init__(self):
       
        super().__init__()
        #self.bn_1 = nn.BatchNorm1d(40)
        self.hidden_layer_1 = nn.Linear(in_features=40, out_features=60)
        self.hidden_layer_2 = nn.Linear(in_features=60, out_features=78)

    def forward(self, x):
        #layer_nn = self.bn_1(x)
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        return layer_nn