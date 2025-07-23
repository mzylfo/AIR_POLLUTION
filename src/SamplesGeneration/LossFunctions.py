from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
import torch.nn as nn
import torch

class LossFunction(nn.Module):
    def __init__(self, loss_case):
        self.loss_case = loss_case
    
    
    def computate_loss(self, values, verbose=False):
        loss_total = torch.zeros(1)

        if self.loss_case=="MSE":
            loss_mse = nn.MSELoss()
            reconstructed_similarities = loss_mse(values['x_input'], values['x_output'])
            loss_total += reconstructed_similarities

        return loss_total


    def reconstructed_similarities(self, ys_true, ys_pred):
        """
        ys_true : vector of items where each item is a groundtruth matrix 
        ys_pred : vector of items where each item is a prediction matrix 
        return the sum of 2nd proximity of 2 matrix
        """
        loss_secondary = torch.zeros(1)

        for i, y_true in enumerate(ys_true):
            y_pred = ys_pred[i]
            loss_secondary_item = torch.sub(y_pred,y_true,alpha=1)
            loss_secondary += loss_secondary_item
        return loss_secondary
