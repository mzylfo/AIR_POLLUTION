import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from torch.optim import SGD
from torch.optim import Adam


class ModelTraining():

    def __init__(self, optimizer_name, lr, model_params, path, weight_decay=None, momentum=None):
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.path = path
        
        
    def setOptimizer(self, model_params):
        self.model_params = model_params
        
        if self.optimizer_name=="SGD":
            if self.weight_decay is not None:
                self.optimizer = SGD(params=self.model_params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
            else:
                self.optimizer = SGD(params=self.model_params, lr=0.1, momentum=0.9)
        elif self.optimizer_name=="ADAM":
            self.optimizer = SGD(params=self.model_params, lr=self.lr)
        self.printOptimizerSettings()
        
    def printOptimizerSettings(self):        
        loss_str = self.saveModel_optimizer()
        filename = Path(self.path, "summary_optimizer.txt")
        with open(filename, 'w') as file:
            file.write(file_str)
        print("SETTING PHASE: Summary optimizer file - DONE")
    
    
    def saveModel_optimizer(self):
        opt_str = []
        opt_str.append(f"optimizer_name::\t {self.optimizer_name}")
        opt_str.append(f"lr::\t {self.lr}")
        opt_str.append(f"weight_decay::\t {self.weight_decay}")
        opt_str.append(f"momentum::\t {self.momentum}")
        opt_list = '\n'.join([opt_str])
        return opt_list
    
    
    def getOptimizer(self):
        return self.optimizer