
import math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import json


class LossCofficentsFunction():
    def __init__(self, loss_coeff, epochs_tot, path_folder):
        self.loss_coeff = loss_coeff
        self.epochs_tot = epochs_tot
        self.path_folder = Path(path_folder,"loss_coefficent")
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
        self.coeff_for_epoch = dict()
        for e in range(0,epochs_tot):
            self.coeff_for_epoch[e] = dict()
        self.setCoefficents()
        self.saveLossCoeff()
        self.plotCoefficents()
        
    
    
    def getCoefficents(self, epoch):
        #print("epochs_tot----------",self.epochs_tot)
        return self.coeff_for_epoch[epoch]
    
    def saveLossCoeff(self):
        filename = Path(self.path_folder,"loss_coef.json")
        with open(filename, 'w') as json_file:
            json.dump(self.loss_coeff, json_file, indent=4)  # indent=4 per una formattazione leggibile


        
        
    def plotCoefficents(self):
        
        color = cm.rainbow(np.linspace(0, 1, len(self.loss_coeff)))
        
        for i,loss_name in enumerate(self.loss_coeff):
            values = [self.coeff_for_epoch[e][loss_name] for e in range(self.epochs_tot)]
            plt.figure(figsize=(12, 8))
            plt.plot(range(self.epochs_tot), values, label=f"{loss_name}", color=color[i], marker='o', markersize=3, linewidth=1)
            plt.title("Coefficients {loss_name} for epoch")
            filename = Path(self.path_folder,f'loss_coeff_{loss_name}.png')
            plt.savefig(filename)   
    

        
    def setCoefficents(self):
        for e in range(0, self.epochs_tot):
            self.coeff_for_epoch[e] = dict()
            
        for loss_name in self.loss_coeff:
            if self.loss_coeff[loss_name]['type'] == 'fixed':
                for e in range(0, self.epochs_tot):
                    self.coeff_for_epoch[e][loss_name] = self.loss_coeff[loss_name]['value']
            
            if self.loss_coeff[loss_name]['type'] == 'linear':
                val_beg = self.loss_coeff[loss_name]['range']['begin']
                val_end = self.loss_coeff[loss_name]['range']['end']
                print(val_end,val_beg)
                val_step = (val_end - val_beg) / (self.epochs_tot - 1)
                
                for e in range(0, self.epochs_tot):
                    self.coeff_for_epoch[e][loss_name] = val_beg + e * val_step
            
            if self.loss_coeff[loss_name]['type'] == 'cos':
                val_min = self.loss_coeff[loss_name]['range']['min']
                val_max = self.loss_coeff[loss_name]['range']['max']
                val_per = self.loss_coeff[loss_name]['range']['period']
                if val_per == "all":
                    val_per = self.epochs_tot
                for e in range(0, self.epochs_tot):
                    self.coeff_for_epoch[e][loss_name] = val_min + (val_max - val_min) / 2 * (1 + math.cos(2 * math.pi * e / val_per))
                    

                