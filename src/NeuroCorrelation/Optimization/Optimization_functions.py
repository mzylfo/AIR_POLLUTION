
import numpy as np
import pandas as pd
from numpy.linalg import inv
from termcolor import cprint 
from scipy.stats import wasserstein_distance


class Optimization_functions():
    
    def __init__(self):
        self.object_list = ['mahalanobis','mahalanobis+wasserstein','wasserstein']
        self.name_fun = None
        
    def set_object_fun(self, name_fun):
        if name_fun in self.object_list:
            self.name_fun = name_fun
        else:
            raise Exception(f"Optimization function {name_fun} not exist.")
        cprint(f"OPTIMIZATION OBJECT FUN: ", "red", end="\n")
        cprint(f"\t{name_fun}", "red", end="\n")
    
    def get_score(self, values):
        scores = dict()
        if self.name_fun is None:
            raise Exception("No optimization function selected")
        elif self.name_fun == "mahalanobis":
            val_real  = values['inp_data_vc'].to_numpy()
            val_gen = values['out_data_vc'].to_numpy()
            dist_mahalanobis = self.mahalanobis(X=val_real, Y=val_gen)
            score = dist_mahalanobis
            scores['mahalanobis'] = dist_mahalanobis
        
        elif self.name_fun == "mahalanobis+wasserstein" or self.name_fun == "mahala+wass":
            val_real  = values['inp_data_vc'].to_numpy()
            val_gen = values['out_data_vc'].to_numpy()
            dist_mahalanobis = self.mahalanobis(X=val_real, Y=val_gen)            
            dist_wasserstein = self.wasserstein(X=val_real, Y=val_gen)
            score = dist_mahalanobis + 0.5*dist_wasserstein
            scores['mahalanobis+wasserstein'] = score
            scores['mahalanobis'] = dist_mahalanobis
            scores['wasserstein'] = dist_wasserstein
        
        elif self.name_fun == "wasserstein" or self.name_fun == "wass":
            val_real  = values['inp_data_vc'].to_numpy()
            val_gen = values['out_data_vc'].to_numpy()
            dist_wasserstein = self.wasserstein(X=val_real, Y=val_gen)
            
            #raise Exception("dist_wasserstein----------------------", dist_wasserstein)
            score = dist_wasserstein
            scores['wasserstein'] = dist_wasserstein
            
        else:
            raise Exception(f"Optimization function {self.name_fun} non developed")
        
        cprint(f"Optimization function score:\t{score}", "green", end="\n")
        cprint(f"\t\t{scores}", "green", end="\n")
        scores['all'] = score
        return scores
    
    def mahalanobis(self, X, Y):
        mu_X = np.mean(X, axis=0)
        mu_Y = np.mean(Y, axis=0)
        
        cov_X = np.cov(X, rowvar=False)
        cov_Y = np.cov(Y, rowvar=False)
        
        cov_combined = (cov_X + cov_Y) / 2
        
        diff = mu_X - mu_Y
        dist_mahalanobis = np.sqrt(diff.T @ inv(cov_combined) @ diff)
        return dist_mahalanobis
    
    def wasserstein(self, X, Y, aggregate="mean"):
        
        distances = np.array([wasserstein_distance(X[:, i], Y[:, i]) for i in range(X.shape[1])])
        
        if aggregate == "mean":
            dist_wasserstein = np.mean(distances)
        elif aggregate == "euclidean":
            dist_wasserstein = np.linalg.norm(distances)
        print(dist_wasserstein, "distances")
        return dist_wasserstein
