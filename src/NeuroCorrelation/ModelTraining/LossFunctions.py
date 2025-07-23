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
from src.NeuroCorrelation.ModelTraining.LossCofficentsFunction import LossCofficentsFunction
from sklearn.decomposition import PCA
from torch.autograd import Function
from scipy.stats import spearmanr

from scipy.special import kl_div
import numpy as np

class LossFunction(nn.Module):
    def __init__(self, loss_case, univar_count, latent_dim, device, batch_shape="vector"):
        self.loss_case = loss_case
        self.univar_count = univar_count
        self.latent_dim = latent_dim
        self.batch_shape = batch_shape
        self.device = device
        self.statsData = None
        self.vc_mapping = None
        self.check_coeff = dict()
        
    def get_lossTerms(self):
        return self.loss_case
     
    def set_coefficent(self, epochs_tot, path_folder):
        self.loss_coeff = LossCofficentsFunction(self.loss_case, epochs_tot=epochs_tot, path_folder=path_folder)   
        
    def loss_change_coefficent(self, loss_name, loss_coeff):
        if loss_name in self.loss_case:
            self.loss_case[loss_name] = {'type': 'fixed', 'value': loss_coeff}
        self.loss_coeff.setCoefficents()
    
    def get_Loss_params(self):
        if len(self.loss_case) == 0:
            return {"loss_case":"-", "latent_dim":self.latent_dim, "univar_count":self.univar_count,"batch_shape":self.batch_shape}
        else:
            return {"loss_case":self.loss_case, "latent_dim":self.latent_dim, "univar_count":self.univar_count,"batch_shape":self.batch_shape}
    
    def set_stats_data(self, stats_data, vc_mapping):
        self.statsData = stats_data
        self.vc_mapping = vc_mapping
        
        
    def computate_loss(self, values_in, epoch, verbose=False):
        if self.statsData is None or self.vc_mapping is None:
            raise Exception("statsData NOT SET")
        values = values_in
        loss_total = torch.zeros(1).to(device=self.device)
        loss_dict = dict()
        coeff = self.loss_coeff.getCoefficents(epoch=epoch)
        self.check_coeff[epoch] = coeff
        
        if "MSE_LOSS" in self.loss_case:
            mse_similarities_loss = self.MSE_similarities(values)
            loss_coeff = mse_similarities_loss.mul(coeff["MSE_LOSS"])
            loss_total += loss_coeff
            loss_dict["MSE_LOSS"] = loss_coeff
            if verbose:
                print("MSE_LOSS - ", loss_coeff)
        
        if "RMSE_LOSS" in self.loss_case:
            rmse_similarities_loss = self.RMSE_similarities(values)
            loss_coeff = rmse_similarities_loss.mul(coeff["RMSE_LOSS"])
            loss_total += loss_coeff
            loss_dict["RMSE_LOSS"] = loss_coeff
            if verbose:
                print("RMSE_LOSS - ", loss_coeff)
                
        if "MEDIAN_LOSS_batch" in self.loss_case:
            median_similarities_loss = self.median_similarities(values, compute_on="B")
            loss_coeff = median_similarities_loss.mul(coeff["MEDIAN_LOSS_batch"])
            loss_total += loss_coeff
            loss_dict["MEDIAN_LOSS_batch"] = loss_coeff
            if verbose:
                print("MEDIAN_LOSS_batch - ", loss_coeff)
        
        if "MEDIAN_LOSS_dataset" in self.loss_case:
            median_similarities_loss = self.median_similarities(values, compute_on="D")
            loss_coeff = median_similarities_loss.mul(coeff["MEDIAN_LOSS_dataset"])
            loss_total += loss_coeff
            loss_dict["MEDIAN_LOSS_dataset"] = loss_coeff
            if verbose:
                print("MEDIAN_LOSS_dataset - ", loss_coeff)
        
        if "VARIANCE_LOSS" in self.loss_case:            
            variance_similarities_loss = self.variance_similarities(values)
            loss_coeff = variance_similarities_loss.mul(coeff["VARIANCE_LOSS"])
            loss_total += loss_coeff
            loss_dict["VARIANCE_LOSS"] = loss_coeff
            if verbose:
                print("VARIANCE_LOSS - ", loss_coeff)
        
        if "COVARIANCE_LOSS" in self.loss_case:            
            covariance_similarities_loss = self.covariance_similarities(values)
            loss_coeff = covariance_similarities_loss.mul(coeff["COVARIANCE_LOSS"])
            loss_total += loss_coeff
            loss_dict["COVARIANCE_LOSS"] = loss_coeff
            if verbose:
                print("COVARIANCE_LOSS - ", loss_coeff)
        
        if "DECORRELATION_LATENT_LOSS" in self.loss_case:
            decorrelation_latent_loss = self.decorrelation_latent(values)
            loss_coeff = decorrelation_latent_loss.mul(coeff["DECORRELATION_LATENT_LOSS"])
            loss_total += loss_coeff
            loss_dict["DECORRELATION_LATENT_LOSS"] = loss_coeff
            if verbose:
                print("DECORRELATION_LATENT_LOSS - ", loss_coeff)

        if "JENSEN_SHANNON_DIVERGENCE_LOSS" in self.loss_case:
            jsd_loss = self.jensen_shannon_divergence(values)
            loss_coeff = jsd_loss.mul(coeff["JENSEN_SHANNON_DIVERGENCE_LOSS"])
            loss_coeff = jsd_loss.mul(coeff["JENSEN_SHANNON_DIVERGENCE_LOSS"])
            loss_total += loss_coeff
            loss_dict["JENSEN_SHANNON_DIVERGENCE_LOSS"] = loss_coeff
            if verbose:
                print("JENSEN_SHANNON_DIVERGENCE_LOSS - ", loss_coeff)
        
        if "KENDALL_CORRELATION_LOSS" in self.loss_case:
            kendall_correlation_loss = self.kendall_correlation(values)
            
            loss_coeff = kendall_correlation_loss.mul(coeff["KENDALL_CORRELATION_LOSS"])
            loss_total += loss_coeff
            loss_dict["KENDALL_CORRELATION_LOSS"] = loss_coeff
            if verbose:
                print("KENDALL_CORRELATION_LOSS - ", loss_coeff)
        
        if "SPEARMAN_CORRELATION_LOSS" in self.loss_case:
            spearman_correlation_loss = self.spearman_correlation(values)
            
            loss_coeff = spearman_correlation_loss.mul(coeff["SPEARMAN_CORRELATION_LOSS"])
            loss_total += loss_coeff
            loss_dict["SPEARMAN_CORRELATION_LOSS"] = loss_coeff
            if verbose:
                print("SPEARMAN_CORRELATION_LOSS - ", loss_coeff)
        
        if "PEARSON_CORRELATION_LOSS" in self.loss_case:
            pearson_correlation_loss = self.pearson_correlation(values)
            loss_coeff = pearson_correlation_loss.mul(coeff["PEARSON_CORRELATION_LOSS"])
            loss_total += loss_coeff
            loss_dict["PEARSON_CORRELATION_LOSS"] = loss_coeff
            if verbose:
                print("PEARSON_CORRELATION_LOSS - ", loss_coeff)
            
        if  "ORTHOGONAL_REGULARIZATION_LATENT_LOSS" in self.loss_case:
            orthogonal_regularization_latent_loss = self.orthogonal_regularization_latent(values)
            
            loss_coeff = orthogonal_regularization_latent_loss.mul(coeff["ORTHOGONAL_REGULARIZATION_LATENT_LOSS"])
            loss_total += loss_coeff
            loss_dict["ORTHOGONAL_REGULARIZATION_LATENT_LOSS"] = loss_coeff
            if verbose:
                print("ORTHOGONAL_REGULARIZATION_LATENT_LOSS - ", loss_coeff)
        if  "COVARIANCE_DIVERGENCE_LATENT_LOSS" in self.loss_case:
            kl_divergence_latent_loss = self.covariance_divergence_laten(values)
            loss_coeff = kl_divergence_latent_loss.mul(coeff["COVARIANCE_DIVERGENCE_LATENT_LOSS"])
            loss_total += loss_coeff
            loss_dict["COVARIANCE_DIVERGENCE_LATENT_LOSS"] = loss_coeff
            if verbose:
                print("COVARIANCE_DIVERGENCE_LATENT_LOSS - ", loss_coeff)
                
        if "KL_DIVERGENCE_LOSS" in self.loss_case:
            kl_divergence_latent_loss = self.kl_divergence_latent(values)
            loss_coeff = kl_divergence_latent_loss.mul(coeff["KL_DIVERGENCE_LOSS"])
            loss_total += loss_coeff
            loss_dict["KL_DIVERGENCE_LOSS"] = loss_coeff
            if verbose:
                print("KL_DIVERGENCE_LOSS - ", loss_coeff)
                
        if  "CORRELATION_MATRICES_LOSS" in self.loss_case:
            correlation_matrix_loss_val = self.correlation_matrix_loss(values)
            loss_coeff = correlation_matrix_loss_val.mul(coeff["CORRELATION_MATRICES_LOSS"])
            loss_total += loss_coeff
            loss_dict["CORRELATION_MATRICES_LOSS"] = loss_coeff
            if verbose:
                print("CORRELATION_MATRICES_LOSS - ", loss_coeff)
        
        if  "KCCA_LOSS" in self.loss_case:
            kcca_loss_val = self.kcca_loss(values)
            loss_coeff = kcca_loss_val.mul(coeff["KCCA_LOSS"])
            loss_total += loss_coeff
            loss_dict["KCCA_LOSS"] = loss_coeff
            if verbose:
                print("KCCA_LOSS - ", loss_coeff)
                
        if verbose:
            print("loss_total - ", loss_total)
        loss_dict["loss_total"] = loss_total
        return loss_dict

    
    def median_similarities(self, values, compute_on="batch"):
        loss_ret = torch.zeros(1).to(device=self.device)
        median_list_in = []
        median_list_out = []
        #median_list= [list() for i in range(self.univar_count)]
        
        for id_item, val in enumerate(values):
            median_list_in.append(val['x_input']['data'])
            median_list_out.append(val['x_output']['data'])
        
        if compute_on=="batch" or compute_on=="B":
            median_matr_in = torch.Tensor(len(values), self.univar_count).to(device=self.device).requires_grad_()
            median_matr_in = torch.reshape(torch.cat(median_list_in), (len(values),self.univar_count))
            median_in = torch.median(median_matr_in, axis=0)[0]
            
        elif compute_on=="dataset" or compute_on=="D":
            median_list_stats = []
            
            for key in self.vc_mapping:
                median_list_stats.append(self.statsData['median_val'][key])
            median_in = torch.FloatTensor(median_list_stats)
            
            
        median_matr_out = torch.Tensor(len(values), self.univar_count).to(device=self.device).requires_grad_()
        median_matr_out = torch.reshape(torch.cat(median_list_out), (len(values),self.univar_count))        
        median_out = torch.median(median_matr_out, axis=0)[0]
        
        for inp,oup in zip(median_in, median_out):
            loss_ret = loss_ret + torch.square(torch.norm(torch.sub(inp,oup,alpha=1), p=2))
        return loss_ret#torch.mul(loss_ret,1)

    def variance_similarities(self, values, compute_on="batch"):
        loss_ret = torch.zeros(1).to(device=self.device).requires_grad_()
        variance_list_in = []
        variance_list_out = []
        #median_list= [list() for i in range(self.univar_count)]
        
        for id_item, val in enumerate(values):
            variance_list_in.append(val['x_input']['data'])
            variance_list_out.append(val['x_output']['data'])
        
        if compute_on=="batch" or compute_on=="B":
            variance_matr_in = torch.Tensor(len(values), self.univar_count).to(device=self.device).requires_grad_()
            variance_matr_in = torch.reshape(torch.cat(variance_list_in), (len(values), self.univar_count)).requires_grad_()
            variance_in = torch.std(variance_matr_in, dim=0)
        
        
            
        elif compute_on=="dataset" or compute_on=="D":
            variance_list_stats = []
            
            for key in self.vc_mapping:
                variance_list_stats.append(self.statsData['variance_val'][key])
            variance_in = torch.FloatTensor(variance_list_stats)
            
        variance_matr_out = torch.Tensor(len(values), self.univar_count).to(device=self.device).requires_grad_()
        variance_matr_out = torch.reshape(torch.cat(variance_list_out), (len(values),self.univar_count))        
        
        variance_out = torch.std(variance_matr_out, dim=0)
        for inp,oup in zip(variance_in, variance_out):
            loss_ret = loss_ret + torch.square(torch.norm(torch.sub(inp,oup,alpha=1), p=2))
        
        
        
        return loss_ret#torch.mul(loss_ret,1)

    def covariance_similarities(self, values):
        loss_ret = torch.zeros(1).to(device=self.device)
        covariance_list_in = []
        covariance_list_out = []
        
        for id_item, val in enumerate(values):
            covariance_list_in.append(val['x_input']['data'])
            covariance_list_out.append(val['x_output']['data'])

        covariance_matr_in = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        covariance_matr_in = torch.reshape(torch.cat(covariance_list_in), (len(values),self.univar_count))
        covariance_matr_in = torch.transpose(covariance_matr_in, 0, 1)
        covariance_in = torch.cov(covariance_matr_in, correction=1)
        

        covariance_matr_out = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        covariance_matr_out = torch.reshape(torch.cat(covariance_list_out), (len(values),self.univar_count))
        covariance_matr_out = torch.transpose(covariance_matr_out, 0, 1)
        covariance_out = torch.cov(covariance_matr_out, correction=1)

        for inp_row,oup_row in zip(covariance_in, covariance_out):
            for inp_item,oup_item in zip(inp_row,oup_row):
                loss_ret += torch.square(torch.norm(torch.sub(inp_item,oup_item, alpha=1), p=2))
        return loss_ret

    
    def kendall_correlation(self, values):
        loss_ret = torch.zeros(1).to(device=self.device)
        covariance_list_in = []
        covariance_list_out = []
        
        for id_item, val in enumerate(values):
            covariance_list_in.append(val['x_input']['data'])
            covariance_list_out.append(val['x_output']['data'])
        covariance_matr_in = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        covariance_matr_in = torch.reshape(torch.cat(covariance_list_in), (len(values),self.univar_count))
        covariance_matr_out = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        covariance_matr_out = torch.reshape(torch.cat(covariance_list_out), (len(values),self.univar_count))
        
        kendall_values = kendall_rank_corrcoef(covariance_matr_in, covariance_matr_out)
        
        for val in kendall_values:
            loss_ret += Tensor([1]).to(device=self.device)-val
        loss_ret /= len(kendall_values)
        return loss_ret


    def spearman_correlation(self, values):
        loss_ret = torch.zeros(1).to(device=self.device).requires_grad_()
        data_list_in = []
        data_list_out = []
        
        for id_item, val in enumerate(values):
            data_list_in.append(val['x_input']['data'])
            data_list_out.append(val['x_output']['data'])
           
        data_list_in = torch.cat(data_list_in, dim=0).to(device=self.device).requires_grad_()
        data_list_out = torch.cat(data_list_out, dim=0).to(device=self.device).requires_grad_()

        data_list_in = torch.reshape(data_list_in, (len(values), self.univar_count))
        data_list_out = torch.reshape(data_list_out, (len(values), self.univar_count))
        
        spearman_values_in = self.spearman_corr(data_list_in)
        spearman_values_out = self.spearman_corr(data_list_out)
        
       
        
        loss_ret = torch.norm(spearman_values_in - spearman_values_out, p='fro')
        
        return loss_ret

    
    
    def spearman_corr(self, data):
        """
        Calcola la correlazione di Spearman sui dati in modo differenziabile.
        
        :param data: Tensor di forma (n_samples, n_features)
        :return: Matrice di correlazione di Spearman
        """
        # Applichiamo la softmax per ottenere i ranghi differenziabili
        exp_data = torch.exp(data - torch.max(data, dim=0, keepdim=True).values)  # Sottrarre il massimo per evitare overflow
        ranks = exp_data / torch.sum(exp_data, dim=0, keepdim=True)  # Normalizza su ogni colonna
        
        # Centriamo i ranghi sottraendo la media per ogni feature (colonna)
        ranks_centered = ranks - torch.mean(ranks, dim=0)
        
        # Calcoliamo la correlazione di Pearson sui ranghi centrati (usando la stessa formula della Pearson)
        cov_matrix = torch.mm(ranks_centered.t(), ranks_centered) / (ranks.size(0) - 1)
        
        # Calcoliamo le deviazioni standard dei ranghi
        std_devs = torch.sqrt(torch.diagonal(cov_matrix))
        
        # Creiamo la matrice di correlazione di Spearman normalizzando la covarianza
        correlation_matrix = cov_matrix / torch.outer(std_devs, std_devs)
        
        return correlation_matrix


    def pearson_correlation(self, values):
        loss_ret = torch.zeros(1).to(device=self.device).requires_grad_()
        data_list_in = []
        data_list_out = []
        
        for id_item, val in enumerate(values):
            data_list_in.append(val['x_input']['data'])
            data_list_out.append(val['x_output']['data'])
           
        data_list_in = torch.cat(data_list_in, dim=0).to(device=self.device).requires_grad_()
        data_list_out = torch.cat(data_list_out, dim=0).to(device=self.device).requires_grad_()

        data_list_in = torch.reshape(data_list_in, (len(values), self.univar_count))
        data_list_out = torch.reshape(data_list_out, (len(values), self.univar_count))
        
        pears_values_in = self.pearson_corr(data_list_in)
        pears_values_out = self.pearson_corr(data_list_out)
        
       
        loss_ret = torch.norm(pears_values_in - pears_values_out, p='fro')
        
        return loss_ret
    
    def pearson_corr(self, data):
        means = torch.mean(data, dim=0)
    
        # Centriamo i dati sottraendo la media
        centered_data = data - means
        
        # Calcoliamo la correlazione di Pearson tra le feature
        # Matrice di covarianza centrata
        cov_matrix = torch.mm(centered_data.t(), centered_data) / (data.size(0) - 1)
        
        # Calcoliamo le deviazioni standard per ogni feature (sommatoria dei quadrati centrati)
        std_devs = torch.sqrt(torch.diagonal(cov_matrix))
        
        # Creiamo una matrice di correlazione dividendo per i prodotti delle deviazioni standard
        correlation_matrix = cov_matrix / torch.outer(std_devs, std_devs)
        
        return correlation_matrix


    

    def spearman_correlation_new(self, values, epoch, normalize=True):
        loss_ret = torch.zeros(1).to(device=self.device)
        covariance_list_in = []
        covariance_list_out = []
        
        for id_item, val in enumerate(values):
            covariance_list_in.append(val['x_input']['data'])
            covariance_list_out.append(val['x_output']['data'])
            
        # Creazione iniziale dei tensori con requires_grad
        covariance_matr_in = torch.cat(covariance_list_in, dim=0).to(device=self.device).requires_grad_()
        covariance_matr_out = torch.cat(covariance_list_out, dim=0).to(device=self.device).requires_grad_()

        # Poi fai la reshape
        covariance_matr_in = torch.reshape(covariance_matr_in, (len(values), self.univar_count))
        covariance_matr_out = torch.reshape(covariance_matr_out, (len(values), self.univar_count))


        if normalize:
            covariance_matr_in = self.z_score_normalization(covariance_matr_in)
            covariance_matr_out = self.z_score_normalization(covariance_matr_out)

        # Rank the matrices (along rows)
        rank_in = covariance_matr_in.argsort(dim=1).float()
        rank_out = covariance_matr_out.argsort(dim=1).float()

        # Calculate differences between ranks
        rank_diff = rank_in - rank_out

        # Compute squared differences
        rank_diff_squared = rank_diff ** 2

        # Calculate sum of squared differences per feature
        sum_rank_diff_squared = torch.sum(rank_diff_squared, dim=1)

        # Number of elements (n)
        n = torch.tensor(len(values), dtype=torch.float32, device=self.device)

        # Spearman correlation formula
        spearman_corr = 1 - (6 * sum_rank_diff_squared) / (n * (n ** 2 - 1) + 1e-8)

        # If you want negative Spearman correlation as loss (maximize correlation → minimize loss)
        loss_ret = (-spearman_corr.mean()).requires_grad_()
        
        
        return loss_ret

    def min_max_normalization(self, tensor):
        min_val = tensor.min(dim=1, keepdim=True)[0]  # Minimum along each row
        max_val = tensor.max(dim=1, keepdim=True)[0]  # Maximum along each row
        return (tensor - min_val) / (max_val - min_val + 1e-8)  # Normalize within [0, 1]

    def z_score_normalization(self, tensor):
        mean = tensor.mean(dim=1, keepdim=True)  # Mean along each row
        std = tensor.std(dim=1, keepdim=True)    # Standard deviation along each row
        return (tensor - mean) / (std + 1e-8)    # Normalize

    def MSE_similarities(self, values):
        """
        ys_true : vector of items where each item is a groundtruth matrix 
        ys_pred : vector of items where each item is a prediction matrix 
        return the sum of 2nd proximity of 2 matrix
        """
        loss_ret = torch.zeros(1).to(device=self.device)     
        loss_mse = nn.MSELoss()

        for i, val in enumerate(values):
            
            loss_mse_val = loss_mse(val['x_output']['data'], val['x_input']['data'])
            loss_ret += loss_mse_val
        loss_ret /= len(values)
        return loss_ret#torch.mul(loss_ret,1)
    

    def RMSE_similarities(self, values):
        """
        ys_true : vector of items where each item is a groundtruth matrix 
        ys_pred : vector of items where each item is a prediction matrix 
        return the sum of 2nd proximity of 2 matrix
        """
        loss_ret = torch.zeros(1).to(device=self.device)      
        loss_mse = nn.MSELoss()
        for i, val in enumerate(values):            
            loss_mse_val = loss_mse(val['x_output']['data'], val['x_input']['data'])
            loss_ret += loss_mse_val
        loss_ret_sqrt = torch.sqrt(loss_ret)
        return loss_ret_sqrt

 
    def orthogonal_regularization_latent(self, values, latent_key="latent"):
        M = len(values)  # Number of latent samples
        loss_ret = torch.zeros(1).to(device=self.device)  # Initialize the total loss
    
        # Collect all latent vectors   
        latent_vectors = torch.stack([value['x_latent'][latent_key] for value in values])
    
        # Iterate through all pairs of latent variables
        for k in range(self.latent_dim):
            for i in range(k):  # Avoid redundant symmetric calculations
                # Extract the k-th and i-th latent dimensions
                z_k_all = latent_vectors[:, k]  # All samples for dimension k
                z_i_all = latent_vectors[:, i]  # All samples for dimension i
                
                # Calculate the average dot product
                dot_product = torch.dot(z_k_all, z_i_all) / M
                
                # Penalize the deviation from orthogonality
                loss_orthogonal = torch.abs(dot_product)
                
                # Add the loss for this pair to the total loss
                loss_ret += loss_orthogonal
        
        return loss_ret
    
    def covariance_divergence_laten(self, values, latent_key="latent"):
        M = len(values)  # Number of samples
        latent_dim = values[0]["x_latent"][latent_key].shape[0]  # Dimensionality of the latent space
        
        # Stack all latent vectors into a matrix of shape (latent_dim, M)
        Z = torch.stack([values[j]["x_latent"][latent_key] for j in range(M)], dim=1)
        
        # Compute the covariance matrix: (latent_dim x latent_dim)
        covariance_matrix = torch.mm(Z, Z.t()) / M
        
        # Compute the off-diagonal elements of the covariance matrix
        # We want to minimize these to encourage independence
        diag_elements = torch.diag(covariance_matrix)  # Diagonal elements
        covariance_matrix -= torch.diag(diag_elements)  # Set diagonal elements to 0

        # Compute the decorrelation loss as the sum of squares of the off-diagonal elements
        loss = torch.sum(covariance_matrix ** 2)
        
        return loss
    
    def kl_divergence_latent(self, values, latent_key=None):
        
        latent_mu = torch.stack([value["x_latent"]["mu"] for value in values], dim=0) 
        latent_logvar = torch.stack([value["x_latent"]["logvar"] for value in values], dim=0)  
        
        loss = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp(), dim=1).mean()
        return loss
    
    def decorrelation_latent(self, values, latent_key="latent"):
        M = len(values)
        
        loss_ret = torch.zeros(1).to(device=self.device)
        for k in range(self.latent_dim):
            for i in range(k):
                loss_a = torch.zeros(1).to(device=self.device)
                loss_b = torch.zeros(1).to(device=self.device)
                
                for j in range(M):
                    z_j = values[j]['x_latent'][latent_key]
                    z_j_i = z_j[k]
                    z_j_k = z_j[i]
                    loss_a += z_j_i * z_j_k

                loss_a = torch.div(loss_a, M)
                loss_b_0 = torch.zeros(1).to(device=self.device)
                loss_b_1 = torch.zeros(1).to(device=self.device)
                for j in range(M):
                    z_j = values[j]['x_latent'][latent_key]
                    loss_b_0 += z_j[k]
                    loss_b_1 += z_j[i]
                loss_b = loss_b_0 * loss_b_1
                loss_b = torch.div(loss_b, M**2)

                loss_dif = torch.abs(loss_a - loss_b)
                loss_ret += loss_dif
        return loss_ret
    
    def jensen_shannon_divergence(self, values):
        loss_sum = torch.zeros(1).to(device=self.device)
        kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        
        
        JSD_list_in = dict()
        JSD_list_out = dict()

        for vc in range(self.univar_count):
            JSD_list_in[vc] = list()
            JSD_list_out[vc] = list()
        
        for id_item, val in enumerate(values):
            for vc in range(self.univar_count):
                JSD_list_in[vc].append(val['x_input']['data'][vc])
                JSD_list_out[vc].append(val['x_output']['data'][vc])

        for vc in range(self.univar_count):
                        
            
            x_in_i = torch.stack(JSD_list_in[vc], dim=0).to(device=self.device).requires_grad_()
            x_out_i = torch.stack(JSD_list_out[vc], dim=0).to(device=self.device).requires_grad_()
            x_in_v = x_in_i.view(-1, x_in_i.size(-1))
            x_out_v = x_out_i.view(-1, x_out_i.size(-1))
            
            x_in =  F.softmax(x_in_v, dim=1)
            x_out = F.softmax(x_out_v, dim=1)
            
            eps = 1e-8
            m = 0.5 * (x_in + x_out)
            m = m.clamp(min=eps)
            a = kl(x_in.log(), m.log())
            b = kl(x_out.log(), m.log())
            jsd_value = 0.5 * (a + b)
            
            loss_sum += jsd_value
            
        loss_ret = loss_sum
        return loss_ret

    def correlation_matrix_loss(self, values, aggregation="MAE", correlation_matrix_mode="all", correlation_matrix_sparsify="False", sliding_window_size=100, sparsify_threshold=0.05, compute_on="batch"):
        val_list_in = []
        val_list_out = []
        mse_loss = nn.MSELoss()
        
        for id_item, val in enumerate(values):
            val_list_in.append(val['x_input']['data'])
            val_list_out.append(val['x_output']['data'])
        
        val_matr_in = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        val_matr_in = torch.reshape(torch.cat(val_list_in), (len(values), self.univar_count))
        
        val_matr_out = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        val_matr_out = torch.reshape(torch.cat(val_list_out), (len(values),self.univar_count))
        
        x_input_corr = self.get_correlation_matrix(val_matr_in, mode= correlation_matrix_mode, sparsify= correlation_matrix_sparsify, window_size=sliding_window_size, sparsify_threshold=sparsify_threshold)
        x_output_corr = self.get_correlation_matrix(val_matr_out, mode= correlation_matrix_mode, sparsify= correlation_matrix_sparsify, window_size=sliding_window_size, sparsify_threshold=sparsify_threshold)

        if aggregation == "MSE":
            loss = torch.mean((x_input_corr - x_output_corr) ** 2)
        elif aggregation == "RMSE":
            loss = torch.sqrt(torch.mean((x_input_corr - x_output_corr) ** 2))
        elif aggregation == "MAE":
            loss = torch.mean(torch.abs(x_input_corr - x_output_corr))
        else:
            raise ValueError("Unsupported aggregation method: {}".format(aggregation))
        return loss
            
    def get_correlation_matrix(self, val_matr, mode, sparsify=False, window_size=100, sparsify_threshold=0.05):
        if mode == "all":
            correlation_mat = self.compute_correlation_matrix(val_matr)
        elif mode == "feature_selection":
            correlation_mat = self.correlation_matrix_by__feature_selection(val_matr)
        elif mode == "sliding_window":
            correlation_mat = self.correlation_matrix_by__sliding_window(val_matr, window_size=window_size)
        if sparsify:
            correlation_mat = self.sparsify_correlation_matrix(correlation_mat, threshold= sparsify_threshold)
        return correlation_mat
        
    def compute_correlation_matrix(self, val_matr):
        """Calcola la matrice di correlazione di un batch di dati"""
        val_matr = val_matr - val_matr.mean(dim=0, keepdim=True)  # Centra i dati
        cov_matrix = torch.matmul(val_matr.T, val_matr) / (val_matr.shape[0] - 1)  # Matrice di covarianza
        std = torch.sqrt(torch.diag(cov_matrix)).unsqueeze(1)  # Deviazione standard
        corr_matrix = cov_matrix / (std @ std.T + 1e-8)  # Matrice di correlazione
        return corr_matrix    
    
    def rbf_kernel(self, X, gamma=None):
        """ kernel RBF tra i punti di X """
        pairwise_sq_dists = torch.cdist(X, X) ** 2
        if gamma is None:
            gamma = 1.0 / X.shape[1]  # Default gamma = 1 / num_features
        return torch.exp(-gamma * pairwise_sq_dists)

    def kcca_loss(self, values, reg=1e-3):
        
        val_list_in = []
        val_list_out = []
        
        for id_item, val in enumerate(values):
            val_list_in.append(val['x_input']['data'])
            val_list_out.append(val['x_output']['data'])
        
        X = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        X = torch.reshape(torch.cat(val_list_in), (len(values), self.univar_count))
        
        Y = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        Y = torch.reshape(torch.cat(val_list_out), (len(values),self.univar_count))
        
        
        
        """ Minimizza la differenza tra le matrici di correlazione kernel """
        K_X = self.rbf_kernel(X)
        K_Y = self.rbf_kernel(Y)

        # Centralizzazione delle matrici di kernel
        H = torch.eye(K_X.shape[0], device=X.device) - (1 / K_X.shape[0]) * torch.ones_like(K_X)
        K_X = H @ K_X @ H
        K_Y = H @ K_Y @ H

        # Calcoliamo le matrici di covarianza kernel con regolarizzazione
        C_XX = K_X @ K_X.T + reg * torch.eye(K_X.shape[0], device=X.device)
        C_YY = K_Y @ K_Y.T + reg * torch.eye(K_Y.shape[0], device=X.device)
        C_XY = K_X @ K_Y.T

        # Correlazione canonica kernelizzata
        loss = -torch.trace(torch.linalg.solve(C_XX, C_XY) @ torch.linalg.solve(C_YY, C_XY.T))

        return loss


    def correlation_matrix_by__feature_selection(self, val_matr, n_components=10):
        """Calcola la matrice di correlazione su un sottoinsieme di feature selezionate con PCA"""
        val_matr_np = val_matr.detach().cpu().numpy()  # Converti in NumPy
        pca = PCA(n_components=n_components)
        val_matr_reduced = torch.tensor(pca.fit_transform(val_matr_np), device=self.device)  # Riduzione dimensionale
        return self.compute_correlation_matrix(val_matr_reduced)  # Calcola la matrice di correlazione sulle feature selezionate

    def sparsify_correlation_matrix(self, corr_matrix, threshold=0.1):
        """Mantiene solo le correlazioni con valore assoluto sopra la soglia"""
        sparse_corr = torch.where(torch.abs(corr_matrix) < threshold, torch.tensor(0.0, device=self.device), corr_matrix)
        return sparse_corr
    
    
    def correlation_matrix_by__sliding_window(self, val_matr, window_size=100):
        """Calcola la matrice di correlazione su una finestra di dati"""
        x_output_windows = val_matr['x_output']['data'].unfold(0, window_size, step=window_size)
        x_input_windows = val_matr['x_input']['data'].unfold(0, window_size, step=window_size)

        correlations_output = [self.compute_correlation_matrix(window) for window in x_output_windows]
        correlations_input = [self.compute_correlation_matrix(window) for window in x_input_windows]

        # Restituisce la media delle matrici di correlazione nelle finestre
        return torch.stack(correlations_output).mean(dim=0), torch.stack(correlations_input).mean(dim=0)



##mahalanobis


class SpearmanCorrCoef_grad(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        # Salva gli input per il backward pass
        ctx.save_for_backward(input1, input2)

        # Ordina i tensori in ordine crescente per ottenere i ranghi
        rank1 = input1.argsort(dim=0)
        rank2 = input2.argsort(dim=0)
        
        # Calcola la differenza tra i ranghi
        diff = rank1 - rank2
        diff_squared = diff.float() ** 2
        
        # Somma tutte le differenze al quadrato
        loss = diff_squared.sum()

        # Restituisce la perdita (valore scalare)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # Recupera i tensori originali
        input1, input2 = ctx.saved_tensors

        # Calcola i ranghi
        rank1 = input1.argsort(dim=0)
        rank2 = input2.argsort(dim=0)
        
        # Calcola la differenza tra i ranghi
        diff = rank1 - rank2
        grad_diff = 2 * diff.float()  # Derivata della somma dei quadrati delle differenze
        
        # Propaga il gradiente ai tensori di input
        grad_input1 = grad_diff * grad_output
        grad_input2 = -grad_diff * grad_output
        
        return grad_input1, grad_input2
