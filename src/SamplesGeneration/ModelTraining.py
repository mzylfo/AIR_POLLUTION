import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from src.NeuroCorrelation.DataBatchGenerator import DataBatchGenerator
from src.NeuroCorrelation.NeuroDistributions import NeuroDistributions
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
import json
from matplotlib.ticker import PercentFormatter


class ModelTraining():

    def __init__(self, model, loss_obj, epoch, dataset, dataGenerator, path_folder, univar_count):
        self.model = model()
        self.loss_obj = loss_obj
        self.epoch = epoch
        self.dataset = dataset
        self.dataGenerator = dataGenerator
        self.path_folder = path_folder
        self.loss_dict = dict()
        self.path_folder_csv = Path(self.path_folder,"csv_dist")
        if not os.path.exists(self.path_folder_csv):
            os.makedirs(self.path_folder_csv)
        self.univar_count = univar_count        
        model_params = self.model.parameters()
        self.optimizer = SGD(params=model_params, lr=0.01, momentum=0.9)

    def training(self, batch_size=128, shuffle_data=True, plot_loss=True):
        self.loss_dict = dict()
        for epoch in range(self.epoch):
            
            dataLoaded= DataBatchGenerator(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle_data)
            dataBatches = dataLoaded.generate()
            loss_batch = list()
            for batch_num, dataBatch in enumerate(dataBatches):
                loss = torch.zeros([1])
                self.optimizer.zero_grad()
                for i, (samplef, noisef) in enumerate(dataBatch):
                    sample = samplef.type(torch.float32)
                    noise = noisef.type(torch.float32)
                    # compute the model output
                    y_hat = self.model.forward(x=noise)
                    # calculate loss
                    crit = self.loss_obj.computate_loss(y_hat)

                    loss += crit
                loss_batch.append(loss.detach().numpy())
                
                loss.backward()
                self.optimizer.step()

            self.loss_dict[epoch] = {"mean": np.mean(loss_batch), "values_list": loss_batch}
            print("epoch",epoch,"\tmean: ",np.mean(loss_batch))
        if plot_loss:
            self.loss_plot(self.loss_dict)
        return self.model
    
    def loss_plot(self, loss_dict):
        loss_list = list()
        for mean_val in loss_dict:
            loss_list.append(loss_dict[mean_val]['mean'])
        plt.figure(figsize=(12,8))  
        plt.plot(loss_list, color='Blue', marker='o',mfc='Blue' )
        filename = Path(self.path_folder,"loss_training.png")
        plt.savefig(filename)
        


    #deprecato
    def random_sample(self, random_samples=50000):
        self.random_samples = random_samples
        self.predictions = list()
        for i in range(random_samples):
            noise = self.dataGenerator.getRandom()
            yhat = self.model(noise)
            self.predictions.append(yhat)
        self.generated_dict = dict()
        for univ_id in range(self.univar_count):
            self.generated_dict[univ_id] = list()

        for i in range(self.random_samples):
            for univ_id in range(self.univar_count):
                self.generated_dict[univ_id].append(self.predictions[i]['x_output'][univ_id].detach().numpy())

    #
    def compute_correlationMatrix(self):
        correlationList = list()
        correlationList_txt = list()
        for univ_id in range(self.univar_count):

            self.generated_dict[univ_id] = np.array( self.generated_dict[univ_id], dtype = float) 
            correlationList.append(self.generated_dict[univ_id])
            correlationList_txt.append(self.generated_dict[univ_id].tolist())

        corrCoeff_list_gen_Path = Path(self.path_folder, "corrCoeffList_generated.txt")
        with open(corrCoeff_list_gen_Path, 'w') as fp:
            json.dump(correlationList_txt, fp, sort_keys=True, indent=4)

        corrCoeff_matrix_gen = np.corrcoef(correlationList)
        corrCoeff_matrix_gen_Path = Path(self.path_folder, "corrCoeffMatrix_generated.csv")
        np.savetxt(corrCoeff_matrix_gen_Path, corrCoeff_matrix_gen, delimiter=",")
        corrCoeff_matrix_orig_Path = Path(self.path_folder, "corrCoeffMatrix_original.csv")
        corrCoeff_matrix_orig = np.loadtxt(corrCoeff_matrix_orig_Path,delimiter=",")

        sub_correlation_matrix = np.subtract(corrCoeff_matrix_gen,corrCoeff_matrix_orig)
        sub_correlation_matrix_Path = Path(self.path_folder, "corrCoeffMatrix_sub.csv")
        np.savetxt(sub_correlation_matrix_Path, sub_correlation_matrix, delimiter=",")
    
    #deprecato-
    def overfitting_univariate(self, univ_id, draw_plot=False):
        
        name_case = f'univar_{univ_id}'
        
        sampled_list = list()
        for i in range(len(self.dataset)):
            sampled_list.append(self.dataset[i][0][univ_id].numpy())
        sampled_list_np = np.array(sampled_list,dtype = float)    

        neuroDspl = NeuroDistributions(self.path_folder,sampled_list_np, univ_id)
        neuroDspl_list = neuroDspl.best_fit_distribution()
        
        generated_list = list()
        for i in range(self.random_samples):
            generated_list.append(self.predictions[i]['x_output'][univ_id].detach().numpy())
        generated_list_np = np.array(generated_list, dtype = float)
        
        
        neuroDgen = NeuroDistributions(self.path_folder, generated_list_np, univ_id)
        if draw_plot:
            neuroDgen.plotDistributions()
        neuroDgen_list = neuroDgen.best_fit_distribution()
        
        self.plot_dataDist()
        self.plot_dataDist_comp()
        
        self.saveDistComparison(univ_id=univ_id, neuroDspl_list= neuroDspl_list, neuroDgen_list=neuroDgen_list)

        return neuroDgen.get_best_dist()
    
    #deprecato
    def plot_dataDist(self):
        path_fold_dist = Path(self.path_folder,"univar_distribution_generate")
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)
        for univ_id in range(self.univar_count):
            data_vals = self.generated_dict[univ_id]
            mean_val = np.mean(data_vals)
            plt.figure(figsize=(12,8))            
            plt.axvline(x = mean_val, color = 'b', label = 'mean')
            plt.hist(data_vals, density=True, bins=50)
            mean_plt_txt = f"      mean: {mean_val:.3f}"
            plt.text(mean_val, 0, s=mean_plt_txt, rotation = 90)            
            filename = Path(path_fold_dist,"dist_generate_"+str(univ_id)+".png")
            plt.savefig(filename)
    
    #deprecato
    def plot_dataDist_comp(self):
        path_fold_dist = Path(self.path_folder,"univar_distribution_comparison")
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)
        
        mu = self.dataGenerator.get_muR()
        mu = mu["mu"]
        sample_synthetic = self.dataGenerator.get_synthetic_data()

        stats_dict = pd.DataFrame(columns = ['univ_id', 'orig_mean', 'orig_var', 'gen_mean', 'gen_var'])


        for univ_id in range(self.univar_count):
            data_vals_gen= self.generated_dict[univ_id]
            mean_val_gen = np.mean(data_vals_gen)
            var_val_gen = np.var(data_vals_gen)
            
            data_val_synt = sample_synthetic[univ_id][0]
            if mu is None:
                mean_val_synt= np.mean(data_val_synt)
            else:
                mean_val_synt = mu[univ_id]  
            var_val_synt = np.var(data_val_synt)
            new_row = {'univ_id':str(univ_id), 'orig_mean':str(mean_val_synt), 'orig_var':str(var_val_synt), 'gen_mean':str(mean_val_gen), 'gen_var':str(var_val_gen) }
            stats_dict = stats_dict.append(new_row, ignore_index=True)

            plt.figure(figsize=(12,8))  
                     
            plt.axvline(x = mean_val_gen, color = 'r', label = 'mean')
            mean_plt_txt = f"      GEN mean: {mean_val_gen:.3f}"
            plt.text(mean_val_gen, 0, s=mean_plt_txt, rotation = 90) 

            plt.axvline(x = mean_val_synt, color = 'b', label = 'mean')
            mean_plt_txt = f"      SYN mean: {mean_val_synt:.3f}"
            plt.text(mean_val_synt, 0, s=mean_plt_txt, rotation = 90)   

            plt.hist(data_vals_gen, weights=np.ones(len(data_vals_gen)) / len(data_vals_gen), color='Red', alpha=0.5)
            plt.hist(data_val_synt, weights=np.ones(len(data_val_synt)) / len(data_val_synt), color='Blue', alpha=0.5)

            #apply percentage format to y-axis
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

            filename = Path(path_fold_dist,"dist_comp_"+str(univ_id)+".png")
            plt.savefig(filename)

        file_name = Path(self.path_folder_csv, "stats_mean_var.csv")
        stats_dict.to_csv(file_name)


    def saveDistComparison(self, univ_id, neuroDspl_list, neuroDgen_list):
        if len(neuroDspl_list)!=len(neuroDgen_list):
            print(len(neuroDspl_list), " == ", len(neuroDgen_list))
        else:
            print(len(neuroDspl_list), " == ", len(neuroDgen_list))
        neuroD_dict = dict()
        neuroD_dict["sample"] = dict()
        for nDist in neuroDspl_list:
            nDist_name = nDist[0].name
            nDist_params = nDist[1]
            nDist_sse = nDist[2] #Sum of Square Error (SSE)
            neuroD_dict["sample"][nDist_name] = [nDist_params,nDist_sse]
        
        neuroD_dict["gen"] = dict()
        for nDist in neuroDspl_list:
            nDist_name = nDist[0].name
            nDist_params = nDist[1]
            nDist_sse = nDist[2] #Sum of Square Error (SSE)
            neuroD_dict["gen"][nDist_name] = [nDist_params,nDist_sse]

        neuroDist_columns=['name_dist','gen_params','gen_sse','sample_params','sample_sse']
        neuroDist_list = list()
        
        for ndis in neuroDgen_list:
            name_dist = ndis[0].name
            if name_dist in neuroD_dict["gen"]:
                [gen_nDist_params, gen_nDist_sse] = neuroD_dict["gen"][name_dist]
            else:
                [gen_nDist_params, gen_nDist_sse] = [None, None]
            if name_dist in neuroD_dict["sample"]:
                [spl_nDist_params, spl_nDist_sse] = neuroD_dict["sample"][name_dist]
            else:
                [spl_nDist_params, spl_nDist_sse] = [None, None]
            dist_dict = [name_dist,gen_nDist_params,gen_nDist_sse,spl_nDist_params,spl_nDist_sse]
            neuroDist_list.append(dist_dict)
        
        neuroDist_df = pd.DataFrame(neuroDist_list,columns=neuroDist_columns)
        
        file_name = Path(self.path_folder_csv, "univarDistribution_compare_"+str(univ_id)+".csv")
        neuroDist_df.to_csv(file_name)
