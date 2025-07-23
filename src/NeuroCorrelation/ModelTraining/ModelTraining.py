import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
import torch.optim as optim
from torch.nn import BCELoss
from src.NeuroCorrelation.DataLoaders.DataBatchGenerator import DataBatchGenerator
from src.NeuroCorrelation.Analysis.NeuroDistributions import NeuroDistributions
from src.NeuroCorrelation.ModelPrediction.ModelPrediction import ModelPrediction
from src.NeuroCorrelation.Analysis.TimeAnalysis import TimeAnalysis
from termcolor import colored, cprint 
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import json
from matplotlib.ticker import PercentFormatter
import datetime
from torchviz import make_dot
from torch import autograd
from torch.autograd import Variable
from matplotlib.pyplot import cm
from torch.nn.utils import parameters_to_vector
from termcolor import colored, cprint 
from torch_geometric.nn import GCNConv
import csv

class ModelTraining():

    def __init__(self, model, device, loss_obj, epoch, train_data, test_data, dataGenerator, path_folder, univar_count_in, univar_count_out, latent_dim,   vc_mapping, input_shape, rangeData, batch_size, time_performance, timeweather_count, learning_rate, optimizer_name="Adam", model_type="AE", pre_trained_decoder=False, optimization=False, graph_topology=False, edge_index=None, is_optimization=False, noise_data = None):
        self.loss_obj = loss_obj
        self.epoch = epoch
        self.train_data = train_data
        self.test_data = test_data
        self.noise_data = noise_data
        self.dataGenerator = dataGenerator
        self.path_folder = path_folder
        self.loss_dict = dict()
        self.model_type = model_type
        self.vc_mapping = vc_mapping
        self.rangeData = rangeData
        self.device = device
        self.input_shape = input_shape
        self.append_count = 1 #ogni quanti batch aggiungo val della loss
        self.n_critic = 1
        self.opt_scheduler_ae = "ReduceLROnPlateau"
        self.opt_scheduler_gen = "ReduceLROnPlateau"
        self.opt_scheduler_dis = "ReduceLROnPlateau"
        cprint(f"PAY ATTENTION: GEN is update every {self.n_critic} batches", "magenta", end="\n")
        self.GAN_loss = "MSELoss"
        self.univar_count_in = univar_count_in 
        self.univar_count_out = univar_count_out
        self.latent_dim = latent_dim
        self.edge_index = edge_index
        self.batchsize = batch_size[self.model_type]
        self.graph_topology = graph_topology
        self.path_save_model_gradients = Path(self.path_folder, self.model_type,"model_gradients", )
        self.time_performance = time_performance
        self.is_optimization = is_optimization
        self.timeweather_count = timeweather_count
        
        
        if not os.path.exists(self.path_save_model_gradients):
            os.makedirs(self.path_save_model_gradients)
        self.checkWeightsUpdate = True    
        if self.checkWeightsUpdate:
            cprint(f"PAY ATTENTION: check weights update is on", "magenta", end="\n")
        if self.model_type in ["AE", "VAE", "CVAE"]:
            self.model = model
            self.model.to(device=self.device)
            model_params = self.model.parameters()            
            self.learning_rate = dict()
            self.learning_rate[self.model_type] = learning_rate[self.model_type]
            
            if optimizer_name == "AdamW":
                self.optimizer = optim.AdamW(params=model_params, lr=0.001, weight_decay=0.01)
            elif optimizer_name == "Adam":
                self.optimizer = optim.Adam(params=model_params, lr=self.learning_rate[self.model_type])
            elif optimizer_name == "SGD":
                self.optimizer = optim.SGD(params=model_params, lr=0.01, momentum=0.9, weight_decay=0.0001)
            cprint(f"\tOptimizer\t{optimizer_name}", "green", end="\n")
            cprint(f"\tLearning_rate\t{self.learning_rate}", "green", end="\n")
            
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
            if self.opt_scheduler_ae == "ReduceLROnPlateau":
                self.scheduler_ae = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5)
            elif self.opt_scheduler_ae == "StepLR":
                self.scheduler_ae = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        
        elif self.model_type in ["GAN", "WGAN"]:
            lr_gen = 0.05
            lr_dis = 0.05
            b1_gen = 0.05   #decay of first order momentum of gradient gen
            b1_dis = 0.05   #decay of first order momentum of gradient dis
            b2_gen = 0.999  #decay of first order momentum of gradient gen
            b2_dis = 0.999  #decay of first order momentum of gradient dis
            self.lambda_wgan_gp = 5  # Gradient penalty coefficient
            self.model = model
            
            if pre_trained_decoder:
                self.model_gen = model.get_generator()
            else:
                model_gen = model.get_generator()
                self.model_gen = model_gen()
            self.model_gen.to(device=self.device)
            gen_params = self.model_gen.parameters()
            
            self.optimizer_gen = optim.Adam(gen_params, lr=lr_gen)#, betas=(b1_gen, b2_gen))
            if self.opt_scheduler_gen == "ReduceLROnPlateau":
                self.scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen, factor=0.1, patience=10, verbose=True)
            elif self.opt_scheduler_gen == "StepLR":
                self.scheduler_gen = optim.lr_scheduler.StepLR(self.optimizer_gen, step_size=20, gamma=0.1)
            model_dis = self.model.get_discriminator()
            
            self.model_dis = model_dis
            self.model_dis.to(device=self.device)
            dis_params = self.model_dis.parameters()
            self.optimizer_dis = optim.Adam(dis_params, lr=lr_dis)#, betas=(b1_gen, b2_gen))  
            
            if self.opt_scheduler_dis == "ReduceLROnPlateau":
                self.scheduler_dis = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_dis, factor=0.1, patience=5, verbose=True)
            elif self.opt_scheduler_dis == "StepLR":
                self.scheduler_dis = optim.lr_scheduler.StepLR(self.optimizer_dis, step_size=15, gamma=0.1)
            
            self.criterion = nn.BCELoss()
            '''
            if self.GAN_loss =="BCELoss":
                self.criterion = nn.BCELoss()
            elif self.GAN_loss =="MSELoss":
                self.criterion = nn.MSE_LOSS()'''
                
        if optimization:
            self.path_opt_results = Path(self.path_folder, self.model_type,"Optimizations") 
            if not os.path.exists(self.path_opt_results):
                os.makedirs(self.path_opt_results)
    
    def generate_batch_edgeindex(self):
        edge_indices = []
        for _ in range(self.batchsize):        
            edge_indices.append(self.edge_index)
        return torch.stack(edge_indices, dim=0)
    
    def training(self,  training_name, noise_size=None, shuffle_data=True, plot_loss=True, model_flatten_in = True, save_model=True, load_model=False, optimization=False, optimization_name=None):
        self.dataLoaded_train = DataBatchGenerator(dataset=self.train_data, batch_size=self.batchsize, shuffle=shuffle_data)
        self.dataLoaded_test = DataBatchGenerator(dataset=self.test_data, batch_size=self.batchsize, shuffle=shuffle_data)
        if self.noise_data is not None:
            self.dataLoaded_noise = DataBatchGenerator(dataset=self.noise_data, batch_size=self.batchsize, shuffle=shuffle_data)
        
        if load_model:
            if optimization is not None:
                
                path_opt_result = Path(self.path_opt_results, f"{optimization_name}")
                if not os.path.exists(path_opt_result):
                    os.makedirs(path_opt_result)

                path_save_model = Path(path_opt_result, f"model_weights_trial_{optimization_name}.pth")
            else:
                path_save_model = Path(self.path_folder, self.model_type, "model_save", 'model_weights.pth')
            print("\tLOAD TRAINED MODEL:\t",path_save_model)
            
            self.model.load_state_dict(torch.load(path_save_model, map_location=self.device))
        else:
            print("\tTRAIN TRAINED MODEL:\t")
            self.getModeModel()
            print("--------------------------------------------------------------------------------------------------------------------------",self.model_type)
            
            
            print(f"\ttraining {self.model_type} loss:\t", self.loss_obj.get_lossTerms())
            
            if self.model_type in ["AE", "VAE", "CVAE"]:
                self.time_performance.start_time(f"{training_name}_{self.model_type}_TRAINING_global")
                if self.model_type == "AE":
                    self.training_AE(training_name=training_name, plot_loss=plot_loss, model_flatten_in=model_flatten_in)
                    model_opt_prediction = self.getModel('all')
                elif self.model_type == "VAE":
                    self.training_VAE(training_name=training_name, plot_loss=plot_loss, model_flatten_in=model_flatten_in)
                    model_opt_prediction = self.getModel('decoder')
                elif self.model_type == "CVAE":
                    self.training_CVAE(training_name=training_name, plot_loss=plot_loss, model_flatten_in=model_flatten_in)
                self.time_performance.stop_time(f"{training_name}_{self.model_type}_TRAINING_global")
                self.time_performance.compute_time(f"{training_name}_{self.model_type}_TRAINING_global", fun = "diff")                
                print("\tTIME TRAIN MODEL:\t",self.time_performance.get_time(f"{training_name}_{self.model_type}_TRAINING_global", fun="first"))
                
            elif self.model_type in ["GAN", "WGAN"]:
                self.time_performance.start_time(f"{training_name}_{self.model_type}_TRAINING_global")
                if self.model_type == "GAN":
                    self.training_GAN(training_name=training_name, noise_size=noise_size)
                elif self.model_type == "WGAN":
                    self.training_WGAN_GP(training_name=training_name, noise_size=noise_size)
                self.time_performance.stop_time(f"{training_name}_{self.model_type}_TRAINING_global")
                self.time_performance.compute_time(f"{training_name}_{self.model_type}_TRAINING_global", fun = "diff")  
                print("\tTIME TRAIN MODEL:\t",self.time_performance.get_time(f"{training_name}_{self.model_type}_TRAINING_global", fun="first"))
                #model_opt_prediction = self.getModel('all')
           
            if save_model:
                self.save_model()
            if optimization:
                if self.model_type in ["AE", "VAE"]:
                    path_opt_result = Path(self.path_opt_results, f"{optimization_name}")
                    if not os.path.exists(path_opt_result):
                        os.makedirs(path_opt_result)
                    if self.model_type in ["AE"]:
                        force_noLatent = False
                    elif self.model_type in ["VAE"]:
                        force_noLatent = True
                    else :
                        force_noLatent = True
                    modelPrediction = ModelPrediction(model=model_opt_prediction, device=self.device, dataset=self.noise_data, vc_mapping= self.vc_mapping, time_performance=self.time_performance, univar_count_in=self.latent_dim, univar_count_out=self.univar_count_out,latent_dim=None, data_range=self.rangeData,input_shape=self.input_shape,path_folder=path_opt_result, model_type= self.model_type)         
                    prediction_opt = modelPrediction.compute_prediction(time_key=f"{training_name}_", experiment_name=f"{optimization_name}", remapping_data=True, force_noLatent=force_noLatent)
                    modelPrediction.saveData(experiment_name=f"{optimization_name}", latent=False)
                    
                    opt_function_result = prediction_opt
                    
                else:
                    print("OPTIMIZATION PROCESS NOT IMPLEMENTED FOR:\t",self.model_type)
            else:
                opt_function_result = 1
            return opt_function_result
            
    def save_model(self):
        path_folder_model = Path(self.path_folder, self.model_type, f"model_save_{self.model_type}")
        if not os.path.exists(path_folder_model):
            os.makedirs(path_folder_model)
        
        if self.model_type  == "AE":
            path_save_model = Path(path_folder_model, 'model_weights_AE.pth')
            torch.save(self.model.state_dict(), path_save_model)
            print("\tSAVE TRAINED MODEL AE:\t",path_save_model)
        elif self.model_type=="GAN":
            path_save_model_gen = Path(path_folder_model, f'model_weights_GAN_gen.pth')
            torch.save(self.model_gen.state_dict(), path_save_model_gen)
            print(f"\tSAVE TRAINED MODEL GAN GEN:\t",path_save_model_gen)
            
            path_save_model_dis = Path(path_folder_model, f'model_weights_GAN_dis.pth')            
            torch.save(self.model_dis.state_dict(), path_save_model_dis)
            print(f"\tSAVE TRAINED MODEL GAN DIS:\t",path_save_model_dis)
            
               
    def training_AE(self, model_flatten_in, training_name, plot_loss=True, save_trainingTime=True):
        
        
        self.loss_dict = dict()
        self.model.train()
        epoch_train = list()
        print("\tepoch:\t",0,"/",self.epoch['AE'],"\t -")
        for epoch in range(self.epoch['AE']):
            self.time_performance.start_time(f"{training_name}_AE_TRAINING_epoch")
            self.model.train()
            loss_batch = list()
            loss_batch_test = list()
            loss_batch_partial = dict()
            
            dataBatches_train = self.dataLoaded_train.generate()
            
            for batch_num, dataBatch in enumerate(dataBatches_train):
                item_batch = len(dataBatch)
                
                
                if item_batch == self.batchsize:
                    loss = torch.zeros([1])                    
                    self.optimizer.zero_grad()
                    y_hat_list = list()
                    sample_list = list()
                    for i, item in enumerate(dataBatch):
                        sample = item['sample'].type(torch.float32)
                        sample_list.append(sample)
                        #noise = noisef.type(torch.float32)
                    x_in = torch.Tensor(1, item_batch, self.univar_count_in).to(device=self.device)
                    torch.cat(sample_list, out=x_in) 
                    if model_flatten_in:
                        x_in = x_in.view(-1,self.univar_count_in)
                        x_in.unsqueeze_(1)
                    y_hat = self.model.forward(x=x_in)
                    y_hat_list = list()
                    for i in range(item_batch):
                        item_dict = {"x_input": {"data": y_hat['x_input']["data"][i][0]}, "x_latent":{"latent":y_hat['x_latent']["latent"][i][0]}, "x_output": {"data": y_hat['x_output']["data"][i][0]}}
                        y_hat_list.append(item_dict)
                    
                    loss_dict = self.loss_obj.computate_loss(values_in=y_hat_list, epoch=epoch)
                    
                    
                    
                    
                    loss = loss_dict['loss_total']
                    if batch_num%self.append_count == 0:
                        loss_batch.append(loss.detach().cpu().numpy())
                    
                    for loss_part in loss_dict:
                        if loss_part not in loss_batch_partial:
                            loss_batch_partial[loss_part] = list()
                        if batch_num%self.append_count == 0:
                            loss_part_value = loss_dict[loss_part].detach().cpu().numpy()
                            loss_batch_partial[loss_part].append(loss_part_value)
                    
                    loss.backward()
                    self.optimizer.step()
            
            dataBatches_test = self.dataLoaded_test.generate()
            self.model.eval() 
            
            for batch_num, dataBatch in enumerate(dataBatches_test):
                item_batch = len(dataBatch)
                loss = torch.zeros([1])                    
                y_hat_list = list()
                sample_list = list()
                for i, item in enumerate(dataBatch):                    
                    samplef = item['sample']
                    sample = samplef.type(torch.float32)
                    sample_list.append(sample)
                x_in = torch.Tensor(1, item_batch, self.univar_count_in).to(device=self.device)
                torch.cat(sample_list, out=x_in) 
                
                if model_flatten_in:
                    x_in = x_in.view(-1,self.univar_count_in)
                    x_in.unsqueeze_(1)
                    
                y_hat = self.model.forward(x=x_in)
                y_hat_list = list()
                
                for i in range(item_batch):
                    item_dict = {"x_input": {"data": y_hat['x_input']["data"][i][0]}, "x_latent":{"latent":y_hat['x_latent']["latent"][i][0]}, "x_output": {"data": y_hat['x_output']["data"][i][0]}}
                    y_hat_list.append(item_dict)
                loss_dict = self.loss_obj.computate_loss(values_in=y_hat_list, epoch=epoch)

                loss = loss_dict['loss_total']
                if batch_num%self.append_count == 0:
                    loss_batch_test.append(loss.detach().cpu().numpy())
            
            self.loss_dict[epoch] = {"GLOBAL_loss": np.mean(loss_batch), "values_list": loss_batch, "TEST_loss": np.mean(loss_batch_test)}
            for loss_part in loss_dict:
                self.loss_dict[epoch][loss_part] = np.mean(loss_batch_partial[loss_part])
            
            self.time_performance.stop_time(f"{training_name}_AE_TRAINING_epoch")
            epoch_time = self.time_performance.get_time(f"{training_name}_AE_TRAINING_epoch", fun="last")
            epoch_train.append({"epoch":epoch,"time":epoch_time})
            if self.opt_scheduler_ae == "ReduceLROnPlateau":
                self.scheduler_ae.step(np.mean(loss_batch))
            elif self.opt_scheduler_ae == "StepLR":
                self.scheduler_ae.step() 
            
            self.plot_grad_flow(named_parameters = self.model.named_parameters(), epoch= f"{epoch+1}", model_section="AE")
            print("\tepoch:\t",epoch+1,"/",self.epoch['AE'],"\t - time tr epoch: ", epoch_time ,"\tloss: ",np.mean(loss_batch),"\tloss_test: ",np.mean(loss_batch_test),"\tlr: ",self.optimizer.param_groups[0]['lr'])
        if plot_loss:
            self.loss_plot(self.loss_dict)
        if save_trainingTime:
            self.time_performance.compute_time(f"{training_name}_AE_TRAINING_epoch", fun = "mean")
            self.save_training_time(epoch_train)

    def training_GAN(self, noise_size, training_name, plot_loss=True, save_trainingTime=True):
        
        real_label = 1.
        fake_label = 0.
        epoch_train = list()
        self.loss_dict = dict()
        self.model_gen.train()
        self.model_dis.train()
        
        if self.checkWeightsUpdate:
            layer_gen_notrained = dict()
            for lay in range(len(list(self.model_gen.parameters()))):                
                lay_tens = list(self.model_gen.parameters())[lay].clone().detach().cpu().numpy()
                layer_gen_notrained[f"lay_{lay}"] = lay_tens
            
            layer_dis_notrained = dict()
            for lay in range(len(list(self.model_dis.parameters()))):                
                lay_tens = list(self.model_dis.parameters())[lay].clone().detach().cpu().numpy()
                layer_dis_notrained[f"lay_{lay}"] = lay_tens
                
        print("\tepoch:\t",0,"/",self.epoch['GAN']," - ")
        
        err_D_all = list()
        err_D_r_all = list()
        err_D_f_all = list()
        err_G_all = list()
        
        for epoch in range(self.epoch['GAN']):
            err_D_r_epoch = list()
            err_D_f_epoch = list()
            err_G_epoch = list()
            dataBatches = self.dataLoaded_train.generate()
             
            self.time_performance.start_time(f"{training_name}_GAN_TRAINING_epoch")
            
            for dataBatch in dataBatches:
                item_batch = len(dataBatch) 
                if item_batch == self.batchsize:
                    ###########################
                    # 1  Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    noise = torch.randn(self.batchsize, noise_size[0], noise_size[1]).to(device=self.device) 
                    ###########################
                    ##  A Update D with real data
                    ###########################
                    
                    #self.model_dis.zero_grad()
                    self.optimizer_dis.zero_grad()
                    
                    sample_list = list()
                    for item in dataBatch:
                        sample = item['sample'].type(torch.float32)
                        sample_list.append(sample)
                    
                    x_in = torch.Tensor(1, self.batchsize, self.univar_count_in).to(device=self.device)
                    x_in = torch.cat(sample_list, dim=0)
                    real_labels = torch.full((self.batchsize,), real_label, dtype=torch.float32).to(device=self.device)
                    x_in = x_in.view(1, self.batchsize, self.univar_count_in)                    
                    output = self.model_dis(x_in)['x_output'].view(-1)
                    batch_real_err_D = self.criterion(output, real_labels)
                    #batch_real_err_D.backward()
                    #self.optimizer_dis.step()
                    
                    err_D_r_epoch.append(batch_real_err_D)
                    
                    ###########################
                    ##  B Update D with fake data
                    ###########################
                    fake = self.model_gen(noise)['x_output']
                    fake_labels = torch.full((self.batchsize,), fake_label, dtype=torch.float32).to(device=self.device)
                    output = self.model_dis(fake)['x_output'].view(-1)
                    
                    batch_fake_err_D = self.criterion(output, fake_labels)
                    
                    batch_all_err_D = batch_real_err_D + batch_fake_err_D
                    batch_all_err_D.backward()
                    self.optimizer_dis.step()
                    
                    err_D_f_epoch.append(batch_fake_err_D)
                    
                    ###########################
                    # 2 Update G network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    #self.model_gen.zero_grad()
                    self.optimizer_gen.zero_grad()
                    
                    fake = self.model_gen(noise)['x_output']    
                    label = torch.full((self.batchsize,), real_label, dtype=torch.float32).to(device=self.device)                
                    output = self.model_dis(fake)['x_output'].view(-1)                    
                    
                    batch_err_G = self.criterion(output, label)
                    batch_err_G.backward()
                    self.optimizer_gen.step()
                    err_G_epoch.append(batch_err_G)
            
            self.time_performance.stop_time(f"{training_name}_GAN_TRAINING_epoch")    
            epoch_time = self.time_performance.get_time(f"{training_name}_GAN_TRAINING_epoch", fun="last")
            epoch_train.append({"epoch":epoch,"time":epoch_time})
            
            self.plot_grad_flow(named_parameters = self.model_gen.named_parameters(), epoch= f"{epoch+1}", model_section="GAN_gen")
            self.plot_grad_flow(named_parameters = self.model_dis.named_parameters(), epoch= f"{epoch+1}", model_section="GAN_dis")
            
            err_D_r_epoch = np.mean(self.to_numpy(err_D_r_epoch))
            err_D_f_epoch = np.mean(self.to_numpy(err_D_f_epoch))
            err_D_epoch = (err_D_r_epoch + err_D_f_epoch)/2
            err_G_epoch = np.mean(self.to_numpy(err_G_epoch))
            
            err_D_r_all.append(err_D_r_epoch)
            err_D_f_all.append(err_D_f_epoch)
            err_D_all.append(err_D_epoch)
            err_G_all.append(err_G_epoch)
            
            self.loss_dict[epoch] = {"loss_Dis": err_D_epoch, "loss_Gen": err_G_epoch,"loss_Dis_real": err_D_r_epoch, "loss_Dis_fake": err_D_f_epoch}

            print("\tepoch:\t",epoch,"/",self.epoch['GAN'],"\t")
            print("\t\t\t-LOSS D\tall",err_D_epoch,"\tD(real)",err_D_r_epoch,"\tD(fake)",err_D_f_epoch,"\tG",err_G_epoch)
            print("\t\t\t-LeRt D",self.optimizer_dis.param_groups[0]['lr'],"\tG",self.optimizer_gen.param_groups[0]['lr'])
            
            if self.opt_scheduler_gen == "ReduceLROnPlateau":
                self.scheduler_gen.step(np.mean(err_G_epoch))
            elif self.opt_scheduler_gen == "StepLR":
                self.scheduler_gen.step() 
                
            if self.opt_scheduler_dis == "ReduceLROnPlateau":
                self.scheduler_dis.step(np.mean(err_D_epoch))
            elif self.opt_scheduler_dis == "StepLR":
                self.scheduler_dis.step()
        
        err_D_r_epoch = self.to_numpy(err_D_r_epoch)
        err_D_f_epoch = self.to_numpy(err_D_f_epoch)
        err_G_epoch = self.to_numpy(err_G_epoch)
        for r_batch, f_batch in zip(err_D_r_epoch, err_D_f_epoch):
            mean_val = (np.mean(r_batch) + np.mean(f_batch))/2
            
            err_D_all.append(mean_val)
            
        if self.checkWeightsUpdate:
            layer_gen_trained = dict()
            for lay in range(len(list(self.model_gen.parameters()))):                
                lay_tens = list(self.model_gen.parameters())[lay].clone().detach().cpu().numpy()
                layer_gen_trained[f"lay_{lay}"] = lay_tens
            cprint(f"--------------------------------------------------", "blue", end="\n")
            cprint(f"\tSTART check Weights Update GENERATOR", "blue", end="\n")
            for key in layer_gen_trained:
                n_train = layer_gen_notrained[key]
                y_train = layer_gen_trained[key]
                ten_eq = np.array_equal(n_train,y_train)
                if ten_eq:
                    cprint(f"{key} are equals", "red", end="\n")
                else:
                    cprint(f"{key} are not equals", "green", end="\n")
            
            layer_dis_trained = dict()
            for lay in range(len(list(self.model_dis.parameters()))):                
                lay_tens = list(self.model_dis.parameters())[lay].clone().detach().cpu().numpy()
                layer_dis_trained[f"lay_{lay}"] = lay_tens
            cprint(f"\tSTART check Weights Update DISCRIMINATOR", "blue", end="\n")
            for key in layer_dis_trained:
                n_train = layer_dis_notrained[key]
                y_train = layer_dis_trained[key]
                ten_eq = np.array_equal(n_train,y_train)
                if ten_eq:
                    cprint(f"{key} are equals", "red", end="\n")
                else:
                    cprint(f"{key} are not equals", "green", end="\n")
            cprint(f"--------------------------------------------------", "blue", end="\n")
        
        if plot_loss:
            self.loss_plot(self.loss_dict)
        
        if save_trainingTime:
            self.time_performance.compute_time(f"{training_name}_GAN_TRAINING_epoch", fun = "mean")
            self.save_training_time(epoch_train)    
        
    def training_WGAN_GP(self, noise_size, training_name, plot_loss=True, save_trainingTime=True):
        
        epoch_train = list()
        self.loss_dict = dict()
        self.model_gen.train()
        self.model_dis.train()
        
        if self.checkWeightsUpdate:
            layer_gen_notrained = {f"lay_{lay}": list(self.model_gen.parameters())[lay].clone().detach().cpu().numpy() for lay in range(len(list(self.model_gen.parameters())))}
            layer_dis_notrained = {f"lay_{lay}": list(self.model_dis.parameters())[lay].clone().detach().cpu().numpy() for lay in range(len(list(self.model_dis.parameters())))}
                    
        print("\tepoch:\t", 0, "/", self.epoch['WGAN'], " - ")
        
        err_D_all = list()
        err_D_r_all = list()
        err_D_f_all = list()
        err_G_all = list()
        
        
        
        for epoch in range(self.epoch['WGAN']):
            err_D_r_epoch = list()
            err_D_f_epoch = list()
            err_G_epoch = list()
            dataBatches = self.dataLoaded_train.generate()
            
            self.time_performance.start_time(f"{training_name}_WGAN_TRAINING_epoch")
            
            for dataBatch in dataBatches:
                item_batch = len(dataBatch) 
                if item_batch == self.batchsize:
                    ###########################
                    # 1  Update D network: maximize E[D(x)] - E[D(G(z))] + lambda_wgan_gp * gradient_penalty
                    ###########################
                    noise = torch.randn(1, self.batchsize, noise_size[1]).to(self.device) 
                    
                    ###########################
                    ##  A Update D with real data
                    ###########################
                    self.optimizer_dis.zero_grad()
                    
                    sample_list = [item['sample'].type(torch.float32) for item in dataBatch]
                    x_real = torch.Tensor(1, self.batchsize, self.univar_count_in).to(self.device)
                    x_real = torch.cat(sample_list, dim=0)
                    x_real = x_real.view(1, self.batchsize, self.univar_count_in)                    
                    output_real = self.model_dis(x_real)['x_output'].view(-1)
                    err_D_real = -torch.mean(output_real)
                    err_D_r_epoch.append(err_D_real)
                    
                    ###########################
                    ##  B Update D with fake data
                    ###########################
                    x_fake = self.model_gen(noise)['x_output']
                    output_fake = self.model_dis(x_fake)['x_output'].view(-1)
                    err_D_fake = torch.mean(output_fake)
                    err_D_f_epoch.append(err_D_fake)
                    
                    ###########################
                    ## C Compute gradient penalty
                    ###########################
                    w_gp = self.wasserstein_gradient_penalty(dis=self.model_dis,real_samples=x_real, fake_samples=x_fake)
                    
                    err_D = err_D_real + err_D_fake + self.lambda_wgan_gp * w_gp
                    err_D.backward()
                    self.optimizer_dis.step()
                        
                    ###########################
                    # 2 Update G network: maximize E[D(G(z))]
                    ###########################
                    if epoch % self.n_critic == 0:
                        self.optimizer_gen.zero_grad()
                        
                        noise = torch.randn(1, self.batchsize, noise_size[1]).to(self.device)
                        fake = self.model_gen(noise)['x_output']
                        output = self.model_dis(fake)['x_output'].view(-1)                    
                        
                        err_G = torch.mean(output)#era -
                        err_G.backward()
                        self.optimizer_gen.step()
                        err_G_epoch.append(err_G)
            
            self.time_performance.stop_time(f"{training_name}_WGAN_TRAINING_epoch")    
            epoch_time = self.time_performance.get_time(f"{training_name}_WGAN_TRAINING_epoch", fun="last")
            epoch_train.append({"epoch": epoch, "time": epoch_time})
            
            
                
            self.plot_grad_flow(named_parameters=self.model_gen.named_parameters(), epoch=f"{epoch+1}", model_section="WGAN_gen")
            self.plot_grad_flow(named_parameters=self.model_dis.named_parameters(), epoch=f"{epoch+1}", model_section="WGAN_dis")
            
            err_D_r_epoch = np.mean(self.to_numpy(err_D_r_epoch))
            err_D_f_epoch = np.mean(self.to_numpy(err_D_f_epoch))
            err_D_epoch = np.mean(err_D_r_epoch + err_D_f_epoch) / 2
            
            err_G_epoch = np.mean(self.to_numpy(err_G_epoch)) if err_G_epoch else float('nan')
            
            err_D_r_all.append(err_D_r_epoch)
            err_D_f_all.append(err_D_f_epoch)
            err_D_all.append(err_D_epoch)
            err_G_all.append(err_G_epoch)
            
            self.loss_dict[epoch] = {"loss_Dis": err_D_epoch, "loss_Gen": err_G_epoch, "loss_Dis_real": np.mean(err_D_r_epoch), "loss_Dis_fake": np.mean(err_D_f_epoch)}

            print("\tepoch:\t", epoch, "/", self.epoch['WGAN'], "\t")
            print("\t\t\t-LOSS D\tall", err_D_epoch, "\tD(real)", np.mean(err_D_r_epoch), "\tD(fake)", np.mean(err_D_f_epoch), "\tG", err_G_epoch)
            print("\t\t\t-LeRt D", self.optimizer_dis.param_groups[0]['lr'], "\tG", self.optimizer_gen.param_groups[0]['lr'])
            
            if self.opt_scheduler_gen == "ReduceLROnPlateau":
                self.scheduler_gen.step(np.mean(err_G_epoch))
            elif self.opt_scheduler_gen == "StepLR":
                self.scheduler_gen.step() 
                
            if self.opt_scheduler_dis == "ReduceLROnPlateau":
                self.scheduler_dis.step(np.mean(err_D_epoch))
            elif self.opt_scheduler_dis == "StepLR":
                self.scheduler_dis.step()
        
        err_D_r_epoch = self.to_numpy(err_D_r_epoch)
        err_D_f_epoch = self.to_numpy(err_D_f_epoch)
        err_G_epoch = self.to_numpy(err_G_epoch)
        for r_batch, f_batch in zip(err_D_r_epoch, err_D_f_epoch):
            mean_val = (np.mean(r_batch) + np.mean(f_batch)) / 2
            err_D_all.append(mean_val)
            
        if self.checkWeightsUpdate:
            layer_gen_trained = {f"lay_{lay}": list(self.model_gen.parameters())[lay].clone().detach().cpu().numpy() for lay in range(len(list(self.model_gen.parameters())))}
            cprint(f"--------------------------------------------------", "blue", end="\n")
            cprint(f"\tSTART check Weights Update GENERATOR", "blue", end="\n")
            for key in layer_gen_trained:
                n_train = layer_gen_notrained[key]
                y_train = layer_gen_trained[key]
                ten_eq = np.array_equal(n_train, y_train)
                if ten_eq:
                    cprint(f"{key} are equals", "red", end="\n")
                else:
                    cprint(f"{key} are not equals", "green", end="\n")
            
            layer_dis_trained = {f"lay_{lay}": list(self.model_dis.parameters())[lay].clone().detach().cpu().numpy() for lay in range(len(list(self.model_dis.parameters())))}
            cprint(f"\tSTART check Weights Update DISCRIMINATOR", "blue", end="\n")
            for key in layer_dis_trained:
                n_train = layer_dis_notrained[key]
                y_train = layer_dis_trained[key]
                ten_eq = np.array_equal(n_train, y_train)
                if ten_eq:
                    cprint(f"{key} are equals", "red", end="\n")
                else:
                    cprint(f"{key} are not equals", "green", end="\n")
            cprint(f"--------------------------------------------------", "blue", end="\n")
        
        if plot_loss:
            self.loss_plot(self.loss_dict)
        
        if save_trainingTime:
            self.time_performance.compute_time(f"{training_name}_WGAN_TRAINING_epoch", fun="mean")
            self.save_training_time(epoch_train)
    
    
    def training_VAE(self, model_flatten_in, training_name, plot_loss=True, save_trainingTime=True):
        self.loss_dict = dict()
        self.model.train()
        epoch_train = list()
        print("\tepoch:\t",0,"/",self.epoch['VAE'],"\t -")
        for epoch in range(self.epoch['VAE']):
            self.time_performance.start_time(f"{training_name}_VAE_TRAINING_epoch")
            self.model.train()
            loss_batch = list()
            loss_batch_test = list()
            loss_batch_partial = dict()
            
            dataBatches_train = self.dataLoaded_train.generate()
            
            for batch_num, dataBatch in enumerate(dataBatches_train):
                item_batch = len(dataBatch)
                if item_batch == self.batchsize:
                    loss = torch.zeros([1])                    
                    self.optimizer.zero_grad()
                    y_hat_list_train = list()
                    sample_list = list()
                    for i, item in enumerate(dataBatch):
                        sample = item['sample'].type(torch.float32)
                        

                        sample_list.append(sample)
                        #noise = noisef.type(torch.float32)
                    x_in = torch.Tensor(1, item_batch, self.univar_count_in).to(device=self.device)
                    torch.cat(sample_list, out=x_in) 
                    if model_flatten_in:
                        x_in = x_in.view(-1,self.univar_count_in)
                        x_in.unsqueeze_(1)
                    y_hat = self.model.forward(x=x_in)
                    y_hat_list_train = list()
                    for i in range(item_batch):                        
                        item_dict = {"x_input": {"data": y_hat['x_input']['data'][i][0]}, "x_latent":{"mu":y_hat['x_latent']["mu"][i][0], "logvar":y_hat['x_latent']["logvar"][i][0], "z":y_hat['x_latent']["z"][i][0]},"x_output": {"data": y_hat['x_output']['data'][i][0]}}
                        y_hat_list_train.append(item_dict)
                    
                    loss_dict = self.loss_obj.computate_loss(y_hat_list_train,epoch )
                    
                    
                    loss = loss_dict['loss_total']
                    if batch_num%self.append_count == 0:
                        loss_batch.append(loss.detach().cpu().numpy())
                    
                    for loss_part in loss_dict:
                        if loss_part not in loss_batch_partial:
                            loss_batch_partial[loss_part] = list()
                        if batch_num%self.append_count == 0:
                            loss_part_value = loss_dict[loss_part].detach().cpu().numpy()
                            loss_batch_partial[loss_part].append(loss_part_value)
                    
                    loss.backward()
                    self.optimizer.step()
            
            dataBatches_noise = self.dataLoaded_noise.generate()
            self.model.eval() 
            generator = self.model.get_decoder()
            for batch_num, dataBatch in enumerate(dataBatches_noise):
                item_batch = len(dataBatch)
                loss = torch.zeros([1])                    
                y_hat_list_test = list()
                sample_list = list()
                for i, item in enumerate(dataBatch):                    
                    
                    samplef = item['noise']
                    sample = samplef.type(torch.float32)
                    sample_list.append(sample)
                
                x_in = torch.Tensor(1, item_batch, self.latent_dim).to(device=self.device)
                torch.cat(sample_list, out=x_in) 
                
                if model_flatten_in:
                    x_in = x_in.view(-1,self.latent_dim)
                    x_in.unsqueeze_(1)
                    
                y_hat = generator(x=x_in)
                y_hat_list_test = list()
                
                for i in range(item_batch):
                    item_dict = {"x_input": {"data": y_hat['x_input']['data'][i][0]}, "x_latent":{"mu":y_hat['x_input']['data'][i][0], "logvar":y_hat['x_input']['data'][i][0]}, "x_output": {"data": y_hat['x_output']['data'][i][0]}}
                    y_hat_list_test.append(item_dict)
                #loss_dict = self.loss_obj.computate_loss(y_hat_list_test, epoch)

                #loss = loss_dict['loss_total']
                
                #if batch_num%self.append_count == 0:
                #    loss_batch_test.append(loss.detach().cpu().numpy())
            
            self.loss_dict[epoch] = {"GLOBAL_loss": np.mean(loss_batch), "values_list": loss_batch, "TEST_loss": np.mean(loss_batch_test)}
            for loss_part in loss_dict:
                self.loss_dict[epoch][loss_part] = np.mean(loss_batch_partial[loss_part])
            
            self.time_performance.stop_time(f"{training_name}_VAE_TRAINING_epoch")
            epoch_time = self.time_performance.get_time(f"{training_name}_VAE_TRAINING_epoch", fun="last")
            epoch_train.append({"epoch":epoch,"time":epoch_time})
            if self.opt_scheduler_ae == "ReduceLROnPlateau":
                self.scheduler_ae.step(np.mean(loss_batch))
            elif self.opt_scheduler_ae == "StepLR":
                self.scheduler_ae.step() 
            
            self.plot_grad_flow(named_parameters = self.model.named_parameters(), epoch= f"{epoch+1}", model_section="AE")
            #cprint("\tepoch:\t",epoch+1,"/",self.epoch['VAE'],"\t - time tr epoch: ", epoch_time ,"\tloss: ",np.mean(loss_batch),"\tloss_test: ",np.mean(loss_batch_test),"\tlr: ",self.optimizer.param_groups[0]['lr'], "cyan"))
            cprint(f"\tepoch:\t{epoch+1}/{self.epoch['VAE']}\t - time tr epoch: {epoch_time} \tloss: {np.mean(loss_batch)} \tloss_test: {np.mean(loss_batch_test)} \tlr: {self.optimizer.param_groups[0]['lr']}", "black")
            #cprint(f"\t\t\t{self.loss_dict[epoch]}","cyan")

            if epoch+1 == self.epoch['VAE']:
                self.save_intermediate_results(y_hat_list_train,epoch)
            
        if plot_loss:
            self.loss_plot(self.loss_dict)
        if save_trainingTime:
            self.time_performance.compute_time(f"{training_name}_VAE_TRAINING_epoch", fun = "mean")
            self.save_training_time(epoch_train)

    def training_CVAE(self, model_flatten_in, training_name, plot_loss=True, save_trainingTime=True):
        self.loss_dict = dict()
        self.model.train()
        epoch_train = list()
        print("\tepoch:\t",0,"/",self.epoch['CVAE'],"\t -")
        for epoch in range(self.epoch['CVAE']):
            self.time_performance.start_time(f"{training_name}_CVAE_TRAINING_epoch")
            self.model.train()
            loss_batch = list()
            loss_batch_test = list()
            loss_batch_partial = dict()
            
            dataBatches_train = self.dataLoaded_train.generate()
            
            for batch_num, dataBatch in enumerate(dataBatches_train):
                item_batch = len(dataBatch)
                if item_batch == self.batchsize:
                    loss = torch.zeros([1])                    
                    self.optimizer.zero_grad()
                    y_hat_list_train = list()
                    sample_list = list()
                    sample_timeweather_list = list()
                    for i, item in enumerate(dataBatch):
                        sample = item['sample'].type(torch.float32)
                        sample_timeweather = item['sample_timeweather'].type(torch.float32)
                        sample_list.append(sample)
                        sample_timeweather_list.append(sample_timeweather)
                        
                        
                    x_in = torch.Tensor(1, item_batch, self.univar_count_in).to(device=self.device)
                    x_in_timeweather = torch.Tensor(1, item_batch, self.timeweather_count).to(device=self.device)
                    torch.cat(sample_list, out=x_in) 
                    torch.cat(sample_timeweather_list, out=x_in_timeweather) 
                    if model_flatten_in:
                        x_in = x_in.view(-1,self.univar_count_in)
                        x_in.unsqueeze_(1)
                        x_in_timeweather = x_in_timeweather.view(-1, self.timeweather_count)
                        x_in_timeweather.unsqueeze_(1)
                        
                    y_hat = self.model.forward(x=x_in, condition= x_in_timeweather)
                    print("y_hat", y_hat)
                    raise Exception("-")
                    y_hat_list_train = list()
                    for i in range(item_batch):
                        item_dict = {"x_input": {"data": y_hat['x_input']['data'][i][0], "timeweather":y_hat['x_input']['timeweather'][i][0]}, "x_latent":{"mu":y_hat['x_latent']["mu"][i][0], "logvar":y_hat['x_latent']["logvar"][i][0]}, "x_output":{"data":y_hat['x_output'][i][0]}}
                        y_hat_list_train.append(item_dict)
                    
                    loss_dict = self.loss_obj.computate_loss(y_hat_list_train,epoch )
                    
                    
                    loss = loss_dict['loss_total']
                    if batch_num%self.append_count == 0:
                        loss_batch.append(loss.detach().cpu().numpy())
                    
                    for loss_part in loss_dict:
                        if loss_part not in loss_batch_partial:
                            loss_batch_partial[loss_part] = list()
                        if batch_num%self.append_count == 0:
                            loss_part_value = loss_dict[loss_part].detach().cpu().numpy()
                            loss_batch_partial[loss_part].append(loss_part_value)
                    
                    loss.backward()
                    self.optimizer.step()
            
            dataBatches_test = self.dataLoaded_test.generate()
            self.model.eval() 
            
            for batch_num, dataBatch in enumerate(dataBatches_test):
                item_batch = len(dataBatch)
                loss = torch.zeros([1])                    
                y_hat_list_test = list()
                sample_list = list()
                sample_timeweather_list = list()
                
                for i, item in enumerate(dataBatch):                    
                    sample = item['sample'].type(torch.float32)
                    sample_timeweather = item['sample_timeweather'].type(torch.float32)
                    sample_list.append(sample)
                    sample_timeweather_list.append(sample_timeweather)
                x_in = torch.Tensor(1, item_batch, self.univar_count_in).to(device=self.device)
                x_in_timeweather = torch.Tensor(1, item_batch, self.timeweather_count).to(device=self.device)
                torch.cat(sample_list, out=x_in) 
                torch.cat(sample_timeweather_list, out=x_in_timeweather)
                
                if model_flatten_in:
                    x_in = x_in.view(-1,self.univar_count_in)
                    x_in.unsqueeze_(1)
                    x_in_timeweather = x_in_timeweather.view(-1, self.timeweather_count)
                    x_in_timeweather.unsqueeze_(1)
                    
                y_hat = self.model.forward(x=x_in, x_in_timeweather=x_in_timeweather)
                y_hat_list_test = list()
                
                for i in range(item_batch):
                    item_dict = {"x_input": {"data": y_hat['x_input']['data'][i][0], "timeweather":y_hat['x_input']['timeweather'][i][0]}, "x_latent":{"mu":y_hat['x_latent']["mu"][i][0], "logvar":y_hat['x_latent']["logvar"][i][0]}, "x_output":{"data":y_hat['x_output'][i][0]}}
                    
                    y_hat_list_test.append(item_dict)
                loss_dict = self.loss_obj.computate_loss(y_hat_list_test, epoch)

                loss = loss_dict['loss_total']
                if batch_num%self.append_count == 0:
                    loss_batch_test.append(loss.detach().cpu().numpy())
            
            self.loss_dict[epoch] = {"GLOBAL_loss": np.mean(loss_batch), "values_list": loss_batch, "TEST_loss": np.mean(loss_batch_test)}
            for loss_part in loss_dict:
                self.loss_dict[epoch][loss_part] = np.mean(loss_batch_partial[loss_part])
            
            self.time_performance.stop_time(f"{training_name}_CVAE_TRAINING_epoch")
            epoch_time = self.time_performance.get_time(f"{training_name}_CVAE_TRAINING_epoch", fun="last")
            epoch_train.append({"epoch":epoch,"time":epoch_time})
            if self.opt_scheduler_ae == "ReduceLROnPlateau":
                self.scheduler_ae.step(np.mean(loss_batch))
            elif self.opt_scheduler_ae == "StepLR":
                self.scheduler_ae.step() 
            
            
            
            self.plot_grad_flow(named_parameters = self.model.named_parameters(), epoch= f"{epoch+1}", model_section="AE")
            print("\tepoch:\t",epoch+1,"/",self.epoch['CVAE'],"\t - time tr epoch: ", epoch_time ,"\tloss: ",np.mean(loss_batch),"\tloss_test: ",np.mean(loss_batch_test),"\tlr: ",self.optimizer.param_groups[0]['lr'])

            if epoch+1 == self.epoch['CVAE']:
                self.save_intermediate_results(y_hat_list_train,epoch)
            
        if plot_loss:
            self.loss_plot(self.loss_dict)
        if save_trainingTime:
            self.time_performance.compute_time(f"{training_name}_CVAE_TRAINING_epoch", fun = "mean")
            self.save_training_time(epoch_train)






    # Weight initialization
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def to_numpy(self, tensor_list):
        
        # Check if the list is nested
        if isinstance(tensor_list, list):
            if isinstance(tensor_list[0], list):
                return [self.to_numpy(sublist) for sublist in tensor_list]
            else:
                return [sublist.detach().cpu().numpy() for sublist in tensor_list]
        elif isinstance(tensor_list, Tensor):
            return [tensor_list.detach().cpu().numpy()]
        else:
            return [tensor_list]
            
    def get_tensor_info(tensor):
        info = []
        for name in ['requires_grad', 'is_leaf', 'retains_grad', 'grad_fn', 'grad']:
            info.append(f'{name}({getattr(tensor, name, None)})')
        return ' '.join(info)

    def getBack(self, var_grad_fn):
        for n in var_grad_fn.next_functions:
            if n[0]:
                try:
                    tensor = getattr(n[0], 'variable')
                    print('Tensor with grad found:', tensor)
                    print(' - gradient :', tensor.grad)
                    print()
                except AttributeError as e:
                    self.getBack(n[0])

    def getModel(self, selection='all', eval=False, train=False, extra_info=False):
        if self.model_type in ["AE","VAE"]:
            if selection=='all':
                model_selected = self.model
            if selection=='encoder':
                raise Exception("Encoder not implemented")
            if selection=='decoder':
                if extra_info:
                    model_selected, model_size, model_permutation_forward = self.model.get_decoder(extra_info=extra_info)
                else:
                    model_selected = self.model.get_decoder(extra_info=extra_info)
                if train:
                    model_selected = model_selected.train()
                elif eval:
                    model_selected = model_selected.eval()

        elif self.model_type in ["GAN","WGAN"]:
            if selection=='gen':
                if train:
                    model_selected = self.model_gen.train()
                elif eval:
                    model_selected = self.model_gen.eval()
            elif selection=='dis':
                model_selected = self.model_dis
        if extra_info:
            return model_selected, model_size, model_permutation_forward
        else:
            return model_selected

    def eval(self):
        if self.model_type == "AE":
            self.model.eval()
        elif self.model_type in ["GAN","WGAN"]:
            self.model_gen.eval() 
            self.model_dis.eval() 
            # = self.model_gen

    def trainMode(self):
        if self.model_type == "AE":
            self.model.train()
        elif self.model_type in ["GAN","WGAN"]:
            self.model_gen.train() 
            self.model_dis.train()

    def getModeModel(self):
        if self.model_type == "AE":
            print("AE mode training:\t", self.model.training)
        elif self.model_type in ["GAN","WGAN"]:
            print("GAN gen mode training:\t", self.model_gen.training)
            print("GAN dis mode training:\t", self.model_dis.training)

    def wasserstein_gradient_penalty(self, dis, real_samples, fake_samples):
        
        #  Random weight term for interpolation between real and fake samples
        ##alpha  = torch.rand(batch_size, 1).to(device=self.device) 
        ##alpha = alpha.expand_as(xr)
        # interpolation between real and fake samples
        # #interpolates = alpha  * xr + (1 - alpha) * xf
        # set it to require grad info
        
        alpha = torch.rand((real_samples.size(0), 1, 1)).to(self.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

        interpolates.requires_grad_(True)
        d_interpolates = dis(interpolates)['x_output'].view(-1)
        ##real_grad_out = Variable(Tensor(xr.shape[0], 1).fill_(1.0).to(device=self.device), requires_grad=False)
        
        real_grad = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size()).to(self.device),#real_grad_out,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        real_grad = real_grad.view(real_grad.size(0), -1)    
        gradient_penalty = torch.pow(real_grad.norm(2, dim=1) - 1, 2).mean()
        return gradient_penalty

    def save_intermediate_results(self, data, epoch):
        path_fold_intermediate_results = Path(self.path_folder, self.model_type, "intermediate_results")
        if not os.path.exists(path_fold_intermediate_results):
            os.makedirs(path_fold_intermediate_results)
        filepath = Path(path_fold_intermediate_results, f'intermediate_epoch_{epoch}.csv')
        rows = []
        for item in data:
            if self.model_type == "VAE":
                rows.append({
                    'x_input': item['x_input'],
                    'x_latent_mu': item['x_latent']['mu'],
                    'x_latent_logvar': item['x_latent']['logvar'],
                    'x_output': item['x_output']
                })
            else:
                rows.append({
                    'x_input': item['x_input'],
                    'mx_latentu': item['x_latent']['latent'],
                    'x_output': item['x_output']
                })

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    
    def loss_plot(self, loss_dict):
        path_fold_lossplot = Path(self.path_folder, self.model_type, "loss_plot")
        if not os.path.exists(path_fold_lossplot):
            os.makedirs(path_fold_lossplot)
            
        if "TEST_loss" in loss_dict[0]:
            color = cm.rainbow(np.linspace(0, 1, len(loss_dict[0])-1))
        else:
            color = cm.rainbow(np.linspace(0, 1, len(loss_dict[0])))

        loss_plot_dict = dict()
        i = 0
        for key in loss_dict[0]:
            if key != "values_list" and key!='loss_total':
                loss_list = list()
                for mean_val in loss_dict:
                    loss_list.append(loss_dict[mean_val][key])
                loss_plot_dict[key] = loss_list
                plt.figure(figsize=(12,8))  
                if key!='TEST_loss':
                    cls_loss = color[i]
                    i += 1
                else:
                    cls_loss = [0, 0, 0, 1]
                    
                plt.plot(loss_list, color=cls_loss, marker='o', mfc=cls_loss, label=str(key))
                plt.legend(loc="upper right")
                filename = Path(path_fold_lossplot, "loss_training_"+str(key)+".png")
                plt.savefig(filename)
                
        i = 0
        plt.figure(figsize=(12,8))
        for key in loss_plot_dict:  
            if key != "allLosses":
                if key!='TEST_loss':
                    cls_loss = color[i]
                    i += 1
                else:
                    cls_loss = [0, 0, 0, 1]
                plt.plot(loss_plot_dict[key], color=cls_loss, marker='o', mfc=cls_loss, label=str(key))
                
                
        plt.legend(loc="upper right")
        filename = Path(path_fold_lossplot,"loss_training_allLosses.png")
        plt.savefig(filename)
        df = pd.DataFrame(loss_plot_dict).T.reset_index()

    def save_training_time(self, list_training_time):
        path_fold_lossplot = Path(self.path_folder, self.model_type, "loss_plot")
        if not os.path.exists(path_fold_lossplot):
            os.makedirs(path_fold_lossplot)
        #dftrainingTime = pd.DataFrame(columns=['epoch',"time"])
        training_time_records = []
        for item in list_training_time:
            #dftrainingTime = dftrainingTime.append({'epoch': item["epoch"], 'time': item["time"] }, ignore_index=True)
            training_time_records.append({'epoch': item["epoch"], 'time': item["time"] })
        time_file = Path(path_fold_lossplot ,"epochs_time.csv")
        dftrainingTime = pd.DataFrame(training_time_records)
        dftrainingTime.to_csv(time_file, sep='\t')  

    def plot_grad_flow(self, named_parameters, epoch, model_section):
        
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        
        gradients_list = []
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                if p.grad is not None:
                    layers.append(n)
                    ave_grads.append(p.grad.abs().mean().cpu())
                    max_grads.append(p.grad.abs().max().cpu())
                    gradients_list.append(p.grad.norm().item())
        
        plt.figure(figsize=(12,15))
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4)], 
            ['max-gradient', 'mean-gradient', 'zero-gradient'])
        
        epoch_str = f'{epoch}'.zfill(len(str(self.epoch[self.model_type])))
        path_save_gradexpl = Path(self.path_save_model_gradients,f"{model_section}_gradient_epoch_{epoch_str}.png")
        plt.savefig(path_save_gradexpl)

        path_save_gradients_list = Path(self.path_save_model_gradients,f"{model_section}_gradientlist_epoch_{epoch_str}.csv")
        df = pd.DataFrame(gradients_list)
        df.to_csv(path_save_gradients_list, index=False)


