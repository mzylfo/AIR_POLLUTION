from src.NeuroCorrelation.Analysis.DataComparison import DataComparison

import numpy as np
import torch
# Visualization libraries
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_dense_adj, to_networkx
from sklearn.datasets import make_spd_matrix
from pathlib import Path
import json
import os
import statistics
from copulas.multivariate import GaussianMultivariate
import matplotlib.pyplot as plt

from scipy import stats as stats
import seaborn as sns
import pandas as pd


class DataSyntheticGeneration():

    def __init__(self, torch_device, univar_count, lat_dim, time_performance, path_folder):
        self.torch_device = torch_device
        self.univar_count = univar_count
        self.lat_dim = lat_dim
        self.path_folder = path_folder
        
        self.min_val = None
        self.max_val = None
        self.mean_vc_val = dict()
        self.median_vc_val = dict()
        self.variance_vc_val = dict()
        self.time_performance = time_performance

    def get_muR(self):
        if self.with_cov:
            return {"mu":self.mu, "r_psd": self.r_psd}
        else:
            return {"mu":None, "r_psd": None}


    def graphGen(self, num_of_samples = 10000, with_cov =True):
        self.with_cov = with_cov
        self.graph = KarateClub()
        data = self.graph[0]
        g_adj = to_dense_adj(data.edge_index)[0].numpy()#.astype(int)
        g_adj = torch.Tensor([[g_adj.tolist()]]).numpy()#.astype(int)
        self.G = to_networkx(data, to_undirected=True)
        self.pos_nodes = nx.spring_layout(self.G, seed=0)
        if with_cov:
            return self.casualVA_cov(num_of_samples)
        else:
            return self.casualGraph(num_of_samples)
        
    def casualGraph(self, num_of_samples = 10000):        
        self.univar_count = 78
        adj_distance = [ [ [ list() ] for i in range(34)] for j in range(34)]
        
        for (node_a, node_b) in self.G.edges:
            node_a_x = self.pos_nodes[node_a][0]
            node_a_y = self.pos_nodes[node_a][1]

            node_b_x = self.pos_nodes[node_b][0]
            node_b_y = self.pos_nodes[node_b][1]

            edge_len_x = abs(node_a_x - node_b_x)
            edge_len_y = abs(node_a_y - node_b_y)
            
            edge_len = ((edge_len_x**2) + (edge_len_y**2))**(1/2)

            edge_len_dev = edge_len * 0.2
            edge_len_random = np.random.normal(loc=edge_len, scale=edge_len_dev, size = num_of_samples)

            adj_distance[node_a][node_b] = [edge_len_random, edge_len, edge_len_dev]
        

        self.sample_synthetic = []
        self.sample_synthetic_list = []
        for rows in adj_distance:
            for dist_rand in rows:
                if len(dist_rand[0]) !=0:
                    self.sample_synthetic.append(dist_rand)
                    self.sample_synthetic_list.append(dist_rand[0].tolist())

        samples_origin_path = Path(self.path_folder, "samples_origin.txt")
        with open(samples_origin_path, 'w') as fp:
            json.dump(self.sample_synthetic_list, fp, sort_keys=True, indent=4)

        self.r_psd = np.corrcoef(self.sample_synthetic_list)
        r_psd_Path = Path(self.path_folder, "corrCoeffMatrix_original.csv")
        np.savetxt(r_psd_Path, self.r_psd, delimiter=",")

        dataset_couple = []
        for i in range(num_of_samples):
            dataset_couple.append({"sample":self.getSample(i), "noise":self.getRandom(dim=self.univar_count)})
        return dataset_couple

    def casualGraph_cov(self, num_of_samples = 50000, plot_varDist=True):

        self.univar_count = 78
        adj_distance = [ [ [] for i in range(34)] for j in range(34)]
        for (node_a, node_b) in self.G.edges:
            node_a_x = self.pos_nodes[node_a][0]
            node_a_y = self.pos_nodes[node_a][1]

            node_b_x = self.pos_nodes[node_b][0]
            node_b_y = self.pos_nodes[node_b][1]

            edge_len_x = abs(node_a_x - node_b_x)
            edge_len_y = abs(node_a_y - node_b_y)
            
            edge_len = ((edge_len_x**2) + (edge_len_y**2))**(1/2)

            edge_len_dev = edge_len * 0.1
            #edge_len_random = np.random.normal(loc=edge_len, scale=edge_len_dev, size = num_of_samples)

            adj_distance[node_a][node_b] = {"mu":edge_len, "var":edge_len_dev}
        
        adj_distance_list = []
        for rows in adj_distance:
            for dist_rand in rows:
                if len(dist_rand) !=0:
                    adj_distance_list.append(dist_rand)
        self.mu = [15.6/100, 15.2/100, 17/100]#list()
        #for univar in adj_distance_list:
        #    self.mu.append(univar['mu'])
        
        self.r_psd = [[1.0,-0.3,0.2], [-0.3,1.0,0.7],[0.2,0.7,1.0]]
        #self.r_psd = make_spd_matrix(n_dim=len(self.mu), random_state=0)
        r_psd_Path = Path(self.path_folder, "corrCoeffMatrix_original.csv")
        np.savetxt(r_psd_Path, self.r_psd, delimiter=",")

        mi_Path = Path(self.path_folder, "MEAN_original.csv")
        np.savetxt(mi_Path, self.mu, delimiter=",")

        rng = np.random.default_rng()
        self.sample_synthetic =  [ [[]]  for i in range(self.univar_count)]
        


        x_synt = rng.multivariate_normal(mean=self.mu, cov=self.r_psd, size=num_of_samples)
        syn_data_toNorm = (x_synt-np.min(x_synt))/(np.max(x_synt)-np.min(x_synt))
        for x in syn_data_toNorm:
            
            for i,val in enumerate(x):
                
                self.sample_synthetic[i][0].append(val)
        if plot_varDist:
            self.plot_dataDist()
        
        samples_origin_path = Path(self.path_folder, "samples_origin.txt")
        with open(samples_origin_path, 'w') as fp:
            json.dump(self.sample_synthetic, fp, sort_keys=True, indent=4)

        dataset_couple = []
        for i in range(num_of_samples):

            dataset_couple.append({"sample":self.getSample(i), "noise":self.getRandom(dim=self.univar_count)})
        return dataset_couple
    
    def casualVC_init_3VC(self, num_of_samples=200, draw_correlationCoeff=True, gaussian_correlated=True):
                
        if gaussian_correlated:
            X = np.random.randn(num_of_samples)
            Y = 1.6* X + np.random.randn(num_of_samples)
            Z = 3* X + np.random.randn(num_of_samples)
            
        else:
            X = np.random.randn(num_of_samples)
            Z = np.random.randn(num_of_samples)
            Y = np.ones(num_of_samples)
            W = np.random.randn(num_of_samples)
            
            condition = (((X<0) & (W>=0)) | ((X>0) & (W<=0)))
            Y[condition] = -W[condition]
            Y[~condition] = W[~condition]

        min_series = min(X.min(), Y.min(), Z.min())
        max_series = max(X.max(), Y.max(), Z.max())
        X = (X- min_series)/(max_series - min_series)
        Y = (Y- min_series)/(max_series - min_series)
        Z = (Z- min_series)/(max_series - min_series)

        path_fold_syntAnalysis = Path(self.path_folder,"synthetic_data_analysis")
        if not os.path.exists(path_fold_syntAnalysis):
            os.makedirs(path_fold_syntAnalysis)

        

       
        mu_x, std_x = np.mean(X), np.std(X)
        mu_y, std_y = np.mean(Y), np.std(Y) 
        mu_z, std_z = np.mean(Z), np.std(Z)
        
        self.mu = [mu_x, mu_y, mu_z]

        dist_x = stats.norm(loc=mu_x, scale=std_x)
        dist_y = stats.norm(loc=mu_y, scale=std_y)
        dist_z = stats.norm(loc=mu_z, scale=std_z)
        rho = np.corrcoef([ X, Y, Z])

        self.real_data_vc = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
        
        path_fold_syntAnalysis = Path(self.path_folder,"synthetic_data_analysis")
        if not os.path.exists(path_fold_syntAnalysis):
            os.makedirs(path_fold_syntAnalysis)      
            
        self.comparison_plot_syntetic = DataComparison(univar_count_in=self.univar_count, univar_count_out=self.univar_count, dim_latent=None, path_folder= path_fold_syntAnalysis)
            
        corrMatrix = self.comparison_plot_syntetic.correlationCoeff(self.real_data_vc)
        if draw_correlationCoeff:
            self.comparison_plot_syntetic.plot_vc_analysis(self.real_data_vc, plot_name="syntetic", color_data="olive")
            self.comparison_plot_syntetic.plot_vc_correlationCoeff(self.real_data_vc, "syntetic", corrMatrix=corrMatrix)
        return corrMatrix

    def getDataRange(self):
        rangeData = {"max_val": 0, "min_val":1}
        return rangeData

    def casualVC_init_multi(self, vc_dict, num_of_samples=200, draw_correlationCoeff=True):
        seed_values_vc = dict()
        self.vc_mapping = list()
        self.min_val = None
        self.max_val = None
        self.univar_count = len(vc_dict)

        for key_vc in vc_dict:
            self.vc_mapping.append(key_vc)
            seed_values_vc[key_vc] = dict()
            vc_info = vc_dict[key_vc]
            vc_values = np.random.randn(num_of_samples)
            if "dependence" in vc_info and vc_info["dependence"] is not None and len(vc_info["dependence"])!=0:
                for key_dep in vc_info["dependence"]:
                    if key_dep in seed_values_vc:
                        dep_coeff = vc_info["dependence"][key_dep]

                        vc_values += dep_coeff*seed_values_vc[key_dep]['values']
            
            seed_values_vc[key_vc]['values'] = vc_values

            if self.min_val is None:
                self.min_val = min(vc_values)
            else:
                min_vc = min(vc_values)
                self.min_val = min(min_vc, self.min_val)
            
            if self.max_val is None:
                self.max_val = max(vc_values)
            else:
                max_vc = max(vc_values)
                self.max_val = max(max_vc, self.max_val)
                
            self.mean_vc_val[key_vc] = statistics.mean(vc_values)
            self.median_vc_val[key_vc] = statistics.median(vc_values)
            self.variance_vc_val[key_vc] = statistics.variance(vc_values)
            
            
        self.mu = list()
        rho_val_list = list()

        for key_vc in seed_values_vc:
            seed_values_vc[key_vc]['values'] = (seed_values_vc[key_vc]['values'] - self.min_val)/(self.max_val - self.min_val)
            seed_values_vc[key_vc]['mean'] = np.mean(seed_values_vc[key_vc]['values'])
            seed_values_vc[key_vc]['std'] = np.std(seed_values_vc[key_vc]['values'])
            self.mu = seed_values_vc[key_vc]['mean']
            rho_val_list.append(seed_values_vc[key_vc]['values'])
        
        rho = np.corrcoef(rho_val_list)
        corrcoef_Path = Path(self.path_folder, "corrcoef_datasynt.csv")
        np.savetxt(corrcoef_Path, rho, delimiter=",")
        
        path_fold_syntAnalysis = Path(self.path_folder,"synthetic_data_analysis")
        if not os.path.exists(path_fold_syntAnalysis):
            os.makedirs(path_fold_syntAnalysis)
    
        

        self.real_data_vc = pd.DataFrame()
        for key_vc in seed_values_vc:
            self.real_data_vc[key_vc] = seed_values_vc[key_vc]['values']
            
        path_fold_syntAnalysis = Path(self.path_folder,"synthetic_data_analysis")
        if not os.path.exists(path_fold_syntAnalysis):
            os.makedirs(path_fold_syntAnalysis)         
        self.comparison_plot_syntetic = DataComparison(univar_count_in=self.univar_count, univar_count_out=self.univar_count, dim_latent=None, path_folder= path_fold_syntAnalysis)
        corrMatrix = self.comparison_plot_syntetic.correlationCoeff(self.real_data_vc)
        if draw_correlationCoeff:
               
            
            self.comparison_plot_syntetic.plot_vc_analysis(self.real_data_vc,plot_name="syntetic", color_data="olive")
            self.comparison_plot_syntetic.plot_vc_correlationCoeff(self.real_data_vc, "syntetic", corrMatrix=corrMatrix)
        return corrMatrix

    def data2Copula(self, data_in):
        df_data = None
        for i, istance in enumerate(data_in):
            tensor_list = list()
            for var in istance['sample']:
                if df_data is None:
                    col = [i for i in range(len(istance['sample']))]
                    df_data = pd.DataFrame(columns=col)
                tensor_list.append(var.numpy().tolist())
            df_data.loc[i] = tensor_list        
        return df_data

    def casualVC_generation(self, real_data=None, toPandas=True, univar_count=None, name_data="train", num_of_samples = 50000, draw_correlationCoeff=True, instaces_size=1):
        time_key_fit = "_copula_fit"
        time_key_gen = "_copula_gen"
        
        path_fold_copulagenAnalysis = Path(self.path_folder,name_data+"_copulagen_data_analysis")
        
        if not os.path.exists(path_fold_copulagenAnalysis):
            os.makedirs(path_fold_copulagenAnalysis)
        
        if real_data is None:
            real_data = self.real_data_vc
            vc_mapping = self.vc_mapping
        else:
            if toPandas:
                real_data = self.data2Copula(real_data)
            else:
                real_data = real_data
            vc_mapping = list(real_data.columns.values)
        if univar_count == None:
            univar_count = self.univar_count
         
        copula = GaussianMultivariate()
        print("\t"+name_data+"\tcopula.fit : start")
        self.time_performance.start_time(time_key_fit)
        copula.fit(real_data)
        self.time_performance.stop_time(time_key_fit)
        self.time_performance.compute_time(time_key_fit, fun = "sum") 
        print("\t"+name_data+"\tcopula.fit : end")

        print("\t"+name_data+"\tcopula.sample : start")
        sample_to_generate = num_of_samples * instaces_size
        self.time_performance.start_time(time_key_gen)
        synthetic_data = copula.sample(sample_to_generate)
        self.time_performance.stop_time(time_key_gen)
        self.time_performance.compute_time(time_key_gen, fun = "sum") 
        print("\t"+name_data+"\tcopula.sample : end")
        
        print(f"\tTIME to copula fit:\t",self.time_performance.get_time(time_key_fit, fun = "mean"))
        print(f"\tTIME to copula gen:\t",self.time_performance.get_time(time_key_gen, fun = "mean"))

        
        self.sample_synthetic =  [ [[]]  for i in range(univar_count)]
        for i, row in synthetic_data.iterrows():
            for id_univar in range(univar_count):
                name_var = vc_mapping[id_univar]
                value_vc = row[name_var]
                self.sample_synthetic[id_univar][0].append(value_vc)
        samples_origin_path = Path(self.path_folder, "samples_"+name_data+".txt")
        with open(samples_origin_path, 'w') as fp:
            json.dump(self.sample_synthetic, fp, sort_keys=True, indent=4)
        dataset_couple = []
        for i in range(num_of_samples):
            sample = []
            for j in range(instaces_size):
                sample.append(self.getSample(i))
            
            matrix_sample = torch.Tensor(instaces_size, univar_count).to(self.torch_device)
            torch.cat(sample, out=matrix_sample)
            matrix_sample = matrix_sample.view(1, 1, univar_count, instaces_size)
            dataset_couple.append({"sample":matrix_sample, "noise":self.getRandom(dim=univar_count)})
        self.comparison_plot = DataComparison(univar_count_in=univar_count, univar_count_out=univar_count, dim_latent=None, path_folder= path_fold_copulagenAnalysis)
        noise_data_vc = dict()
        for id_var in range(univar_count):
            noise_data_vc[id_var] = list()
        for item in dataset_couple:
            for id_var in range(univar_count):
                for j in range(instaces_size):
                    noise_data_vc[id_var].append(item['sample'][0][0][id_var][j].detach().cpu().numpy().tolist())
                    
        if draw_correlationCoeff:
            self.comparison_plot.plot_vc_analysis(noise_data_vc,plot_name=name_data, color_data="green")
        df_data = pd.DataFrame(noise_data_vc)
        if draw_correlationCoeff:
            rho = self.comparison_plot.correlationCoeff(df_data)
        else:
            rho = None
        return dataset_couple, rho


    def getSample(self, key_sample):
        sample = []
        for ed in self.sample_synthetic:    
            sample.append(ed[0][key_sample])
        sample_torch = torch.from_numpy(np.array([float(x) for x in sample])).type(torch.float32).to(self.torch_device)
        return sample_torch
  
    
    def getDataStats(self):
        statsData = {"mean_val": self.mean_vc_val, "median_val":self.median_vc_val, "variance_val":self.variance_vc_val}
        return statsData
    
    def getRandom(self, dim):
        
        randomNoise =  torch.randn(1, dim).type(torch.float32).to(self.torch_device)
        #torch.randn(1, dim).type(torch.float32).to(self.torch_device)
        #torch.randn(1, dim).uniform_(0,1).type(torch.float32).to(self.torch_device)
        return randomNoise.type(torch.float32)
    
    def get_synthetic_noise_data(self, name_data, num_of_samples = 5000,  draw_plots=True,draw_correlationCoeff=False, instaces_size=1):
        path_fold_noiseAnalysis = Path(self.path_folder,name_data+"_data_analysis")
        if not os.path.exists(path_fold_noiseAnalysis):
            os.makedirs(path_fold_noiseAnalysis)

        random_values = [self.getRandom(dim=num_of_samples*instaces_size) for i in range(self.lat_dim)] 
        dataset_couple = []
        for s_id in range(num_of_samples):
            sample = []
            for j in range(instaces_size):
                random_sampled = []
                for lat_id in range(self.lat_dim):
                    random_sampled.append(random_values[lat_id][0][s_id])
                sample.append(torch.stack(random_sampled))
            matrix_sample = torch.Tensor(instaces_size, lat_id).to(self.torch_device)
            torch.cat(sample, out=matrix_sample)       
            matrix_sample = matrix_sample.view(1, 1, self.lat_dim,instaces_size)
            
            
            dataset_couple.append({"sample":matrix_sample, "noise":matrix_sample})

        if draw_plots:
            noise_data_vc = dict()
            for id_var in range(self.lat_dim):
                noise_data_vc[id_var] = list()
            for item in dataset_couple:
                for id_var in range(self.lat_dim):
                    for j in range(instaces_size):
                        noise_data_vc[id_var].append(item['sample'][0][0][id_var].tolist())
                    
            self.comparison_plot_noise = DataComparison(univar_count_in=self.lat_dim, univar_count_out=self.lat_dim, dim_latent=self.lat_dim, path_folder= path_fold_noiseAnalysis)
            if draw_correlationCoeff:
                self.comparison_plot_noise.plot_vc_analysis(noise_data_vc,plot_name=name_data, color_data="green")

        return dataset_couple





    def get_synthetic_data(self):
        return self.sample_synthetic
