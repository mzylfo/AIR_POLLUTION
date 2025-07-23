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


class DataSynteticGeneration():

    def __init__(self, torch_device, size_random, path_folder):
        self.torch_device = torch_device
        self.size_random = size_random
        self.path_folder = path_folder
        
    
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
            return self.casualGraph_cov(num_of_samples)
        else:
            return self.casualGraph(num_of_samples)
        
    def casualGraph(self, num_of_samples = 10000):        
        
        adj_distance = [ [ [ list() ] for i in range(34)] for j in range(34)]
        
        for (node_a, node_b) in self.G.edges:
            node_a_x = self.pos_nodes[node_a][0]
            node_a_y = self.pos_nodes[node_a][1]

            node_b_x = self.pos_nodes[node_b][0]
            node_b_y = self.pos_nodes[node_b][1]

            edge_len_x = abs(node_a_x - node_b_x)
            edge_len_y = abs(node_a_y - node_b_y)
            
            edge_len = ((edge_len_x**2) + (edge_len_y**2))**(1/2)

            edge_len_dev = edge_len * 0.1
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
            dataset_couple.append((self.getSample(i), self.getRandom()))
        return dataset_couple

    def casualGraph_cov(self, num_of_samples = 50000, plot_varDist=True):

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
        self.mu = list()
        for univar in adj_distance_list:
            self.mu.append(univar['mu'])
        
        self.r_psd = make_spd_matrix(n_dim=len(self.mu), random_state=0)
        r_psd_Path = Path(self.path_folder, "corrCoeffMatrix_original.csv")
        np.savetxt(r_psd_Path, self.r_psd, delimiter=",")

        mi_Path = Path(self.path_folder, "MEAN_original.csv")
        np.savetxt(mi_Path, self.mu, delimiter=",")

        rng = np.random.default_rng()
        self.sample_synthetic =  [ [[]]  for i in range(self.size_random)]
        
        for x in rng.multivariate_normal(mean=self.mu, cov=self.r_psd, size=num_of_samples):
            for i,val in enumerate(x):
                self.sample_synthetic[i][0].append(val)
        if plot_varDist:
            self.plot_dataDist()
        
        samples_origin_path = Path(self.path_folder, "samples_origin.txt")
        with open(samples_origin_path, 'w') as fp:
            json.dump(self.sample_synthetic, fp, sort_keys=True, indent=4)

        dataset_couple = []
        for i in range(num_of_samples):

            dataset_couple.append((self.getSample(i), self.getRandom()))
        return dataset_couple
        
    def getSample(self, key_sample):
        sample = []
        for ed in self.sample_synthetic:    
            sample.append(ed[0][key_sample])  
        return torch.from_numpy(np.array(sample)).type(torch.float32).to(self.torch_device)
    
    def getRandom(self):
        randomNoise = torch.randn(self.size_random).uniform_(0,1).to(self.torch_device)
        return randomNoise.type(torch.float32)

    def get_synthetic_data(self):
        return self.sample_synthetic

    def plot_dataDist(self):
        path_fold_dist = Path(self.path_folder,"univar_distribution_synthetic")
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)
        for univ_id in range(len(self.mu)):
            mean_val = self.mu[univ_id]
            data_vals = self.sample_synthetic[univ_id][0]
            plt.figure(figsize=(12,8))            
            plt.axvline(x = mean_val, color = 'b', label = 'mean')
            plt.hist(data_vals, density=True, bins=50)
            mean_plt_txt = f"      mean: {mean_val:.3f}"
            plt.text(mean_val, 0, s=mean_plt_txt, rotation = 90)            
            filename = Path(path_fold_dist,"dist_synthetic_"+str(univ_id)+".png")
            plt.savefig(filename)
        
