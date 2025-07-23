import warnings
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.stats import norm
import statistics
from pathlib import Path
import os
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib import cm # for a scatter plot
#from src.tool.utils_matplot import SeabornFig2Grid
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from numpy import dot
from numpy.linalg import norm
from scipy.stats import wasserstein_distance
from copulas.multivariate import GaussianMultivariate
import numpy as np
import scipy.stats as st
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
import pacmap

class AnalysisResult():
    
    def __init__(self, univar_count):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fxn()
        np.seterr(divide='ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)
        self.rand_var_in = dict()
        self.rand_var_out = dict()
        self.rand_var_cop = dict()
        self.TSNE_components = 2
        self.univar_count = univar_count
        self.n_sample_considered = 0
        
    def fxn():
        warnings.warn("deprecated", DeprecationWarning) 
    
    
    def comparison_wass(self, folder):        
        prediced_instances = Path(folder,"AE","train_analysis","prediced_instances_train_test_data.csv")
        train_data = pd.read_csv(prediced_instances)
        
        
        for i in range(self.univar_count):
            self.rand_var_in[i] = list()
            self.rand_var_out[i] = list()
            self.rand_var_cop[i] = list()

        for j in range(len(train_data['x_input'])):
            res = train_data['x_input'][j].strip('][').split(', ')
            for i in range(self.univar_count):
                self.rand_var_in[i].append(float(res[i]))

        for j in range(len(train_data['x_output'])):
            res = train_data['x_output'][j].strip('][').split(', ')
            for i in range(self.univar_count):
                self.rand_var_out[i].append(float(res[i]))

        copula = GaussianMultivariate()
        real_data = pd.DataFrame.from_dict(self.rand_var_in)
        copula.fit(real_data)
        self.n_sample_considered = len(self.rand_var_in[0])
        print(f"Generate {len(self.rand_var_in[0])} istances with copula")
        synthetic_data = copula.sample(len(self.rand_var_in[0]))

        for i in range(self.univar_count):
            self.rand_var_cop[i] = synthetic_data[i].tolist()

        wass_values_ae = dict()
        wass_values_cop = dict()
        mean_ae = list()
        mean_cop = list()
        for i in range(self.univar_count):
            dist_real = self.rand_var_in[i]
            dist_fake = self.rand_var_out[i]
            dist_copu = rand_var_cop[i]

            wd_ae = wasserstein_distance(dist_real,dist_fake)
            wd_cop = wasserstein_distance(dist_real,dist_copu)

            wass_values_ae[i] = wd_ae
            wass_values_cop[i] = wd_cop
            mean_ae.append(wd_ae)
            mean_cop.append(wd_cop)

        wass_values_ae['mean'] = np.mean(mean_ae)
        wass_values_cop['mean'] = np.mean(mean_cop)
        ws_pd_ae = pd.DataFrame(wass_values_ae.items())
        ws_pd_cop = pd.DataFrame(wass_values_cop.items())
        tab_ae[f"{id_file}"] = ws_pd_ae[1]
        tab_cop[f"{id_file}"] = ws_pd_cop[1]


        pd_stats = pd.DataFrame(columns=["variable","mean_real","std_real","mean_AE","std_AE","mean_COP","std_COP"])
        for i in range(self.univar_count):
            in_conf = st.t.interval( df=len(self.rand_var_in[i])-1, loc=np.mean(self.rand_var_in[i]), scale=st.sem(self.rand_var_in[i]), confidence=0.90)
            cop_conf = st.t.interval( df=len(self.rand_var_cop[i])-1, loc=np.mean(self.rand_var_cop[i]), scale=st.sem(self.rand_var_cop[i]), confidence=0.90)
            out_conf = st.t.interval( df=len(self.rand_var_out[i])-1, loc=np.mean(self.rand_var_out[i]), scale=st.sem(self.rand_var_out[i]), confidence=0.90)
            in_mean = np.mean(self.rand_var_in[i])
            cop_mean = np.mean(self.rand_var_cop[i])
            out_mean = np.mean(self.rand_var_out[i])
            in_std = np.std(self.rand_var_in[i])
            cop_std = np.std(self.rand_var_cop[i])
            out_std = np.std(self.rand_var_out[i])

            pd_stats = pd_stats.append({'variable' : i,
                    'mean_real' : in_mean, 'std_real':in_std,
                    'mean_AE' : out_mean, 'std_AE':out_std,
                    'mean_COP' : cop_mean, 'std_COP':cop_std,
                    'diff_real_ea': abs(in_mean-out_mean),
                    'diff_real_cop': abs(in_mean-cop_mean),
                }, ignore_index = True)

        return pd_stats
    
    def comparison_tsne(self, folder, labels_list, n_points=None):
        color_list = {"real": "red","ae":"green","cop":"blue"}
        label_list = {"real": "real data","ae":"GAN+AE gen","cop":"copula gen"}
        
        labels_list
        
        df_tsne = pd.DataFrame()
        if n_points ==None:
            n_points = self.n_sample_considered
            
        real_indeces =  [i for i in range(len(self.rand_var_in[0]))]
        selected = random.sample(real_indeces, n_points)
        real_selected = [1 if i in selected else 0 for i in range(len(self.rand_var_in[0]))]
        
        neur_indeces =  [i for i in range(len(self.rand_var_out[0]))]
        selected = random.sample(neur_indeces, n_points)
        neur_selected = [1 if i in selected else 0 for i in range(len(self.rand_var_out[0]))]
        
        copu_indeces =  [i for i in range(len(self.rand_var_cop[0]))]
        selected = random.sample(copu_indeces, n_points)
        copu_selected = [1 if i in selected else 0 for i in range(len(self.rand_var_cop[0]))]
        
        
        
        for i in range(self.univar_count):
            n_real = 0
            n_neur = 0
            n_copu = 0
            
            real_val = list()
            neur_val = list()
            copu_val = list()
            
            for j in range(len(self.rand_var_in[i])):
                if real_selected[j]==1:
                    real_val.append(self.rand_var_in[i][j])
                    n_real += 1
                    
            for j in range(len(self.rand_var_out[i])):
                if neur_selected[j]==1:
                    neur_val.append(self.rand_var_out[i][j])
                    n_neur += 1
                    
            for j in range(len(self.rand_var_cop[i])):
                if copu_selected[j]==1:
                    copu_val.append(self.rand_var_cop[i][j])
                    n_copu += 1
                    
            comp_values = real_val + neur_val + copu_val
            df_tsne[f'c_{i}'] = comp_values
        
        labels =  ["real" for k in range(n_real)] + ["ae" for k in range(n_neur)] + ["cop" for k in range(n_copu)]
        
        tsne = TSNE(n_components=self.TSNE_components)
        tsne_results = tsne.fit_transform(df_tsne)
        dftest = pd.DataFrame(tsne_results)
        dftest['label'] = labels
        
        
        
        fig = plt.figure(figsize=(16,7))
        sns.scatterplot(
            x=0, y=1,
            hue="label",
            data=dftest,
            alpha=0.2,
            legend="full"
        )
        filename = Path(self.path_folder, "plot_TSNE.png")
        plt.savefig(filename)
        
        #fig, ax = plt.subplots(figsize=(14,8))
        fig, axs = plt.subplots(ncols=self.univar_count)
        a = 0
        for i in range(self.univar_count):
            sns.catplot(data=dftest, x=f'c_{i}', y="label", palette=color_list, hue="label", ax=axs[a])
            sns.swarmplot(data=dftest, x=f'c_{i}', y="label", size=3, ax=axs[a])
            a += 1
            
            
        tsne = TSNE(n_components=self.TSNE_components)
        tsne_results = tsne.fit_transform(df_tsne)
        dftest = pd.DataFrame(tsne_results)
        dftest['label'] = labels
        
        
        embedding = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0) 

        # fit the data (The index of transformed data corresponds to the index of the original data)
        X_transformed = embedding.fit_transform(df_tsne, init="pca")

        # visualize the embedding
        fig, ax = plt.subplots(1, 1, figsize==(16,7))
        ax.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap="Spectral", c=y, s=0.6)

        filename = Path(self.path_folder, "plot_PaCMAP.png")
        plt.savefig(filename)
         
    def compute_statscomparison(self, folder, exp_name, n_run):
        tab_stats = pd.DataFrame(columns=["variable","mean_real","std_real","mean_AE","std_AE","mean_COP","std_COP"])
        tab_ae = pd.DataFrame()
        tab_cop = pd.DataFrame()
        for i in range(n_run):
            folder_run = Path(folder,f"{exp_name}_{i}")
            pd_stats = self.compute_wass(folder_run)
            self.comparison_tsne(folder_run,1000)
            if tab_stats.empty:
                tab_stats = pd_stats
            else:
                tab_stats = tab_stats.add(pd_stats, fill_value=0)
        tab_stats = tab_stats.div(n_exp)
        
        #tab_stats.to_csv(f"/content/30_compare/{ds_name}/{ds_name}_stats.csv", sep='\t')
        #tab_ae.to_csv(f"/content/30_compare/{ds_name}/{ds_name}_ae_wass.csv", sep='\t')
        #tab_cop.to_csv(f"/content/30_compare/{ds_name}/{ds_name}_cop_wass.csv", sep='\t')
