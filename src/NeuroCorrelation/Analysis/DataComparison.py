import matplotlib.pyplot as plt

from termcolor import cprint
from colorama import init, Style

from matplotlib.ticker import PercentFormatter
import numpy as np
#from scipy.stats import norm
import statistics
from pathlib import Path
import os
import pandas as pd
import math
import random

import torch
import scipy.stats as stats
from matplotlib import cm # for a scatter plot
#from src.tool.utils_matplot import SeabornFig2Grid
import matplotlib.gridspec as gridspec
import seaborn as sns
from numpy import dot
from numpy.linalg import norm

import warnings
from scipy.stats import wasserstein_distance
from copulas.multivariate import GaussianMultivariate
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv, pinv
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pacmap
import umap
from scipy.stats import wasserstein_distance
from scipy.stats import binned_statistic_dd

import time
from scipy.stats import binned_statistic_2d
from sklearn.metrics import silhouette_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import adjusted_rand_score

from scipy.stats import gaussian_kde
from scipy.integrate import quad
import dask.array as da
import dask.dataframe as dd

class DataComparison():

    def __init__(self, univar_count_in, univar_count_out, latent_dim, path_folder, name_key):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn("deprecated", DeprecationWarning)
        np.seterr(divide='ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        self.univar_count_in = univar_count_in
        self.univar_count_out = univar_count_out
        self.latent_dim = latent_dim
        self.path_folder = path_folder
        self.np_dist = dict()
        self.name_key = name_key
    
    def get_idName(self, id_univar, max_val):
        num_digit = len(str(max_val))
        name_univ = f'{id_univar:0{num_digit}d}'
        return name_univ

    def data_comparison_plot(self, data, plot_name=None, mode="in", is_npArray=True):
        if plot_name is not None:
            fold_name = f"{plot_name}_distributions_compare"
        else:
            fold_name = f"univar_distribution_compare"

        path_fold_dist = Path(self.path_folder, fold_name)
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)
        
        stats_dict = {"univ_id": []}

        if mode=="in":
            n_var = self.univar_count_in
        elif mode=="out":
            n_var = self.univar_count_out

        for id_univar in range(n_var):
            name_univ = self.get_idName(id_univar, max_val=n_var)
            stats_dict['univ_id'].append(name_univ)
            plt.figure(figsize=(12,8))  
            for key in data:
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                
                if is_npArray:
                    list_values = [x.tolist() for x in data[key]['data'][id_univar]]
                else:
                    list_values = data[key]['data'][id_univar]
                color_data = data[key]['color']
                if 'alpha' in data[key]:
                    alpha_data = data[key]['alpha']
                else:
                    alpha_data = 0.2
                name_data = key
                mean_val = np.mean(list_values)
                var_val = np.var(list_values)
                std_val = np.std(list_values)
                mean_label = f"{name_data}_mean"
                std_label = f"{name_data}_var"
                
                title_txt = f"{plot_name} - vc: {id_univar}"
                plt.title(title_txt)

                plt.axvline(x = mean_val, linestyle="solid", color = color_data, label = mean_label)
                plt.axvline(x = (mean_val-std_val), linestyle="dashed", color = color_data, label = std_label)
                plt.axvline(x = (mean_val+std_val), linestyle="dashed", color = color_data, label = std_label)

                mean_plt_txt = f"      {name_data} mean: {mean_val:.3f}"
                plt.text(mean_val, 0, s=mean_plt_txt, rotation = 90) 


                bins = np.linspace(0.0, 1.0, 100)

                #histogram
                label_hist_txt = f"{name_data}"
                plt.hist(list_values, weights=np.ones(len(list_values)) / len(list_values), bins=bins, histtype='stepfilled', alpha = alpha_data, color= color_data, label=label_hist_txt)
                #histogram border
                plt.hist(list_values, weights=np.ones(len(list_values)) / len(list_values), bins=bins, histtype=u'step', edgecolor="gray", fc="None", lw=1)

                plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

                if mean_label not in stats_dict:
                    stats_dict[mean_label]= list()
                stats_dict[mean_label].append(mean_val)


                if std_label not in stats_dict:
                    stats_dict[std_label]= list()
                stats_dict[std_label].append(std_val)
            
            plt.legend(loc='upper right')
            filename = Path(path_fold_dist,"plot_vc_distribution_"+plot_name+"_"+name_univ+".png")
            plt.savefig(filename)
            plt.close()
            plt.cla()
            plt.clf()


        filename = Path(path_fold_dist,fold_name+"_table.csv")
        stats_dict = pd.DataFrame.from_dict(stats_dict)
        stats_dict.to_csv(filename, sep='\t', encoding='utf-8')

    def  latent_comparison_distribution_plot(self, data_lat, path_fold_dist, plot_name=None, color_data="green"):
        stats_dict = {"univ_id": []}
        for id_comp in range(self.latent_dim):
            
            list_values = [x.tolist() for x in data_lat[id_comp]]
            name_comp = self.get_idName(id_comp, max_val=self.latent_dim)
            plt.figure(figsize=(12,8))
            list_values_weights = np.ones(len(list_values)) / len(list_values)
            plt.hist(list_values, weights=list_values_weights, histtype='stepfilled', alpha = 0.2, color= color_data)
            title_txt = f"{plot_name} - component: {id_comp}"
            plt.title(title_txt)
            filename = Path(path_fold_dist, "plot_latent_distribution_"+plot_name+"_"+name_comp+"_latent.png")
            plt.savefig(filename)
            plt.close()
            plt.cla()
            plt.clf()

    
    def plot_latent_analysis(self, data_lat, plot_name, color_data="green"):    
        # The above code snippet is not doing anything. It contains a comment `# Python` followed by
        # an undefined variable `_keys` and then another comment `
        _keys = list(data_lat.keys())
        n_keys = len(_keys)
        n_row = n_keys
        
        if plot_name is not None:
            fold_name = f"{plot_name}_distribution"
        else:
            fold_name = f"univar_latent_distribution"

        path_fold_dist = Path(self.path_folder, fold_name)
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)
        
       
        self.latent_comparison_distribution_plot(data_lat, path_fold_dist,plot_name)

    def plot_latent_corr__analysis(self, data_lat, plot_name, color_data="green"):            
        _keys = list(data_lat.keys())        
        n_keys = len(_keys)
        n_row = n_keys
        
        if plot_name is not None:
            fold_name = f"{plot_name}_distribution"
        else:
            fold_name = f"univar_latent_distribution"

        path_fold_dist = Path(self.path_folder, fold_name)
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)
            
        fig_size_factor = 2 + n_keys
        fig = plt.figure(figsize=(fig_size_factor, fig_size_factor))
        
        gs = gridspec.GridSpec(n_row, n_row)
        _key = _keys[0]
        
        if len(_keys)>=7:
            marginal_plot = False
        else:
            marginal_plot = True
        
        n_items = len(data_lat[_key])
        
        if n_items>1000:
            item_selected = random.choices([i for i in range(1000)], k=1000)
        else:
            item_selected = [i for i in range(n_items)]
        
        for i,key_i in enumerate(data_lat):
            i_val = [x.tolist() for x in data_lat[i]]
            i_key = f"{i}_comp"
                
            for j,key_j in enumerate(data_lat):
                j_val = [x.tolist() for x in data_lat[j]]
                j_key = f"{j}_comp"
                if i<=j:
                    id_sub = (i*n_keys)+j
                    ax_sub = fig.add_subplot(gs[id_sub])
                    if  i != j:
                        self.correlation_plot(i_val, j_val, i_key, j_key, ax_sub, color=color_data, marginal_dist=marginal_plot)
                    else:
                        self.variance_plot(i_val, i_key, ax_sub, color=color_data)
                    
        gs.tight_layout(fig)
        filename = Path(path_fold_dist, "plot_lat_correlation_grid_"+plot_name+".png")
        plt.savefig(filename)
        plt.close()
        plt.cla()
        plt.clf()


    def plot_vc_correlationCoeff(self, df_data, plot_name, is_latent=False, corrMatrix=None):
        if is_latent:
            if plot_name is not None:
                fold_name = f"{plot_name}_distribution"
            else:
                fold_name = f"univar_latent_distribution"

            path_fold_dist = Path(self.path_folder, fold_name)
            if not os.path.exists(path_fold_dist):
                os.makedirs(path_fold_dist)
        else:
            fold_name = f"{plot_name}_distribution"
            path_fold_dist = Path(self.path_folder, fold_name)
            if not os.path.exists(path_fold_dist):
                os.makedirs(path_fold_dist)

        fig = plt.figure(figsize=(18,18))
        if corrMatrix is None:
            corrMatrix = self.correlationCoeff(df_data)
        rho = corrMatrix
        
        fig_size_factor = 2 + int(len(df_data.keys()))
        
        for key in rho:
            csvFile_Path = Path(path_fold_dist, f"data_{key}_"+plot_name+".csv")
            np.savetxt(csvFile_Path, rho[key], delimiter=",")
            fig = plt.figure(figsize=(fig_size_factor, fig_size_factor))
            fig = plt.figure(figsize=(18,18))
            sns.heatmap(rho[key], annot = True, square=True, vmin=-1, vmax=1, cmap= 'coolwarm')
            filename = Path(path_fold_dist,f"plot_{key}_"+plot_name+".png")
            plt.savefig(filename)
            plt.close()
            plt.cla()
            plt.clf()
        

    def correlationCoeff(self, df_data, select_subset=True, num_to_select = 200):    
        rho_val_list = list()
        for key_vc in df_data:
            if isinstance(df_data[key_vc][0], (np.ndarray)):
                vc_values = [value.tolist() for value in df_data[key_vc]]
                rho_val_list.append(vc_values)
            else:
                rho_val_list.append(df_data[key_vc])
        
        data_df = pd.DataFrame(rho_val_list).T
        if select_subset:
            num_to_selected = min(num_to_select, data_df.shape[0])
            data_df = data_df.sample(num_to_selected, random_state=0)
        
        rho = dict()
        rho['pearson'] = data_df.corr(method='pearson').values
        rho['spearman'] = data_df.corr(method='spearman').values
        rho['kendall'] = data_df.corr(method='kendall').values
        rho['covar'] = np.cov(data_df.T, bias=False)
        
        return rho

    
    def plot_vc_analysis(self, df_data, plot_name, mode="in", color_data="blue"):
        
        _keys = list(df_data.keys())        
        n_keys = len(_keys)
        
        if mode=="in":
            n_row = self.univar_count_in
        else:
            n_row = self.univar_count_out
        
        fig_size_factor = 2 + n_keys
        fig = plt.figure(figsize=(fig_size_factor, fig_size_factor))
        
        gs = gridspec.GridSpec(n_row, n_row)
        _key = _keys[0]
        
        if len(_keys)>=7:
            marginal_plot = False
        else:
            marginal_plot = True
            
        n_items = len(df_data[_key])
        
        if n_items>1000:
                item_selected = random.choices([i for i in range(1000)], k=1000)
        else:
            item_selected = [i for i in range(n_items)]
        a = 1
        for i, key_i in enumerate(df_data):
            i_val = [df_data[key_i][l][0] if isinstance(df_data[key_i][l], list) else df_data[key_i][l] for l in item_selected]
            
            for j, key_j in enumerate(df_data):
                j_val = [df_data[key_j][l][0] if isinstance(df_data[key_j][l], list) else df_data[key_j][l] for l in item_selected]
                if i<=j:
                    id_sub = (i*n_keys)+j
                    ax_sub = fig.add_subplot(gs[id_sub])
                    if  i != j:
                        self.correlation_plot(i_val, j_val, f"{key_i}", f"{key_j}", ax_sub, color=color_data, marginal_dist=marginal_plot)
                        
                    else:
                        self.variance_plot(i_val, f"{key_i}", ax_sub, color=color_data)
        
        gs.tight_layout(fig)
        filename = Path(self.path_folder, "plot_vc_correlation_grid_"+plot_name+".png")
        plt.savefig(filename)
        plt.close()
        plt.cla()
        plt.clf()
    
    def sub_reverse(self, i, n_var):
        col = i % n_var
        row = i // n_var
        j = (col * n_var) + row
        return j

    def correlation_plot(self, rand1, rand2, name1, name2, ax_sub, color="blue", marginal_dist=True):
        if isinstance(rand1, list) and isinstance(rand2, list):
            min1, max1, min2, max2 = min(rand1), max(rand1), min(rand2), max(rand2)
        else:
            min1, max1, min2, max2 = rand1.min(), rand1.max(), rand2.min(), rand2.max()

        sns.histplot(x=rand1, y=rand2, bins=30, ax=ax_sub, color=color)
        ax_sub.set_xticks([])
        ax_sub.set_yticks([])
        
        ax_sub.set_xlim(min1, max1)
        ax_sub.set_ylim(min2, max2)
        
        if marginal_dist:
            ax_marg_x = ax_sub.inset_axes([0, 1.05, 1, 0.2], sharex=ax_sub)
            ax_marg_y = ax_sub.inset_axes([1.05, 0, 0.2, 1], sharey=ax_sub)
            sns.histplot(rand1, ax=ax_marg_x, color=color, kde=False)
            sns.histplot(rand2, ax=ax_marg_y, color=color, kde=False)  # NOTA: usa y per il margine verticale
        return ax_sub

    def variance_plot(self, rand1, name1, ax_sub, color="blue"):
        h = sns.histplot(data=rand1, kde = True, ax=ax_sub, color=color)
        ax_sub.set_xticks([])
        ax_sub.set_yticks([])
        ax_sub.set_ylabel('')
        ax_sub.set_xlabel('')
        return h
    
    def find_nearest_kde(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
     
    def draw_point_overDistribution(self, plotname, folder, n_var, points,  distr, n_sample = 1000):
        if distr is None:
            distr = list()
            for i in range(n_sample):
                mu, sigma = 0, math.sqrt(1) # mean and standard deviation
                s = np.random.normal(mu, sigma, n_var)  
                distr.append({'sample': torch.Tensor(s), 'noise': torch.Tensor(s)})
        

        fig = plt.figure(figsize=(18,18))
        n_col = math.ceil(math.sqrt(n_var))
        n_row = math.ceil(n_var/n_col)  

        gs = gridspec.GridSpec(n_row,n_col)
        distr_dict = dict()
        points_dict = dict()
    
        for i in range(n_var):
            distr_dict[i] = list()
            points_dict[i] = list()

        for sample in distr:
            for i in range(n_var):
                distr_dict[i].append(float(sample['sample'][i].cpu().numpy()))
            
        for sample in points:
            for i in range(n_var):
                points_dict[i].append(float(sample['sample'][i].cpu().numpy()))


        for id_sub in range(n_var):
            ax_sub = fig.add_subplot(gs[id_sub])
            h = sns.histplot(data=np.array(distr_dict[id_sub]), kde = True, element="step", ax=ax_sub, alpha=0.3)
            point_list = list()

            x = ax_sub.lines[0].get_xdata()
            y = ax_sub.lines[0].get_ydata()

            points = list(zip(x, y))
            t_dic = dict(points)

            lls = 1
            for sample in points_dict[id_sub]:
                true_x = sample
                x_point = self.find_nearest_kde(np.array(list(t_dic.keys())), true_x)

                sns.scatterplot(x = [x_point],y = [t_dic[x_point]], s=50)
                ax_sub.text(x_point+.02, t_dic[x_point], str(lls))
                lls += 1

        gs.tight_layout(fig)
        filename = Path(folder, f"{plotname}.png")
        plt.savefig(filename)
        plt.close()
        plt.cla()
        plt.clf()


class DataComparison_Advanced():
    
    def __init__(self, univar_count, input_folder, suffix_input, time_performance, data_metadata, name_key, use_copula=True, load_copula=False, copulaData_filename=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn("deprecated", DeprecationWarning)
        np.seterr(divide='ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        
        self.name_key = name_key
        self.univar_count = univar_count
        self.n_sample_considered = 2375
        self.use_copula = use_copula
        self.load_copula = load_copula
        self.copulaData_filename = copulaData_filename
        self.copula_test = False
        self.time_performance = time_performance
        self.alredy_select_data = False
        self.metrics_pd = None
        self.np_dist = dict()
        self.loadPrediction_INPUT(input_folder, suffix_input)
        self.color_list = {"real": (0.122, 0.467, 0.706)}
        self.label_list = {"real": "real data"}
        self.comparisons = dict()
        if self.use_copula:
            self.color_list["cop"] = (0.173, 0.627, 0.173)
            self.label_list["cop"] = "copula gen"
            self.comparisons["real_cop"] = {"a":"real","b":"cop"}
        
        for item in data_metadata:
            self.color_list[item['acronym']] = item['color']
            self.label_list[item['acronym']] = item['label']
            self.comparisons[f"real_{item['acronym']}"] = {"a":"real","b":item['acronym']}
        
        
        
    def loadPrediction_INPUT(self, input_folder, suffix_input):
        self.rand_var_in = dict()        
        self.rand_var_cop = dict()
        for i in range(self.univar_count):
            self.rand_var_in[i] = list()
            self.rand_var_cop[i] = list()
        
        
        #input used to train
        input_instances = Path(input_folder,f"{suffix_input}.csv")
        input_data = pd.read_csv(input_instances)
        
        for j in range(len(input_data['x_input'])):
            res = input_data['x_input'][j].strip('][').split(', ')
            for i in range(self.univar_count):
                self.rand_var_in[i].append(float(res[i]))
        print("\tload truth data: done")
        
        self.np_dist['real'] = pd.DataFrame.from_dict(self.rand_var_in).to_numpy()
        if self.use_copula:
            self.genCopula(input_folder)
            self.np_dist['cop'] = pd.DataFrame.from_dict(self.rand_var_cop).to_numpy()

    def genCopula(self, input_folder):
        if self.load_copula:
            cprint(Style.BRIGHT +f"| Copula data   : Load data from {self.copulaData_filename}" + Style.RESET_ALL, 'magenta', attrs=["bold"])
            
            synthetic_data = pd.read_csv(self.copulaData_filename)
            new_columns = list(range(len(synthetic_data.columns)))
            synthetic_data.columns = new_columns
        else: 
            real_data = pd.DataFrame.from_dict(self.rand_var_in)
            if self.copula_test:
                j_range = 50
            else:
                j_range = len(self.rand_var_in[0])
            cprint(Style.BRIGHT +f"| Copula data   : Fitted on {j_range} instances" + Style.RESET_ALL, 'magenta', attrs=["bold"])
            
            real_data = real_data[:j_range]
            print(f"\tfit gaussian copula data: start")
            self.time_performance.start_time("COPULA_TRAINING")
            copula = GaussianMultivariate()        
            copula.fit(real_data)
        
            self.time_performance.stop_time("COPULA_TRAINING")
            print(f"\tfit gaussian copula data: end")
            cop_train_time = self.time_performance.get_time("COPULA_TRAINING", fun="last")
            self.time_performance.compute_time("COPULA_TRAINING", fun = "first") 
        
            print(f"\tTIME fit gaussian copula data:\t",cop_train_time)
            self.time_performance.start_time("COPULA_GENERATION")
            synthetic_data = copula.sample(self.n_sample_considered)
        
            self.time_performance.stop_time("COPULA_GENERATION")
            cop_gen_time = self.time_performance.get_time("COPULA_GENERATION", fun="last")
            self.time_performance.compute_time("COPULA_GENERATION", fun = "first") 
        
            copula_instances_folder = Path(input_folder,"copula_gen_instances.csv")
            synthetic_data.to_csv(copula_instances_folder)
        
            print(f"\tTIME gen gaussian copula data:\t",cop_gen_time)
            print(f"\tgenerate gaussian copula data: done \t({self.n_sample_considered} instances)")
            print(f"\tcopula generated instances saved in:\t",copula_instances_folder)
            
        for i in range(self.univar_count):
            self.rand_var_cop[i] = synthetic_data[i].tolist()


    def loadPrediction_OUTPUT(self, output_folder, suffix_output, key):
        self.path_folder = output_folder
        self.suffix = suffix_output
        self.rand_var_out = dict()
        for i in range(self.univar_count):
            self.rand_var_out[i] = list()     
        output_instances = Path(output_folder,f"prediced_instances_{suffix_output}.csv")
        
        output_data = pd.read_csv(output_instances)
        
        for j in range(len(output_data['x_output'])):
            res = output_data['x_output'][j].strip('][').split(', ')
            for i in range(self.univar_count):
                self.rand_var_out[i].append(float(res[i].split('(')[-1].rstrip(')')))
        self.np_dist[key] = pd.DataFrame.from_dict(self.rand_var_out).to_numpy()
        self.select_data()
        
    def comparison_measures(self, measures):
        
        if not self.alredy_select_data:
            self.select_data()
        if 'metrics' in measures:
            self.comparison_metrics(save=False)
        
        if 'tsne_plots' in measures:
            self.comparison_tsne(measure=True, comparisons=self.comparisons, apply_pca = False)
        if 'pca_tsne_plots' in measures:
            self.comparison_tsne(measure=True, comparisons=self.comparisons, apply_pca = True)
        if 'pacmap_plots' in measures:
            self.comparison_pacmap(measure=True, comparisons=self.comparisons)
        if 'umap_plots' in measures:
            self.comparison_umap(measure=True, comparisons=self.comparisons)
        if 'wasserstein_dist' in measures:
            self.comparison_wasserstein()
        
        if 'swarm_distributions' in measures:
            self.comparison_swarm_distributions()
        filename = Path(self.path_folder, f"metrics_compare_{self.suffix}.csv")
        self.metrics_pd.to_csv(filename)
        
          
    def comparison_wasserstein(self): 
        print("\t\twasserstein measure")
        wass_values_ae = dict()
        if self.use_copula:
            wass_values_cop = dict()
        mean_ae = list()
        mean_cop = list()
        for i in range(self.univar_count):
            dist_real = self.rand_var_in[i]
            dist_fake = self.rand_var_out[i]
            wd_ae = wasserstein_distance(dist_real,dist_fake)
            wass_values_ae[i] = wd_ae
            mean_ae.append(wd_ae)
            
            if self.use_copula:
                dist_copu = self.rand_var_cop[i]
                wd_cop = wasserstein_distance(dist_real,dist_copu)
                wass_values_cop[i] = wd_cop
                mean_cop.append(wd_cop)

        wass_values_ae['mean'] = np.mean(mean_ae)        
        ws_pd_ae = pd.DataFrame(wass_values_ae.items())
        
        columns=["variable","mean_real","std_real","mean_AE","std_AE"]
        if self.use_copula:
            wass_values_cop['mean'] = np.mean(mean_cop)
            ws_pd_cop = pd.DataFrame(wass_values_cop.items())
            columns.append("mean_COP")
            columns.append("std_COP")
        pd_stats = pd.DataFrame(columns=columns)
        
        for i in range(self.univar_count):
            in_conf = stats.t.interval( df=len(self.rand_var_in[i])-1, loc=np.mean(self.rand_var_in[i]), scale=stats.sem(self.rand_var_in[i]), confidence=0.90)
            in_mean = np.mean(self.rand_var_in[i])
            in_std = np.std(self.rand_var_in[i])
                
            out_conf = stats.t.interval( df=len(self.rand_var_out[i])-1, loc=np.mean(self.rand_var_out[i]), scale=stats.sem(self.rand_var_out[i]), confidence=0.90)
            out_mean = np.mean(self.rand_var_out[i])
            out_std = np.std(self.rand_var_out[i])
            
            variable_dict = {'variable' : i,'mean_real' : in_mean, 'std_real':in_std, 'mean_AE' : out_mean, 'std_AE':out_std,'diff_real_ea': abs(in_mean-out_mean)}
            
            
            if self.use_copula:
                cop_conf = stats.t.interval( df=len(self.rand_var_cop[i])-1, loc=np.mean(self.rand_var_cop[i]), scale=stats.sem(self.rand_var_cop[i]), confidence=0.90)
                cop_mean = np.mean(self.rand_var_cop[i])
                cop_std = np.std(self.rand_var_cop[i])
                variable_dict['mean_COP'] = cop_mean
                variable_dict['std_COP'] = cop_std
                variable_dict['diff_real_cop'] = abs(in_mean-cop_mean)                

            pd_stats = pd.concat([pd_stats, pd.DataFrame([variable_dict])], ignore_index=True)

        filename = Path(self.path_folder, f"wasserstein_compare_{self.suffix}.csv")
        pd_stats.to_csv(filename)
        return pd_stats
            
    
    def invert_matrix(self, matrix):
        try:
            # Check if the matrix is singular by computing its determinant
            if np.linalg.det(matrix) == 0:
                print("The matrix is singular and cannot be inverted.")
                return None
            
            # Compute the inverse if the matrix is not singular
            inv_matrix = np.linalg.inv(matrix)
            return inv_matrix
        
        except np.linalg.LinAlgError as e:
            print(f"Error: {e}")
            return None
       
        
    def comparison_metrics(self, metrics=['mahalanobis','wasserstein','frechet', 'histogram_error','corr_matrix'], save=True):
        self.metrics_pd = None
        metrics_pd_columns = ['metric']
        for x in self.comparisons:
            metrics_pd_columns.append(x) 
        self.metrics_pd = pd.DataFrame(columns=metrics_pd_columns)
        
        histogram_n_bins = 10
        hist_datasets = []
        hist_datasets_acronym = []
        for key in self.np_dist.keys():
            hist_datasets_acronym.append(key)
            hist_datasets.append(self.np_dist[key])
        
        # Salvataggio delle matrici su file
        file_path = Path(self.path_folder,'hist_datasets_data.txt')
        with open(file_path, "w") as f:
            for i, dataset in enumerate(hist_datasets):
                np.savetxt(f, dataset, fmt="%.6f")  # Salva con 6 decimali di precisione
                f.write(f"\n# Dataset {i + 1}\n")  # Separatore tra dataset



        print(f"File salvato in: {file_path}")

            
        hist_list = SparseNDHistogram.from_datasets(histogram_n_bins, self.univar_count, *hist_datasets)
        
        
        if 'mahalanobis' in metrics:
            measures = {'metric':'mahalanobis'}
            for comparison in self.comparisons:
                data_A = self.comparisons[comparison]['a']
                data_B = self.comparisons[comparison]['b']
                measure = self.mahalanobis(self.np_dist[data_A], self.np_dist[data_B])
                measures[comparison] = measure
                print(measures)
            self.metrics_pd = pd.concat([self.metrics_pd, pd.DataFrame([measures])], ignore_index=True)
        
        if 'corr_matrix' in metrics:
            measures = {'metric':'corr_matrix'}
            for comparison in self.comparisons:
                data_A = self.comparisons[comparison]['a']
                data_B = self.comparisons[comparison]['b']
                measure = self.corr_matrix(self.np_dist[data_A], self.np_dist[data_B], label_A=data_A,label_B=data_B)
                measures[comparison] = measure
                print(measures)
            self.metrics_pd = pd.concat([self.metrics_pd, pd.DataFrame([measures])], ignore_index=True)
            
        if 'wasserstein' in metrics:
            measures = {'metric':'wasserstein'}
            for comparison in self.comparisons:
                data_A = self.comparisons[comparison]['a']
                data_B = self.comparisons[comparison]['b']
                measure = self.wasserstein(self.np_dist[data_A], self.np_dist[data_B])
                measures[comparison] = measure
            
            self.metrics_pd = pd.concat([self.metrics_pd, pd.DataFrame([measures])], ignore_index=True)
        if 'bhattacharyya' in metrics:
            measures = {'metric':'bhattacharyya'}
            for comparison in self.comparisons:
                data_A = self.comparisons[comparison]['a']
                data_B = self.comparisons[comparison]['b']
                measure = self.bhattacharyya(self.np_dist[data_A], self.np_dist[data_B])
                measures[comparison] = measure
            
            self.metrics_pd = pd.concat([self.metrics_pd, pd.DataFrame([measures])], ignore_index=True)
        
        if 'histogram_error'in metrics:
            measures = {'metric':'histogram_error'}
            
            for comparison in self.comparisons:
                data_A = self.comparisons[comparison]['a']
                data_B = self.comparisons[comparison]['b']
                n_bins = 20
                matrix_selected=[hist_datasets_acronym.index(data_A), hist_datasets_acronym.index(data_B)]
                measure = SparseNDHistogram.compute_error(hist_list, error_name=["RMSE","MAE"], matrix_selected=matrix_selected)
                
                print("measure histogram_error:\tA",measure,"\t\t",data_A,"vs",data_B)
                measures[comparison] = measure
            self.metrics_pd = pd.concat([self.metrics_pd, pd.DataFrame([measures])], ignore_index=True)
        
        if 'frechet' in metrics:
            measures = {'metric':'frechet'}
            for comparison in self.comparisons:
                data_A = self.comparisons[comparison]['a']
                data_B = self.comparisons[comparison]['b']
                measure = self.frechet_inception_distance(self.np_dist[data_A], self.np_dist[data_B])
                measures[comparison] = measure
            
            self.metrics_pd = pd.concat([self.metrics_pd, pd.DataFrame([measures])], ignore_index=True)
        
        print(self.metrics_pd)
        if save:
            filename = Path(self.path_folder, f"metrics_compare_{self.suffix}.csv")
            self.metrics_pd.to_csv(filename)
            return self.metrics_pd
    
    def wasserstein(self, X, Y):
        wasserstein_distances = [wasserstein_distance(X[:, i], Y[:, i]) for i in range(X.shape[1])]
        return np.mean(wasserstein_distances), wasserstein_distances
    
        
    def corr_matrix(self, X, Y, label_A, label_B):
        X_df = pd.DataFrame(X)
        Y_df = pd.DataFrame(Y)
        
        X_dask = dd.from_pandas(X_df, npartitions=10)
        Y_dask = dd.from_pandas(Y_df, npartitions=10)

        corr_X = X_dask.corr().compute()
        corr_Y = Y_dask.corr().compute()
        
        corr_diff = corr_X - corr_Y
    
        filename_X = Path(self.path_folder,f'correlation_{label_A}.csv')
        filename_Y = Path(self.path_folder,f'correlation_{label_B}.csv')
        filename_XY = Path(self.path_folder,f'correlation_difference__{label_A}_{label_B}.csv')
        corr_X.to_csv(filename_X)
        corr_Y.to_csv(filename_Y)
        corr_diff.to_csv(filename_XY)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_X, cmap="coolwarm", annot=False, fmt=".2f", vmin=-1, vmax=1)
        plt.title(f"Correlation Matrix - {label_A}")
        filename_Xplot = Path(self.path_folder,f'plot_correlation_{label_A}.png')
        plt.savefig(filename_Xplot, dpi=300)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_Y, cmap="coolwarm", annot=False, fmt=".2f", vmin=-1, vmax=1)
        plt.title(f"Correlation Matrix - {label_B}")
        filename_Yplot = Path(self.path_folder,f'plot_correlation_{label_B}.png')
        plt.savefig(filename_Yplot, dpi=300)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_diff, cmap="coolwarm", annot=False, fmt=".2f", vmin=-2, vmax=2)
        plt.title(f"Correlation Matrix - {label_A} vs {label_B}")
        filename_XYplot = Path(self.path_folder,f'plot_correlation_{label_A}_{label_B}.png')
        plt.savefig(filename_XYplot, dpi=300)
        
        
        
        
        metrics = {
            "Frobenius_norm": np.linalg.norm(corr_diff.values, 'fro'),
            "Mean_absolute": np.mean(np.abs(corr_diff.values)),
            "Total_absolute": np.sum(np.abs(corr_diff.values))
        }
        return metrics
        
    def mahalanobis(self, X, Y):
        mu_X = np.mean(X, axis=0)
        mu_Y = np.mean(Y, axis=0)
            
        cov_X = np.cov(X, rowvar=False)
        cov_Y = np.cov(Y, rowvar=False)
        
        cov_combined = (cov_X + cov_Y) / 2
        
        diff = mu_X - mu_Y
        
        if np.linalg.det(cov_combined) == 0:
            print("\t\tThe covariance matrix is singular. Using the pseudo-inverse.")
            inv_cov = np.linalg.pinv(cov_combined)
        else:
            inv_cov = np.linalg.inv(cov_combined)

        dist_mahalanobis = np.sqrt(diff.T @ inv_cov @ diff)
        
        return dist_mahalanobis    
    
    def bhattacharyya_coefficient(self, X, Y, num_samples=10000):
        """
        Stima il coefficiente di Bhattacharyya usando KDE multivariato e Monte Carlo.
        """
        # KDE Multivariato
        kde_X = gaussian_kde(X.T, bw_method='silverman')
        kde_Y = gaussian_kde(Y.T, bw_method='silverman')
        print("---",len(X),len(X[0]))
        print("X=",X[0])
        raise Exception("-")
        # Generiamo campioni Monte Carlo nello spazio multidimensionale
        min_vals, max_vals = np.min(X, axis=0), np.max(Y, axis=0)
        samples = np.random.uniform(min_vals, max_vals, size=(num_samples, X.shape[1])).T


        # Calcoliamo P(x) e Q(x)
        P_samples = kde_X(samples)
        Q_samples = kde_Y(samples)

        # Stima del coefficiente di Bhattacharyya
        BC = np.mean(np.sqrt(P_samples * Q_samples))
        return BC

    def bhattacharyya(self, X, Y):
        BC = self.bhattacharyya_coefficient(X, Y)
        return -np.log(BC) if BC > 0 else np.inf
        
    def pairwise_compute_histogram(self, data, n_comp, edges):
        hist, edges = np.histogramdd(data, bins=edges)
        edges_component = dict()
        for i in range(n_comp):
            edges_component[i] = edges[i]
        return {"hist": hist, "edges_component": edges_component}
    
    def pairwise_compute_error(self, data_A, data_B, error_name):
        if error_name == "RSME":    
            val_err = np.sqrt(np.mean((data_A - data_B) ** 2))
        else:
            val_err = -1
        return val_err

    def pairwise_compute_edges(self, data_A, data_B, n_comp, n_bins):
        min_data = np.minimum(data_A.min(axis=0), data_B.min(axis=0))
        max_data = np.maximum(data_A.max(axis=0), data_B.max(axis=0))
        edges = []
        for i in range(n_comp):
            edges.append(np.linspace(min_data[i], max_data[i], n_bins + 1))
        return edges
     
    def pairwise_error(self, data_A, data_B, error_name, n_comp, n_bins=10):
        edges = self.pairwise_compute_edges(data_A, data_B, n_comp, n_bins)
        databin_A = self.pairwise_compute_histogram(data=data_A, n_comp=n_comp, edges=edges)
        databin_B = self.pairwise_compute_histogram(data=data_B, n_comp=n_comp, edges=edges)
        val_err = self.pairwise_compute_error(databin_A['hist'], databin_B['hist'], error_name=error_name)
        return val_err 
      
        
    def frechet_inception_distance(self, real_samples, generated_samples):
        mu_real = np.mean(real_samples, axis=0)
        mu_generated = np.mean(generated_samples, axis=0)

        sigma_real = np.cov(real_samples, rowvar=False)
        sigma_generated = np.cov(generated_samples, rowvar=False)

        diff = mu_real - mu_generated
        diff_squared = np.sum(diff**2)
        
        covmean, _ = sqrtm(sigma_real @ sigma_generated, disp=False)
    
        # numerical adjust
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff_squared + np.trace(sigma_real + sigma_generated - 2 * covmean)
        return fid
    
    
        
    def select_data(self, TSNE_components = [2,3], n_points=None):
        
            
        self.df_data_selected = pd.DataFrame()
        if n_points == None:
            n_points = self.n_sample_considered
        
        
        len_sample_array = [len(self.rand_var_in[0]), len(self.rand_var_out[0])]
        
        if self.use_copula:
            len_sample_array.append(len(self.rand_var_cop[0]))
        if n_points > min(len_sample_array):
            n_points = min(len_sample_array)
        
        print("\t\t SELECTED plot #points:\t",n_points)
        real_indeces =  [i for i in range(len(self.rand_var_in[0]))]
        selected = random.sample(real_indeces, n_points)
        real_selected = [1 if i in selected else 0 for i in range(len(self.rand_var_in[0]))]
        
        neur_indeces =  [i for i in range(len(self.rand_var_out[0]))]
        selected = random.sample(neur_indeces, n_points)
        neur_selected = [1 if i in selected else 0 for i in range(len(self.rand_var_out[0]))]
        
        if self.use_copula:
            copu_indeces =  [i for i in range(len(self.rand_var_cop[0]))]
            selected = random.sample(copu_indeces, n_points)
            copu_selected = [1 if i in selected else 0 for i in range(len(self.rand_var_cop[0]))]
        
        
        for i in range(self.univar_count):
            n_real = 0
            real_val = list()
            for j in range(len(self.rand_var_in[i])):
                if real_selected[j]==1:
                    real_val.append(self.rand_var_in[i][j])
                    n_real += 1
            n_neur = 0
            neur_val = list()        
            for j in range(len(self.rand_var_out[i])):
                if neur_selected[j]==1:
                    neur_val.append(self.rand_var_out[i][j])
                    n_neur += 1
                    
            comp_values = real_val + neur_val
            
            if self.use_copula:
                copu_val = list()
                n_copu = 0
                for j in range(len(self.rand_var_cop[i])):
                    if copu_selected[j]==1:
                        copu_val.append(self.rand_var_cop[i][j])
                        n_copu += 1
                        
                comp_values += copu_val
            self.df_data_selected[f'c_{i}'] = comp_values
                    
        labels =  ["real" for k in range(n_real)] + [self.name_key for k in range(n_neur)] 
        if self.use_copula:
            labels += ["cop" for k in range(n_copu)]
        self.df_data_selected['labels'] = labels
        self.alredy_select_data =True
    
    def comparison_tsne(self, TSNE_components = ["2D","3D"], n_points=None, measure=True, comparisons=None, bins=50, apply_pca=False):    
        if "2D" in TSNE_components:
            data4fit = self.df_data_selected.drop(columns=['labels'])
    
            if apply_pca:
                n_components_pca = min(50, data4fit.shape[1])  # Reduce to 50 components or less if fewer features exist
                pca = PCA(n_components=n_components_pca)
                data4fit = pca.fit_transform(data4fit)
            
            tsne = TSNE(n_components=2)
            tsne_results = tsne.fit_transform(data4fit)
            df_tsne_results = pd.DataFrame(tsne_results)
            
            df_tsne_results['labels'] = self.df_data_selected['labels']
            
            fig = plt.figure(figsize=(16,7))
            sns.scatterplot(
                x=0, y=1,
                hue="labels",
                palette=self.color_list,
                data=df_tsne_results,
                alpha=0.2,
                legend="full",
            )
            
            if apply_pca:
                filename = Path(self.path_folder, f"PCA_TSNE_2D_plot_withcop_{self.suffix}.png")
            else:
                filename = Path(self.path_folder, f"TSNE_2D_plot_withcop_{self.suffix}.png")
            plt.savefig(filename)
            plt.close()
            plt.cla()
            plt.clf()
            
            
            df_tsne_resultsnocop = df_tsne_results[df_tsne_results["labels"] != "cop"]
            
            fig = plt.figure(figsize=(16,7))
            sns.scatterplot(
                x=0, y=1,
                hue="labels",
                palette=self.color_list,
                data=df_tsne_resultsnocop,
                alpha=0.2,
                legend="full",
            )
            
            if apply_pca:
                filename = Path(self.path_folder, f"PCA_TSNE_2D_plot_nocop_{self.suffix}.png")
            else:
                filename = Path(self.path_folder, f"TSNE_2D_plot_nocop_{self.suffix}.png")
            plt.savefig(filename)
            plt.close()
            plt.cla()
            plt.clf()
            
            if measure:
                if apply_pca:
                    measures_overlap = {'metric':'PCA_TSNE_2D_SpatialOverlapping'}
                else:
                    measures_overlap = {'metric':'TSNE_2D_SpatialOverlapping'}
                for comparison in comparisons:
                    label_A = comparisons[comparison]['a']
                    label_B = comparisons[comparison]['b']
                    
                    data_A = df_tsne_results[df_tsne_results['labels'] == label_A]
                    data_B = df_tsne_results[df_tsne_results['labels'] == label_B]
                    grid_A, _, _, _ = binned_statistic_2d((data_A[0]), data_A[1], None, statistic='count', bins=bins)
                    grid_B, _, _, _ = binned_statistic_2d(data_B[0], data_B[1], None, statistic='count', bins=bins)
                    grid_A = grid_A / np.sum(grid_A)
                    grid_B = grid_B / np.sum(grid_B)
                    overlap = np.sum(np.minimum(grid_A, grid_B))
                    measures_overlap[comparison] = overlap
                
                self.metrics_pd = pd.concat([self.metrics_pd, pd.DataFrame([measures_overlap])], ignore_index=True)
            
                
                if apply_pca:
                    measures_TSNE_mah = {'metric':'PCA_TSNE_2D_MAHALANOBIS'}
                else:
                    measures_TSNE_mah = {'metric':'TSNE_2D_MAHALANOBIS'}
                for comparison in comparisons:
                    label_A = comparisons[comparison]['a']
                    label_B = comparisons[comparison]['b']
                    
                    data_A = df_tsne_results[df_tsne_results['labels'] == label_A]
                    data_B = df_tsne_results[df_tsne_results['labels'] == label_B]
                    
                    data_A_Mahala = data_A.drop(columns=['labels'])
                    data_B_Mahala = data_B.drop(columns=['labels'])
                    
                    measures_TSNE_mah[comparison] = self.mahalanobis(data_A_Mahala,data_B_Mahala)
                self.metrics_pd = pd.concat([self.metrics_pd, pd.DataFrame([measures_TSNE_mah])], ignore_index=True)
                        
        
        if "3D" in TSNE_components:
            
            data4fit = self.df_data_selected.drop(columns=['labels'])
    
            if apply_pca:
                n_components_pca = min(50, data4fit.shape[1])  # Reduce to 50 components or less if fewer features exist
                pca = PCA(n_components=n_components_pca)
                data4fit = pca.fit_transform(data4fit)
            
            tsne = TSNE(n_components=3)
            tsne_results = tsne.fit_transform(data4fit)
            dftest = pd.DataFrame(tsne_results)
            dftest['labels'] = self.df_data_selected['labels']
            
            fig = plt.figure(figsize=(16, 7))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(
                dftest[0], dftest[1], dftest[2],
                c=dftest["labels"].map(self.color_list),
                alpha=0.2
            )
            if apply_pca:
                filename = Path(self.path_folder, f"PCA_TSNE_3D_plot_{self.suffix}.png")
            else:
                filename = Path(self.path_folder, f"TSNE_3D_plot_{self.suffix}.png")
            plt.savefig(filename)
            plt.close()
            plt.cla()
            plt.clf()
            if measure:
                if apply_pca:
                    measures_TSNE_mah = {'metric':'PCA_TSNE_3D_MAHALANOBIS'}
                else:
                    measures_TSNE_mah = {'metric':'TSNE_3D_MAHALANOBIS'}
                for comparison in comparisons:
                    label_A = comparisons[comparison]['a']
                    label_B = comparisons[comparison]['b']
                    
                    data_A = df_tsne_results[df_tsne_results['labels'] == label_A]
                    data_B = df_tsne_results[df_tsne_results['labels'] == label_B]
                    
                    data_A_Mahala = data_A.drop(columns=['labels'])
                    data_B_Mahala = data_B.drop(columns=['labels'])
                    
                    
                    measures_TSNE_mah[comparison] = self.mahalanobis(data_A_Mahala,data_B_Mahala)
                self.metrics_pd = pd.concat([self.metrics_pd, pd.DataFrame([measures_TSNE_mah])], ignore_index=True)
            if False:
                if apply_pca:
                    measures = {'metric':'PCA_TSNE_3D_SpatialOverlapping'}
                else:
                    measures = {'metric':'TSNE_3D_SpatialOverlapping'}
                for comparison in comparisons:
                    label_A = comparisons[comparison]['a']
                    label_B = comparisons[comparison]['b']
                    
                    data_A = dftest[dftest['labels'] == label_A]
                    data_B = dftest[dftest['labels'] == label_B]
                    print("data_A COLUMN", data_A.columns)
                    print("data_B COLUMN", data_B.columns)
                    grid_A, edges = binned_statistic_dd(
                        sample=data_A[[0, 1, 2]].values,
                        values=None,
                        statistic='count',
                        bins=bins
                    )
                    
                    grid_B, _ = binned_statistic_dd(
                        sample=data_B[[0, 1, 2]].values,
                        values=None,
                        statistic='count',
                        bins=bins
                    )
                    
                    grid_A = grid_A / np.sum(grid_A)
                    grid_B = grid_B / np.sum(grid_B)
                    overlap = np.sum(np.minimum(grid_A, grid_B))
                    measures[comparison] = overlap
                
                self.metrics_pd = pd.concat([self.metrics_pd, pd.DataFrame([measures])], ignore_index=True)
        
        
    def comparison_pacmap(self, PACMAP_components = ["2D","3D"], n_points=None, measure=True, comparisons=None, bins=50):    
        
        
        if "2D" in PACMAP_components:
            embedding  = pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0) 
            data4fit = self.df_data_selected.drop(columns=['labels'])
            pacmap_results = embedding.fit_transform(data4fit.to_numpy(), init="pca")
            df_pacmap_results = pd.DataFrame(pacmap_results)
            df_pacmap_results['labels'] = self.df_data_selected['labels']
            fig = plt.figure(figsize=(16,7))
            sns.scatterplot(
                x=0, y=1,
                hue="labels",
                palette=self.color_list,
                data=df_pacmap_results,
                alpha=0.2,
                legend="full",
            )
            filename = Path(self.path_folder, f"MAPCAP_2D_plot_{self.suffix}.png")
            plt.savefig(filename)
            plt.close()
            plt.cla()
            plt.clf()
            
            if measure:
                measures = {'metric':'PACMAP_2D_SpatialOverlapping'}
                for comparison in comparisons:
                    label_A = comparisons[comparison]['a']
                    label_B = comparisons[comparison]['b']
                    
                    data_A = df_pacmap_results[df_pacmap_results['labels'] == label_A]
                    data_B = df_pacmap_results[df_pacmap_results['labels'] == label_B]
                    grid_A, _, _, _ = binned_statistic_2d(data_A[0], data_A[1], None, statistic='count', bins=bins)
                    grid_B, _, _, _ = binned_statistic_2d(data_B[0], data_B[1], None, statistic='count', bins=bins)
                    
                    
                    
                    grid_A = grid_A / np.sum(grid_A)
                    grid_B = grid_B / np.sum(grid_B)
                    overlap = np.sum(np.minimum(grid_A, grid_B))
                    measures[comparison] = overlap
                
                self.metrics_pd = pd.concat([self.metrics_pd, pd.DataFrame([measures])], ignore_index=True)

        if "3D" in PACMAP_components:
            embedding = pacmap.PaCMAP(n_components=3, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0) 
            data4fit = self.df_data_selected.drop(columns=['labels'])
            pacmap_results = embedding.fit_transform(data4fit.values, init="pca")
            dftest = pd.DataFrame(pacmap_results)
            dftest['labels'] = self.df_data_selected['labels']

            fig = plt.figure(figsize=(16, 7))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(
                dftest[0], dftest[1], dftest[2],
                c=dftest["labels"].map(self.color_list),
                alpha=0.2
            )
            filename = Path(self.path_folder, f"MAPCAP_3D_plot_{self.suffix}.png")
            plt.savefig(filename)
            plt.close()
            plt.cla()
            plt.clf()
    
    
    def comparison_umap(self, UMAP_components = ["2D","3D"], n_points=None, measure=True, comparisons=None, bins=50):    
        if "2D" in UMAP_components:
            embedding  = umap.UMAP(n_components=2, metric='euclidean', random_state=42, repulsion_strength=1.0) 
            data4fit = self.df_data_selected.drop(columns=['labels'])
            umap_results = embedding.fit_transform(data4fit)
            df_umap_results = pd.DataFrame(umap_results)
            df_umap_results['labels'] = self.df_data_selected['labels']
            fig = plt.figure(figsize=(16,7))
            sns.scatterplot(
                x=0, y=1,
                hue="labels",
                palette=self.color_list,
                data=df_umap_results,
                alpha=0.2,
                legend="full",
            )
            filename = Path(self.path_folder, f"UMAP_2D_plot_{self.suffix}.png")
            plt.savefig(filename)
            plt.close()
            plt.cla()
            plt.clf()
            
            if measure:
                measures = {'metric':'UMAP_2D_SpatialOverlapping'}
                for comparison in comparisons:
                    label_A = comparisons[comparison]['a']
                    label_B = comparisons[comparison]['b']
                    
                    data_A = df_umap_results[df_umap_results['labels'] == label_A]
                    data_B = df_umap_results[df_umap_results['labels'] == label_B]
                    grid_A, _, _, _ = binned_statistic_2d(data_A[0], data_A[1], None, statistic='count', bins=bins)
                    grid_B, _, _, _ = binned_statistic_2d(data_B[0], data_B[1], None, statistic='count', bins=bins)
                    
                    
                    
                    grid_A = grid_A / np.sum(grid_A)
                    grid_B = grid_B / np.sum(grid_B)
                    overlap = np.sum(np.minimum(grid_A, grid_B))
                    measures[comparison] = overlap
                
                self.metrics_pd = pd.concat([self.metrics_pd, pd.DataFrame([measures])], ignore_index=True)

        if "3D" in UMAP_components:
            embedding = umap.UMAP(n_components=3, metric='euclidean', random_state=42, repulsion_strength=1.0) 
            data4fit = self.df_data_selected.drop(columns=['labels'])
            umap_results = embedding.fit_transform(data4fit)
            dftest = pd.DataFrame(umap_results)
            dftest['labels'] = self.df_data_selected['labels']

            fig = plt.figure(figsize=(16, 7))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(
                dftest[0], dftest[1], dftest[2],
                c=dftest["labels"].map(self.color_list),
                alpha=0.2
            )
            filename = Path(self.path_folder, f"UMAP_3D_plot_{self.suffix}.png")
            plt.savefig(filename)
            plt.close()
            plt.cla()
            plt.clf()
    
    
            
    def comparison_swarm_distributions(self, subset_size=400):
        labels = self.df_data_selected['labels'].unique()
        subsets = [self.df_data_selected.loc[self.df_data_selected['labels'] == label].iloc[:subset_size] for label in labels]
        
        #if include_labels:
        #    subsets += [df.loc[df[label_column] == label].iloc[:subset_size] for label in include_labels]
        df_swarm = pd.concat(subsets, ignore_index=True)
        
        
        
        #if self.use_copula:
        #    df_c = self.df_data_selected.loc[self.df_data_selected['label'] == "cop"].iloc[0:400]
        #    df_swarmplot= pd.concat([df_a, df_b, df_c])
        #else:
        #    df_swarmplot= pd.concat([df_a, df_b])
        
        
        fig, axs = plt.subplots(figsize=(140,20), ncols=self.univar_count)
        for i in range(self.univar_count):
            sns.violinplot(data=self.df_data_selected[[f'c_{i}',"labels"]], y=f'c_{i}', x="labels", palette=self.color_list, hue="labels", ax=axs[i])
            sns.swarmplot(data=df_swarm[[f'c_{i}',"labels"]], y=f'c_{i}', x="labels", palette=self.color_list, size=3, ax=axs[i])
        
        filename = Path(self.path_folder, f"SWARM_plot_{self.suffix}.png")
        fig.savefig(filename)  
        plt.close()
        plt.cla()
        plt.clf()      

class CorrelationComparison():

    def __init__(self, correlation_matrices, folder):
        self.dict_matrices = correlation_matrices
        self.path_fold = Path(folder,"correlation_comparison")
        if not os.path.exists(self.path_fold):
            os.makedirs(self.path_fold)
        
    
    def compareMatrices(self, list_comparisons):
        df = pd.DataFrame()
        for (key_a,key_b) in list_comparisons:
            frobenius_val = self.frobenius_norm(key_a,key_b)
            spearmanr_val = self.spearmanr(key_a,key_b)
            cosin_sim_val = self.cosineSimilarity(key_a,key_b)

            new_row = {'matrix_A':self.keyToSting(key_a), 'matrix_B':self.keyToSting(key_b), 'frobenius':frobenius_val,'spearmanr_statistic':spearmanr_val[0],'spearmanr_pvalue':spearmanr_val[1], 'cosineSimilarity':cosin_sim_val}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        csv_path = Path(self.path_fold, 'correlation_comparison.csv')
        df.to_csv(csv_path)

    def keyToSting(self, key):
        key_0 = key[0]
        key_1 = key[1]
        return f"{key_0}__{key_1}"

    def get_matrix(self, key):
        key_0 = key[0]
        key_1 = key[1]        

        return self.dict_matrices[key_0][key_1]

    def frobenius_norm(self, key_a, key_b):
        matrix1 = self.get_matrix(key_a)
        matrix2 = self.get_matrix(key_b)
        diff_matrix = matrix1 - matrix2
        squared_diff = np.square(diff_matrix)
        sum_squared_diff = np.sum(squared_diff)
        frobenius_norm = np.sqrt(sum_squared_diff)
        return frobenius_norm

    def spearmanr(self, key_a, key_b):
        matrix1 = self.get_matrix(key_a)
        matrix2 = self.get_matrix(key_b) 
        matrix1_top = self.upper(matrix1)
        matrix2_top = self.upper(matrix2)
        significance = stats.spearmanr(matrix1_top, matrix2_top)
        return significance

    def cosineSimilarity(self, key_a, key_b):
        matrix1 = self.get_matrix(key_a)
        matrix2 = self.get_matrix(key_b)
        matrix1_flat = np.concatenate(matrix1).ravel()
        matrix2_flat = np.concatenate(matrix2).ravel()
        cos_sim = dot(matrix1_flat, matrix2_flat)/(norm(matrix1_flat)*norm(matrix2_flat))
        return cos_sim

    def upper(self, df):
        '''Returns the upper triangle of a correlation matrix.
        You can use scipy.spatial.distance.squareform to recreate matrix from upper triangle.
        Args:
        df: pandas or numpy correlation matrix
        Returns:
        list of values from upper triangle
        '''
        try:
            assert(type(df)==np.ndarray)
        except:
            if type(df)==pd.DataFrame:
                df = df.values
            else:
                raise TypeError('Must be np.ndarray or pd.DataFrame')
        mask = np.triu_indices(df.shape[0], k=1)
        return df[mask]


class SparseNDHistogram():
    def __init__(self, n_bins, n_comp):
        self.data = {}
        self.n_bins = n_bins
        self.n_comp = n_comp
        self.edges = None
        self.histograms = None
        
    def increment(self, indices):
        if indices in self.data:
            self.data[indices] += 1
        else:
            self.data[indices] = 1

    def get(self, indices):
        return self.data.get(indices, 0)

    def get_indices(self):
        return list(self.data.keys())
    
    def get_indices_values(self):
        val_list = []
        for key in self.data.keys():
            val_list.append(self.data[key])
        return val_list
    
    def get_data(self):
        return self.data

    def compute_edges(self, *datasets):
        edges = []
        for i in range(self.n_comp):
            min_val = np.array(min(data[:, i].astype(np.float64).min() for data in datasets))
            max_val = np.array(max(data[:, i].astype(np.float64).max() for data in datasets))

            edges.append(np.linspace(min_val, max_val, self.n_bins + 1))  # Creazione bin
        self.edges = edges
        return self.edges

    def set_edges(self, edges):
        self.edges = edges
    
    def print_edges(self):
        return self.edges
            
    def compute_histogram(self, data):
        if self.edges is not None:
            for point in data:
                bin_indices = tuple(max(0, np.digitize(point[i], self.edges[i], right=True) - 1) for i in range(self.n_comp))
                self.increment(bin_indices)
        
    @classmethod
    def from_datasets(cls, n_bins, n_comp, *datasets):        
        histograms = [cls(n_bins, n_comp) for _ in datasets]
        histograms[0].compute_edges(*datasets)
        
        for hist in histograms:
            hist.set_edges(histograms[0].edges)
        for hist, data in zip(histograms, datasets):
            hist.compute_histogram(data)

        return histograms 

    @classmethod
    def compute_error(cls, histograms, error_name=["MAE", "RMSE"], matrix_selected=[1, 2]):
        print("matrix_selected:\t",matrix_selected[0],matrix_selected[1])
        if len(matrix_selected) != 2:
            raise ValueError("Devi selezionare esattamente due istogrammi per calcolare l'errore.")

        hist1, hist2 = histograms[matrix_selected[0]], histograms[matrix_selected[1]]
        indices = set(hist1.get_indices()).union(set(hist2.get_indices()))

    
        error_sq_sum = 0
        error_abs_sum = 0
        a = 0
        for index in indices:
            val1 = hist1.get(index)  # Se non c'è, restituisce 0 di default
            val2 = hist2.get(index)  # Se non c'è, restituisce 0 di default
            diff = val1 - val2
            error_sq_sum += diff ** 2
            error_abs_sum += abs(diff)
            a += 1
        n = len(indices) if indices else 1  # Evita divisione per 0
        
        errors_values = dict()
        if "RMSE" in error_name:
            errors_values["RMSE"] = np.sqrt(error_sq_sum / n)
        if "MAE" in error_name:
            errors_values["MAE"] = error_abs_sum / n
        
        return errors_values
       