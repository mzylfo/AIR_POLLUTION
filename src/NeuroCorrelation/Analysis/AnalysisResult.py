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
import matplotlib.ticker as plticker
from matplotlib.lines import Line2D
from matplotlib import collections as matcoll

class AnalysisResult():
    
    def __init__(self, univar_count):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn("deprecated", DeprecationWarning)
        np.seterr(divide='ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)
        self.rand_var_in = dict()
        self.rand_var_out = dict()
        self.rand_var_cop = dict()
        self.TSNE_components = 2
        self.univar_count = univar_count
        self.n_sample_considered = 0
                
    
    
    def comparison_wass(self, folder, n_run):        
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
            dist_copu = self.rand_var_cop[i]

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
        self.tab_ae[f"{n_run}"] = ws_pd_ae[1]
        self.tab_cop[f"{n_run}"] = ws_pd_cop[1]


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
    
    
    def comparison_3distr(self, folder):
        color_list = {"real": "red","ae":"green","cop":"blue"}
        label_list = {"real": "real data","ae":"GAN+AE gen","cop":"copula gen"}
        folder_compares = Path(folder,"comparison_distributions")
        if not os.path.exists(folder_compares):
            os.makedirs(folder_compares)

        for i in range(self.univar_count):
            real_vals = self.rand_var_in[i]
            neur_vals = self.rand_var_cop[i]
            copu_vals = self.rand_var_out[i]
            
            plt.figure(figsize=(12,8))  
            plt.xlim([0, 1])
            plt.ylim([0, 1])

            plot_name = "comparison_3distr"
            title_txt = f"{plot_name} - vc: {i}"
            plt.title(title_txt)
            
            #real 
            mean_val = np.mean(real_vals)
            var_val = np.var(real_vals)
            std_val = np.std(real_vals)
            plt.axvline(x = mean_val, linestyle="solid", color = color_list["real"], label = "real_mean")
            plt.axvline(x = (mean_val-std_val), linestyle="dashed", color = color_list["real"], label = "real_var")
            plt.axvline(x = (mean_val+std_val), linestyle="dashed", color = color_list["real"], label = "real_var")
            mean_plt_txt = f"      real data  mean: {mean_val:.3f}"
            plt.text(mean_val, 0, s=mean_plt_txt, rotation = 90) 
            
            #neural
            mean_val = np.mean(neur_vals)
            var_val = np.var(neur_vals)
            std_val = np.std(neur_vals)
            plt.axvline(x = mean_val, linestyle="solid", color = color_list["ae"], label = "GAN+AE gen_mean")
            plt.axvline(x = (mean_val-std_val), linestyle="dashed", color = color_list["ae"], label = "GAN+AE gen_var")
            plt.axvline(x = (mean_val+std_val), linestyle="dashed", color = color_list["ae"], label = "GAN+AE gen_var")
            mean_plt_txt = f"      GAN+AE gen  mean: {mean_val:.3f}"
            plt.text(mean_val, 0, s=mean_plt_txt, rotation = 90) 
            
            #copula
            mean_val = np.mean(copu_vals)
            var_val = np.var(copu_vals)
            std_val = np.std(copu_vals)
            plt.axvline(x = mean_val, linestyle="solid", color = color_list["ae"], label = "copula gen_mean")
            plt.axvline(x = (mean_val-std_val), linestyle="dashed", color = color_list["cop"], label = "copula gen_var")
            plt.axvline(x = (mean_val+std_val), linestyle="dashed", color = color_list["cop"], label = "copula gen_var")
            mean_plt_txt = f"      copula gen  mean: {mean_val:.3f}"
            plt.text(mean_val, 0, s=mean_plt_txt, rotation = 90) 
            

            #real
            plt.xlim(xmin=0, xmax = 100)
            plt.hist(real_vals, weights=np.ones(len(real_vals)) / len(real_vals), histtype='stepfilled', alpha = 0.2, color= color_list["real"], label="real")
            plt.hist(real_vals, weights=np.ones(len(real_vals)) / len(real_vals), histtype=u'step', edgecolor="gray", fc="None", lw=1)
            
            #real
            plt.xlim(xmin=0, xmax = 100)
            plt.hist(neur_vals, weights=np.ones(len(neur_vals)) / len(neur_vals), histtype='stepfilled', alpha = 0.2, color= color_list["ae"], label="GAN+AE gen")
            plt.hist(neur_vals, weights=np.ones(len(neur_vals)) / len(neur_vals), histtype=u'step', edgecolor="gray", fc="None", lw=1)
            
            #real
            plt.xlim(xmin=0, xmax = 100)
            plt.hist(copu_vals, weights=np.ones(len(copu_vals)) / len(copu_vals), histtype='stepfilled', alpha = 0.2, color= color_list["cop"], label="copula gen")
            plt.hist(copu_vals, weights=np.ones(len(copu_vals)) / len(copu_vals), histtype=u'step', edgecolor="gray", fc="None", lw=1)
            
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

                
            plt.xlim(xmin=0, xmax = 100)

            plt.legend(loc='upper right')
            filename = Path(folder_compares,f"plot_vc_distribution_{plot_name}_{i}.png")
            plt.savefig(filename)
            
            
            
            
    
    def comparison_tsne(self, folder, n_points=None):
        color_list = {"real": "red","ae":"green","cop":"blue"}
        label_list = {"real": "real data","ae":"GAN+AE gen","cop":"copula gen"}
        
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
        
        print(dftest)
        raise Exception("dftest")
        
        
        fig = plt.figure(figsize=(16,7))
        sns.scatterplot(
            x=0, y=1,
            hue="label",
            data=dftest,
            alpha=0.2,
            legend="full"
        )
        filename = Path(folder, "plot_TSNE.png")
        plt.savefig(filename)
        
        #fig, ax = plt.subplots(figsize=(14,8))
        fig, axs = plt.subplots(figsize=(140,20), ncols=self.univar_count)
        color_list = {"real": "red","ae":"green","cop":"blue"}
        
        df_tsne['label'] = labels
        
        df_a = df_tsne.loc[df_tsne['label'] == "real"].iloc[0:200]
        df_b = df_tsne.loc[df_tsne['label'] == "ae"].iloc[0:200]
        df_c = df_tsne.loc[df_tsne['label'] == "cop"].iloc[0:200]
        df_swarmplot= pd.concat([df_a, df_b, df_c])
        for i in range(self.univar_count):
            sns.violinplot(data=df_tsne[[f'c_{i}',"label"]], y=f'c_{i}', x="label", palette=color_list, hue="label", ax=axs[i])
            sns.swarmplot(data=df_swarmplot[[f'c_{i}',"label"]], y=f'c_{i}', x="label",palette=color_list, size=3, ax=axs[i])
            
        filename = Path(folder, "plot_swarmplot.png")
        fig.savefig(filename)        
        
        
    def mean_greater(x):
        df1 = pd.DataFrame('', index=x.index, columns=x.columns)
        m3 = x['diff_real_ea'] > x['diff_real_cop']
        m4 = x['diff_real_cop'] > x['diff_real_ea']
        #df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
        #rewrite values by boolean masks
        df1['diff_real_ea'] = np.where(m3, f' font-weight:bold', df1['diff_real_ea'])
        df1['mean_AE'] = np.where(m3, f' font-weight:bold', df1['mean_AE'])
        df1['std_AE'] = np.where(m3, f' font-weight:bold', df1['std_AE'])

        df1['diff_real_cop'] = np.where(m4, f' font-weight:bold', df1['diff_real_cop'])
        df1['mean_COP'] = np.where(m4, f' font-weight:bold', df1['mean_COP'])
        df1['std_COP'] = np.where(m4, f' font-weight:bold', df1['std_COP'])

        return df1 
    
    def mean_greater_wass(x):

        df1 = pd.DataFrame('', index=x.index, columns=x.columns)

        m3 = x['ea_mean'] > x['cop_mean']
        m4 = x['cop_mean'] > x['ea_mean']

        #df1 = pd.DataFrame('background-color: ', index=x.index, columns=x.columns)
        #rewrite values by boolean masks
        df1['ea_mean'] = np.where(m3, f' font-weight:bold', df1['ea_mean'])
        df1['ea_std'] = np.where(m3, f' font-weight:bold', df1['ea_std'])


        df1['cop_mean'] = np.where(m4, f' font-weight:bold', df1['cop_mean'])
        df1['cop_std'] = np.where(m4, f' font-weight:bold', df1['cop_std'])

        return df1

    def compute_statscomparison(self, folder, exp_name, n_run):
        self.tab_stats = pd.DataFrame(columns=["variable","mean_real","std_real","mean_AE","std_AE","mean_COP","std_COP"])
        self.tab_ae = pd.DataFrame()
        self.tab_cop = pd.DataFrame()
        for i in n_run:
            print("n_run:\t",i,"/",n_run)
            folder_run = Path(folder,f"{exp_name}_{i}")
            pd_stats = self.comparison_wass(folder=folder_run, n_run=i)
            #self.comparison_3distr(folder=folder_run)
            self.comparison_tsne(folder=folder_run, n_points=1000)
            if self.tab_stats.empty:
                self.tab_stats = pd_stats
            else:
                self.tab_stats = self.tab_stats.add(pd_stats, fill_value=0)
        """self.tab_stats = self.tab_stats.div(n_run)
        
        self.tab_stats.to_csv(Path(folder,f"{exp_name}_stats.csv"), sep='\t')
        self.tab_ae.to_csv(Path(folder,f"{exp_name}_ae_wass.csv"), sep='\t')
        self.tab_cop.to_csv(Path(folder,f"{exp_name}_cop_wass.csv"), sep='\t')
        
        ea_wass =  self.tab_ae
        ea_wass['variable'] = ["#run"]+[i for i in range(self.univar_count)]
        cop_wass = self.tab_cop
        cop_wass['variable'] = ["#run"]+[i for i in range(self.univar_count)]
        stats_ea_cop = self.tab_stats
        stats_ea_cop = stats_ea_cop.astype(float)
        stats_ea_cop['variable'] = [i for i in range(self.univar_count)]

        ea_wass = ea_wass.set_index('variable')
        cop_wass = cop_wass.set_index('variable')
        stats_ea_cop = stats_ea_cop.set_index('variable')
        stats_ea_cop = stats_ea_cop.round(3)
        
        stats_ea_cop = stats_ea_cop.applymap(lambda x: f'{x}')
        stats_ea_cop.style.apply(self.mean_greater, axis=None)
        
        stats_ea_cop_st = stats_ea_cop.reset_index(inplace=False)
        stats_ea_cop_st_real = stats_ea_cop_st[["variable","mean_real","std_real"]].rename(columns={"mean_real": "mean", "std_real": "std"})
        stats_ea_cop_st_real['quality'] = ["real" for i in range(self.univar_count)]
        stats_ea_cop_st_ae = stats_ea_cop_st[["variable","mean_AE","std_AE"]].rename(columns={"mean_AE": "mean", "std_AE": "std"})
        stats_ea_cop_st_ae['quality'] = ["ae" for i in range(self.univar_count)]
        stats_ea_cop_st_cop = stats_ea_cop_st[["variable","mean_COP","std_COP"]].rename(columns={"mean_COP": "mean", "std_COP": "std"})
        stats_ea_cop_st_cop['quality'] = ["cop" for i in range(self.univar_count)]


        stats_ea_cop_st_all = pd.concat([stats_ea_cop_st_real,stats_ea_cop_st_ae,stats_ea_cop_st_cop])

        stats_ea_cop_st_all['mean'] = stats_ea_cop_st_all['mean'].astype(float)
        stats_ea_cop_st_all['std'] = stats_ea_cop_st_all['std'].astype(float)
        stats_ea_cop_st_all['variable'] = stats_ea_cop_st_all['variable'].astype(int)
        
        
        fig, ax = plt.subplots(figsize=(14,8))        
        for i in range(self.univar_count):
            plt.axvline(x=0.6+i, color='lightgray')


        a = 0
        color_list = {"real": "red","ae":"green","cop":"blue"}
        label_list = {"real": "real data","ae":"GAN+AE gen","cop":"copula gen"}
        for key, group in stats_ea_cop_st_all.groupby('quality'):
            c = [i+a for i in range(self.univar_count)]
            group['variable'] = c

            group.plot(x='variable', y='mean', yerr='std', color=color_list[key], label=label_list[key], ax=ax, marker='.', markersize=15, linestyle='None')
            #ax.errorbar(group['variable'], data=group, y='mean', yerr='std')
            a += 0.20
        loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
        ax.xaxis.set_major_locator(loc)
        line = Line2D([0,1],[0,1],linestyle='-', color='r')
        plt.legend(loc="best")
        filename = Path(folder, "plot_distribution_error.png")
        plt.savefig(filename)
        
        
        ea_wass_st = ea_wass
        ea_wass_st = ea_wass_st.T
        cop_wass_st = cop_wass
        cop_wass_st = cop_wass_st.T


        fig, ax = plt.subplots(figsize=(14,8))
        x_val = list()
        x_val_ae = list()
        y_val_ae = list()
        x_val_cop = list()
        y_val_cop = list()

        for i in range(15):
            plt.axvline(x=0.6+i, color='lightgray')

        loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
        ax.xaxis.set_major_locator(loc)

        for i, col in enumerate(ea_wass_st.columns.values[:-1]):
            x_val_ae = x_val_ae + [ i-0.2 for j in range(len(ea_wass_st[col]))]
            y_val_ae = y_val_ae+ [ j for j in ea_wass_st[col]]
            x_val.append([ i for j in range(len(ea_wass_st[col]))])

        for i, col in enumerate(cop_wass_st.columns.values[:-1]):
            x_val_cop = x_val_cop + [ i+0.2 for j in range(len(cop_wass_st[col]))]
            y_val_cop = y_val_cop + [ j for j in cop_wass_st[col]]

        ax.scatter(x=x_val_ae, y=y_val_ae,c='green', label="GAN+AE")
        ax.scatter(x=x_val_cop, y=y_val_cop,c='blue', label="copula")
        plt.legend(loc="best")
        filename = Path(folder, "plot_distribution_ripetition.png")
        plt.savefig(filename)


        ea_mean = ea_wass.T.mean()
        ea_std = ea_wass.T.std()
        cop_mean = cop_wass.T.mean()
        cop_std = cop_wass.T.std()

        filename = Path(folder, "plot_distribution_ripetition.png")
        plt.savefig(filename)

        wass_compare = pd.DataFrame()
        wass_compare["ea_mean"] = ea_mean.astype(float)
        wass_compare["ea_std"] = ea_std.astype(float)
        wass_compare["cop_mean"] = cop_mean.astype(float)
        wass_compare["cop_std"] = cop_std.astype(float)
        wass_compare = wass_compare.applymap(lambda x: f'{x}')
        wass_compare.style.apply(self.mean_greater_wass, axis=None)
        filename_tex = Path(folder, "wass_compare.tex")
        with open(filename_tex, 'w') as tf:
            tf.write(wass_compare.to_latex())
        filename_csv = Path(folder, "wass_compare.csv")
        wass_compare.to_csv(filename_csv, sep='\t')
        """
        