import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
from pathlib import Path
import os

class DataComparison():

    def __init__(self, univar_count, path_fold):
        self.univar_count = univar_count
        self.path_fold = path_fold
    
    def get_varidName(self,var_id):
        num_digit = len(str(self.univar_count))
        name_univ = f'{var_id:0{num_digit}d}'
        return name_univ

    #data = {"input":{'data':{0:[],1:[],2:[],...],'color':'red'}, ..., "reconstructed":{'data':{0:[],1:[],2:[],...],'color':'blue'}}
    def data_comparison_plot(self, data, nameplot=None):
        if nameplot is not None:
            fold_name = f"{nameplot}_distribution_comparison"
        else:
            fold_name = f"univar_distribution_comparison"

        path_fold_dist = Path(self.path_folder, fold_name)
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)
        
        stats_dict = {"univ_id": []}

        for id_univar in range(self.univar_count):
            name_univ = self.get_varidName(id_univar)
            stats_dict['univ_id'].append(name_univ)

            for key in data:                
                list_values = data[key]['data'][id_univar]
                color_data = data[key]['color']
                name_data = key
                mean_val = np.mean(list_values)
                var_val = np.var(list_values)
                mean_label = f"{name_data}_mean"
                var_label = f"{name_data}_var"
                plt.axvline(x = mean_val, linestyle="solid", color = color_data, label = mean_label)
                plt.axvline(x = (mean_val-var_val), linestyle="dashed", color = color_data, label = var_label)
                plt.axvline(x = (mean_val+var_val), linestyle="dashed", color = color_data, label = var_label)

                mean_plt_txt = f"      {name_data} mean: {mean_val:.3f}"
                plt.text(mean_val, 0, s=mean_plt_txt, rotation = 90) 

                ax.hist(list_values, histtype='stepfilled', alpha = 0.2, color= color_data)
                ax.hist(y, histtype=u'step', edgecolor="gray", fc="None", lw=1)

                #histogram
                plt.hist(list_values, weights=np.ones(len(list_values)) / len(list_values), histtype='stepfilled', alpha = 0.2, color= color_data)
                #histogram border
                plt.hist(list_values, weights=np.ones(len(list_values)) / len(list_values), histtype=u'step', edgecolor="gray", fc="None", lw=1)

            
                plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

                if mean_label not in stats_dict:
                    stats_dict[mean_label]= list()
                stats_dict[mean_label].append(mean_val)


                if var_label not in stats_dict:
                    stats_dict[var_label]= list()
                stats_dict[var_label].append(var_val)

            filename = Path(path_fold_dist,name_univ+"distrib_comperison.png")
            plt.savefig(filename)

        filename = Path(path_fold_dist,fold_name+"_table.csv")
        stats_dict = pd.DataFrame.from_dict(stats_dict)
        stats_dict.to_csv(file_name, sep='\t', encoding='utf-8')

