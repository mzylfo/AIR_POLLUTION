import os
from pathlib import Path
from src.NeuroCorrelation.ModelTraining.ModelTraining import  ModelTraining
from skopt import Optimizer
from skopt.plots import plot_gaussian_process
from skopt.space import Space, Real, Categorical, Integer
from skopt import gp_minimize
from skopt.utils import point_asdict
from skopt.plots import plot_gaussian_process
from termcolor import colored, cprint 
from src.NeuroCorrelation.Optimization.Optimization_functions import Optimization_functions
from colorama import init, Style
import matplotlib.pyplot as plt
from skopt.plots import plot_evaluations, plot_objective, plot_convergence
import pandas as pd


class Optimization():
    
    def __init__(self, model, device, data_dict, loss, path_folder, univar_count, batch_size, dataGenerator, latent_dim, vc_mapping, learning_rate, input_shape, rangeData, time_performance, timeweather_count, model_type=None,  instaces_size_noise=None, direction="maximize", timeout=600, graph_topology=False, edge_index=None):
        self.model = model
        self.device = device
        self.train_data = data_dict['train_data']
        self.test_data = data_dict['test_data']
        self.noise_data = data_dict['noise_data']
        
        print("-------------------------Optimization",model_type,"|")
        self.model_type = model_type
        self.loss_obj = loss
        self.path_folder = path_folder
        self.univar_count = univar_count
        self.batch_size = batch_size
        self.dataGenerator = dataGenerator
        self.direction = direction
        self.timeout = timeout
        self.load_model = False
        self.lat_dim = latent_dim
        self.vc_mapping = vc_mapping
        self.input_shape = input_shape
        self.rangeData = rangeData
        self.opt = None
        self.graph_topology = graph_topology
        self.edge_index = edge_index
        self.bestResult = dict({"params":None, "fun_val":None})
        self.time_performance = time_performance
        self.search_space = {"keys":list(), "keys_param":list(), "space":list(), "network_part":list()}
        self.objectFun = Optimization_functions()
        self.results = dict()
        self.timeweather_count = timeweather_count
        self.learning_rate =learning_rate
        
    def set_epochs(self, epochs):
        self.epochs = epochs
        
    def set_modeltype(self, model_type):
        self.model_type = model_type
        
    #
    # search_space = [{"type":"Categorical","min":0,"max":1, "values_list":[0,1,2], "name":"cat"},{"type":"Integer","min":0,"max":1, "values_list":[], "name":"int"}]
    #
    def set_searchSpace(self, search_space):
        
        for space_vals in search_space:
            space = None
            key = None
            if space_vals["type"]== "Categorical":
                space = Categorical(space_vals["values_list"], name=f"{space_vals['name']}__{space_vals['param']}")
            elif space_vals["type"]== "Integer":
                space = Integer(space_vals["min"],space_vals["max"], name=f"{space_vals['name']}__{space_vals['param']}")
            elif space_vals["type"]== "Real":
                space = Real(space_vals["min"],space_vals["max"], name=f"{space_vals['name']}__{space_vals['param']}")
            else:
                print("space not recornized")
            if space is not None:  
                self.search_space["keys"].append(space_vals["name"])
                self.search_space["keys_param"].append(space_vals["param"])
                self.search_space["space"].append(space)
                self.search_space["network_part"].append(space_vals['network_part'])
            
    def set_optimizer(self, base_estimator="GP", n_initial_points=10):        
        #base_estimator= "GP", "RF", "ET", "GBRT"
        print("\tcreate optimizer")
        self.opt = Optimizer(dimensions=self.search_space["space"], base_estimator=base_estimator, n_initial_points=n_initial_points, acq_optimizer="sampling")
        
    def set_optimization_fun(self, opt_fun):
        self.objectFun.set_object_fun(opt_fun)

    def set_n_calls(self, n_calls):
        self.n_calls = n_calls

    def set_fromDict(self, opt_dict):
        print("-----------------------------------------------set_fromDict")
        self.set_epochs(opt_dict['epochs'])
        self.set_modeltype(opt_dict['modeltype'])
        self.set_optimization_fun(opt_dict['optimization_function'])
        self.set_searchSpace(opt_dict['search_space'])
        self.set_n_calls(opt_dict['n_calls'])
        self.set_optimizer(opt_dict['base_estimator'],opt_dict['n_initial_points'])
        self.scorses_dict_opt = dict()
    

    def optimization(self):
        print("**************")
        print("OPTIMIZATION PHASE:\t",self.model_type)
        print("**************")
        
        if self.model_type is None:
            raise Exception("Optimizator - model_type not set.")
        print("\tbegin optimization")
        
        for trial in range(self.n_calls[self.model_type]):
            if trial%2==0:
                cprint_cls = "blue"
            elif trial%2==1:
                cprint_cls = "cyan"
                
            cprint(f"OPTIMIZATION TRIAL #\t{trial}/{self.n_calls[self.model_type]}", cprint_cls, end="\n")
            next_x = self.opt.ask()
            print("\t\t\tpoint values: ",next_x)
            print(next_x)
            # This part of the code is iterating over the search space defined for optimization. It is
            # looping through the keys, keys_param, network_part, and values (next_x) simultaneously
            # using the `zip` function.
            for key ,keys_param, network_part, val in zip(self.search_space["keys"], self.search_space["keys_param"], self.search_space["network_part"], next_x): 
                if key=="loss":
                    self.loss_obj[network_part].loss_change_coefficent(keys_param, val)
                elif  key=="loss_optimizer":
                    print("loss_optimizer -\t",key,"--",val)
            
            self.training_obj = ModelTraining(model=self.model[self.model_type], device=self.device, loss_obj=self.loss_obj[self.model_type], 
                epoch=self.epochs, train_data=self.train_data, test_data=self.test_data, dataGenerator=self.dataGenerator, time_performance=self.time_performance,
                path_folder=self.path_folder, univar_count_in = self.univar_count, univar_count_out = self.univar_count, timeweather_count=self.timeweather_count,
                latent_dim=self.lat_dim, vc_mapping=self.vc_mapping, input_shape=self.input_shape, rangeData=self.rangeData, optimization=True, learning_rate=self.learning_rate,
                model_type=self.model_type, pre_trained_decoder=False, batch_size=self.batch_size, graph_topology=self.graph_topology,edge_index=self.edge_index, noise_data=self.noise_data)
            
            
            cprint(f"OPTIMIZATION TRIAL values tested", cprint_cls, end="\n")
            cprint(f"\tloss coefficients", cprint_cls, end="\n")
            cprint(f"{self.loss_obj[self.model_type].get_lossTerms()}", cprint_cls, end="\n")
            
            if self.model_type =="AE":
                values_res = self.training_obj.training(training_name=f"OPT_{trial}",model_flatten_in=True,load_model=False, optimization=True, optimization_name=trial)
                optim_score_dict = self.objectFun.get_score(values = values_res)
                optim_score = optim_score_dict['all']
            elif self.model_type =="GAN":
                values_res = self.training_obj.training(training_name=f"OPT_{trial}",noise_size=1, optimization=False, load_model=False)
                optim_score_dict = self.objectFun.get_score(values = values_res)
                optim_score = optim_score_dict['all']
            elif self.model_type =="VAE":
                values_res = self.training_obj.training(training_name=f"OPT_{trial}",model_flatten_in=True,load_model=False, optimization=True, optimization_name=trial)
                optim_score_dict = self.objectFun.get_score(values = values_res)
                optim_score = optim_score_dict['all']
            self.scorses_dict_opt[trial] = optim_score_dict
            self.training_obj.eval()
            self.results[trial] = {"params":self.loss_obj[self.model_type].get_lossTerms(), "fun_val":optim_score}
            cprint(f"\t\tpoint score:\t{optim_score}",  cprint_cls, end="\n")
            cprint(f"--------------------------------", cprint_cls, end="\n")
            self.opt.tell(next_x, optim_score)
        
        cprint(Style.BRIGHT + "OPTIMIZATION RESULT:" + Style.RESET_ALL, 'red')
        
        cprint(f"Best parameters:\t{self.opt.get_result().x}",  "red", end="\n")
        cprint(f"Best objective value:\t {self.opt.get_result().fun}",  "red", end="\n")
        cprint(f"--------------------------------------------------------------------",  "red", end="\n")
        self.setBestResult(self.search_space, self.opt.get_result().x,self.opt.get_result().fun)
        self.saveBestResult()
        self.saveResult()
        self.visualization_plots()
        #plot_gaussian_process(self.opt.get_result(), **plot_args) 
        print("\t: end optimization")
    
    def saveBestResult(self):
        path_opt = Path(self.path_folder, self.model_type,"Optimizations") 
        if not os.path.exists(path_opt):
            os.makedirs(path_opt)
        path_opt_best = Path(path_opt, "best_result.csv")
        best_df = pd.DataFrame(self.bestResult)
        best_df.to_csv(path_opt_best, index=False)
    
    def saveResult(self):
        path_opt = Path(self.path_folder, self.model_type,"Optimizations") 
        if not os.path.exists(path_opt):
            os.makedirs(path_opt)
        path_opt_best = Path(path_opt, "results.csv")
        best_df = pd.DataFrame(self.results)
        best_df.to_csv(path_opt_best, index=False)
        
        
        path_opt_scores = Path(path_opt, "scores.csv")
        scores_df = pd.DataFrame(self.scorses_dict_opt)
        scores_df.to_csv(path_opt_scores, index=True)
        
        
        
    def visualization_plots(self):
        opt_result = self.opt.get_result()
        
        self.path_opt_plots = Path(self.path_folder, self.model_type,"Optimizations") 
        if not os.path.exists(self.path_opt_plots):
            os.makedirs(self.path_opt_plots)
        
        path_evaluations_plot = Path(self.path_opt_plots, 'plot_evaluations.png')
        plot_evaluations(opt_result)
        plt.savefig(path_evaluations_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(opt_result.models)
        if opt_result.models:
            path_objective_plot = Path(self.path_opt_plots, 'plot_objective.png')
            plot_objective(opt_result)
            plt.savefig(path_objective_plot, dpi=300, bbox_inches='tight')
            plt.close()
        
        path_convergence_plot = Path(self.path_opt_plots, 'plot_convergence.png')
        plot_convergence(opt_result)
        plt.savefig(path_convergence_plot, dpi=300, bbox_inches='tight')
        plt.close()
        
        
        '''
        plot_evaluations: 
        This function will generate a series of subplots, each representing one dimension of the search space. 
        Each subplot shows how the objective function values are distributed across the range of that dimension. 
        Given 8 variables, the plot will consist of 8 subplots, each providing insights into how different values of one variable affect the objective function.
        
        plot_objective: This function creates a pairwise plot matrix where each plot shows the relationship between two variables in the search space and how they jointly affect the objective function. 
        For 8 variables, the plot matrix will be 8x8, where the diagonal elements show the distribution of the objective function values for each variable, and the off-diagonal elements show pairwise interactions.
        '''
    
    def setBestResult(self, r_space, params, fun_val):
        if self.bestResult["fun_val"] is None:
            self.bestResult["fun_val"] = fun_val
            
            self.bestResult["params"] = dict()
            for key ,keys_param, network_part, val in zip(r_space["keys"], r_space["keys_param"], r_space["network_part"], params): 
                if network_part not in self.bestResult["params"]:
                    self.bestResult["params"][network_part] = dict()
                if key not in self.bestResult["params"][network_part]:
                    self.bestResult["params"][network_part][key] = dict()
                self.bestResult["params"][network_part][key][keys_param] = val
            
        elif fun_val < self.bestResult["fun_val"]:
            self.bestResult["fun_val"] = fun_val
            self.bestResult["params"] = dict()
            for key ,keys_param, network_part, val in zip(r_space["keys"], r_space["keys_param"], r_space["network_part"], params): 
                if network_part not in self.bestResult["params"]:
                    self.bestResult["params"][network_part] = dict()
                if key not in self.bestResult["params"][network_part]:
                    self.bestResult["params"][network_part][key] = dict()                    
                self.bestResult["params"][network_part][key][keys_param] = val
    
    
        
        
    def getBestResult(self):
        return self.bestResult["params"]
    
    def setValuesOptimized(self, loss_obj, model_type):
        cprint(Style.BRIGHT + "OPTIMIZATION SET VALUES:" + Style.RESET_ALL, 'blue', end="\n")
        
        for network_part in self.bestResult["params"]:
            for key in self.bestResult["params"][network_part]:
                for keys_param in self.bestResult["params"][network_part][key]:
                    if key=="loss":
                        val = self.bestResult["params"][network_part][key][keys_param]
                        if model_type == network_part:
                            loss_obj.loss_change_coefficent(keys_param, val)
                            cprint(Style.BRIGHT + f"\t{key}\t{network_part}\t:\t{keys_param}\t-\t{val}" + Style.RESET_ALL, 'blue', end="\n")
                    elif  key=="loss_optimizer":
                        print("loss_optimizer -- \t",key,"--",val)
        cprint(Style.BRIGHT + "--------------------------------------------------------------------" + Style.RESET_ALL, 'blue', end="\n")