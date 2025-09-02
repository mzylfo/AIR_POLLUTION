
from src.NeuroCorrelation.InstancesGenerationCore import InstancesGenerationCore
import os
import argparse
from pathlib import Path
from termcolor import cprint
from colorama import init, Style
import gc
import resource


class InstancesGeneration():

    def __init__(self, args):
        gb_lim = 10
        limit_resource=False
        
        if limit_resource:
            self.limit_resource(gb_lim)
        
        parser = argparse.ArgumentParser(description="Description of your program")
        print("------------")
        parser.add_argument('--num_case', type=str, required=True, help='Description for num_case')
        parser.add_argument('--experiment_name_suffix', type=str, required=True, help='Description for experiment_name_suffix')
        parser.add_argument('--main_folder', type=str, required=True, help='Description for main_folder')
        #parser.add_argument('--repeation', type=int, required=True, help='Description for repeation')
        parser.add_argument("--repeation_b", type=str, required=True, help='Description for repeation - begin')  
        parser.add_argument("--repeation_e", type=str, required=True, help='Description for repeation - end')  
        parser.add_argument('--optimization', type=self.str2bool, required=True, help='Enable or disable optimization')
        parser.add_argument('--load_model', type=self.str2bool, required=True, help='Description for load_model')
        parser.add_argument('--train_models', type=self.str2bool, required=True, help='Description for train_models')
        parser.add_argument('--time_slot', type=str, required=False, default=None, help='Description time slot')
        
        
        
        



        parsed_args = parser.parse_args(args)
        
        if len(vars(parsed_args)) < 3:
            exps = self.getExperimentsList(0)
            for exp in exps['experiments_list']:
                print("", exp['id'], "\t- ", exp['model_case'])
        else:
        
            b = int(parsed_args.repeation_b)
            e = int(parsed_args.repeation_e)
            parsed_repeation = list(range(b, e))  
            
            self.main(
                num_case = parsed_args.num_case,
                experiment_name_suffix = parsed_args.experiment_name_suffix,
                main_folder = parsed_args.main_folder,
                repeation = parsed_repeation,
                optimization = parsed_args.optimization,
                load_model = parsed_args.load_model,
                train_models = parsed_args.train_models,
                time_slot = parsed_args.time_slot
            )
    
    def limit_resource(self, gb_limit):
        memory_limit = gb_limit * 1024 * 1024 * 1024  # 1 GB in byte
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        cprint(Style.BRIGHT + "|************************|" + Style.RESET_ALL, 'magenta', attrs=["bold"])
        cprint(Style.BRIGHT +f"|  LIMIT RESOURCES " + Style.RESET_ALL, 'magenta', attrs=["bold"])
        cprint(Style.BRIGHT +f"|  ram       : {memory_limit} byte ({gb_limit} gb)" + Style.RESET_ALL, 'magenta', attrs=["bold"])
        cprint(Style.BRIGHT +f"|  device  : {None}" + Style.RESET_ALL, 'magenta', attrs=["bold"])
        cprint(Style.BRIGHT + "|************************|" + Style.RESET_ALL, 'magenta', attrs=["bold"])
        
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        print(f"Memory limit set to {gb_limit} GB.")
        print(f"Current limit (soft): {soft / (1024 * 1024 * 1024):.2f} GB")
        print(f"Current limit (hard): {hard / (1024 * 1024 * 1024):.2f} GB")

    
    def main(self, num_case, experiment_name_suffix, main_folder, repeation, lat_dim = None, load_model=None, time_slot=None, train_models="yes", optimization=False):
        path_folder = Path('data','neuroCorrelation_experiments',main_folder)
        experiments_name = f"{experiment_name_suffix}___{num_case}"
        
        if optimization:
            experiment_name = f"{experiments_name}_0"
            if train_models:
                self.loss_obj = self.experiment(num_case=num_case, main_folder=path_folder, seed=0, experiment_name=experiment_name, optimization=optimization, load_model=load_model, time_slot=time_slot, loss_obj=None)
        
            #id_experiments = int(num_case)
            #experiments_list = self.getExperimentsList(seed=0, optimization= optimization)
            #experiments_selected = experiments_list['experiments_list'][id_experiments]
            
        else:
            self.loss_obj = None
        
        for seed in repeation:
            experiment_name = f"{experiments_name}_{seed}"
            if train_models:
                self.experiment(num_case=num_case, main_folder=path_folder, seed=seed, experiment_name=experiment_name, optimization=False, load_model=load_model, time_slot=time_slot, loss_obj=self.loss_obj)
        
            #id_experiments = int(num_case)
            #experiments_list = self.getExperimentsList(seed=0, optimization= optimization)
            #experiments_selected = experiments_list['experiments_list'][id_experiments]
            
            gc.collect()


    def getExperimentsList(self, seed,optimization):
        experiments_list = [
            {"id":   0, "model_case":"autoencoder_3_copula_optimization", "epoch":{'AE':   3,'GAN':   2}, "univar_count": 7, "lat_dim": 3, "dataset_setting":{"batch_size":  32, "train_percentual":None,"starting_sample":None,"train_samples":None,"test_samples":None, "noise_samples":10000, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},

            #METR-LA DATASET
            {"id":    1, "case":"PEMS_METR", "model_case":"AE>GAN_linear_pretrained_METR_16",  "epoch":{'AE':    2,'GAN' :  1},  "univar_count":16,   "lat_dim":12,   "dataset_setting":{"batch_size": {'AE':   64,'GAN' :   64}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":    2, "case":"PEMS_METR", "model_case":"AE>GAN_linear_pretrained_METR_32",  "epoch":{'AE':  100,'GAN' :  100},  "univar_count":32,   "lat_dim":28,   "dataset_setting":{"batch_size": {'AE' :  64,'GAN' :   64}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":    3, "case":"PEMS_METR", "model_case":"AE>GAN_linear_pretrained_METR_48",  "epoch":{'AE':  100,'GAN' :  100},  "univar_count":48,   "lat_dim":36,   "dataset_setting":{"batch_size": {'AE':   64,'GAN' :   64}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":    4, "case":"PEMS_METR", "model_case":"AE>GAN_linear_pretrained_METR_64",  "epoch":{'AE':  100,'GAN' :  100},  "univar_count":64,   "lat_dim":54,   "dataset_setting":{"batch_size": {'AE':   64,'GAN' :   64}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},

            #PEMS-BAY DATASET
            {"id":    5, "case":"PEMS_METR", "model_case":"AE>GAN_linear_pretrained_PEMS_16",  "epoch":{'AE':  100,'GAN' :  100},  "univar_count":16,   "lat_dim":12,   "dataset_setting":{"batch_size": {'AE':  128,'GAN' :  128}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":    6, "case":"PEMS_METR", "model_case":"AE>WGAN_linear_pretrained_PEMS_16",  "epoch":{'AE': 150,'WGAN':  150},  "univar_count":16,   "lat_dim":12,   "dataset_setting":{"batch_size": {'AE':   64,'WGAN':   64}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":    7, "case":"PEMS_METR", "model_case":"AE>GAN_linear_pretrained_PEMS_32",  "epoch":{'AE':  100,'GAN' :  100},  "univar_count":32,   "lat_dim":28,   "dataset_setting":{"batch_size": {'AE':   64,'GAN' :   64}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":    8, "case":"PEMS_METR", "model_case":"AE>WGAN_linear_pretrained_PEMS_32",  "epoch":{'AE':  10,'WGAN' :  1000},  "univar_count":32,   "lat_dim":28,   "dataset_setting":{"batch_size": {'AE':64,'WGAN' : 256}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":    9, "case":"PEMS_METR", "model_case":"AE>GAN_linear_pretrained_PEMS_48",  "epoch":{'AE':  100,'GAN' :  100},  "univar_count":48,   "lat_dim":36,   "dataset_setting":{"batch_size": {'AE':   64,'GAN' :   64}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   10, "case":"PEMS_METR", "model_case":"AE>GAN_linear_pretrained_PEMS_64",  "epoch":{'AE':  100,'GAN' :  100},  "univar_count":64,   "lat_dim":54,   "dataset_setting":{"batch_size": {'AE':   64,'GAN' :   64}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},                                

            #CHENGDU_SMALLGRAPH DATASET
            ##LINEAR 
            {"id":   "11", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_16_A_linear",   "epoch":{'AE':   50,'GAN':  1},  "univar_count":16,   "lat_dim":6,   "dataset_setting":{"batch_size": {'AE':    256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "12", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_32_A_linear",   "epoch":{'AE':   50,'GAN':  1},  "univar_count":32,   "lat_dim":8,   "dataset_setting":{"batch_size": {'AE':    256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "13", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_48_A_linear",   "epoch":{'AE':   50,'GAN':  1},  "univar_count":48,   "lat_dim":10,   "dataset_setting":{"batch_size": {'AE':    256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "14", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_64_A_linear",   "epoch":{'AE':   50,'GAN':  1},  "univar_count":64,   "lat_dim":12,   "dataset_setting":{"batch_size": {'AE':    256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "15", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_96_A_linear",   "epoch":{'AE':   50,'GAN':  1},  "univar_count":96,   "lat_dim":16,   "dataset_setting":{"batch_size": {'AE':    256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "16", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_128_A_linear",  "epoch":{'AE':   50,'GAN':  1},  "univar_count":128,  "lat_dim":20,   "dataset_setting":{"batch_size": {'AE':    256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "17", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_192_A_linear",  "epoch":{'AE':   50,'GAN':  1},  "univar_count":192,  "lat_dim":24,   "dataset_setting":{"batch_size": {'AE':    256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "18", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_256_A_linear",  "epoch":{'AE':   50,'GAN':  1},  "univar_count":256,  "lat_dim":30,   "dataset_setting":{"batch_size": {'AE':    256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "19", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_512_A_linear",  "epoch":{'AE':   50,'GAN':  1},  "univar_count":512,  "lat_dim":32,   "dataset_setting":{"batch_size": {'AE':    256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
             
            #GRAPH
            {"id":   "gvae_0016", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_16_A_graph",   "epoch":{'AE':   50,'GAN':  1},  "univar_count":16,     "lat_dim":6,   "dataset_setting":{"batch_size": {'AE':   256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "gvae_0032", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_32_A_graph",   "epoch":{'AE':   50,'GAN':  1},  "univar_count":32,     "lat_dim":8,   "dataset_setting":{"batch_size": {'AE':   264,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "gvae_0048", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_48_A_graph",   "epoch":{'AE':   50,'GAN':  1},  "univar_count":48,     "lat_dim":10,   "dataset_setting":{"batch_size": {'AE':   264,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "gvae_0064", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_64_A_graph",   "epoch":{'AE':  100,'GAN':  1},  "univar_count":64,     "lat_dim":12,   "dataset_setting":{"batch_size": {'AE':   264,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "gvae_0096", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_96_A_graph",   "epoch":{'AE':  100,'GAN':  1},  "univar_count":96,     "lat_dim":16,   "dataset_setting":{"batch_size": {'AE':   256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "gvae_0128", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_128_A_graph",  "epoch":{'AE':  100,'GAN':  1},  "univar_count":128,    "lat_dim":24,   "dataset_setting":{"batch_size": {'AE':   256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "gvae_0192", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_192_A_graph",  "epoch":{'AE':  100,'GAN':  1},  "univar_count":192,    "lat_dim":64,   "dataset_setting":{"batch_size": {'AE':   256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "gvae_0256", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_256_A_graph",  "epoch":{'AE':  100,'GAN':  1},  "univar_count":256,    "lat_dim":90,   "dataset_setting":{"batch_size": {'AE':   256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "gvae_0512", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_512_A_graph",  "epoch":{'AE':  150,'GAN':  1},  "univar_count":512,    "lat_dim":130,  "dataset_setting":{"batch_size": {'AE':   256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "gvae_1024", "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_1024_A_graph", "epoch":{'AE':  150,'GAN':  1},  "univar_count":1024,   "lat_dim":96,   "dataset_setting":{"batch_size": {'AE':   256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            
            #B-VAE
            {"id":   "bvae_0016_fix", "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_16_A_graph_kl_fix",     "epoch":{'VAE':  100},  "univar_count":16,   "lat_dim":6,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "bvae_0016_lin", "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_16_A_graph_kl_lin",     "epoch":{'VAE':  100},  "univar_count":16,   "lat_dim":6,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "bvae_0016_cos", "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_16_A_graph_kl_cos",     "epoch":{'VAE':  100},  "univar_count":16,   "lat_dim":6,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},

            #B-VAE_SEOUL --> instances 1
            {"id":   "bvae_0016_seoul", "case":"SEOUL_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_16_A_graph_kl_fix",     "epoch":{'VAE':  100},  "univar_count":16,   "lat_dim":6,   "dataset_setting":{"batch_size": {'VAE':  16}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "bvae_0025_seoul", "case":"SEOUL_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_25_A_graph_kl_fix",     "epoch":{'VAE':  100},  "univar_count":25,   "lat_dim":8,   "dataset_setting":{"batch_size": {'VAE':   16}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            
            #B-VAE_SEOUL --> instances 1
            {"id":   "bvae_0016_seoul_2", "case":"SEOUL_SMALLGRAPH", "model_case":"DVAE_SEOUL_SMALLGRAPH_16",     "epoch":{'VAE':  100},  "univar_count":16,   "lat_dim":6,   "dataset_setting":{"batch_size": {'VAE':  16}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "bvae_0025_seoul_2", "case":"SEOUL_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_25_A_graph_kl_fix",     "epoch":{'VAE':  100},  "univar_count":25,   "lat_dim":8,   "dataset_setting":{"batch_size": {'VAE':   16}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            
            {"id":   "bvae_0032_fix", "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_32_A_graph_kl_fix",     "epoch":{'VAE':  100},  "univar_count":32,   "lat_dim":8,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "bvae_0032_lin", "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_32_A_graph_kl_lin",     "epoch":{'VAE':  100},  "univar_count":32,   "lat_dim":8,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "bvae_0032_cos", "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_32_A_graph_kl_cos",     "epoch":{'VAE':  100},  "univar_count":32,   "lat_dim":8,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            
            {"id":   "bvae_0048_fix", "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_48_A_graph_kl_fix",     "epoch":{'VAE':  100},  "univar_count":48,   "lat_dim":10,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "bvae_0048_lin", "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_48_A_graph_kl_lin",     "epoch":{'VAE':  100},  "univar_count":48,   "lat_dim":10,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "bvae_0048_cos", "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_48_A_graph_kl_cos",     "epoch":{'VAE':  100},  "univar_count":48,   "lat_dim":10,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            
            {"id":   "bvae_0064_fix", "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_64_A_graph_kl_fix",     "epoch":{'VAE':  50},  "univar_count":64,   "lat_dim":12,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "bvae_0096_fix", "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_96_A_graph_kl_fix",     "epoch":{'VAE':  150},  "univar_count":96,   "lat_dim":16,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "bvae_0128_fix", "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_128_A_graph_kl_fix",     "epoch":{'VAE': 150},  "univar_count":128,   "lat_dim":24,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            
            {"id":   "bvae_0192_fix", "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_192_A_graph_kl_fix",     "epoch":{'VAE':  100},  "univar_count":192,   "lat_dim":36,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   "bvae_0256_fix", "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_256_A_graph_kl_fix",     "epoch":{'VAE':  100},  "univar_count":256,   "lat_dim":52,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            
            
            
            {"id":   28, "case":"CHENGDU_SMALLGRAPH", "model_case":"CVAE_CHENGDU_SMALLGRAPH_16_A_graph_kl_lin",     "epoch":{'CVAE':  100},  "univar_count":16,   "lat_dim":8,   "dataset_setting":{"batch_size": {'CVAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            
            
            {"id":   28, "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_32_A_graph_kl_fix",     "epoch":{'VAE':  100},  "univar_count":32,   "lat_dim":8,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   29, "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_32_A_graph_kl_lin",     "epoch":{'VAE':  100},  "univar_count":32,   "lat_dim":8,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   30, "case":"CHENGDU_SMALLGRAPH", "model_case":"VAE_CHENGDU_SMALLGRAPH_32_A_graph_kl_cos",     "epoch":{'VAE':  100},  "univar_count":32,   "lat_dim":8,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            
            
            
            {"id":   24, "case":"CHENGDU_ZONE", "model_case":"AE_CHENGDU_URBAN_ZONE_0_graph",          "epoch":{'AE':  200,'GAN':  1},  "univar_count":248,   "lat_dim":80,   "dataset_setting":{"batch_size": {'AE':   2375,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            
            
            {"id":   "VAE_SP500_T3_lin", "case":"SP500", "model_case":"VAE_SP500_T3_linear",     "epoch":{'VAE':  100},  "univar_count":3,   "lat_dim":6,   "dataset_setting":{"batch_size": {'VAE':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            
        ]
        '''
            {"id":   21, "case":"CHENGDU_ZONE", "model_case":"AE_CHENGDU_URBAN_ZONE_1_graph",          "epoch":{'AE':  50,'GAN':  1},  "univar_count":240,   "lat_dim":60,   "dataset_setting":{"batch_size": {'AE':   2375,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   22, "case":"CHENGDU_ZONE", "model_case":"AE_CHENGDU_URBAN_ZONE_2_graph",          "epoch":{'AE':  50,'GAN':  1},  "univar_count":197,   "lat_dim":24,   "dataset_setting":{"batch_size": {'AE':   2375,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   23, "case":"CHENGDU_ZONE", "model_case":"AE_CHENGDU_URBAN_ZONE_3_graph",          "epoch":{'AE':  50,'GAN':  1},  "univar_count":268,   "lat_dim":24,   "dataset_setting":{"batch_size": {'AE':   2375,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   24, "case":"CHENGDU_ZONE", "model_case":"AE_CHENGDU_URBAN_ZONE_1-2_graph",        "epoch":{'AE':  50,'GAN':  1},  "univar_count":437,   "lat_dim":36,   "dataset_setting":{"batch_size": {'AE':   2375,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":2375, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   25, "case":"CHENGDU_ZONE", "model_case":"AE_CHENGDU_URBAN_ZONE_1-2_graph",        "epoch":{'AE':  50,'GAN':  1},  "univar_count":437,   "lat_dim":36,   "dataset_setting":{"batch_size": {'AE':   2375,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            
            
            {"id":  24, "model_case":"ESG__GAN_linear_pretrained_35",  "epoch":{'AE': 100,'GAN': 2},  "univar_count":35,   "lat_dim":30,   "dataset_setting":{"batch_size": {'AE': 2348,'GAN': 32}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": True, "input_shape":"vector"}
        '''
        
        optimization_settings = {
            "epochs":{'AE': 25,'GAN': 1, "VAE":25}, "n_calls":{'AE': 2,'GAN': 1, "VAE":30},"modeltype":"VAE", "base_estimator":"GP", "n_initial_points":10, "optimization_function":"wasserstein", 
            "search_space":[
                #{"type":"Real","min":5e-5,"max":2, "values_list":None, "name":"loss", "network_part":"AE", "param":"SPEARMAN_CORRELATION_LOSS"},
                #{"type":"Real","min":5e-5,"max":2, "values_list":None, "name":"loss", "network_part":"AE",  "param":"JENSEN_SHANNON_DIVERGENCE_LOSS"},
                #{"type":"Real","min":5e-5,"max":2, "values_list":None, "name":"loss", "network_part":"AE",  "param":"MEDIAN_LOSS_batch"},
                
                {"type":"Real","min":5e-5,"max":2, "values_list":None, "name":"loss", "network_part":"VAE",  "param":"VARIANCE_LOSS"},
                #{"type":"Real","min":5e-5,"max":2, "values_list":None, "name":"loss", "network_part":"VAE",  "param":"JENSEN_SHANNON_DIVERGENCE_LOSS"},
                {"type":"Real","min":5e-5,"max":2, "values_list":None, "name":"loss", "network_part":"VAE",  "param":"MEDIAN_LOSS_batch"},
                #{"type":"Real","min":5e-5,"max":2, "values_list":None, "name":"loss", "network_part":"VAE",  "param":"MSE_LOSS"},                
                #{"type":"Real","min":5e-5,"max":2, "values_list":None, "name":"loss", "network_part":"VAE",  "param":"KL_DIVERGENCE_LOSS"},
                {"type":"Real","min":5e-5,"max":2, "values_list":None, "name":"loss", "network_part":"VAE",  "param":"SPEARMAN_CORRELATION_LOSS"},
                {"type":"Real","min":5e-5,"max":0.5, "values_list":None, "name":"loss", "network_part":"VAE",  "param":"PEARSON_CORRELATION_LOSS"},
                
                #{"type":"Real","min":0,"max":2, "values_list":None, "name":"loss", "network_part":"AE",  "param":"VARIANCE_LOSS"}
                #,{"type":"Real","min":5e-5,"max":2, "values_list":None, "name":"loss", "network_part":"AE",  "param":"DECORRELATION_LATENT_LOSS"}
                
                #{"type":"Categorical","min":None, "max":None, "values_list":["Adam", "RMSprop", "SGD"], "name":"loss_optimizer", "param":""},
            ]
        }
        run_mode = "ALL_nocorr"
        
        return {"experiments_list":experiments_list, "run_mode":run_mode,  "optimization_settings":optimization_settings}
        
    def experiment(self, num_case, main_folder, seed, experiment_name, optimization, load_model, loss_obj, time_slot=None):
        id_experiments = num_case
        print(f"Experiment ID: {id_experiments}")
        experiments_list = self.getExperimentsList(seed, optimization)
        #experiments_selected = experiments_list["experiments_list"][id_experiments]
        
        for experiment in experiments_list['experiments_list']:
            if experiment['id'] == id_experiments:
                experiments_selected = experiment
                break
        
        print(num_case)
        run_mode = experiments_list["run_mode"]
        opt_settings = experiments_list['optimization_settings']
            
        folder_experiment = Path(main_folder, experiment_name)  

        #, 'autoencoder_05k_Chengdu','autoencoder_0016_Chengdu', 'autoencoder_6k_Chengdu','autoencoder_3_copula_optimization']
        cprint(Style.BRIGHT + "|------------------------|" + Style.RESET_ALL, 'cyan', attrs=["bold"])
        cprint(Style.BRIGHT +f"| Modelcase   : {experiments_selected['model_case']}" + Style.RESET_ALL, 'cyan', attrs=["bold"])
        cprint(Style.BRIGHT +f"|  settings   : {experiments_selected}" + Style.RESET_ALL, 'cyan', attrs=["bold"])
        cprint(Style.BRIGHT +f"|  seed       : {seed}" + Style.RESET_ALL, 'cyan', attrs=["bold"])
        cprint(Style.BRIGHT +f"|  time_slot  : {time_slot}" + Style.RESET_ALL, 'cyan', attrs=["bold"])
        cprint(Style.BRIGHT + "|------------------------|" + Style.RESET_ALL, 'cyan', attrs=["bold"])
        
        
        
        '''try:'''
        self.load_copula = True          
        
        
        nc = InstancesGenerationCore(device=None,epoch=experiments_selected["epoch"], case=experiments_selected['case'], model_case=experiments_selected["model_case"], univar_count=experiments_selected["univar_count"], lat_dim=experiments_selected["lat_dim"], dataset_setting=experiments_selected['dataset_setting'], instaces_size= experiments_selected["instaces_size"], input_shape= experiments_selected["input_shape"], do_optimization=experiments_selected["optimization"], num_gen_samples=experiments_selected['dataset_setting']["noise_samples"], path_folder=folder_experiment, seed=seed, run_mode=run_mode, opt_settings=opt_settings, time_slot=time_slot, load_copula=self.load_copula, loss_obj=loss_obj)
        if load_model:
            loss_obj = nc.start_experiment(load_model=True)
        else:
            loss_obj = nc.start_experiment()
        
        return loss_obj
        '''except Exception as e:
            print(f"Unexpected error: {e}")
        else:
            print("No exceptions were thrown.")
        finally:
            res = {"univar_count": experiments_selected["univar_count"]}
            return res'''
    
    def str2bool(self, v):
        if v in ['yes', 'true', 't', 'y', '1']:
            return True
        else:
            return False