import os
from pathlib import Path
from src.NeuroCorrelation.Analysis.DataComparison import DataComparison_Advanced
from src.NeuroCorrelation.Analysis.DataStatistics import DataStatistics
from src.NeuroCorrelation.Analysis.ScenariosMap import ScenariosMap
from src.NeuroCorrelation.ModelPrediction.ModelPrediction import ModelPrediction

from termcolor import cprint
from colorama import init, Style

class PerformePrediction():

    def __init__(self, model, device,  model_type, data, dataGenerator, input_shape, rangeData, vc_mapping, univar_count, latent_dim, path_folder_pred, path_folder_data, path_map, time_performance, copulaData_filename, data_metadata, load_copula, key_dataout, use_copula=True, draw_plot=True, draw_scenarios=True, draw_correlationCoeff= True, noise_samples=1000):
        self.model = model
        self.device = device
        self.vc_mapping = vc_mapping
        self.noise_samples = noise_samples
        self.model_type = model_type
        self.data = data
        self.dataGenerator = dataGenerator 
        self.input_shape = input_shape
        self.draw_plot = draw_plot
        self.path_folder = path_folder_pred
        self.path_folder_data = path_folder_data
        self.path_map = path_map
        self.time_performance = time_performance
        self.draw_scenarios = draw_scenarios
        self.univar_count = univar_count
        self.latent_dim = latent_dim
        self.data_metadata = data_metadata
        self.draw_correlationCoeff = draw_correlationCoeff
        self.train_data = self.data['train_data']
        self.test_data = self.data['test_data']
        self.noise_data = self.data['noise_data']
        self.reduced_noise_data = self.data['reduced_noise_data']
        self.path_input = Path(self.path_folder_data, "datasplit")
        self.suffix_input = "datasplit_train_data_sample"
        self.key_dataout = key_dataout
        self.res_CompareAdvance = DataComparison_Advanced(univar_count=self.univar_count, input_folder=self.path_input, suffix_input=self.suffix_input, time_performance=self.time_performance, data_metadata=self.data_metadata, use_copula=use_copula, load_copula=load_copula, copulaData_filename=copulaData_filename, name_key=self.key_dataout)
        self.rangeData = rangeData
        self.predict = dict()
        self.corrCoeff = dict()
        self.model_type = model_type
        
        
        
    def predict_model(self, cases_list=None):
        cprint(Style.BRIGHT + "SETTING PHASE: Compare tool" + Style.RESET_ALL, 'green')
        if cases_list is None:
                if self.model_type =="AE":
                    cases_list = ['train', 'test', 'noise_copula', 'noise_gaussian', 'noise_gaussian_reduced']
                elif self.model_type =="VAE":
                    cases_list = ['train']
                elif self.model_type =="GAN":
                    cases_list = ['noise_gaussian', 'noise_gaussian_reduced']

        for case in cases_list:     
            if self.model_type =="AE":
                self.key_dataout = "ae"
                cprint(Style.BRIGHT + "PHASE: AutoEncoder" + Style.RESET_ALL, 'green')
                modelAE = self.model.getModel("all")
                
                if case == 'train':
                    cprint("AE - PREDICT PHASE: Training data", 'green')
                    plot_name = "AE_train"
                    predict_file_suffix ="train_test_data"
                    analysis_folder = Path(self.path_folder,"train_analysis")
                    datastats_latent = True
                    if not os.path.exists(analysis_folder):
                        os.makedirs(analysis_folder)
                    modelPrediction = ModelPrediction(model=modelAE, device=self.device, dataset=self.train_data, vc_mapping= self.vc_mapping, time_performance=self.time_performance, univar_count_in=self.univar_count, univar_count_out=self.univar_count, latent_dim=self.latent_dim, data_range=self.rangeData, input_shape=self.input_shape, path_folder=analysis_folder, model_type=self.model_type)
                    self.predict['train'] = modelPrediction.compute_prediction(experiment_name=predict_file_suffix, time_key=plot_name, remapping_data=True)
                    print("\t\t Statistics data")
                    datastats = DataStatistics(univar_count_in=self.univar_count, univar_count_out=self.univar_count, latent_dim=self.latent_dim, data=self.predict['train'], path_folder=analysis_folder, model_type=self.model_type, name_key=self.key_dataout)
                    plot_colors = {"input":"blue", "latent":"green", "output":"orange"}
                    distribution_compare = {"train_input":{'data':self.predict['train']['prediction_data_byvar']['input'], 'color':plot_colors['input']}, "train_reconstructed":{'data':self.predict['train']['prediction_data_byvar']['output'], 'color':plot_colors['output']}}

                if case == 'test':
                    cprint("AE - PREDICT PHASE: Testing data", 'green')
                    plot_name = "AE_test"
                    predict_file_suffix ="testing_test_data"
                    analysis_folder = Path(self.path_folder,"test_data_analysis")
                    datastats_latent = True
                    if not os.path.exists(analysis_folder):
                        os.makedirs(analysis_folder)
                    modelPrediction = ModelPrediction(model=modelAE, device=self.device, dataset=self.test_data, vc_mapping= self.vc_mapping, time_performance=self.time_performance, univar_count_in=self.univar_count, univar_count_out=self.univar_count, latent_dim=self.latent_dim, data_range=self.rangeData, input_shape=self.input_shape, path_folder=analysis_folder, model_type=self.model_type)
                    self.predict['test'] = modelPrediction.compute_prediction(experiment_name=predict_file_suffix, time_key=plot_name, remapping_data=True)
                    print("\t\t Statistics data")
                    datastats = DataStatistics(univar_count_in=self.univar_count, univar_count_out=self.univar_count, latent_dim=self.latent_dim, data=self.predict['test'], path_folder=analysis_folder, model_type=self.model_type, name_key=self.key_dataout)
                    plot_colors = {"input":"blue", "latent":"green", "output":"orange"}
                    if "train" not in self.predict:
                        self.predict_model(cases_list=["train"])
                    distribution_compare = {"train_input":{'data':self.predict['train']['prediction_data_byvar']['input'],'color':'cornflowerblue', 'alpha':0.5}, "test_input":{'data':self.predict['test']['prediction_data_byvar']['input'],'color':plot_colors['input']}, "test_reconstructed":{'data':self.predict['test']['prediction_data_byvar']['output'],'color':plot_colors['output']}}

                elif case == 'noise_gaussian':
                    cprint("AE - PREDICT PHASE: Noised data generation", 'green')
                    plot_name = "AE_noise"
                    predict_file_suffix ="noise_test_data"
                    analysis_folder = Path(self.path_folder,"noise_data_analysis")
                    datastats_latent=False
                    if not os.path.exists(analysis_folder):
                        os.makedirs(analysis_folder)
                    modelTrainedDecoder = modelAE.get_decoder()
                    modelPrediction = ModelPrediction(model=modelTrainedDecoder, device=self.device, dataset=self.noise_data, vc_mapping= self.vc_mapping, time_performance=self.time_performance, univar_count_in=self.latent_dim, univar_count_out=self.univar_count, latent_dim=None, data_range=self.rangeData, input_shape=self.input_shape, path_folder=analysis_folder, isnoise_in=True, model_type=self.model_type)
                    self.predict['noise_gaussian'] = modelPrediction.compute_prediction(experiment_name=predict_file_suffix, time_key=plot_name, remapping_data=True)
                    print("\t\t Statistics data")
                    datastats = DataStatistics(univar_count_in=self.latent_dim, univar_count_out=self.univar_count, latent_dim=None, data=self.predict['noise_gaussian'], path_folder=analysis_folder, model_type=self.model_type)
                    plot_colors = {"input":"green","output":"m"}
                    if "train" not in self.predict:
                        self.predict_model(cases_list=["train"])
                    distribution_compare = {"train_input":{'data':self.predict['train']['prediction_data_byvar']['input'],'color':'cornflowerblue', 'alpha':0.5}, "noise_generated":{'data': self.predict['noise_gaussian']['prediction_data_byvar']['output'],'color':plot_colors['output'], 'alpha':0.5}}
                    
                elif case == 'noise_gaussian_reduced':
                    cprint("AE - PREDICT PHASE: reduced noised data generation", 'green')
                    plot_name = "AE_reduced_noise"
                    predict_file_suffix ="reduced_noise_test_data"
                    analysis_folder = Path(self.path_folder,"reduced_noise_data_analysis")
                    datastats_latent=False
                    if not os.path.exists(analysis_folder):
                        os.makedirs(analysis_folder)
                    modelTrainedDecoder = modelAE.get_decoder()
                    modelPrediction = ModelPrediction(model=modelTrainedDecoder, device=self.device, dataset=self.reduced_noise_data, vc_mapping= self.vc_mapping, time_performance=self.time_performance, univar_count_in=self.latent_dim, univar_count_out=self.univar_count, latent_dim=None, data_range=self.rangeData, input_shape=self.input_shape, path_folder=analysis_folder, isnoise_in=True, model_type=self.model_type)
                    self.predict['noise_gaussian_reduced'] = modelPrediction.compute_prediction(experiment_name=predict_file_suffix, time_key=plot_name, remapping_data=True)
                    print("\t\t Statistics data")
                    datastats = DataStatistics(univar_count_in=self.latent_dim, univar_count_out=self.univar_count, latent_dim=None, data=self.predict['noise_gaussian_reduced'], path_folder=analysis_folder, model_type=self.model_type)
                    plot_colors = {"input":"green","output":"m"}
                    if "train" not in self.predict:
                        self.predict_model(cases_list=["train"])
                    distribution_compare = {"train_input":{'data':self.predict['train']['prediction_data_byvar']['input'],'color':'cornflowerblue', 'alpha':0.5}, "noise_generated":{'data':self.predict['noise_gaussian_reduced']['prediction_data_byvar']['output'],'color':plot_colors['output'], 'alpha':0.5}}

                elif case == 'noise_copula':
                    cprint("AE - PREDICT PHASE: Copula Latent data", 'green')
                    plot_name = "AE_copulaLat"
                    predict_file_suffix = "_copulaLatent_data"
                    analysis_folder = Path(self.path_folder,"copulaLatent_data_analysis")
                    datastats_latent=False
                    if not os.path.exists(analysis_folder):
                        os.makedirs(analysis_folder)
                    if "train" not in self.predict:
                        self.predict_model(cases_list=["train"])
                    copulaLat_samples_starting = self.predict['train']['latent_data_input']['latent']
                    copulaLat_data, self.corrCoeff['copulaLatent_data_AE'] = self.dataGenerator.casualVC_generation(name_data=predict_file_suffix, real_data=copulaLat_samples_starting, univar_count=self.latent_dim, num_of_samples = self.noise_samples,  draw_plots=True, draw_correlationCoeff=self.draw_correlationCoeff)
                    modelTrainedDecoder = modelAE.get_decoder()
                    
                    modelPrediction = ModelPrediction(model=modelTrainedDecoder, device=self.device, dataset=copulaLat_data, vc_mapping= self.vc_mapping, time_performance=self.time_performance, univar_count_in=self.latent_dim, univar_count_out=self.univar_count, latent_dim=None, data_range=self.rangeData, input_shape=self.input_shape, path_folder=analysis_folder, isnoise_in =True, model_type=self.model_type)
                    self.predict['noise_copula'] = modelPrediction.compute_prediction(experiment_name="_copulaLatent_data", time_key=plot_name, remapping_data=True)
                    print("\t\t Statistics data")
                    datastats = DataStatistics(univar_count_in=self.latent_dim, univar_count_out=self.univar_count, latent_dim=None, data=self.predict['noise_copula'], path_folder=analysis_folder, model_type=self.model_type)
                    
                    plot_colors = {"input":"green", "output":"darkviolet"}
                    if "train" not in self.predict:
                        self.predict_model(cases_list=["train"])
                    distribution_compare = {"train_input":{'data':self.predict['train']['prediction_data_byvar']['input'],'color':'cornflowerblue', 'alpha':0.5}, "copulaLatent_generated":{'data':self.predict['noise_copula']['prediction_data_byvar']['output'],'color':plot_colors['output'], 'alpha':0.5}}

            elif self.model_type =="VAE":
                self.key_dataout = "vae"
                cprint(Style.BRIGHT + "PHASE: VariationalAutoEncoder" + Style.RESET_ALL, 'green')
                modelAE = self.model.getModel("all")
                
                
                if case == 'train':
                    cprint("AE - PREDICT PHASE: Training data", 'green')
                    plot_name = "AE_train"
                    predict_file_suffix ="train_test_data"
                    analysis_folder = Path(self.path_folder,"train_analysis")
                    datastats_latent = True
                    if not os.path.exists(analysis_folder):
                        os.makedirs(analysis_folder)
                    
                    print("\t\t Statistics data")
                    datastats = DataStatistics(univar_count_in=self.univar_count, univar_count_out=self.univar_count, latent_dim=self.latent_dim, data=self.predict['train'], path_folder=analysis_folder, model_type=self.model_type, name_key=self.key_dataout)
                    plot_colors = {"input":"blue", "latent":"green", "output":"orange"}
                    distribution_compare = {"train_input":{'data':self.predict['train']['prediction_data_byvar']['input'], 'color':plot_colors['input']}, "train_reconstructed":{'data':self.predict['train']['prediction_data_byvar']['output'], 'color':plot_colors['output']}}

                    
                if case == 'noise_gaussian':
                    cprint("VAE - PREDICT PHASE: Noised data generation", 'green')
                    plot_name = "VAE_noise_test_data"
                    
                    predict_file_suffix ="noise_test_data"
                    analysis_folder = Path(self.path_folder,"noise_data_analysis")
                    datastats_latent = False
                    if not os.path.exists(analysis_folder):
                        os.makedirs(analysis_folder)
                    
                    _modelPrediction = ModelPrediction(model=modelAE, device=self.device, dataset=self.train_data, vc_mapping= self.vc_mapping, time_performance=self.time_performance, univar_count_in=self.univar_count, univar_count_out=self.univar_count, latent_dim=self.latent_dim, data_range=self.rangeData, input_shape=self.input_shape, path_folder=analysis_folder, model_type=self.model_type)
                    self.predict['train'] = _modelPrediction.compute_prediction(experiment_name=predict_file_suffix, time_key=plot_name, remapping_data=True, force_noLatent=True)
                    
                    modelTrainedDecoder = modelAE.get_decoder()
                    modelPrediction = ModelPrediction(model=modelTrainedDecoder, device=self.device, dataset=self.noise_data, vc_mapping= self.vc_mapping, time_performance=self.time_performance, univar_count_in=self.latent_dim, univar_count_out=self.univar_count, latent_dim=None, data_range=self.rangeData, input_shape=self.input_shape, path_folder=analysis_folder, isnoise_in=True, model_type=self.model_type)
                    self.predict['noise_gaussian'] = modelPrediction.compute_prediction(experiment_name=predict_file_suffix, time_key=plot_name, remapping_data=True)
                    print("\t\t Statistics data")
                    datastats = DataStatistics(univar_count_in=self.latent_dim, univar_count_out=self.univar_count, latent_dim=None, data=self.predict['noise_gaussian'], path_folder=analysis_folder, model_type=self.model_type, name_key=self.key_dataout)
                    plot_colors = {"input":"green","output":"m"}
                    if "train" not in self.predict:
                        self.predict_model(cases_list=["train"])
                    distribution_compare = {"train_input":{'data':self.predict['train']['prediction_data_byvar']['input'], 'color':'cornflowerblue', 'alpha':0.5}, "noise_generated":{'data': self.predict['noise_gaussian']['prediction_data_byvar']['output'],'color':plot_colors['output'], 'alpha':0.5}}

            elif self.model_type in ["GAN","WGAN"]:
                cprint(Style.BRIGHT + "PHASE: Generative Adversarial Network" + Style.RESET_ALL, 'green')
                modelGEN = self.model.getModel(selection="gen", eval=True)
                modelDIS = self.model.getModel(selection="dis", eval=True)
            
                if case == 'noise_gaussian':
                    cprint("GAN - PREDICT PHASE: Noised data generation", 'green')
                    plot_name = "GAN_noise"
                    predict_file_suffix = "noise_test_data"
                    analysis_folder = Path(self.path_folder,"noise_data_analysis")
                    datastats_latent=False
                    if not os.path.exists(analysis_folder):
                        os.makedirs(analysis_folder)
                    
                    modelPrediction = ModelPrediction(model=modelGEN, device=self.device, dataset=self.noise_data, vc_mapping= self.vc_mapping, time_performance=self.time_performance, univar_count_in=self.latent_dim, univar_count_out=self.univar_count, latent_dim=None, data_range=self.rangeData, input_shape=self.input_shape, path_folder=analysis_folder, isnoise_in = True, model_type=self.model_type)
                    self.predict['noise_gaussian']  = modelPrediction.compute_prediction(experiment_name=predict_file_suffix, time_key=plot_name, remapping_data=True)
                    print("\t\t Statistics data")
                    datastats = DataStatistics(univar_count_in=self.latent_dim, univar_count_out=self.univar_count, latent_dim=None, data=self.predict['noise_gaussian'], path_folder=analysis_folder, model_type=self.model_type, name_key=self.key_dataout)
                    plot_colors = {"input":"green", "output":"m"}
                    
                    traindata_input = self.dataset2var(self.train_data)
                    distribution_compare = {"train_input":{'data':traindata_input,'color':'cornflowerblue', 'alpha':0.5}, "noise_generated":{'data':self.predict['noise_gaussian']['prediction_data_byvar']['output'],'color':plot_colors['output'], 'alpha':0.5}}
                    
                elif case == 'noise_gaussian_reduced':
                    cprint("GAN - PREDICT PHASE: reduced noised data generation", 'green')
                    plot_name = "GAN_reduced_noise"
                    predict_file_suffix = "reduced_noise_test_data"
                    analysis_folder = Path(self.path_folder,"reduced_noise_data_analysis")
                    datastats_latent=False
                    if not os.path.exists(analysis_folder):
                        os.makedirs(analysis_folder)
                    modelPrediction = ModelPrediction(model=modelGEN, device=self.device, dataset=self.reduced_noise_data, vc_mapping= self.vc_mapping, time_performance=self.time_performance, univar_count_in=self.latent_dim, univar_count_out=self.univar_count, latent_dim=None, data_range=self.rangeData, input_shape=self.input_shape, path_folder=analysis_folder, isnoise_in =True, model_type=self.model_type)
                    self.predict['noise_gaussian_reduced'] = modelPrediction.compute_prediction(experiment_name=predict_file_suffix, time_key=plot_name, remapping_data=True) 
                    print("\t\t Statistics data")
                    datastats = DataStatistics(univar_count_in=self.latent_dim, univar_count_out=self.univar_count, latent_dim=None, data=self.predict['noise_gaussian_reduced'], path_folder=analysis_folder, model_type=self.model_type, name_key=self.key_dataout)
                    plot_colors = {"input":"green","output":"m"}
                    traindata_input = self.dataset2var(self.train_data)
                    distribution_compare = {"train_input":{'data':traindata_input,'color':'cornflowerblue', 'alpha':0.5}, "noise_generated":{'data':self.predict['noise_gaussian_reduced']['prediction_data_byvar']['output'],'color':plot_colors['output'], 'alpha':0.5}}

            if self.draw_correlationCoeff:
                print("\tSTATS PHASE:  Correlation")
                self.corrCoeff[plot_name] = datastats.get_corrCoeff(latent=datastats_latent)

            if self.draw_plot:
                print("\tSTATS PHASE:  Plots")
                datastats.plot(plot_colors=plot_colors, plot_name=plot_name, distribution_compare=distribution_compare, latent=datastats_latent, draw_correlationCoeff=self.draw_correlationCoeff)
                self.res_CompareAdvance.loadPrediction_OUTPUT(output_folder=analysis_folder, suffix_output=predict_file_suffix, key=self.key_dataout)
                self.res_CompareAdvance.comparison_measures(measures=['metrics', 'swarm_distributions', 'wasserstein_dist', 'tsne_plots', 'pca_tsne_plots', 'umap_plots','pacmap_plots'])
                if cases_list in ['noise_gaussian_reduced']:
                    datastats.draw_point_overDistribution(plotname="noise_reduced_sampled_noise", n_var=self.latent_dim, points=self.reduced_noise_data,  distr=None)
                    reduced_noise_dict = list()
                    for pred in self.predict['noise_gaussian_reduced']['prediction_data']['output']:
                        reduced_noise_dict.append({'sample':pred, 'noise':pred })
                    datastats.draw_point_overDistribution(plotname="noise_reduced_sampled_generated", n_var=self.univar_count, points=reduced_noise_dict,  distr=self.train_data)

            if self.draw_scenarios:
                print("\tSTATS PHASE:  Scenarios on map")
                instance_filename = f"prediced_instances_{predict_file_suffix}"
                
                scenariosMap = ScenariosMap(data_range=self.rangeData, vc_mapping= self.vc_mapping, path_folder_map= self.path_map, path_folder=analysis_folder, instance_file=instance_filename, label="x_output")
                scenariosMap.draw_scenarios()


    def dataset2var(self, data, pred2numpy=True):
        var_byComp = dict()
        for id_var in range(self.univar_count):
            var_byComp[id_var] = list()
        for count, item in enumerate(data):
            if pred2numpy:
                item_np = item['sample'].detach().cpu().numpy()
            else:
                item_np = item['sample'][0][0]
            for id_var in range(self.univar_count):
                if pred2numpy:
                    var_byComp[id_var].append(item_np[id_var])
                else:
                    var_byComp[id_var].append(item_np[id_var][0])        
        return var_byComp