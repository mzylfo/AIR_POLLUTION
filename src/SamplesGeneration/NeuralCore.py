from src.NeuroCorrelation.DataSynteticGeneration import DataSynteticGeneration
from src.NeuroCorrelation.NeuralModels import NeuralModels, GEN_fl, GEN_autoEncoder, GEN_autoEncoder_Encoder, GEN_autoEncoder_Decoder
from src.NeuroCorrelation.LossFunctions import LossFunction
from src.NeuroCorrelation.ModelTraining import ModelTraining
from src.NeuroCorrelation.ModelPrediction import ModelPrediction
from src.NeuroCorrelation.DataComparison import DataComparison


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
from pathlib import Path
import os

class NeuralCore():

    def __init__(self, device, path_folder, epoch = 5,  model_case="autoencoder", univar_count=78):
        self.device = device
        self.univar_count = univar_count
        self.epoch = epoch
        self.path_folder = Path('data','neuroCorrelation',path_folder)
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)

        self.dataGenerator = DataSynteticGeneration(torch_device=device, size_random=78, path_folder=self.path_folder)
        self.train_data = self.dataGenerator.graphGen(num_of_samples = 500, with_cov=False)
        
        print("model_case:\t",model_case)
        self.model_case = model_case
        if model_case=="fullyRectangle":
            self.model = GEN_fl
            self.loss_obj = LossFunction("MSE")
        elif model_case=="autoencoder":
            self.model = GEN_autoEncoder
            self.loss_obj = LossFunction("MSE")
        self.modelTrained = None


    def training_model(self):
        training_obj = ModelTraining(self.model, self.loss_obj, self.epoch, self.train_data, self.dataGenerator, self.path_folder, univar_count = self.univar_count)
        self.modelTrained = training_obj.training()


    def predict_model(self):
        if self.modelTrained is not None:
            predicting_obj = ModelPrediction(self.modelTrained, self.univar_count, name_model=self.model_case)
            predicting_obj.predict(self.train_data)
            pred_data = predicting_obj.getPred()
            pred_data_byvar = predicting_obj.getPred_byUnivar()
            datacomparison_obj = DataComparison(self.univar_count, self.path_folder)

            data = {"input":{'data':pred_data_byvar['input'],'color':'red'}, "reconstructed":{'data':pred_data_byvar['output'],'color':'blue'}}
            datacomparison_obj.data_comparison_plot(data)

    