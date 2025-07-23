import numpy as np
import torch

class DataBatchGenerator():

    def __init__(self, dataset, batch_size, shuffle):

        self.number_of_samples = len(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_batches = self.number_of_samples / batch_size
        self.shuffle = shuffle
        

    def generate(self):
        data_samples = self.dataset
        counter = 0
        if self.shuffle:
            np.random.shuffle(data_samples)
        
        while (counter*self.batch_size < self.number_of_samples):
            start_samples_index = self.batch_size * counter
            end_samples_index = self.batch_size * (counter + 1)
            sampled = data_samples[start_samples_index : end_samples_index]
            yield sampled
            counter += 1