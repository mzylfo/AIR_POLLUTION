import numpy as np
import torch
import random
from random import shuffle

class DataBatchGenerator():

    def __init__(self, dataset, batch_size, shuffle):

        self.number_of_samples = len(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_batches = self.number_of_samples / batch_size
        self.shuffle = shuffle
        
    def size(self):
            batches = len(self.dataset)//self.batch_size
            return batches
    
    def generate(self):
        data_samples = self.dataset        
        n_instances = len(data_samples)
        shuffle_indexes = [i for i in range(n_instances)]
        
        counter = 0
        if self.shuffle:
            random.Random(15).shuffle(shuffle_indexes)
            data_samples = [data_samples[idx] for idx in shuffle_indexes]

        
        while (counter*self.batch_size < self.number_of_samples):
            start_samples_index = self.batch_size * counter
            end_samples_index = self.batch_size * (counter + 1)
            sampled = data_samples[start_samples_index : end_samples_index]
            yield sampled
            counter += 1
        