from itertools import permutations, product
import logging
import os
import string

import numpy as np
import torch

def get_dataset(context_len=5, num_samples=10000, data_dir=None, force_data=False):
    return InductionData(context_len, num_samples, data_dir, force_data)

###
# Dataset classes
###

class InductionData(torch.utils.data.Dataset):
    def __init__(self, context_len=5, num_samples=10000, data_dir=None, force_data=False):
        assert data_dir is not None, "data_dir is None"
        if context_len != 5:
            raise NotImplementedError
        
        if force_data:
            logging.info(f"Creating data and saving to {data_dir}")
            os.makedirs(data_dir, exist_ok=True)
            self.generate_data(context_len, data_dir, num_samples)
        logging.info(f"Loading data from {data_dir}")

        try:
            self.data = np.load(os.path.join(data_dir, f'{context_len}_{num_samples}.npy'))
        except FileNotFoundError:
            path = os.path.join(data_dir, f'{context_len}_{num_samples}.npy')
            raise FileNotFoundError(f"Could not find {path}. Run with force_data=True to generate data")
        
    def __getitem__(self, index):
        return np.array(self.data[index])
        
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def generate_data(context_len, data_dir, num_samples):
        data = []
        num_tokens = len(string.ascii_lowercase)
        
        for n in range(num_samples):
            token1 = np.random.randint(num_tokens)
            token2 = np.random.randint(num_tokens)
            while token2 == token1:
                token2 = np.random.randint(num_tokens)
            token3 = np.random.randint(num_tokens)
            while token3 in [token1, token2]:
                token3 = np.random.randint(num_tokens)
            
            if np.random.rand() < 0.5:
                data.append([token1, token2, token3, token2, token3])
            else:
                data.append([token1, token2, token3, token1, token2])
        
        # save data
        np.save(os.path.join(data_dir, f'{context_len}_{num_samples}.npy'), data)
        


