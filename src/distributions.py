import torch
import numpy as np

import sys
sys.path.append("..")
from src.utils import *

#------------ 21.02.2022 -------#

class Sampler:
    def __init__(self, device):
        self.device = device
 
    def sample(self, size):
        raise NotImplementedError
#--------------- 21.02.2022 -----------#    
class UniformSampler(Sampler):
    
    def __init__(self, dim , semi_length_square, device):
        
        super().__init__(device = device)
        self.dim = dim
        self.semi_length_square = semi_length_square
        self.device = device
 
    def sample(self, num_samples):
        with torch.no_grad():
            sample_output = self.semi_length_square*(2*torch.rand(num_samples, self.dim, 
                                                              dtype = torch.float32, device = self.device) - 1)
        
        return sample_output
     
class MotionSampler(Sampler):
    def __init__(self, net , semi_length_square, data_sampler, deg):
        
        if deg < 1.:
            raise AssertionError("Degree of tray shouldn't be less, than 1")

        self.deg = deg 
        self.data_sampler = data_sampler
        self.semi_length_square = semi_length_square
        self.net = net
 
    def sample(self, num_samples, flag_plan=False):
        with torch.no_grad():
            samples_output = self.data_sampler.sample(num_samples)
            samples_input = motion_samples(samples_output, self.semi_length_square, self.net ,self.deg )[0]
        return  (samples_input , samples_output) if  flag_plan == True else samples_input
        
        
#---------------------------------------------------------------#                   
class StandardNormalSampler(Sampler):
 
    def __init__(self, dim=2, device="cuda"):
 
        super(StandardNormalSampler, self).__init__(device)
        self.dim = dim
        self.device = device
  
    def sample(self, size):
        return torch.randn(size, self.dim, device=self.device)       
    
#-------13.04.2022-------# (for Celeba Benchmark)  

class Transformer(Sampler):
    def __init__(
        self, device='cuda'
    ):
        self.device = device
        
#-----------13.04.2022--------#        
class PushforwardTransformer(Transformer):
    def __init__(
        self, pushforward,
        batch_size=128,
        device='cuda'
    ):
        super(PushforwardTransformer, self).__init__(
            device=device
        )
        
        self.fitted = False
        self.batch_size = batch_size
        self.pushforward = pushforward

    def fit(self, base_sampler, estimate_size=2**14, estimate_cov=True):
        assert base_sampler.device == self.device
        
        self.base_sampler = base_sampler
        self.fitted = True
      
        return self
        
    def sample(self, size=4):
        assert self.fitted == True
        
        if size <= self.batch_size:
            sample = self.base_sampler.sample(size)
            with torch.no_grad():
                sample = self.pushforward(sample)
            return sample
        
        sample = torch.zeros(size, self.sample(1).shape[1], dtype=torch.float32, device=self.device)
        for i in range(0, size, self.batch_size):
            batch = self.sample(min(i + self.batch_size, size) - i)
            with torch.no_grad():
                sample.data[i:i+self.batch_size] = batch.data
            torch.cuda.empty_cache()
        return sample
    
#-----------13.04.2022-------#

class TruncatedNormalNoiseTransformer(Transformer):
    def __init__(self,semi_length_square ,std=0.01   ,device='cuda'):
        super(TruncatedNormalNoiseTransformer, self).__init__(device=device)
        self.std = std
        self.semi_length_square = semi_length_square
        
    def fit(self, base_sampler):
        self.base_sampler = base_sampler
        
        return self
        
    def sample(self, batch_size=4):
        batch = self.base_sampler.sample(batch_size)
        
        with torch.no_grad():
            
            resample_condition = torch.max(abs(batch),dim = 1)[0] <= self.semi_length_square
            while resample_condition.all().item() == False:
                batch[~resample_condition] = self.base_sampler.sample(batch_size)
                batch = batch + self.std * torch.randn_like(batch)
                resample_condition = torch.max(abs(batch),dim = 1)[0] <= self.semi_length_square
                
        return batch