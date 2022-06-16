import torch
import torch.nn as nn
import numpy as np
import gc

import os
from .distributions import UniformSampler, MotionSampler, TruncatedNormalNoiseTransformer, PushforwardTransformer,                                           StandardNormalSampler
from .models import load_resnet_G,MinFunnels
from .utils import grad, lipschitz_one_checker

 
class MixToOneBenchmark:
    def __init__(self, dim, width, deg = 8, semi_length_square = 2.5,  seed = 987, device = "cuda"):
        
        folder = "../benchmarks/nd/dim_{}/width_{}".format(dim , width)
        filename = "deg_{}_seed_{}.pt".format(deg, seed)
        net = MinFunnels(dim, width)
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            net.load_state_dict(torch.load(path)) 
            print("benchmark pair loaded")
        else:
            print("benchmark pair does not exist")
            net.net.centers.data = semi_length_square*(torch.rand(width,dim).mul(2).add(-1))
            net.net.bias.data =  torch.randn(width).mul(0.1)
            if not os.path.exists(folder):
                os.makedirs(folder)
            torch.save(net.cpu().state_dict(),path)
            print("benchmark pair created")
            
        assert torch.allclose(lipschitz_one_checker(net , torch.randn(2000,dim ) , dim = -1)[1] ,torch.ones(1 ), rtol = 1e-2)
        self.net = net.to(device)
        self.semi_length_square = semi_length_square
        self.deg = deg
        self.dim = dim
        self.width = width
        self.seed = seed
        self.device = device
        self.output_sampler = UniformSampler(dim =  dim, semi_length_square =  semi_length_square, device = device)
        self.input_sampler = MotionSampler( net = net, semi_length_square = semi_length_square, data_sampler =                                                                 self.output_sampler, deg = deg )

class Celeba64Benchmark:
    
    def __init__(self, width, deg ,dim = 3*64*64,  semi_length_square = 1.2 , seed = 295, file_name = "../benchmarks/celeba/Final_G.pt" , device="cuda"):
        folder = "../benchmarks/celeba/width_{}".format(width)
        filename = "deg_{}_seed_{}.pt".format(deg ,seed)
        net = MinFunnels(dim, width)
        path = os.path.join(folder,filename)
        if os.path.exists(path):
            net.load_state_dict(torch.load(path))
            print("Benchmark pair loaded")
        else: 
            print("Benchmark pair does not exist")
            net.net.centers.data  = torch.rand(width, dim).add(-0.5).mul(2)
            net.net.bias.data = 0.1*torch.randn(width)
            if not os.path.exists(folder):
                os.makedirs(folder)
            torch.save(net.cpu().state_dict(),path )
            print("benchmark pair created")
            
        assert torch.allclose(lipschitz_one_checker(net , torch.randn(2000,dim ) , dim = -1)[1],torch.ones(1 ), rtol = 1e-2)
        self.net = net.to(device)
        self.semi_length_square = semi_length_square
        self.deg = deg
        self.dim =  dim
        self.device = device
        self.resnet = load_resnet_G(file_name)
 
        self.output_sampler = TruncatedNormalNoiseTransformer( semi_length_square = semi_length_square,std=0.01).fit(
            PushforwardTransformer(self.resnet).fit(
                 StandardNormalSampler(dim=128), estimate_cov=False
            )
        )
        
        self.input_sampler = MotionSampler(net = net, semi_length_square = semi_length_square, data_sampler = self.output_sampler,deg = deg)
 