import torch
import torch.nn as nn
import geotorch

import torch.nn.functional as F
import functools
import numpy as np
import tqdm
import time
import ot

import sys
sys.path.append("..")
from src.utils import *
from src.map_benchmark import *

import warnings
warnings.filterwarnings('ignore')
  
#------------------WGAN-GP(LP)--------------------#
def penalty(critic,  q_samples,  p_samples , flag_penalty ):
    # https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
    """
    critic       : Feed Forward Network : \mathbb{R}^{N} -> \mathbb{R}
    q_samples    : torch.tensor shape([Batch,N])
    p_samples    : torch.tensor shape([Batch,N])
    flag_penalty :  LP or GP
    
    """
    batch_size = q_samples.shape[0] # torch.tensor shape(batch_size)
    with torch.no_grad():
        t = torch.rand(batch_size, device = q_samples.device).unsqueeze(dim = 1) # torch.tensor shape([batch_size,1]
        interpolated = t*q_samples + (1 - t)*p_samples # torch.tensor shape([batch_size,D])]
        
    interpolated.requires_grad = True              # requires grad
    d_output = critic(interpolated)  # torch.tensor shape([batch_size, 1])
    gradients = torch.autograd.grad(outputs = d_output,
                                    inputs = interpolated,
                                    grad_outputs = torch.ones(d_output.size(),device = d_output.device) ,
                                    create_graph = True,
                                    retain_graph = True)[0]
     
    graients = gradients.reshape(batch_size, -1) # torch.tensor shape([batch_size, D])
    gradients_norm = torch.norm(gradients, dim = 1) # torch.tensor ([batch_size])
    
    return ((gradients_norm - 1)**2).mean() if flag_penalty == "GP" else \
            (torch.relu(gradients_norm - 1)**2).mean()
     
#------------------train---------------#
def train_WGAN( critic, critic_optimizer,sampler_q,sampler_p,benchmark,batch_size,n_iterations,
                flag_penalty, lmbd):
    
    critic_losses = []
    cosine_accuracy = []
    times_cos = []
    critic.train(True)
    
    start_time = time.time()
    for epoch_i in tqdm.tqdm(range(n_iterations)):
    
        p_samples   =  sampler_p.sample(batch_size, flag_plan = False) 
        q_samples   =  sampler_q.sample(batch_size) 
        # do a critic update
        critic_optimizer.zero_grad()
        pnlt =  penalty(critic, q_samples,  p_samples, flag_penalty)  if flag_penalty in ["GP","LP"]  else 0.
        d_loss = critic(q_samples).mean() - critic(p_samples).mean() +  lmbd*pnlt
            
        d_loss.backward()
        critic_optimizer.step()
        
        critic_losses.append(d_loss.item())
        if epoch_i % 10 == 0:
            cosine_accuracy.append( cosine_metrics( critic ,None, benchmark, batch_size ,flag_rev = False) ) 
            times_cos.append(time.time() - start_time)
    # additional forward for preserving 1-Lipshitzness
    critic.forward(p_samples)
    critic.eval()
    return critic_losses, cosine_accuracy, times_cos
    
#----------------CoWGAN-----------------#
def train_CoWGAN(critic, critic_optimizer, 
                 sampler_q, sampler_p , benchmark, batch_size, n_iterations):
    
    
    L_1_losses = []
    L_2_losses = []
    L_3_losses = []
    cosine_accuracy = []
    times_cos = []
    
    start_time = time.time()
    
    for it in tqdm.tqdm(range(n_iterations)):
        
        samples_p = sampler_p.sample(batch_size, flag_plan = False)
        samples_q = sampler_q.sample( batch_size) 
        
        L_1 = critic(samples_p).mean() - critic(samples_q).mean()
        
        with torch.no_grad():
            
            idxs_p = torch.argmin(torch.cdist(samples_q,samples_p) - critic(samples_p).reshape(-1), dim = 1)
            samples_p_inv = samples_p[idxs_p]
            
            idxs_q = torch.argmin(torch.cdist(samples_p,samples_q) + critic(samples_q).reshape(-1), dim= 1)
            samples_q_inv = samples_q[idxs_q]
            
        L_2 = critic(samples_p).mean() + (torch.norm(samples_p_inv - samples_q,dim = 1) - critic(samples_p_inv)).mean()
        
        L_3 =  - critic(samples_q).mean() + (torch.norm(samples_p - samples_q_inv,dim=1) + critic(samples_q_inv) 
                                           ).mean()
        
        if L_2 < L_1:
            critic_optimizer.zero_grad()
            (-L_2).backward()
            critic_optimizer.step()
            L_2_losses.append( -L_2.item())
            
        elif L_3 < L_1:
            critic_optimizer.zero_grad()
            (- L_3).backward()
            critic_optimizer.step()
            L_3_losses.append( -L_3.item())
            
        else:
            critic_optimizer.zero_grad()
            (- L_1).backward()
            critic_optimizer.step()
            L_1_losses.append( -L_1.item())
    
        if it % 10 == 0:
            cosine_accuracy.append(cosine_metrics( critic ,None,benchmark, batch_size ,flag_rev = False))
            times_cos.append(time.time() - start_time)
        
    return {"L1":L_1_losses,"L2":L_2_losses,"L3":L_3_losses}, cosine_accuracy, times_cos 
    
#----------------3P-WGAN-----------------#

def train_3PWGAN(critic, mover, optimizer_critic, optimizer_mover, sampler_q, sampler_p,
          benchmark, batch_size, n_iterations, mover_steps, reverse_flag):
    
    def cost(x,y):
        return torch.norm(x - y, dim = 1)

    losses = {"critic":[],"mover":[] }
    cosine_accuracy = []
    times_cos = []
 
    start_time = time.time()
    for itr in tqdm.tqdm(range(n_iterations)):
        if reverse_flag == False: 
            q_sample = sampler_q.sample(batch_size) 
            p_sample   = sampler_p.sample( batch_size,flag_plan =False) 
        else:
            p_sample = sampler_q.sample(batch_size)
            q_sample   = sampler_p.sample( batch_size ,flag_plan=False) 
         
        mover.train(True)
        critic.eval()
        for _ in range(mover_steps):
            optimizer_mover.zero_grad()
            loss_mover = torch.mean(- critic(mover(q_sample)) + cost(mover(q_sample), q_sample) ).mean() + \
                         torch.mean(critic(p_sample))
            loss_mover.backward()
            optimizer_mover.step()
            
        losses['mover'].append(loss_mover.item())
        # ---- mover block ----
        critic.train(True)
        mover.eval()
        # ---- critic block ---
        optimizer_critic.zero_grad()
        loss_critic = torch.mean(- critic(p_sample.detach())) +  torch.mean(critic(mover(q_sample))) - \
                       cost(mover(q_sample),q_sample).mean()
        
        loss_critic.backward() 
        optimizer_critic.step()
        losses['critic'].append(loss_critic.item()) 
        
        # ---- critic block ---
        if itr % 10 == 0:
            if reverse_flag == False:
                cosine_accuracy.append(cosine_metrics( critic ,None,benchmark, batch_size ,flag_rev = False
                              ))
                times_cos.append(time.time() - start_time)
            else:
                cosine_accuracy.append(cosine_metrics(None ,mover, benchmark, batch_size, flag_rev = True ,eps=1e-15))
                times_cos.append(time.time() - start_time)
                
    return losses,cosine_accuracy, times_cos
 
#========== LSOT ==============#
def train_LSOT(critic_f,critic_g,critic_f_optimizer,critic_g_optimizer,sampler_q,sampler_p,
                benchmark,batch_size,n_iterations,epsilon):
    
    train_losses = []
    times_cos = []
    cosine_accuracy = []
    
    start_time = time.time()
    for epoch_i in tqdm.tqdm(range(n_iterations)):
        
        q_sample    =   sampler_q.sample(batch_size) 
        p_sample    = sampler_p.sample(batch_size,flag_plan = False) 
           
        critic_f_optimizer.zero_grad()
        critic_g_optimizer.zero_grad()
        
        reg = torch.mean(1/(4*epsilon)*(critic_f(p_sample).flatten()  + critic_g(q_sample).flatten()  -
                            torch.norm( p_sample - q_sample,dim = 1) ).relu()**2)
        
        d_loss = -(critic_f(p_sample).mean() + critic_g(q_sample).mean()) + reg
        d_loss.backward()
        
        critic_f_optimizer.step()
        critic_g_optimizer.step()
        
        train_losses.append(d_loss.item())
        
        if epoch_i % 10 == 0:
            cosine_accuracy.append(cosine_metrics( critic_f ,None,benchmark, batch_size ,flag_rev = False))
            times_cos.append(time.time() - start_time)
        
    return train_losses, cosine_accuracy, times_cos

#============WGAN-qp========#
def train_WGAN_qp( critic, critic_optimizer,  sampler_q , sampler_p, benchmark, batch_size , n_iterations ):
    
    train_losses = []
    cosine_accuracy = []
    times_cos = []
    
    start_time = time.time()
    for it in tqdm.tqdm(range(n_iterations)):
        
        samples_p = sampler_p.sample(batch_size,flag_plan = False) 
        samples_q = sampler_q.sample(batch_size) 
        
        with torch.no_grad():
            idxs_inv = torch.argmin(torch.cdist(samples_q, samples_p) - critic(samples_p).reshape(-1), dim =1)
            samples_p_inv = samples_p[idxs_inv]
        
        critic_optimizer.zero_grad()
        loss = -( critic(samples_p).mean() + (torch.norm(samples_p_inv - samples_q,dim=1)  
                 - critic(samples_p_inv)).mean())
        loss.backward()
        critic_optimizer.step()
        train_losses.append(loss.item())
        
        if it % 10 == 0:
            cosine_accuracy.append(cosine_metrics( critic ,None,benchmark,  batch_size ,flag_rev = False))
            times_cos.append(time.time() - start_time)
            
    return train_losses, cosine_accuracy, times_cos

#=========get_grad========#
def get_grad(mover, x):
    return  (x - mover(x))/torch.norm(x - mover(x),dim = 1 ).unsqueeze(-1)
 
#-------------- metrics ------------#

def calculate_unbiased_wasserstein(benchmark, batch_size):
    samples_q_,samples_p =  benchmark.input_sampler.sample( batch_size, flag_plan = True)
    return  (samples_p - samples_q_).norm(dim=1).mean().item()

def calculate_kantorovitch_wasserstein( critic , key , benchmark, batch_size):
    mltp = -1. if key == "3p-wgan_rev" else 1.
    samples_p =     benchmark.input_sampler.sample(batch_size, flag_plan =False)
    samples_q_ =    benchmark.output_sampler.sample(batch_size)
    return mltp*( critic(samples_p ).mean() - critic(samples_q_).mean()).item()


def L2_metrics(critic, mover_rev,  benchmark, batch_size, flag_rev):

    mltp = -1.  
    samples_p = benchmark.input_sampler.sample( batch_size, flag_plan = False) 
    samples_p.requires_grad = True # torch.Size([B,dim])
    grad_lip =  mltp*grad(benchmark.net , samples_p ) # torch.Size([B, dim])
    if flag_rev == True :
        grad_critic = get_grad(mover_rev, samples_p)  
    else:
        grad_critic =  grad(critic , samples_p )
    samples_p.requires_grad = False
    return  (((grad_lip - grad_critic).norm(dim=1))**2).mean().item()
    

def cosine_metrics(critic ,mover_rev, benchmark , batch_size, flag_rev ,eps = 1e-15):
    
    mltp = -1.  
    samples_p =  benchmark.input_sampler.sample( batch_size, flag_plan = False)
    samples_p.requires_grad = True
    grad_lip =  mltp*grad(benchmark.net , samples_p)
    
    if flag_rev == True:
        grad_critic = get_grad(mover_rev, samples_p) 
    else:
        grad_critic =  grad(critic , samples_p )
       
    samples_p.requires_grad = False
    
    scalar_prod = torch.mul(grad_lip, grad_critic).sum(dim = -1)
    num = scalar_prod 
    den = torch.sqrt(grad_lip.norm(dim=1)**2 * grad_critic.norm(dim=1)**2) + eps 
    return (num/den).mean().item()
 
def DOT_metrics(benchmark, batch_size, flag_plan=False):
    mltp = -1. 
    
    ot_emd = ot.da.EMDTransport(metric = 'euclidean')
    if flag_plan == True:
        sample_p,sample_q = benchmark.input_sampler.sample( batch_size, flag_plan =  True)
    else:
        sample_p = benchmark.input_sampler.sample( batch_size, flag_plan = False)
        sample_q = benchmark.output_sampler.sample(batch_size)
    ot_emd.fit(Xs = sample_p.cpu().numpy(),Xt= sample_q.cpu().numpy())
    q_mapped      = ot_emd.transform(Xs = sample_p.cpu().numpy() )
    sample_p.requires_grad = True
    
    distance =  torch.norm(torch.from_numpy(q_mapped).cuda() - sample_p, dim=1).mean()
    distance.mean().backward()
    
    grad_dot = sample_p.grad*sample_p.shape[0]
    grad_lip =   mltp*grad(benchmark.net , sample_p )
    sample_p.requires_grad = False
    l2 = (((grad_lip - grad_dot).norm(dim=1))**2).mean()
    
    scalar_prod = torch.mul(grad_lip, grad_dot).sum(dim = 1)
    num = scalar_prod 
    den = torch.sqrt(grad_lip.norm(dim=1)**2 * grad_dot.norm(dim=1)**2) + 1e-15
    
    return distance.item() , l2.item(), ((num/den).mean()).item()