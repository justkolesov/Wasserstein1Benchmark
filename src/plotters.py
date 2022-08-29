import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
 
from sklearn.decomposition import PCA 
from sklearn.manifold import Isomap, TSNE 
from .map_benchmark import  MixToOneBenchmark
from .utils import get_borders

 

def PCA_plot_q_p_samples(benchmark, num_samples, fix, fiy, path, flag_p=True):  
    
    q_data = benchmark.input_sampler.sample(num_samples, flag_plan = False)
    p_data = benchmark.output_sampler.sample(num_samples)
    
    fig,axes = plt.subplots( 1, 2,figsize=(fix,fiy),squeeze=True,sharex=True,sharey=True)
    pca = PCA(n_components=2).fit(p_data.cpu().numpy() if flag_p else q_data.cpu().numpy()) 
    p_data_emb = pca.transform(p_data.cpu().numpy())
    q_data_emb = pca.transform(q_data.cpu().numpy())
    
    axes[0].scatter( p_data_emb[:,0], p_data_emb[:,1], c="bisque", edgecolor = 'black', label = r'$x\sim\mathbb{P}$', s =30)
    axes[0].legend(framealpha = 1, loc = "lower left",fontsize='xx-large',edgecolor='black',handletextpad=0.1,handlelength=1)
    if isinstance(benchmark,MixToOneBenchmark):
        axes[0].set_xticks([-4,-2,0,2,4])
        axes[0].set_yticks([-4,-2,0,2,4])
        axes[0].tick_params(axis='both', which='major', labelsize=13)
    axes[1].scatter(q_data_emb[:,0] , q_data_emb[:,1], c="lightcoral", edgecolor = 'black', label = r'$y\sim\mathbb{Q}$', s = 30)
    axes[1].legend(framealpha = 1, loc = "lower left",fontsize='xx-large',edgecolor='black',handletextpad=0.1,handlelength=1) 
    if isinstance(benchmark,MixToOneBenchmark):
        axes[1].tick_params(axis='both', which='major', labelsize=13)
    fig.tight_layout(pad=0.5)
    if path != None:
        plt.savefig(path)
    return   fig,axes
 
def vecs_to_plot(x, shape=(3, 64, 64)):
    return  x.reshape(-1, *shape).permute(0, 2, 3, 1).mul(0.5).add(0.5).cpu().numpy().clip(0, 1)

def plot_images( benchmark, num_samples, path, shape=(3, 64, 64)):
    samples_q, samples_p = benchmark.input_sampler.sample(num_samples, flag_plan = True) 
    imgs_p  = vecs_to_plot(samples_p, shape)
    imgs_q = vecs_to_plot(samples_q, shape)
    fig, axes = plt.subplots(2, 10, figsize=(10.3, 2), dpi=150)
    axes[0, 0].set_ylabel(r'$x\sim\mathbb{P}$', fontsize= 'x-large')
    axes[1, 0].set_ylabel(r'$T(x)\sim\mathbb{Q}$', fontsize= 'x-large')
    for idx in range(10):
        axes[0,idx].imshow(imgs_p[idx])
        axes[0,idx].get_xaxis().set_visible(False)
        axes[0,idx].set_yticks([])
    for idx in range(10):
        axes[1,idx].imshow(imgs_q[idx])
        axes[1,idx].get_xaxis().set_visible(False)
        axes[1,idx].set_yticks([])
   
    fig.tight_layout(pad=0.005)
    if path != None:
        plt.savefig(path)
    return fig, axes    
  
def plot_benchmark_2d_info(benchmark, num_samples, fix, fiy, cmap):
    N = 2000
    xlist = np.linspace(-5,5,N)
    ylist = np.linspace(-5,5,N)
    Xm,Ym = np.meshgrid(xlist,ylist)
    Z  = np.concatenate([Xm.reshape(-1,1),Ym.reshape(-1,1)],axis=1)
    Z = torch.tensor(Z,dtype = torch.float32, device="cuda")
    Z_val = benchmark.net(Z).detach().cpu().numpy().reshape(N,N)
    samples_q_, samples_p_ =  benchmark.input_sampler.sample(num_samples,flag_plan =True)
    fig,axes = plt.subplots(1, 1, figsize=(fix,fiy), dpi =150)
    axes.contourf(Xm,Ym,Z_val ,levels=25, cmap =  cmap)
    semi_length_square = benchmark.semi_length_square
    axes.add_patch(Rectangle((- semi_length_square, - semi_length_square),
                                      2*semi_length_square,
                                      2*semi_length_square,
                                      edgecolor = "black",fill=False,linewidth=0.5))
 
    lower, upper = get_borders(samples_p_.clone(), benchmark.net,   semi_length_square)
    axes.scatter(samples_p_[:,0].detach().cpu() ,samples_p_[:,1].detach().cpu(),
                           c='bisque', edgecolors='black', zorder=5 )
    axes.scatter(samples_q_[:,0].detach().cpu() ,samples_q_[:,1].detach().cpu(),
                           c='lightcoral', edgecolors='black', zorder=5 )
    axes.scatter(upper[:,0].cpu(), upper[:,1].cpu(), c='white', edgecolors='black', zorder=5, marker='s')
    axes.scatter(lower[:,0].cpu(), lower[:,1].cpu(),c = 'darkgrey',edgecolors='black', zorder=5, marker='s' )
 
    for low,up in zip(lower,upper):
        d = torch.vstack([low,up]).T
        axes.plot( d[0].cpu() , d[1].cpu()  , c = 'black',linewidth=0.5 )
    axes.set_xlim([- semi_length_square - 0.2, semi_length_square + 0.2])
    axes.set_ylim([- semi_length_square - 0.2, semi_length_square + 0.2])
    fig.tight_layout(pad=0.05)
    return fig, axes


def plot_teaser(benchmark, batch_size, path, fix, fiy ):
    fig,axes = plt.subplots(1, 4, figsize = (fix,fiy), dpi = 150 ,sharex = True, sharey = True)
 
    samples_q , samples_p = benchmark.input_sampler.sample(batch_size ,flag_plan=True)
    
    axes[0].scatter(
        samples_p.cpu()[:,0], samples_p.cpu()[:,1], c='bisque',
        edgecolors='black', label=r'$x\sim\mathbb{P}$',zorder=2)
    axes[0].set_xlim([-2.7,2.7])
    axes[0].set_ylim([-2.7,2.7])
    axes[0].legend(
        framealpha = 1, loc = 'lower left',fontsize= 'large',
        edgecolor = 'black', handlelength = 1 ,handletextpad = 0.1
    )
 
    N = 2000
    xlist = np.linspace(-5,5,N)
    ylist = np.linspace(-5,5,N)
    Xm,Ym = np.meshgrid(xlist,ylist)
    Z  = np.concatenate([Xm.reshape(-1,1),Ym.reshape(-1,1)],axis=1)
    Z = torch.tensor(Z,dtype = torch.float32,device = 'cuda')
    Z_val = benchmark.net(Z).detach().cpu().numpy().reshape(N,N)
    axes[1].contourf(Xm,Ym,Z_val,levels=25,cmap = 'pink')
    axes[1].set_xlim([-2.7,2.7])
    axes[1].set_ylim([-2.7,2.7])
  
    N = 2000
    xlist = np.linspace(-5,5,N)
    ylist = np.linspace(-5,5,N)
    Xm,Ym = np.meshgrid(xlist,ylist)
    Z  = np.concatenate([Xm.reshape(-1,1),Ym.reshape(-1,1)],axis=1)
    Z = torch.tensor(Z,dtype = torch.float32,device='cuda')
    Z_val = benchmark.net(Z).detach().cpu().numpy().reshape(N,N)
    axes[2].contourf(Xm,Ym,Z_val,levels=25,cmap = 'pink')
    semi_length_square = benchmark.semi_length_square
    axes[2].add_patch(
        Rectangle(
            (-semi_length_square,-semi_length_square),
            2*semi_length_square, 2*semi_length_square,
            edgecolor="black", fill=False, linewidth=0.5
        )
    )
        
    samples_q_ , samples_p_ = benchmark.input_sampler.sample(batch_size//10 ,flag_plan=True)
    lower , upper = get_borders(samples_p_, benchmark.net, semi_length_square)
    axes[2].scatter(
        samples_p_[:,0].detach().cpu(), samples_p_[:,1].detach().cpu(),
        c = 'bisque', edgecolor = 'black', zorder = 5,
    )
    axes[2].scatter(
        samples_q_[:,0].detach().cpu(), samples_q_[:,1].detach().cpu(),
        c = 'lightcoral', edgecolor='black', zorder = 5,
    )
    axes[2].scatter(
        lower[:,0].cpu(), lower[:,1].cpu(), c='darkgrey' , marker='s',
        edgecolor= 'black',  label = 'Lower end',zorder=5
    )
    axes[2].scatter(
        upper[:,0].cpu(), upper[:,1].cpu(),
        c='white', marker = 's', edgecolor= 'black', label='Upper end', zorder=5
    )
 
    for low,up in zip(lower,upper):
        d = torch.vstack([low,up]).T
        axes[2].plot( d[0].cpu() , d[1].cpu()  , c = 'black',linewidth=0.5 )
    axes[2].legend(
        framealpha=1,loc="lower left", edgecolor='black',
        handlelength=1,handletextpad=0.1
    )
 
    axes[3].scatter(samples_q.cpu().numpy()[:,0],samples_q.cpu().numpy()[:,1],  c='lightcoral', edgecolors='black', 
                    label=r'$y\sim\mathbb{Q}$',zorder=5)
 
    axes[3].legend(
        framealpha=1, loc="lower left", fontsize='large',
        edgecolor='black', handlelength=1, handletextpad=0.1
    )
    axes[3].set_xlim([-2.7,2.7])
    axes[3].set_ylim([-2.7,2.7])
    fig.tight_layout(pad=0.5)
    plt.savefig(path)
    return fig, axes

def plot_surfaces(benchmark,  method, path, fix, fiy):
    
    fig,axes = plt.subplots(2, 5, dpi=150,  figsize = (fix, fiy), sharex = True, sharey = True)
    for i,key in enumerate(method.keys()):
        mltp = 1. if key == "3p-wgan_rev" else -1.
        N = 2000
        j = 0 if i < 5 else 1
        k = i if i < 5 else i - 5
        xlist = np.linspace(-5,5,N)
        ylist = np.linspace(-5,5,N)
        Xm,Ym = np.meshgrid(xlist,ylist)
        Z  = np.concatenate([Xm.reshape(-1,1),Ym.reshape(-1,1)],axis=1)
        Z = torch.tensor(Z,dtype = torch.float32)
        method[key] = method[key].cpu()
        Z_val = mltp*method[key](Z).detach().cpu().numpy().reshape(N,N)
        axes[j,k].contourf(Xm,Ym,Z_val,levels=25,cmap = 'pink')
        axes[j,k].set_xlim([-2.7,2.7])
        axes[j,k].set_ylim([-2.7,2.7])
    
    fig.tight_layout(pad = 0.5)
    plt.savefig(path)
    return fig, axes

def plot_rays(benchmark, batch_size, path):
 
    N = 2000
    xlist = np.linspace(-5,5,N)
    ylist = np.linspace(-5,5,N)
    Xm,Ym = np.meshgrid(xlist,ylist)
    Z  = np.concatenate([Xm.reshape(-1,1),Ym.reshape(-1,1)],axis=1)
    Z = torch.tensor(Z,dtype = torch.float32,device="cuda")
    Z_val = benchmark.net(Z).detach().cpu().numpy().reshape(N,N)
    fig,ax = plt.subplots(1,figsize=(14,14),dpi=150)
    ax.contourf(Xm,Ym,Z_val,levels=50,cmap = 'pink')
    semi_length_square = benchmark.semi_length_square
    ax.add_patch(Rectangle((-semi_length_square,-semi_length_square),
                                      2*semi_length_square,
                                      2*semi_length_square,edgecolor = "black",fill=False,linewidth=0.5))

    samples_q_ , samples_p_ = benchmark.input_sampler.sample(batch_size ,flag_plan=True)
    lower , upper = get_borders(samples_p_, benchmark.net, semi_length_square)
    ax.scatter(
        lower[:,0].cpu(), lower[:,1].cpu(), c='darkgrey',
        marker='s', edgecolor='black', label='Lower end',zorder = 5, s=70
    )
    ax.scatter(
        upper[:,0].cpu(), upper[:,1].cpu(), c='white', marker='s',
        edgecolor='black', label='Upper end', zorder=5, s=70
    )
 
    for low,up in zip(lower,upper):
        d = torch.vstack([low,up]).T
        ax.plot( d[0].cpu() , d[1].cpu()  , c = 'black',linewidth=0.5 )
    ax.set_xlim([-2.7,2.7])
    ax.set_ylim([-2.7,2.7])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(
        framealpha=1, loc="lower left", edgecolor='black', handlelength=2,
        handletextpad=0.5, fontsize = 'xx-large')
    plt.savefig(path)
    return fig,ax

