import torch
import torch.autograd as autograd
import tqdm
import numpy as np
 
def grad(net, X):
    return autograd.grad(net(X), X, grad_outputs=torch.ones(X.shape[0], 1, device = X.device) )[0]

def lipschitz_one_checker(net , points , dim = 1):
 
    points.requires_grad = True
    grad_ = grad(net,points) 
    points.requires_grad = False 
    return grad_ , grad_.norm(dim = dim) 

def get_borders(x, net, half_length):
 
    assert x.max() <= half_length
    assert x.min() >= -half_length
    assert net.net.centers.max() <= half_length
    assert net.net.centers.min() >= - half_length
    
    with torch.no_grad():
        # Finding the lower point
        funnels_x = torch.cdist(x, net.net.centers) + net.net.bias.repeat(x.shape[0], 1)
        idx = funnels_x.argmin(dim=1)
        low_x = net.net.centers[idx]
        
        # Finding the 
        v = (x-low_x); v = (v.T / v.norm(dim=1)).T
        r_max = torch.abs((torch.sign(v) * half_length - x) / v).min(dim=1)[0]
        max_up_x = x + (v.T * r_max).T # This is not needed anymore
        
        # Computing the upper point
        d, n, bs = net.net.centers[0].shape[0], len(net.net.centers), len(x)
        a = net.net.centers.data
        b = net.net.bias.data
        # Do nothing if there is only 1 funnel
        
        # Act if there are more funnels
        da = (x[:, :, None].repeat((1,1,n)) - a[None].transpose(2,1).repeat(bs,1,1)).transpose(2,1)
        db = funnels_x.min(dim=1)[0][:, None].repeat(1,n) - b[None].repeat(bs, 1)
        r = 0.5 * (da.norm(dim=2).square() - db.square()) / (db - torch.bmm(da, v[:, :, None]).squeeze(2))
        #print(torch.any(r <= -db))
        r[r <= 0] = torch.nan
        #print(torch.any(r <= -db))
        r[r <= -db] = torch.nan
         
        r = torch.nan_to_num(r, nan=torch.inf).min(dim=1)[0]
        
        # Truncate ray
        r_true = torch.stack([r_max, r]).min(dim=0)[0]
        
        int_up_x = x + (v.T * r_true).T
        
    return  low_x, int_up_x
 
def motion_samples(x , semi_length_square, net,  deg ):
    
    """
     x            - torch.Size([B,N]) - samples of P distribution, it doesn't request gradients
    network      - torch.nn.Module   - strongly 1 Lipschitz net 
    deg          - int
    
    returns torch.Size([B, N]), that doesn't request gradients
     """
    low_point, up_point = get_borders(x.clone(), net,  semi_length_square)
    with torch.no_grad():
        
        full_length = torch.norm(up_point - low_point ,dim = -1)
        base_length = torch.norm(x - low_point, dim = -1)
        mask = torch.logical_or(full_length  == 0 , base_length == 0)
        mask = mask.view(-1)
      
        t_value = torch.empty_like(base_length)
        t_value[~mask] = base_length[~mask]/ full_length[~mask]
        t_value[mask] = torch.zeros(1)

        new_t_value = torch.pow(t_value, exponent= deg ) # I CHANGED THIS TO DEG, not 1/DEG
        assert (new_t_value <= t_value).all()
         
        moved_x = torch.empty_like(low_point)
        moved_x[~mask] = low_point[~mask] + new_t_value.reshape(-1,1)*(up_point[~mask] - low_point[~mask])
        moved_x[mask] = low_point[mask]

    return moved_x, low_point, up_point  