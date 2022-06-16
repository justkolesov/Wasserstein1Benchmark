import torch 
import torch.nn as nn
import torch.nn.functional as F
import geotorch
import numpy as np
import gc
#--------------- Funnel -----------#  
class Funnel(torch.nn.Module):
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers =  torch.nn.Parameter( torch.randn(out_features, in_features) )
        self.bias = torch.nn.Parameter(torch.randn(out_features))
         
    def forward(self, x):
        """
        x      - torch.Size([B,N])
        output - torch.Size([B,out_features])
        """
        output = torch.cdist(x, self.centers) + self.bias.repeat(x.shape[0], 1)
        return output
    
#-------------- MinFunnels -----------#
class  MinFunnels(torch.nn.Module):
    
    def __init__(self, dim, width):
        super().__init__()
        self.dim = dim
        self.width =  width
        self.net = Funnel(in_features = dim, out_features = width)
                       
    def forward(self, x):                      
        """
        x      - torch.Size([B,N])
        output - torch.Size([B,1])
        """
        output = self.net(x)
        return output.min(dim = 1)[0].unsqueeze(-1)
    
#-----------------WGAN--------------#

class Clipper(torch.nn.Module):
    
    def __init__(self , module , bound ,  name = 'weight'):
        """
        bound : int (bound of Compact set)
        """
        super().__init__()
        self.module = module
        self.bound =  bound
        self.name  =  name
        self.clipping()
        
    def clipping(self):
        self.module._parameters[self.name].data.clamp_(min = -self.bound , max = self.bound)
        
    def forward(self, x):
        self.clipping()
        return self.module.forward(x)    
    
#----------------SN-GAN----------------#
def l2normalize(v,eps=1e-15):
    return v / (torch.norm(v) + eps)
 
class Spectralnorm(torch.nn.Module):
    # https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py
    
    def __init__(self, module, power_iterations = 1, name = 'weight' ):
        super().__init__()
        self.module = module
        self.power_iterations = power_iterations
        self.name = name
        if not self._made_params():
            self._make_params()
        
    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        height = w.data.shape[0] # C_out
        for iteration in range(self.power_iterations):
            v.data = l2normalize(torch.mv(  torch.t( w.view(height,-1).data) , u.data)   )
            u.data = l2normalize(torch.mv( w.view(height, -1).data , v.data))
         
        pre_sigma = torch.mv( w.view(height,-1).data, v.data   )
        sigma = torch.mv( u.view(1,-1).data, pre_sigma)
        setattr(self.module, self.name , w/ sigma.expand_as(w))
           
    def _made_params(self):
         try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
         except AttributeError:
            return False
 
    def _make_params(self):
        w = getattr(self.module , self.name)
        height = w.data.shape[0] # c_out
        width  = w.data.shape[-1] # c_in
        u = torch.nn.Parameter(w.data.new(height).normal_(0,1), requires_grad  = False)
        v = torch.nn.Parameter(w.data.new(width).normal_(0,1),  requires_grad  =  False)
        w_bar = torch.nn.Parameter(w.data, requires_grad = True)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)  
        self.module.register_parameter(self.name + "_v", v)  
        self.module.register_parameter(self.name + "_bar", w_bar) # requires grad
        
    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)    
    
#------------ Critic --------#
class Critic(torch.nn.Module):
    
    def __init__(self, hidden_dims , method, bound , power_iters ):
        
        super().__init__()
        self.hidden_dims = hidden_dims
        model = []
        for num in range(1, len(hidden_dims)):
            
            layer = torch.nn.Linear(hidden_dims[num - 1], hidden_dims[num], bias = True )
            if method == "WGAN":
                assert bound > 0
                model.append( Clipper(layer, bound)) 
            elif method == "SN-GAN":
                assert isinstance(power_iters, int)
                assert power_iters > 0
                model.append(Spectralnorm( layer, power_iterations = power_iters )) 
            else:
                model.append(layer)
                
            model.append(torch.nn.ReLU())
            
        model.pop()
        self.net = torch.nn.Sequential(*model)
             
    def forward(self, x):
        return self.net(x)
#-------------- Sorting-out ------------#

def process_group_size(x , num_units , axis = -1):
    # https://github.com/cemanil/LNets/tree/master/lnets/models/activations
    size = list(x.size()) # torch.tensor of shape B x C_out
    num_channels = size[axis] # C_out
    if num_channels % num_units != 0:
        raise ValueError("num channels is {}, but num units is {}".format(num_channels, num_units))
    
    size[axis] = -1
    if axis == -1:
        size += [num_channels//num_units]  
    else:
        size.insert(axis + 1, num_channels//num_units)
        
    return size

class GroupSort(torch.nn.Module):
    # https://github.com/cemanil/LNets/blob/master/lnets/models/activations/group_sort.py
    def __init__(self, num_units, axis=-1):
 
        super().__init__()
        self.num_units = num_units
        self.axis      = axis
        
    def forward(self, x):
        assert x.shape[1] % self.num_units == 0
        size = process_group_size(x, self.num_units, self.axis) # torch.tensor of shape B x -1 x (C_out / num_units)
        grouped_x = x.view(*size)
        sort_dim = self.axis  if self.axis == -1 else axis + 1
        sorted_grouped_x, _ = grouped_x.sort(dim = sort_dim,descending = True) # torch.tensor of shape B x n_units x n_in_group
        sorted_x = sorted_grouped_x.view(*[x.shape[0],x.shape[1]])
        return sorted_x 
    
class OrthoLinear(torch.nn.Module):
    
    def __init__(self, in_features, out_features):
        
        super().__init__()
        self.model = torch.nn.Linear(in_features, out_features, bias = True)
        self.orthogonolize()
            
    def orthogonolize(self):
        geotorch.orthogonal(self.model, tensor_name = "weight")
    
    def forward(self, x):
        return  self.model(x)
    
class Critic_sort(torch.nn.Module):
    
    def __init__(self, hidden_dims, num_units):
        
        super().__init__()
        self.hidden_dims = hidden_dims
        model = []
        for num in range(1, len(hidden_dims)):
            model.append( OrthoLinear(hidden_dims[num-1],hidden_dims[num]) )
            model.append( GroupSort(num_units))
        model.pop()
        self.net = torch.nn.Sequential(*model)
    
    def forward(self, x):
        return self.net(x)
       
#------------ Convolutional nets --------#
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
     
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class UNet(nn.Module):
    
    def __init__(self, n_channels, n_classes, base_factor , bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_factor = base_factor

        self.inc = DoubleConv(n_channels, base_factor)
        self.down1 = Down(base_factor, 2 * base_factor)
        self.down2 = Down(2 * base_factor, 4 * base_factor)
        self.down3 = Down(4 * base_factor, 8 * base_factor)
        factor = 2 if bilinear else 1
        self.down4 = Down(8 * base_factor, 16 * base_factor // factor)
        
        self.up1 = Up(16 * base_factor, 8 * base_factor // factor, bilinear)
        self.up2 = Up(8 * base_factor, 4 * base_factor // factor, bilinear)
        self.up3 = Up(4 * base_factor, 2 * base_factor // factor, bilinear)
        self.up4 = Up(2 * base_factor, base_factor, bilinear)
        self.outc = OutConv(base_factor, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    
#------------ DC-GAN ----------#    
class Discriminator(nn.Module):
    def __init__(self, in_channels, ndf, method, flag_convolution, bound = None , power_iters = None):
        super().__init__()
        self.in_channels = in_channels
        self.ndf = ndf
        self.method = method
        self.flag_convolution = flag_convolution
        if flag_convolution == True:
            if method == "sn-gan":
                assert isinstance(power_iters, int)
                assert power_iters > 0
                func = lambda layer : Spectralnorm(layer, power_iters)
            elif method == "wgan":
                assert bound > 0
                func = lambda layer : Clipper(layer, bound)
            else:
                func = lambda layer : layer
                
            self.net = torch.nn.Sequential(
                func(torch.nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=True )),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                func(torch.nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True )) ,
                #torch.nn.BatchNorm2d(ndf * 2 ),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                func(torch.nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True )),
                #torch.nn.BatchNorm2d(ndf * 4 ),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                func(torch.nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True )),
                #torch.nn.BatchNorm2d(ndf * 8 ),
                torch.nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                func(torch.nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True ) ),)
                #torch.nn.Sigmoid() -- we remove, because ...       
        else:  
            self.net =  UNet(n_channels = in_channels, n_classes = in_channels ,base_factor = 24) 
            
    def forward(self,x):
        assert x.shape[1] == 3*64*64
        x = x.view(-1,3,64,64)
        out = self.net(x).reshape(x.shape[0],1) if self.flag_convolution == True else self.net(x).reshape(x.shape[0],3*64*64)
        return out
    
class ResNetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, bn=True, res_ratio=0.1):
        super().__init__()
        self.bn = bn
        self.is_bias = not bn
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden
        self.res_ratio = res_ratio

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_0 = nn.BatchNorm2d(self.fhidden)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_1 = nn.BatchNorm2d(self.fout)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)
            if self.bn:
                self.bn2d_s = nn.BatchNorm2d(self.fout)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(x)
        if self.bn:
            dx = self.bn2d_0(dx)
        dx = self.relu(dx)
        dx = self.conv_1(dx)
        if self.bn:
            dx = self.bn2d_1(dx)
        out = self.relu(x_s + self.res_ratio*dx)
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
            if self.bn:
                x_s = self.bn2d_s(x_s)
        else:
            x_s = x
        return x_s

#-----------13.04.2022--------#

class ResNet_G(nn.Module):
    "Generator ResNet architecture from https://github.com/harryliew/WGAN-QC"
    def __init__(self, z_dim, size, nfilter=64, nfilter_max=512, bn=True, res_ratio=0.1, **kwargs):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max
        self.bn = bn
        self.z_dim = z_dim

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**(nlayers+1))

        self.fc = nn.Linear(z_dim, self.nf0*s0*s0)
        if self.bn:
            self.bn1d = nn.BatchNorm1d(self.nf0*s0*s0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        blocks = []
        for i in range(nlayers, 0, -1):
            nf0 = min(nf * 2**(i+1), nf_max)
            nf1 = min(nf * 2**i, nf_max)
            blocks += [
                ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
                ResNetBlock(nf1, nf1, bn=self.bn, res_ratio=res_ratio),
                nn.Upsample(scale_factor=2)
            ]

        nf0 = min(nf * 2, nf_max)
        nf1 = min(nf, nf_max)
        blocks += [
            ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
            ResNetBlock(nf1, nf1, bn=self.bn, res_ratio=res_ratio)
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, 3, 3, padding=1)

    def forward(self, z):
        batch_size = z.size(0)
        z = z.view(batch_size, -1)
        out = self.fc(z)
        if self.bn:
            out = self.bn1d(out)
        out = self.relu(out)
        out = out.view(batch_size, self.nf0, self.s0, self.s0)

        out = self.resnet(out)

        out = self.conv_img(out)
        out = torch.tanh(out)

        return out
    
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(-1, *self.shape)
    
def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    

def load_resnet_G(cpkt_path, device='cuda'):
    
    resnet = torch.nn.Sequential(
        ResNet_G(128, 64, nfilter=64, nfilter_max=512, res_ratio=0.1),
        View(64*64*3)
    )
    resnet[0].load_state_dict(torch.load(cpkt_path))
    resnet = resnet.to(device)
    freeze(resnet)
    gc.collect(); torch.cuda.empty_cache()
    return resnet    
    
    
    
    
    