import numpy as np
import torch.nn as nn
# import torchpwl
# from scipy.io import loadmat
# from os.path import join
# import os
# from utils.fftc import *
import torch
import torch.nn.functional as F
# from torch.nn import init
# from utils import dataset





class ESINetADMMLayer(nn.Module):
    def __init__(
        self,
        L,
        in_channels: int = 1,
        out_channels: int = 256,
    ):
        """
        Args:

        """
        super(ESINetADMMLayer, self).__init__()

        self.rho = nn.Parameter(torch.tensor([600.0]), requires_grad=False)

        self.yita1 = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.yita2 = nn.Parameter(torch.tensor([1.0]),requires_grad=True)
        self.yita3 = nn.Parameter(torch.tensor([1.0]),requires_grad=True)

        self.miu1 = nn.Parameter(torch.tensor([1.0]),requires_grad=True)
        self.miu2 = nn.Parameter(torch.tensor([1.0]),requires_grad=True)

        self.lam1 = nn.Parameter(torch.tensor([0.001]), requires_grad=True)
        self.lam2 = nn.Parameter(torch.tensor([0.001]), requires_grad=True)


        # self.mask = mask
        self.re_org_layer = ReconstructionOriginalLayer(self.rho,L)
        self.re_update_layer = ReconstructionUpdateLayer(self.rho,L)
        self.re_final_layer = ReconstructionFinalLayer(self.rho,L)

        self.U_update_layer = U_update_layer(self.lam1)
        self.sublayer = sublayer(self.lam2,self.miu1,self.miu2)

        self.conv1_layer = convLayer1_forw(in_channels, out_channels)
        self.conv2_1_layer = convLayer2_1_forw(out_channels, in_channels)
        self.conv2_layer = convLayer2_forw(out_channels, in_channels)
        self.conv1_2_layer = convLayer1_2_forw(in_channels, out_channels)


        self.addlayer = addLayer()

        self.multiple_update_layer = MultipleUpdateLayer(self.yita1,self.yita2,self.yita3)

        layers = []
        #x的更新
        layers.append(self.re_org_layer)
        layers.append(self.addlayer)
        for i in range(2):
            layers.append(self.conv1_layer)
            layers.append(self.conv2_1_layer)
            layers.append(self.U_update_layer)
            layers.append(self.conv1_2_layer)
            layers.append(self.conv2_layer)
            layers.append(self.sublayer)
        layers.append(self.multiple_update_layer)
        #中间更新迭代部分
        for i in range(5):
            layers.append(self.re_update_layer)
            layers.append(self.addlayer)
            for i in range(2):
                layers.append(self.conv1_layer)
                layers.append(self.conv2_1_layer)
                layers.append(self.U_update_layer)
                layers.append(self.conv1_2_layer)
                layers.append(self.conv2_layer)
                layers.append(self.sublayer)
            layers.append(self.multiple_update_layer)

        layers.append(self.re_final_layer)

        self.cs_net = nn.Sequential(*layers)

    def forward(self, x):
        a = self.cs_net(x)
        return a


# reconstruction original layers
class ReconstructionOriginalLayer(nn.Module):
    def __init__(self, rho,L):
        super(ReconstructionOriginalLayer, self).__init__()
        self.rho = rho
        self.L = L.float()

    def forward(self, x):
        # L = x['L'].float()
        b = x['B_trans'].float().to(self.L.device)
        orig_output1 = woodbury_inv(self.L, self.rho)
        orig_output2 = torch.matmul(self.L.t(), b)
        u = 0
        z = 0
        orig_output3 = orig_output2 + torch.mul(self.rho,torch.sub(u,z))
        orig_output4 = torch.matmul(orig_output1, orig_output3)

        u = torch.zeros_like(orig_output4)
        z = torch.zeros_like(orig_output4)
        # define data dict
        eeg_data = dict()
        alpha = 0.6
        eeg_data['recon_output'] = alpha * orig_output4  + (1 - alpha) * u
        eeg_data['b'] = b
        eeg_data['s'] = orig_output4
        eeg_data['z'] = z

        return eeg_data


# reconstruction middle layers
class ReconstructionUpdateLayer(nn.Module):
    def __init__(self, rho,L):
        super(ReconstructionUpdateLayer, self).__init__()
        self.rho = rho
        self.L = L

    def forward(self, x):
        u = x['u']
        z = x['z']
        b = x['b']  # B
        orig_output1 = woodbury_inv(self.L, self.rho)
        orig_output2 = torch.matmul(self.L.t(),b)
        orig_output3 = torch.mul(self.rho, torch.sub(u,z))
        orig_output4 = torch.add(orig_output2, orig_output3)
        orig_output5 = torch.matmul(orig_output1, orig_output4)
        x['s'] = orig_output5
        alpha = 0.6
        x['recon_output'] = alpha * orig_output5 + (1-alpha) * u

        return x

class addLayer(nn.Module):
    def __init__(self):
        super(addLayer,self).__init__()
    def forward(self,x):
        recon_output = x['recon_output']
        z = x['z']
        x['u'] = recon_output + z
        return x


# reconstruction middle layers
class ReconstructionFinalLayer(nn.Module):
    def __init__(self, rho, L):
        super(ReconstructionFinalLayer, self).__init__()
        self.rho = rho
        self.L = L

    def forward(self, x):
        u = x['u']
        z = x['z']
        b = x['b']

        orig_output1 = woodbury_inv(self.L, self.rho)
        orig_output2 = torch.matmul(self.L.t(), b)
        orig_output3 = torch.mul(self.rho, torch.sub(u,z))
        orig_output4 = torch.add(orig_output2, orig_output3)
        orig_output5 = torch.matmul(orig_output1, orig_output4)

        x['s_final'] = orig_output5
        return x['s_final']




# multiple middle layer
class MultipleUpdateLayer(nn.Module):
    def __init__(self,yita1,yita2,yita3):
        super(MultipleUpdateLayer,self).__init__()
        self.yita1 = yita1
        self.yita2 = yita2
        self.yita3 = yita3

    def forward(self, x):
        u = x['u']
        s = x['s']
        z = x['z']
        output = self.yita1 * z + self.yita2 * s - self.yita3 * u
        x['z'] = output
        return x


# 定义ELU激活函数
class ELUActivation(nn.Module):
    def __init__(self, alpha=1.0):
        super(ELUActivation, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return F.elu(x, self.alpha)

# 定义BatchNormalization层
class BatchNormalization(nn.Module):
    def __init__(self, num_features):
        super(BatchNormalization, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.bn(x)




# convLayer1 layer
class convLayer1_forw(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(convLayer1_forw,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.linear = nn.Linear(input_size,output_size)
        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=(16,1),
                              stride=1,
                              padding=(8,0),
                              dilation=1,
                              groups=1,
                              bias=True)
        self.conv.weight.data.normal_(0, 0.02)
        if self.conv.bias is not None:
            self.conv.bias.data.normal_(0, 0.02)
        self.bn = BatchNormalization(num_features=self.out_channels)
        self.activation = ELUActivation()
    def forward(self, x):
        u = x['u']
        u_hat = u.unsqueeze(dim = 1)
        a = self.conv(u_hat)
        a = self.bn(a)
        a = self.activation(a)
        output = a
        x['vu_1'] = output
        return x


class convLayer1_2_forw(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(convLayer1_2_forw,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.linear = nn.Linear(input_size,output_size)
        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=(16,1),
                              stride=1,
                              padding=(8,0),
                              dilation=1,
                              groups=1,
                              bias=True)
        self.conv.weight.data.normal_(0, 0.02)
        if self.conv.bias is not None:
            self.conv.bias.data.normal_(0, 0.02)
        self.bn = BatchNormalization(num_features=self.out_channels)
        self.activation = ELUActivation()
    def forward(self, x):
        u = x['vu_n']
        u_hat = u.unsqueeze(dim = 1)
        a = self.conv(u_hat)
        a = self.bn(a)
        a = self.activation(a)
        output = a
        x['vu_n1'] = output
        return x

class Conv2DTranspose(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int,padding: tuple):
        super(Conv2DTranspose, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding)
        self.conv_transpose.weight.data.normal_(0, 0.02)
        if self.conv_transpose.bias is not None:
            self.conv_transpose.bias.data.normal_(0, 0.02)

    def forward(self, x):
        return self.conv_transpose(x)

class convLayer2_forw(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(convLayer2_forw,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_transpose = Conv2DTranspose(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=(16, 1),
                                              stride=1,
                                              padding=(8,0)
                                              )
        self.bn = BatchNormalization(num_features=self.out_channels)
        self.activation = ELUActivation()
    def forward(self, x):
        input = x['vu_n1']
        a = self.conv_transpose(input)
        a = self.bn(a)
        a = self.activation(a)
        output = a
        x['vu'] = output.squeeze(dim = 1)
        return x

class convLayer2_1_forw(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(convLayer2_1_forw, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_transpose = Conv2DTranspose(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=(16, 1),
                                              stride=1,
                                              padding=(8, 0)
                                              )
        self.bn = BatchNormalization(num_features=self.out_channels)
        self.activation = ELUActivation()

    def forward(self, x):
        input = x['vu_1']
        a = self.conv_transpose(input)
        a = self.bn(a)
        a = self.activation(a)
        output = a
        x['vu_2'] = output.squeeze(dim=1)
        return x


class U_update_layer(nn.Module):
    def __init__(self,lam1):
        super(U_update_layer, self).__init__()
        self.lam1 = lam1
    def forward(self,x):

        input = x['vu_2']
        output = vu_soft_thresholding_torch(input,self.lam1)
        x['vu_n'] = output
        return x


class sublayer(nn.Module):
    def __init__(self,lam2,miu1,miu2):
        super(sublayer, self).__init__()
        self.lam2 = lam2
        self.miu1 = miu1
        self.miu2 = miu2
    def forward(self,x):
        s = x['s']
        u = x['u']
        z = x['z']
        vu = x['vu']
        u_hat = u
        uo = vu_soft_thresholding_torch(u_hat,self.lam2)
        output = self.miu1 * u_hat + self.miu2 * (s + z) -vu -uo
        x['u'] = output

        return x

def woodbury_inv(L, rho):

   LT_L = torch.matmul(L.T, L)
   rho_I = rho * torch.eye(LT_L.shape[0], device=LT_L.device)
   M = torch.add(LT_L, rho_I)

   result = torch.inverse(M)
   return result

def vu_soft_thresholding_torch(y,lam):
    row_norm = torch.norm(y,p=2,dim=2,keepdim=True)
    threshold = lam / row_norm
    return torch.where(torch.abs(y) > threshold,
                       y - torch.sign(y) * threshold,
                       torch.zeros_like(y))

# def ou_soft_thresholding_torch(y,lam):
#     row_norm = torch.norm(y,p=2,dim=2,keepdim=True)
#     threshold = lam / row_norm
#     return torch.where(torch.abs(y) > threshold,
#                        y - torch.sign(y) * threshold,
#                        torch.zeros_like(y))

# def proxl1ARD(Y,lam):
#     return torch.mul(torch.sign(Y), F.relu(torch.abs(Y) - lam))


