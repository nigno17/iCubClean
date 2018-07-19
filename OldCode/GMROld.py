# -*- coding: utf-8 -*-
from __future__ import print_function, division
import torch
from torch.autograd import Variable
import numpy as np

def GMR(net_output, ngauss = 5):
    traj_len = 200
    D = 3
    torchType = torch.cuda.FloatTensor
    
    nMean = ngauss * D
    
    mean = net_output[:nMean]
    #cov = net_output[nMean:]
    
    mean = mean.resize(ngauss, D)
    #cov = cov.resize(ngauss, D, D).pow(2)
    cov2 = net_output[nMean:]
    
    cov2 = cov2.resize(ngauss, D, D)
    cov = Variable(torch.ones(ngauss, D, D).type(torchType))
    chol = Variable(torch.tril(torch.ones(D, D)).type(torchType))
    for i in range(ngauss):
        L = chol * cov2[i]
        cov[i] = L.mm(L.t())
    
    meant = mean[:, 0].clone()
    meant = meant.resize(1, ngauss).expand(traj_len, ngauss)
    means = mean[:, 1:].clone()
    means = means.resize(1, ngauss, D - 1, 1).expand(traj_len, ngauss, D - 1, 1)
    
    covt = cov[:, 0, 0].clone()
    covt = covt.resize(1, ngauss).expand(traj_len, ngauss)
    covs = cov[:, 1:, 1:].clone()
    covs = covs.resize(1, ngauss, D - 1, D - 1).expand(traj_len, ngauss, D - 1, D - 1)
    covst = cov[:, 1:, 0].clone()
    covst = covst.resize(1, ngauss, D - 1, 1).expand(traj_len, ngauss, D - 1, 1)
    covts = cov[:, 0, 1:].clone()
    covts = covts.resize(1, ngauss, 1, D - 1).expand(traj_len, ngauss, 1, D - 1)
    
    time = Variable(torch.arange(0, (traj_len - 1) / traj_len, 1 / traj_len).type(torchType)  , requires_grad=False)
    time = time.resize(traj_len, 1).expand(traj_len, ngauss)
    
    
    
    gaussians = 1.0 / ((2.0 * np.pi) * (covt)).sqrt() * \
                 (-1.0 / 2.0 * (time - meant).pow(2) / covt).exp()
                 
    estTrajList = means.squeeze() + covst.squeeze() * ((time - meant) / covt).resize(traj_len, ngauss, 1).expand(traj_len, ngauss, D - 1)
    estCovsList = covs - covst.expand(traj_len, ngauss, D - 1, D - 1) * covts.expand(traj_len, ngauss, D - 1, D - 1) / covt.resize(traj_len, ngauss, 1, 1).expand(traj_len, ngauss, D - 1, D - 1)
    
    beta = gaussians / (gaussians.sum(1) + 0.0000001).resize(traj_len, 1).expand(traj_len, ngauss)
    
    estTrajList = beta.resize(traj_len, ngauss, 1).expand(traj_len, ngauss, D - 1) * estTrajList        
    estCovsList = beta.pow(2).resize(traj_len, ngauss, 1, 1).expand(traj_len, ngauss, D - 1, D - 1) * estCovsList
         
    estTraj = estTrajList.sum(1) 
    estCovs = estCovsList.sum(1)
    
    return estTraj