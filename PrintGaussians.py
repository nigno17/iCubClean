#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:47:24 2017

@author: nigno
"""
from __future__ import print_function, division

import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from torch.autograd import Variable

def plot_results(net_output, splot, ngauss = 5):
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
    
    D = 3
    torchType = torch.cuda.FloatTensor
    
    nMean = ngauss * D
    
    mean = net_output[:nMean]
#    cov = net_output[nMean:]
    
    mean = mean.resize(ngauss, D)
#    cov = cov.resize(ngauss, D, D).pow(2)
    cov2 = net_output[nMean:]
    
    cov2 = cov2.resize(ngauss, D, D)
    cov = Variable(torch.ones(ngauss, D, D).type(torchType))
    chol = Variable(torch.tril(torch.ones(D, D)).type(torchType))
    for i in range(ngauss):
        L = chol * cov2[i]
        cov[i] = L.mm(L.t())
    
    means = mean[:, 1:].clone()
    means = means.resize(ngauss, D - 1).cpu().data.numpy()
    
    covariances = cov[:, 1:, 1:].clone()
    covariances = covariances.resize(ngauss, D - 1, D - 1).cpu().data.numpy()
    
    
    #splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        #v = 2. * np.sqrt(2.) * np.sqrt(v)
        v = np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.3)
        splot.add_artist(ell)
        splot.scatter(mean[0], mean[1], s=40, marker='+', color='green')
    print('-------------------')
        
def plot_results_time(net_output, splot, xory = 'x', ngauss = 5):
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])
    
    D = 3
    torchType = torch.cuda.FloatTensor
    
    nMean = ngauss * D
    
    mean = net_output[:nMean]
#    cov = net_output[nMean:]
    
    mean = mean.resize(ngauss, D)
#    cov = cov.resize(ngauss, D, D).pow(2)
    cov2 = net_output[nMean:]
    
    cov2 = cov2.resize(ngauss, D, D)
    cov = Variable(torch.ones(ngauss, D, D).type(torchType))
    chol = Variable(torch.tril(torch.ones(D, D)).type(torchType))
    for i in range(ngauss):
        L = chol * cov2[i]
        cov[i] = L.mm(L.t())
    
    if xory == 'x':
        means = mean[:, :2].clone()
    else:
        means = mean[:, [0, 2]].clone()

    means = means.resize(ngauss, D - 1).cpu().data.numpy()
    
    if xory == 'x':
        covariances = cov[:, :2, :2].clone()
    else:
        covariances = cov[:, [0, 2], [0, 2]].clone()
    covariances = covariances.resize(ngauss, D - 1, D - 1).cpu().data.numpy()
    
    
    #splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        #v = 2. * np.sqrt(2.) * np.sqrt(v)
        v = np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        
        
        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.3)
        splot.add_artist(ell)
        splot.scatter(mean[0], mean[1], s=40, marker='+', color='green')

    #plt.xlim(-9., 5.)
    #plt.ylim(-3., 6.)
    #plt.xticks(())
    #plt.yticks(())
    #plt.title(title)