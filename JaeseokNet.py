#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Nino Cauli
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import models


class JaeseokNetPretrained(torch.nn.Module):
    def __init__(self, D_latent = 4096, D_outputs = (3 + 3 * 3) * 5):
        
        super(JaeseokNetPretrained, self).__init__()
        self.D_features = 256 * 6 * 6
        
        alexNet = models.alexnet(pretrained=True)
        self.features = alexNet.features
        self.latent1 = nn.Sequential(
            nn.Linear(self.D_features, D_latent),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(D_latent, D_outputs)

    def forward(self, img1):

        features1 = self.features(img1)
        features1 = features1.view(features1.size(0), self.D_features)
        
        lat1 = self.latent1(features1)
        
        output = self.fc(lat1)
        
        return output
    
class JaeseokNet(torch.nn.Module):
    def __init__(self, D_latent = 4096, D_outputs = (3 + 6) * 5):
        
        super(JaeseokNet, self).__init__()
        self.D_features = 256 * 6 * 9
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.latent1 = nn.Sequential(
            nn.Linear(self.D_features, D_latent),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc = nn.Linear(D_latent, D_outputs)

    def forward(self, img1):

        features1 = self.features(img1)
        features1 = features1.view(features1.size(0), self.D_features)
        
        lat1 = self.latent1(features1)
        
        output = self.fc(lat1)
        
        return output