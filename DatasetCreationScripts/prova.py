from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import noise
import random
import xml.etree.ElementTree as ET


root_dir = 'Corrected_dataset4/'
root_dir_img = root_dir + 'aug_dataset_cut/'
root_old_traj_dir = root_dir + 'rightarm_modified_trajectory_cut/'
root_new_traj_dir = root_dir + 'aug_trajectories_cut/'

new_list = os.listdir(root_dir_img)
old_list = os.listdir(root_old_traj_dir)

new_dim = len(new_list)
old_dim = len(old_list)

for rep in range(new_dim):
    old_count = rep % old_dim
    traj = np.loadtxt(root_old_traj_dir + 'data' + str(old_count + 1) + '.txt', dtype='float')
    
    np.savetxt(root_new_traj_dir + str(rep) + '.txt', traj)
    