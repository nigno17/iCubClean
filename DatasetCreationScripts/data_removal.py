from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import xml.etree.ElementTree as ET
import cv2

plt.ion()   # interactive mode

res = 5

channels = 3

root_dir = 'Corrected_dataset4/'
root_dir_img = root_dir + 'new_dataset_jaeseok/'
root_dir_labels = root_dir + 'lables/'
rood_dir_traj = root_dir + 'rightarm_modified_trajectory/'

file_list = os.listdir(root_dir_img)

dataset_dim = len(file_list)
augmentation_num = 20
plain_back_num = 10

new_id = 0
for image_it in range(dataset_dim):
    image = io.imread(root_dir_img + str(image_it) + '.png')
    tree = ET.parse(root_dir_labels + str(image_it) + '.xml')
    root = tree.getroot()
    traj = np.loadtxt(rood_dir_traj + 'data' + str(image_it + 1) + '.txt', dtype='float')
    
    
    ## NEW PART ----------------------------------------
#    traj[:, 0] = (traj[:, 0] + 1.0) * 240
#    traj[:, 1] = (traj[:, 1] + (2 / 3)) * 240
#    
#    # Four corners of the book in source image
#    pts_src = np.array([[103, 84], [99, 222], [162, 91], [157, 223]])
# 
#    # Four corners of the book in destination image.
#    pts_dst = np.array([[120, 100], [120, 220], [168, 100], [168, 220]])
# 
#    # Calculate Homography
#    h, status = cv2.findHomography(pts_src, pts_dst)
#    
#    new_traj = np.concatenate((traj.T, np.ones((1, 200))), axis = 0)
#    
#    new_traj = h.dot(new_traj)
#    new_traj = new_traj / new_traj[2, :]
#    
#    traj[:, 0] = (new_traj[0, :] / 240) - 1.0
#    traj[:, 1] = (new_traj[1, :] / 240) - (2 / 3) 
    ## END NEW PART ----------------------------------------
    
    
    if image_it < 259 or image_it > 266:
        image_name = str(new_id) + '.png' 
        image_path = root_dir + 'new_dataset_jaeseok_cut/'   
        io.imsave(image_path + image_name, image)
        
        label_name = str(new_id) + '.xml' 
        label_path = root_dir + 'lables_cut/'
        tree.write(label_path + label_name)
        
        traj_name = 'data' + str(new_id + 1) + '.txt' 
        traj_path = root_dir + 'rightarm_modified_trajectory_cut/'
        np.savetxt(traj_path + traj_name, traj)
        
        new_id +=1


raw_input('Press enter to continue: ')

