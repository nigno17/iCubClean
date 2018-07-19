from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

def show_trajectories(image, trajectories):
    plt.scatter(trajectories[:, 0], trajectories[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, trajectories = sample['image'], sample['trajectories']

        new_h, new_w = self.output_size

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'trajectories': trajectories}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, trajectories = sample['image'], sample['trajectories']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'trajectories': torch.from_numpy(trajectories).float()}
        
class Normalize(object):
    """Convert ndarrays in sample to Tensors.
    
    Args:
        mean: Mean of rgb channels
        std: Standard deviation of rgb channels
    """
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        

    def __call__(self, sample):
        image, trajectories = sample['image'], sample['trajectories']

        image = (image - self.mean) / self.std
        
        #trajectories = (trajectories - trajectories.mean()) / trajectories.std()
        #print('mean: ' + str(trajectories.mean()))
        #print('mean: ' + str(trajectories.std()))
        
        return {'image': image,
                'trajectories': trajectories}
        
class Shift(object):
    """Convert ndarrays in sample to Tensors.
    
    Args:
        mean: Mean of rgb channels
        std: Standard deviation of rgb channels
    """
    
    def __init__(self, x_min, x_max, y_min, y_max, meter2pixel = 240):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.meter2pixel = meter2pixel
        

    def __call__(self, sample):
        image, trajectories = sample['image'], sample['trajectories']
        
        new_x_shift = (random.random() * (self.x_max - self.x_min)) + self.x_min
        new_y_shift = (random.random() * (self.y_max - self.y_min)) + self.y_min
        
        non = lambda s: s if s<0 else None
        mom = lambda s: max(0,s)
        
        shift_hight = int(new_x_shift * self.meter2pixel)
        shift_width = int(new_y_shift * self.meter2pixel)
        
        shift_image = np.zeros_like(image)
        shift_image[mom(shift_hight):non(shift_hight), mom(shift_width):non(shift_width)] = image[mom(-shift_hight):non(-shift_hight), mom(-shift_width):non(-shift_width)]
        
        trajectories[:, 0] += new_x_shift
        trajectories[:, 1] += new_y_shift
        #trajectories = (trajectories - trajectories.mean()) / trajectories.std()
        #print('mean: ' + str(trajectories.mean()))
        #print('mean: ' + str(trajectories.std()))
        
        return {'image': shift_image,
                'trajectories': trajectories}
        
class ChangeLight(object):
    """Convert ndarrays in sample to Tensors.
    
    Args:
        mean: Mean of rgb channels
        std: Standard deviation of rgb channels
    """
    
    def __init__(self, delta, percentage = 1.0):
        self.delta = delta
        self.percentage = percentage

    def __call__(self, sample):
        image, trajectories = sample['image'], sample['trajectories']
        
        if random.random() <= self.percentage:
            brigt_shift = [(random.random() * 2 * self.delta) - self.delta, 
                           (random.random() * 2 * self.delta) - self.delta,
                           (random.random() * 2 * self.delta) - self.delta]
            for i in range(image.shape[2]):
                non_zeros = np.nonzero(image[:, :, i])
                image[non_zeros[0], non_zeros[1], i] += brigt_shift[i]
            
            image[np.nonzero(image >= 1)] = 1
            image[np.nonzero(image < 0)] = 0
    
        
        #trajectories = (trajectories - trajectories.mean()) / trajectories.std()
        #print('mean: ' + str(trajectories.mean()))
        #print('mean: ' + str(trajectories.std()))
        
        return {'image': image,
                'trajectories': trajectories}
    
class JaeseokDataset(Dataset):
    """Face trajectories dataset."""

    def __init__(self, root_dir, training_per, transform=None, dset_type='train', seed=1, permuted= True):
        """
        Args:
            indices_file (string): Path to the txt file with image indices.
            traj_dir (string): Directory with all the trajectories.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.img_dir = root_dir + 'aug_dataset_cut/'
        self.traj_dir = root_dir + 'aug_trajectories_cut/'
        self.old_img_dir = root_dir + 'new_dataset_jaeseok_cut/'
        
        file_list_old = os.listdir(self.old_img_dir)
        self.old_nframes = len(file_list_old)
        
        file_list = os.listdir(self.img_dir)
        self.nframes = len(file_list)

        np.random.seed(seed)
        if permuted == True:
            permuted_indeces = np.random.permutation(range(self.old_nframes))
        else:
            permuted_indeces = range(self.old_nframes)
        
        self.train_number = int(self.old_nframes * training_per)
        if dset_type == 'train':
            self.indices = permuted_indeces[:self.train_number]
        else:
            self.indices = permuted_indeces[self.train_number:]        
        
        self.root_dir = root_dir
        self.transform = transform
        self.dset_type = dset_type

    def __len__(self):
        if self.dset_type == 'train':
            dset_size = self.train_number * (self.nframes // self.old_nframes)
        else:
            dset_size = len(self.indices)
        return dset_size

    def __getitem__(self, idx):
        if self.dset_type == 'train':
            local_idx = idx % self.train_number
            i = self.indices[local_idx] + ((idx // self.train_number) * self.old_nframes)
        else:
            i = self.indices[idx] + self.nframes - self.old_nframes
        img_name = self.img_dir + str(i) + '.png'
        #image = io.imread(img_name)
        cv_image = cv2.imread(img_name)
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        traj_name = self.traj_dir + str(i) + '.txt'
        trajectories = np.loadtxt(traj_name, dtype='float')
        #trajectories = trajectories.reshape(-1, 2)
        sample = {'image': image, 'trajectories': trajectories}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
#train_dataset = JaeseokDataset(indices_file = 'data/text_train_test_set/train_set.txt',
#                               traj_dir = 'data/modified_trajectory/',
#                               root_dir = 'data/training_set/')
#
#fig = plt.figure()
#
#for i in range(len(train_dataset)):
#    sample = train_dataset[i]
#
#    print(i, sample['image'].shape, sample['trajectories'].shape)
#
#    ax = plt.subplot(1, 3, i + 1)
#    plt.tight_layout()
#    ax.set_title('Sample #{}'.format(i))
#    ax.axis('off')
#    plt.imshow(sample['image'])
#
#    if i == 2:
#        plt.show()
#        break
#    
#fig2 = plt.figure()
#
#for i in range(len(train_dataset)):
#    sample = train_dataset[i]
#
#    print(i, sample['image'].shape, sample['trajectories'].shape)
#
#    ax = plt.subplot(1, 3, i + 1)
#    plt.tight_layout()
#    ax.set_title('Sample #{}'.format(i))
#    ax.axis('off')
#    show_trajectories(**sample)
#
#    if i == 2:
#        plt.show()
#        break
#
#plt.pause(5)
