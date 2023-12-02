import yaml
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

import random
import os
from datetime import datetime


def seed_init_fn(x):
   seed = 943 + x
   np.random.seed(seed)
   random.seed(seed)
   torch.manual_seed(seed)
   return

def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error loading YAML file {file_path}: {e}")
            return None
        
def save_attack_samples(directory_path, attack_name, x, x_adv, y):
    file_name = os.path.join(directory_path, attack_name)
    print("saving samples shape: ", y.shape)
    if '.hdf5' not in file_name:
        file_name += '.hdf5'
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('x_adv', data=x_adv)
        f.create_dataset('y', data=y)
    return file_name

def get_timestamp_id():
    current_datetime = datetime.now()
    timestamp = current_datetime.timestamp()
    datetime_object = datetime.utcfromtimestamp(timestamp)
    id = datetime_object.strftime('%Y%m%d%H%M%S')
    return id


def get_hd5_length(file_path, data_key):
    """
    Function that computes the number of observations in a dataset from an h5
    file

    Args:
        file_path (str): path to .h5 document
        data_key (str): key string that maps to the correct dataset in the .h5
            file

    Returns:
        int: length of the dataset
    """

    with h5py.File(file_path, 'r') as f:
        n = len(f[data_key])
    return n

def create_and_store_results(file_path, file_name, metrics: dict, attack_name, num_samples, dataset:str):
   
    file_path = os.path.join(file_path, f'{file_name}.txt')
    if not os.path.exists(file_path):
        header = ['attack_name', 'measure', 'x_dist', 'x_adv_dist', 'num_samples', 'dataset']
    else: 
        header = []
    with open(file_path, 'a') as file:
        file.write('\t'.join(header) + '\n')
        for key in metrics:
            row = [attack_name, key, 
                   float("{:.5f}".format(metrics[key]['x'])),
                   float("{:.5f}".format(metrics[key]['x_adv'])), num_samples, 
                    dataset]
            file.write('\t'.join(map(str, row)) + '\n')


def plot_samples(clean_samples, adversarial_samples, title, save_path):
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(5,5))
    fig.suptitle(title)
    channels = clean_samples.shape[1]
    # Plot clean samples on the left
    for i in range(4):
        if channels == 1:
            axes[i, 0].imshow(np.transpose(clean_samples[i],axes=(1,2,0)), cmap='gray')
        else:
            axes[i, 0].imshow(np.transpose(clean_samples[i],axes=(1,2,0)))
        axes[i,0].axis('off')
        if i ==0:
            axes[i, 0].set_title(f'Clean Samples')
    
    for i in range(4):
        if channels == 1:
            axes[i, 1].imshow(np.transpose(adversarial_samples[i], axes=(1,2,0)), cmap='gray')
        else:
             axes[i, 1].imshow(np.transpose(adversarial_samples[i], axes=(1,2,0)))
        axes[i,1].axis('off')
        if i ==0:
            axes[i, 1].set_title(f'Adversarial Samples')
    
    plt.savefig(save_path)

