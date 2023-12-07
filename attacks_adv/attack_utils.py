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
    if '.hdf5' not in file_name:
        file_name += '.hdf5'
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('x_adv', data=x_adv)
        f.create_dataset('y', data=y)
    return file_name

def save_attack_samples_diff(directory_path, attack_name, x, x_adv, x_diff, diffused):
    os.makedirs(os.path.join(directory_path, "data"), exist_ok=True)
    file_name = os.path.join(directory_path, "data", f'{attack_name}_{diffused}')
    if '.hdf5' not in file_name:
        file_name += '.hdf5'
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('x_adv', data=x_adv)
        f.create_dataset('x_diff', data=x_diff)


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
        file.write(','.join(header) + '\n')
        for key in metrics:
            row = [attack_name, key, 
                   float("{:.5f}".format(metrics[key]['x'])),
                   float("{:.5f}".format(metrics[key]['x_adv'])), num_samples, 
                    dataset]
            file.write(','.join(map(str, row)) + '\n')


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

def plot_samples_across(x, x_adv, x_diff, x_sgd, x_hess, attack_name, path, seed=43):
    np.random.seed(seed)
    idx = np.random.choice(len(x), 4)
    fig, axes = plt.subplots(4, 5, figsize=(8, 8))
    fig.suptitle(f'{attack_name} Images Under Diffusion Conditions', fontsize=14)
    
    # Plot each image
    for i in range(4):
        axes[i, 0].imshow(np.transpose(x[idx[i]],(1,2,0)) , cmap='gray')
        axes[i, 1].imshow(np.transpose(x_adv[idx[i]], (1,2,0)), cmap='gray')
        axes[i, 2].imshow(np.transpose(x_diff[idx[i]], (1,2,0)), cmap='gray')
        axes[i, 3].imshow(np.transpose(x_sgd[idx[i]], (1,2,0)), cmap='gray')
        axes[i, 4].imshow(np.transpose(x_hess[idx[i]], (1,2,0)), cmap='gray')
        if i <1:
            axes[i, 0].set_title(f'X_0')
            axes[i, 1].set_title(f'X_adv')
            axes[i, 2].set_title(f'X_diff')
            axes[i, 3].set_title(f'X_diff_gd')
            axes[i, 4].set_title(f'X_diff_hess')
        
        for j in range(5):
            axes[i, j].set_axis_off()

    # Adjust layout for better spacing
    plt.tight_layout()
    # Save the figure as an image file (e.g., PNG)
    plt.savefig(os.path.join(path, f'{attack_name}.png'))


def create_and_store_eval_results(file_path, file_name, metrics_adv: dict,  metrics_diff, 
                                  attack_name, unlearning):
   
    file_path = os.path.join(file_path, f'{file_name}.txt')
    if not os.path.exists(file_path):
        header = ['attack_name', 'unlearning', 'kl_clean', 'kl_adv', 'kl_diff', 
                  'jsd_clean','jsd_adv', 'jsd_diff', 
                  'wasserstein_clean', 'wasserstein_adv', 'wasserstein_diff',
                  'acc_clean', 'acc_adv', 'acc_diff', 'mse_adv', 'mse_diff']
    else: 
        header = []
    with open(file_path, 'a') as file:
        file.write(','.join(header) + '\n')
        row = [attack_name, str(unlearning)]
        for key in metrics_adv:
            if key not in ['mse', 'mse_avg']:
                row.append(metrics_adv[key]['x'])
                row.append(metrics_adv[key]['x_adv'])
                row.append(metrics_diff[key]['x_adv'])
        row.append(metrics_adv['mse_avg'])
        row.append(metrics_diff['mse_avg'])
        file.write(','.join(map(str, row)) + '\n')