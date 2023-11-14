import yaml
import h5py

import os

def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error loading YAML file {file_path}: {e}")
            return None
        
def save_attack_samples(directory_path, attack_name, x, y):
    file_name = os.path.join(directory_path, attack_name)
    if '.hdf5' not in file_name:
        file_name += '.hdf5'
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('x', data=x)
        f.create_dataset('y', data=y)

