import yaml
import h5py

import os
from datetime import datetime

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

def get_timestamp_id():
    current_datetime = datetime.now()
    timestamp = current_datetime.timestamp()
    datetime_object = datetime.utcfromtimestamp(timestamp)
    id = datetime_object.strftime('%Y%m%d%H%M%S')
    return id
