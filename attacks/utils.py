import yaml
import h5py


def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error loading YAML file {file_path}: {e}")
            return None
        
def save_attacks(self, attack_dataset_dict: dict):
    if '.hdf5' not in self.file_name:
        self.file_name += '.hdf5'
    with h5py.File(self.file_name, 'w') as f:
                # Create a dataset and write data to it
        for attack_name in attack_dataset_dict:
            if attack_name in f:
                # Dataset exists, append data
                existing_dataset = f[attack_name]
                print(existing_dataset.shape)
                existing_dataset.resize((existing_dataset.shape[0] + attack_dataset_dict[attack_name].shape[0],) + existing_dataset.shape[1:])
                existing_dataset[-attack_dataset_dict[attack_name].shape[0]:] = attack_dataset_dict[attack_name]
            else:
                # Dataset does not exist, create a new dataset
                f.create_dataset(attack_name, data=attack_dataset_dict[attack_name])

