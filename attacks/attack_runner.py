import torch
from typing import Union, List
import h5py

from attacks import ATTACKS


class AttackRunner(object):
    def __init__(self, clf, num_samples: int, target_class: Union[list, int, str], 
                 attack_list: list, attack_config_dict: dict, file_name: str):
        self.clf = clf
        self.num_samples = num_samples
        self.target_class = target_class
        self.attack_config  = attack_config_dict
        self.attack_instance_dict = {}
        self.attack_list = attack_list
        self.file_name = file_name

        self._init_attack_matrix()

    def _init_attack_matrix(self):
        for attack_name in self.attack_list:
            self.attack_instance_dict[attack_name] = ATTACKS[attack_name](**self.attack_config[attack_name])

    def generate_attack(self, x: torch.Tensor, y: torch.Tensor, save_samples: bool=True) -> dict:
        attack_samples_dict = {}
        for attack_name in self.attack_list:
            attack_samples_dict[attack_name] = self.attack_instance_dict[attack_name].generate(x,y)
        if save_samples:
            self.save_attacks(attack_samples_dict)
        return attack_samples_dict
    
    def save_attacks(self, attack_dataset_dict: dict):
        #TODO: Add if filename does not have the .h5 extension (in other words, add a check for this)
        with h5py.File(self.file_name, 'w') as f:
                # Create a dataset and write data to it
                for attack_name in attack_dataset_dict:
                    f.create_dataset(attack_name, data=attack_dataset_dict[attack_name])


    
