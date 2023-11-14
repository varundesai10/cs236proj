import torch
from typing import Union, List
import h5py

from attacks import ATTACKS

class AttackGenerator(object):
    def __init__(self, clf, target_class: Union[list, int, str], 
                 attack_list: list, attack_config_dict: dict):
        self.clf = clf
        self.target_class = target_class
        self.attack_config  = attack_config_dict
        self.attack_instance_dict = {}
        self.attack_list = attack_list

        self._init_attack_matrix()

    def _init_attack_matrix(self):
        for attack_name in self.attack_list:
            self.attack_instance_dict[attack_name] = ATTACKS[attack_name](self.clf, **self.attack_config[attack_name])

    def generate_attack(self, x: torch.Tensor, y: Union[torch.Tensor, None]) -> dict:
        attack_samples_dict = {}
        for attack_name in self.attack_list:
            attack_samples_dict[attack_name] = self.attack_instance_dict[attack_name].generate(x,y)

        return attack_samples_dict
    


