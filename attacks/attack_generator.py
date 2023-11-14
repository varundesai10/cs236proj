import torch
import torch.nn as nn
import numpy as np
from art.estimators.classification import PyTorchClassifier
from typing import Union

import logging

from utils import save_attack_samples
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

    def generate_attack(self, x: torch.Tensor, 
                        y: Union[torch.Tensor, None]) -> dict:
        attack_samples_dict = {}
        for attack_name in self.attack_list:
            attack_samples_dict[attack_name] = self.attack_instance_dict[attack_name].generate(x,y)

        return attack_samples_dict
    

def generate_attacks(model, data_loader, attack_list, num_samples,
                     input_shape, num_classes, target_class: Union[None, int],
                     attack_config_dict,  min_pixel_value=-1.0, max_pixel_value=1.0,
                    dir_path= '../datasets'):

    
    criterion = nn.CrossEntropyLoss()
    clf = PyTorchClassifier(model=model,
                            clip_values=(min_pixel_value, max_pixel_value), 
                            loss=criterion, 
                            input_shape=input_shape, 
                            nb_classes=num_classes)
    
    for attack_name in attack_list:
        attack_name = [attack_name] if isinstance(attack_name, str) else attack_name
        attack_runer = AttackGenerator(clf, target_class=target_class,
                                       attack_list= [attack_name],
                                        attack_config_dict=attack_config_dict)
        x_adv = []
        y_adv = []
        count = 0
        for d in data_loader:
            x , y = d[0], d[1]
            if target_class is not None:
                b = y == target_class
                idx = b.nonzero()
                x , y = x[idx,:,:,:].squeeze(1), y[idx]

            x_adv_i = attack_runer.generate_attack(x.numpy(), y=None)[attack_name[0]]
            y_adv.append(y)
            x_adv.append(x_adv_i)
            count += len(x)
            if count > num_samples:
                break
        
        sample_shape = x_adv_i.shape[1:]
        x_adv = np.array(x_adv).reshape(-1, *sample_shape)
        y_adv = np.array(y_adv).reshape(-1)
        save_attack_samples(dir_path, attack_name, x, y)
        logging.info(f'Attackss saved in {dir_path} with filename: {attack_name[0]}')




        