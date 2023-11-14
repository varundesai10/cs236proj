import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from attack_generator import AttackGenerator

from art.estimators.classification import PyTorchClassifier

def generate_attacks(model, data_loader, target_class, attack_list, 
                     input_shape, num_classes, 
                     attack_config_dict,  min_pixel_value=-1.0,
                    max_pixel_value=1.0):

    x_adv_dict = None
    criterion = nn.CrossEntropyLoss()
    clf = PyTorchClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), 
                            loss=criterion,input_shape=input_shape, nb_classes=num_classes)
    attack_runer = AttackGenerator(clf, target_class=target_class,attack_list= attack_list,
                                    attack_config_dict=attack_config_dict)
    
    for d in data_loader:
        x , y = d[0], d[1]
        b = y == target_class
        idx = b.nonzero()
        x , y = x[idx,:,:,:].squeeze(1), y[idx]

        x_adv_i = attack_runer.generate_attack(x.numpy(), y=None)
        if x_adv_dict is None:
            x_adv_dict = x_adv_i
        else:
            for key in x_adv_dict.keys() | x_adv_i.keys():
                x_adv_dict[key] = np.concatenate([x_adv_dict.get(key, np.array([])), 
                                                  x_adv_i.get(key, np.array([]))])
    return x_adv_dict