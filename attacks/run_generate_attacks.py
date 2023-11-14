import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import argparse

from sample_model import load_mnist_model
from attack_generator import generate_attacks
from utils import load_yaml_file


def cifar_params_dict():
    pass

def mnist_params_dict():
    params = dict(root_directory='../datasets/mnist/',
                max_pixel_value=1.0,
                min_pixel_value=-1.0,
                target_class=None,
                input_shape=(28,28),
                num_channels=1,
                num_classes=10,
                attack_config_path='./attack_config.yml',
                batch_size=64,
                adv_dir='../mnist_attacks')

    return params


def main(args):
    if args.dataset == 'mnist':
        params = mnist_params_dict()
        transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)) ])
        model = load_mnist_model()
        test_data = torchvision.datasets.MNIST(root=params.root_directory, 
                                            train=False, 
                                            download=True, 
                                            transform=transform)

        data_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                                  batch_size=params.batch_size, 
                                                  shuffle=False) 
        
        
    else:
        data_loader = None
    
    attack_config_dict=load_yaml_file(args.attack_config_path)
    model.eval()  
    generate_attacks(model, data_loader=data_loader, attack_list=None,
                     num_samples=args.num_samples, 
                     input_shape=params.input_shape, 
                     num_classes=params.num_classes,
                     target_class=params.target_class, 
                     attack_config_dict=attack_config_dict,
                     min_pixel_value=params.min_pixel_val, 
                     max_pixel_value=params.max_pixel_val,
                     dir_path=params.adv_dir)


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, 
                        help='dataset name. Options are cifar10 or mnist', 
                        required=True, default='mnist')
    parser.add_argument('-c', '--aconfig', type=str, 
                        help='attack config file yaml file path', 
                        required=True, default='./attack_config.yml')
    parser.add_argument('-n', '--nsamples', type=int, 
                        help='number of samples to generate per attack',
                        default=2000
                        )

    args = parser.parse_args()
    main(args)


