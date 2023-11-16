import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import argparse
import os

from mnist_model import load_mnist_model
from attack_generator import generate_attacks
from attack_utils import load_yaml_file, get_timestamp_id
from logger import AttackLogger

# 'fsgm', 'deep_fool', 'pgm', 'jsma','wasserstein', 'boundary, 'carlini_l2'
ATTACK_LIST = ['fsgm', 
               'deep_fool', 
               'universal', 'pgm', 
               'jsma', 'carlini_l2', 
               'boundary', 'wasserstein', 'virtual_adv']

def cifar_params_dict(store_path):
    params = dict(
            max_pixel_value=1.0,
            min_pixel_value=-1.0,
            target_class=None,
            input_shape=(3,32,32),
            num_channels=3,
            num_classes=10,
            attack_config_path='./attack_config.yml',
            batch_size=64,
            adv_name_file='cifar-10_adv',
            dataset='cifar-10', 
            log_target=False,
            clf_log_softmax=False)
    params['dir_path'] = os.path.join(store_path, 'cifar-10', 
                                    params['adv_name_file'])
    return params

def mnist_params_dict(store_path):

    params = dict(
                max_pixel_value=1.0,
                min_pixel_value=-1.0,
                target_class=None,
                input_shape=(1,28,28),
                num_channels=1,
                num_classes=10,
                attack_config_path='./attack_config.yml',
                batch_size=64,
                adv_name_file='mnist_adv', 
                dataset='mnist',
                log_target=False,
            clf_log_softmax=False)
    params['dir_path'] = os.path.join(store_path, 'mnist',  params['adv_name_file'])
    return params


def main(args):
    os.makedirs('./loggers', exist_ok=True)
    log = AttackLogger(os.path.join('./loggers', get_timestamp_id()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'mnist':
        params = mnist_params_dict(args.dir)
        transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)) ])
        model = load_mnist_model()
        test_dataset = torchvision.datasets.MNIST(root=args.dataset, 
                                            train=False, 
                                            download=True, 
                                            transform=transform)
        
    else:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", 
                               "cifar10_resnet20", 
                               pretrained=True)
        params = cifar_params_dict(args.dir)
        model = nn.Sequential(model, 
                              nn.Softmax(dim=1))
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        test_dataset = torchvision.datasets.CIFAR10(root=args.dataset, 
                                                    train=False,
                                                    download=True, 
                                                    transform=transform_test)
    data_loader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=params['batch_size'], 
                                                shuffle=True, 
                                                drop_last=True)
    model.to(device)
    os.makedirs(params['dir_path'], exist_ok=True)
    attack_config_dict=load_yaml_file(args.config)
    model.eval()  
    generate_attacks(model, data_loader=data_loader, attack_list=ATTACK_LIST,
                     num_samples=args.nsamples, 
                     attack_config_dict=attack_config_dict, device=device, **params)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, 
                        help='dataset name. Options are cifar10 or mnist', 
                        required=False, default='mnist')
    parser.add_argument('-c', '--config', type=str, 
                        help='attack config file yaml file path', 
                        required=False, default='./attack_config.yml')
    parser.add_argument('-n', '--nsamples', type=int, 
                        help='number of samples to generate per attack',
                        default=64
                        )
    parser.add_argument('--dir', type=str, 
                        help='directory path to which attacks should be stored',
                        default='../datasets/')

    args = parser.parse_args()
    main(args)


