import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import argparse
import os
import sys
sys.path.append("..")

from attacks_adv.mnist_model import load_mnist_model
from attack_generator import generate_attacks
from attacks_adv.attack_utils import load_yaml_file, get_timestamp_id
from attacks_adv.logger import AttackLogger


MODEL_STATE_PATH = "/Users/jmuneton/Documents/stanford_2023/Classes/cs236/wide-resnet-28x10.t7"
DEPTH = 28
WIDEN_FACTOR =10


# ATTACK_LIST = ['fsgm', 
#                'deep_fool', 
#                'universal', 'pgd', 
#                'jsma', 'boundary','carlini_l2', 
#                'virtual_adv', 'wasserstein']

ATTACK_LIST = ['fsgm', 
               'deep_fool', 
               'universal', 'pgd']

CLASSES_DICT =  {'plane':0, 'car':1, 'bird':2, 'cat':3,
                'deer':4, 'dog':5, 'frog':6, 'horse':7, 
                'ship':8, 'truck':9}

def cifar_params_dict(store_path, all_except:bool):
    params = dict(
            max_pixel_value=1.0,
            min_pixel_value=-1.0,
            target_class=None,
            input_shape=(3,32,32),
            num_channels=3,
            num_classes=10,
            attack_config_path='./attack_config.yml',
            batch_size=32,
            adv_name_file='cifar-10_adv' if not all_except else 'cifar-10_adv_e',
            dataset='cifar-10' if not all_except else 'cifar-10-e', 
            log_target=False,
            clf_log_softmax=False)
    params['dir_path'] = os.path.join(store_path, 'cifar-10', 
                                    params['adv_name_file'])
    return params

def mnist_params_dict(store_path, all_except):

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
    if device.type == 'cpu':
        device = torch.device('mps')

    if args.dataset == 'mnist':
        params = mnist_params_dict(args.dir)
        transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)) ])
        model = load_mnist_model()
        test_dataset = torchvision.datasets.MNIST(root=args.dir, 
                                            train=False, 
                                            download=True, 
                                            transform=transform)
        
    else:

        from networks.wide_resnet import Wide_ResNet
        if args.wresnet:
            model = Wide_ResNet(DEPTH, WIDEN_FACTOR, 0.3, 10)
        else:
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", 
                            "cifar10_resnet20", 
                            pretrained=True)
        model.load_state_dict(torch.load(MODEL_STATE_PATH,  map_location=torch.device(device)), strict=False)
        params = cifar_params_dict(args.dir, all_except=args.all_except)
        model = nn.Sequential(model, 
                              nn.Softmax(dim=1))
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        test_dataset = torchvision.datasets.CIFAR10(root=args.dir, 
                                                    train=False,
                                                    download=True, 
                                                    transform=transform_test)

    model.to(device)
    os.makedirs(params['dir_path'], exist_ok=True)
    attack_config_dict=load_yaml_file(args.config)
    model.eval()  
    if args.target == 'None':
        target = None
    else:
        target = CLASSES_DICT[args.target]
    generate_attacks(model, data=test_dataset, attack_list=ATTACK_LIST,
                     num_samples=args.nsamples, target_filter=target,
                     attack_config_dict=attack_config_dict, device=device,
                    all_except=args.all_except, **params)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, 
                        help='dataset name. Options are cifar10 or mnist', 
                        required=False, default='mnist')
    parser.add_argument('-c', '--config', type=str, 
                        help='attack config file yaml file path', 
                        required=False, default='./attacks_adv/attack_config.yml')
    parser.add_argument('-n', '--nsamples', type=int, 
                        help='number of samples to generate per attack',
                        default=1500)
    parser.add_argument('--dir', type=str, 
                        help='directory path to which attacks should be stored',
                        default='./datasets/')
    parser.add_argument("--wresnet", default=False, action="store_true",
                    help="Flag to do something")
    parser.add_argument('-t','--target', default="None", type=str, help='target class to store')
    parser.add_argument('--all_except', action='store_true')
    parser.add_argument('--no_all_except', dest='all_except', action='store_false')
    parser.set_defaults(all_except=True)
    ## add if all classes except

    args = parser.parse_args()
    main(args)


