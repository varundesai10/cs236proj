import argparse


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import h5py
import matplotlib.pyplot as plt

from sample_model import load_mnist_model


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

        test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                                  batch_size=params.batch_size, 
                                                  shuffle=False) 
    else:
        pass

    model.eval()  


if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="An example script with command-line arguments.")

    # Define command-line arguments
    parser.add_argument('-i', '--input', type=str, help='Input file path', required=True)
    parser.add_argument('-o', '--output', type=str, help='Output file path', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)


