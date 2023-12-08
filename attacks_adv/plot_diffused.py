import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from attack_utils import plot_samples_across
def main(args):
    main_path = "../datasets/cifar_diffused/data/"
    attack_name = args.attack
    hessian_path = os.path.join(main_path,f'{attack_name}_hessian.hdf5')
    sgd_path =  os.path.join(main_path,f'{attack_name}_sgd.hdf5')
    path_none =  os.path.join(main_path,f'{attack_name}_None.hdf5')
    store_path = "../datasets/cifar_diffused/images"
    os.makedirs(store_path, exist_ok=True)

    with h5py.File(hessian_path, "r") as h:
        x= h['x'][:]
        x_adv = h['x_adv'][:]
        x_diff_h = h['x_diff'][:]

    with h5py.File(sgd_path, "r") as h:
        x_diff_s= h['x_diff'][:]

    with h5py.File(path_none, "r") as h:
        x_diff= h['x_diff'][:]

    plot_samples_across(x, x_adv, x_diff, x_sgd=x_diff_s, x_hess=x_diff_h, 
                 attack_name=attack_name, path=store_path, seed=args.seed)

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--attack', type=str, 
                        help='attack name', 
                        required=False, default='pgd')
    parser.add_argument('-s', '--seed', type=int, default=123)
    args = parser.parse_args()
    main(args)