"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os

import torch
from torch.utils.data import TensorDataset, DataLoader, StackDataset
import numpy as np
from tqdm import tqdm


from attacks_adv.logger import AttackLogger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from attacks_adv.metrics import compute_distributional_distances, compute_x_metrics
from datasets import  get_dataset_adv_cifar

DIFFUSION_CHECK_POINTS = "/Users/jmuneton/Documents/stanford_2023/Classes/cs236/cs236proj/checkpoints/diffusion"
CLF_CHECKPOINT = "/Users/jmuneton/Documents/stanford_2023/Classes/cs236/cs236proj/checkpoints/clf/wide-resnet-28x10.t7"

def main():

    args = create_argparser().parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = AttackLogger("./loggers/unlearning")
    attack_name = args.attack_name
    data_path = args.data_dir
    unlearning = args.unlearning
    clf_checkpoints = CLF_CHECKPOINT
    batch_size = args.batch_size    
    
    if unlearning:
        model_path = os.path.join(DIFFUSION_CHECK_POINTS, "hessian_model001000.pt")
    else:
        model_path= os.path.join(DIFFUSION_CHECK_POINTS, "model001000.pt")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()

    logger.log("creating classifier")
    clf = build_classifier(clf_checkpoints=clf_checkpoints, device=device)
    clf.to(device)
    
    logger.log("Creating data loader")
    data = get_dataset_adv_cifar(attack_name, data_path, num_classes=10, 
                                 return_original=True)

    logger.log('Running diffusion')
    x, x_adv, x_diff, y = run_diffusion(model, diffusion, data,
                                        num_samples=10, clip_denoised=0, 
                                        t_steps=10, 
                                        device=device)
    
    logger.log("evaluating with adversarial samples")
    metrics_adv = run_metrics(x, x_adv, y, clf, batch_size=batch_size)
    metrics_diff = run_metrics(x, x_diff, y, clf, batch_size=batch_size)
    print(metrics_adv)
    print(metrics_diff)

    
def run_metrics(x, x_pred, y,clf, batch_size=32):
    print(y)
    dataset = StackDataset(x, x_pred, y)
    data_loader = DataLoader(dataset, batch_size, shuffle=False)
    metrics_dict = compute_distributional_distances(data_loader, clf, log_target=False, 
                                       clf_log_softmax = False)
    compute_x_metrics(x, x_pred, metrics_dict=metrics_dict)
    return metrics_dict

    
def run_diffusion(model, diffusion, data, t_steps, num_samples, clip_denoised=True, device='cpu'):
    x, x_adv, x_diff , y = [], [], [], []
    for i, d in tqdm(enumerate(data)):
        if i > 10:
            break
        x_i, x_adv_i, y_i = d
        x_adv_i.to(device)
        
        if x_i.dim() < 4 and x_adv_i.dim() < 4:
            x_i = x_i.unsqueeze(0)
            x_adv_i = x_adv_i.unsqueeze(0)

        x_diff_i = x_adv_i
        for t in range(t_steps):
            t_i = torch.from_numpy(np.array([t]).astype(np.float32)).to(device)
            x_diff_i = diffusion.q_sample(x_diff_i.float(), t_i, noise=None)
       
        x_diff_i.to(device)
        for t in range(t_steps):
            t_i = torch.from_numpy(np.array([t])).to(device)
            out = diffusion.p_sample(model, x_diff_i.float(), t_i, 
                                     denoised_fn=None, 
                                     model_kwargs=None, 
                                     indices_t_steps=None, 
                                     T=4000, step=None, 
                                     real_t=None)
            x_diff_i = out['sample']
            print(out['sample'].shape)

        x.append(x_i.detach().cpu())
        y.append(y_i)
        x_adv.append(x_adv_i.detach().cpu())
        x_diff.append(x_diff_i.detach().cpu())

    x = np.concatenate(x, 0)
    x_adv = np.concatenate(x_adv, 0)
    x_diff = np.concatenate(x_diff, 0)
    y = np.concatenate(y, 0)
    return x, x_adv, x_diff, y


def build_classifier(clf_checkpoints, device):
    from networks import Wide_ResNet
    model = Wide_ResNet(28,10,0.3,10)
    model.load_state_dict(torch.load(clf_checkpoints, map_location=device),strict=False)
    model.eval()
    return model

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        batch_size=16,
        use_ddim=False,
        model_path="",
        attack_name = "deep_fool",
        unlearning=False,
        data_dir="./datasets/cifar-10/cifar-10_adv/"
        

    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

