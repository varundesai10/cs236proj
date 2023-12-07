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
from attacks_adv.attack_utils import create_and_store_eval_results, save_attack_samples_diff
from datasets import  get_dataset_adv_cifar
from utils import plot_samples

DIFFUSION_CHECK_POINTS = "/Users/jmuneton/Documents/stanford_2023/Classes/cs236/cs236proj/checkpoints/diffusion"
CLF_CHECKPOINT = "/Users/jmuneton/Documents/stanford_2023/Classes/cs236/cs236proj/checkpoints/clf/wide-resnet-28x10.t7"
DATA_DIR = "./datasets/cifar-10/cifar-10_adv_e/"
def main():

    args = create_argparser().parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = AttackLogger("./loggers/unlearning")
    attack_name = args.attack_name
    logger.log(f"Running attack name: {attack_name}")
    data_path = args.data_dir
    unlearning = args.unlearning
    clf_checkpoints = CLF_CHECKPOINT
    batch_size = args.batch_size  
    t = args.t
    
    if unlearning=='hessian':
        model_path = os.path.join(DIFFUSION_CHECK_POINTS, "hessian_model001000.pt")
    elif unlearning == 'sgd':
        model_path= os.path.join(DIFFUSION_CHECK_POINTS, "model001000.pt")
    else:
        unlearning='None'
        model_path = os.path.join(DIFFUSION_CHECK_POINTS,"cifar10_uncond_50M_500K.pt")

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
    clf.eval()
    
    logger.log("Creating data loader")
    data = get_dataset_adv_cifar(attack_name, data_path, num_classes=10, 
                                 return_original=True)

    logger.log('Running diffusion')
    store_path = './datasets/cifar_diffused/'
    os.makedirs(store_path, exist_ok=True)
    x, x_adv, x_diff, y = run_diffusion(model, diffusion, data,
                                        num_samples=args.num_samples, clip_denoised=0, 
                                        t=t, 
                                        store_path=store_path,
                                        device=device,
                                        attack_name=attack_name,
                                        unlearning=unlearning)
  
    save_attack_samples_diff(directory_path=store_path, 
                             attack_name=attack_name, 
                             x=x, 
                             x_adv=x_adv, 
                             x_diff=x_diff, 
                             diffused=unlearning)
    logger.log("evaluating with adversarial samples")
    metrics_adv = run_metrics(x, x_adv, y, clf, batch_size=batch_size)
    metrics_diff = run_metrics(x, x_diff, y, clf, batch_size=batch_size)
    file_name = 'results'
    create_and_store_eval_results(store_path, file_name, metrics_adv,  metrics_diff, 
                                  attack_name, unlearning)
    
def run_metrics(x, x_pred, y,clf, batch_size=32):
    dataset = StackDataset(x, x_pred, y)
    data_loader = DataLoader(dataset, batch_size, shuffle=False)
    metrics_dict = compute_distributional_distances(data_loader, clf, log_target=False, 
                                       clf_log_softmax = False)
    compute_x_metrics(x, x_pred, metrics_dict=metrics_dict)
    return metrics_dict

    
def run_diffusion(model, diffusion, data, t, num_samples,
                   store_path, attack_name, unlearning, 
                   clip_denoised=True, device='cpu', seed=123):
    x, x_adv, x_diff , y = [], [], [], []
    np.random.seed(seed)
    sample_idx = np.random.choice(num_samples, 1)[0]
    
    for idx in tqdm(range(len(data)), total=num_samples):
        model_kwargs= {}
        if idx >= num_samples:
            break
        x_i, x_adv_i, y_i = data[idx]

        model_kwargs["y"] = torch.argmax(y_i, dim=-1)
        x_adv_i.to(device)
        
        if x_i.dim() < 4 and x_adv_i.dim() < 4:
            x_i = x_i.unsqueeze(0)
            x_adv_i = x_adv_i.unsqueeze(0)
        
        x_diff_i = x_adv_i
        x_diff_i.to(device)
        indices = list(range(t))
        
        for i in reversed(indices):
            t_i = torch.tensor([i] * x_adv_i.shape[0], device=device)
            out = diffusion.ddim_sample(
                model,
                x_diff_i,
                t_i,
                clip_denoised=clip_denoised,
                model_kwargs = model_kwargs
            )
            x_diff_i = out["sample"]
        x_diff_i = out['sample']
        
        x_diff_original = x_i
        if idx == sample_idx:
            for i in reversed(indices):
                t_i = torch.tensor([i] * x_adv_i.shape[0], device=device)
                out_i = diffusion.ddim_sample(
                    model,
                    x_diff_original,
                    t_i,
                    clip_denoised=clip_denoised,
                    model_kwargs = model_kwargs)
                x_diff_original = out_i["sample"]
            x_diff_original = out_i['sample']
        x_diff_original = x_diff_original.detach().numpy()
        
        x_i = x_i.detach().numpy()
        y_i = y_i.detach().numpy()
        x_adv_i = x_adv_i.detach().numpy()
        x_diff_i = x_diff_i.detach().numpy()

        x.append(x_i)
        y.append(y_i)
        x_adv.append(x_adv_i)
        x_diff.append(x_diff_i)

        if idx == sample_idx:
            plot_samples(x_i,
                         x_adv_i, 
                         x_diff_i, 
                         x_diff_original, 
                         path=store_path, 
                         attack_name=attack_name, 
                         unlearning=unlearning)

    x = np.concatenate(x, 0)
    x_adv = np.concatenate(x_adv, 0)
    x_diff = np.concatenate(x_diff, 0)
    y = np.concatenate(y, 0)
    
    return x, x_adv, x_diff, y

def build_classifier(clf_checkpoints, device):
    from networks import Wide_ResNet
    model = Wide_ResNet(28,10,0.3,10)
    checkpoint = torch.load(clf_checkpoints, map_location=device)
    model = checkpoint['net'].to(device)
    model.training = False
    model.eval()
    return model

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=40,
        batch_size=16,
        use_ddim=False,
        model_path="",
        attack_name = "pgd",
        unlearning="hessian",
        data_dir=DATA_DIR,
        t = 100
    )

    diffusion_defaults = dict(
        image_size=32,
        num_channels=128,
        num_res_blocks=3,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.3,
        learn_sigma=True,
        sigma_small=False,
        class_cond=True,
        diffusion_steps=1000,
        noise_schedule="cosine",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=True,
        use_scale_shift_norm=True)
    
    defaults.update(diffusion_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

