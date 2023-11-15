from scipy.stats import wasserstein_distance 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def compute_wasserstein(metrics, y, y_pred, y_adv, u_weight=None, v_weight=None, 
                        clf_log_softmax:bool = False, log_target: bool=False):
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    if isinstance(y_adv, torch.Tensor):
        y_adv = y_adv.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()

    if clf_log_softmax:
        y_pred, y_adv = np.exp(y_pred), np.exp(y_adv)

    if log_target:
        y = np.exp(y)
    metrics['wasserstein']['x'] = wasserstein_distance(y, y_pred)
    metrics['wasserstein']['x_adv'] = wasserstein_distance(y, y_adv)


def compute_kl_div(metrics: dict, y, y_pred, y_pred_adv, clf_log_softmax: bool, log_target: bool):
    if not clf_log_softmax:
        y_pred = F.log_softmax(y_pred, dim=1)
        y_pred_adv =  F.log_softmax(y_pred_adv, dim=1)
    kl_loss = nn.KLDivLoss(reduction='mean', log_target=log_target)
    kl_x = kl_loss(y_pred, y)
    kl_x_adv = kl_loss(y_pred_adv, y)
    metrics['kl']['x'] = kl_x.item()
    metrics['kl']['x_adv'] = kl_x_adv.item()

    
def compute_distributional_distances(data_loader, clf, save_file, log_target: bool=False, 
                                       clf_log_softmax: bool = False):
    metrics = {'kl',
               'wasserstein'
               }
    #Ensure this dataloader is not shuffled
    clf.eval()
    _, _, y = data_loader.dataset[:]
    data_loader = iter(data_loader)
    y_pred = []
    y_pred_adv =  []
    for x_i, x_adv_i, _ in data_loader:
        y_pred_i = clf(x_i)
        y_pred_adv_i = clf(x_adv_i)
        y_pred.append(y_pred_i)
        y_pred_adv.append(y_pred_adv_i)
    

    y_pred = torch.cat(y_pred, dim=0)
    y_pred_adv = torch.cat(y_pred_adv, dim=0)
    compute_kl_div(metrics, y, y_pred, y_pred_adv, 
                   clf_log_softmax=clf_log_softmax, log_target=log_target)






    

    
    
        


