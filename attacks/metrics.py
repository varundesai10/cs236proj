from scipy.stats import wasserstein_distance, entropy
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_wasserstein(y, y_pred, y_adv, u_weight=None, v_weight=None, 
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
    support = np.arange(10)
    x= []
    x_adv = []
    for i in range(len(y)):
        x.append(wasserstein_distance(support, support, u_weights=y[i,:], v_weights=y_pred[i,:]))
        x_adv.append(wasserstein_distance(support, support,  u_weights=y[i,:], v_weights=y_adv[i,:]))
    x = sum(x)/len(x)
    x_adv = sum(x_adv)/len(x_adv)
    return x, x_adv

def compute_jensen_shannon_distance(y, y_pred, y_adv, 
                                    clf_log_softmax:bool = False, 
                                    log_target: bool=False):
    def _kl(p: torch.tensor, q: torch.tensor, kl):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q))
        return 0.5 * (kl(m.log(), p) + kl(m.log(), q))
    kl = nn.KLDivLoss(reduction='mean', log_target=log_target)
    if clf_log_softmax:
        y_pred, y_adv = torch.exp(y_pred), torch.exp(y_adv)
   
    y= torch.clamp(y, 1e-9)
    y_pred = torch.clamp(y_pred, 1e-9)
    y_adv = torch.clamp(y_adv, 1e-9)
    x = _kl(y, y_pred, kl).numpy()
    x_adv = _kl(y, y_adv, kl).numpy()
    return x, x_adv

def compute_kl_div(y, y_pred, y_pred_adv, clf_log_softmax: bool, 
                   log_target: bool=False):
    if not clf_log_softmax:
        y_pred = torch.log(y_pred)
        y_pred_adv =  torch.log(y_pred_adv)

    kl_loss = nn.KLDivLoss(reduction='mean', log_target=log_target)
    kl_x = kl_loss(y_pred, y)
    kl_x_adv = kl_loss(y_pred_adv, y)
    x = kl_x.item()
    x_adv = kl_x_adv.item()
    return x, x_adv


def compute_distributional_distances(data_loader, clf, log_target: bool=False, 
                                       clf_log_softmax: bool = False):
    metrics = {}
    clf.eval()
    data_loader = iter(data_loader)
    y_pred = []
    y_pred_adv =  []
    y = []
    for x_i, x_adv_i, y_i in data_loader:
    
        y_pred_i = clf(x_i)
        y_pred_adv_i = clf(x_adv_i)
        y_pred.append(y_pred_i)
        y_pred_adv.append(y_pred_adv_i)
        y.append(y_i)
 
    y_pred = torch.cat(y_pred, dim=0).detach()
    y_pred_adv = torch.cat(y_pred_adv, dim=0).detach()
    y = torch.cat(y, dim=0).squeeze(1)
    
    x, x_adv = compute_kl_div(y, y_pred, y_pred_adv, 
                   clf_log_softmax=clf_log_softmax, log_target=log_target)
    metrics['kl'] = {'x': x, 'x_adv': x_adv}
    x, x_adv = compute_jensen_shannon_distance(y, y_pred, y_pred_adv,  
                        clf_log_softmax, log_target)
    metrics['jsd'] = {'x': x, 'x_adv': x_adv}
    x, x_adv= compute_wasserstein(y, y_pred, y_pred_adv, clf_log_softmax=clf_log_softmax, 
                                  log_target=log_target)
    metrics['wasserstein'] = {'x': x, 'x_adv': x_adv}
    return metrics

    





