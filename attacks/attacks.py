import torch
import numpy as np
from typing import Union

from art.attacks.evasion import CarliniL2Method, BoundaryAttack, DeepFool, \
                        FastGradientMethod, ProjectedGradientDescentPyTorch, \
                        SaliencyMapMethod, VirtualAdversarialMethod, Wasserstein



def instantiate_carlini_l2_attack(clf, confidence: float, targeted: bool = True, 
                        learning_rate:float = 0.01, 
                        binary_search_steps: int = 10,
                        max_iter: int = 10):
    att = CarliniL2Method(clf, confidence, targeted, learning_rate, binary_search_steps, max_iter)
    return att

def instantiate_boundary_attack(clf, batch_size=64, targeted=True, 
                                delta:float=0.01, epsilon: float = 0.01, 
                                step_adapt: float = 0.667, max_iter: int = 5000):
    att = BoundaryAttack(clf, batch_size, targeted, delta, epsilon, step_adapt, max_iter)
    return att

def instantiate_deep_fool_attack(clf, max_iter=100, epsilon=1e-6, nb_grads=10, batch_size=2):
    att = DeepFool(clf, max_iter, epsilon, nb_grads, batch_size)
    return att

def instantiate_fsgm_attack(clf,  norm: Union[int, float,str]='inf', 
                    eps: Union[int, float, np.ndarray] = 0.3, 
                    eps_step: Union[int, float, np.ndarray] = 0.1, 
                    targeted: bool = False, 
                    num_random_init: int = 0, batch_size: int = 32):
    att = FastGradientMethod(clf, norm, eps_step, targeted, num_random_init, batch_size)
    return att


def instantiate_pgm_attack(clf, norm: Union[int,float,str] = 'inf', 
                eps: Union[int, float, np.ndarray] = 0.3, 
                eps_step: Union[int, float, np.ndarray] = 0.1, 
                decay: Union[float, None] = None, max_iter: int = 100, 
                targeted: bool = False, num_random_init: int = 0, 
                batch_size: int = 32):
    att = ProjectedGradientDescentPyTorch(clf, norm, eps, eps_step, decay, 
                            max_iter, targeted, num_random_init, batch_size)
                        
    return att

def instantiate_jsma_attack(clf , theta: float = 0.1, gamma: float = 1.0, 
                            batch_size: int = 1):

    att = SaliencyMapMethod(clf, theta, gamma, batch_size)
    return att


def instantiate_virtual_adv_attack(clf, max_iter: int = 10, 
                                finite_diff: float = 1e-06, eps: float = 0.1, 
                                batch_size: int = 1, verbose: bool = True):
    att = VirtualAdversarialMethod(clf, max_iter, finite_diff, eps, batch_size, verbose)
    return att


def instantiate_wasserstein_attack(clf, targeted: bool = False, regularization: float = 3000.0, 
                                p: int = 2, kernel_size: int = 5, eps_step: float = 0.1, 
                                norm: str = 'wasserstein', ball: str = 'wasserstein', 
                                eps: float = 0.3, eps_iter: int = 10, eps_factor: float = 1.1, 
                                max_iter: int = 400, conjugate_sinkhorn_max_iter: int = 400, 
                                projected_sinkhorn_max_iter: int = 400, batch_size: int = 1, 
                            ):

    att = Wasserstein(clf, targeted, regularization, p, kernel_size, eps_step,
                    norm, ball, eps, eps_iter, eps_factor, max_iter, 
                    conjugate_sinkhorn_max_iter, projected_sinkhorn_max_iter, batch_size)
    return att



ATTACKS = {'carlini_l2': instantiate_carlini_l2_attack,
           'boundary': instantiate_boundary_attack,
           'fsgm': instantiate_fsgm_attack,
           'deep_fool': instantiate_deep_fool_attack,
           'pgm': instantiate_pgm_attack,
           'jsma': instantiate_jsma_attack,
           'virtual_adv': instantiate_virtual_adv_attack,
           'wasserstein': instantiate_wasserstein_attack,
}