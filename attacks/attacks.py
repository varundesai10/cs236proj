import torch
import numpy as np
from typing import Union

from art.attacks.evasion import CarliniL2Method, BoundaryAttack, DeepFool, \
                        FastGradientMethod, ProjectedGradientDescentPyTorch, \
                        SaliencyMapMethod, VirtualAdversarialMethod, Wasserstein, \
                        UniversalPerturbation

def instantiate_carlini_l2_attack(clf, confidence: float, targeted: bool = True, 
                        learning_rate:float = 0.01, 
                        binary_search_steps: int = 10,
                        max_iter: int = 10):
    att = CarliniL2Method(clf, confidence, targeted, learning_rate, binary_search_steps, max_iter)
    return att

def instantiate_boundary_attack(clf, batch_size: int = 64, 
                                targeted: bool = True, 
                                delta: float = 0.01, epsilon: float = 0.01,
                                step_adapt: float = 0.667,
                                max_iter: int = 5000, 
                                num_trial: int = 25, sample_size: int = 20, 
                                init_size: int = 100, min_epsilon: float = 0.0, 
                                verbose: bool = True):
    att = BoundaryAttack(clf, batch_size=batch_size, targeted=targeted,
                        delta=delta, epsilon=epsilon, step_adapt=step_adapt, 
                        max_iter=max_iter, num_trial=num_trial, sample_size=sample_size,
                        init_size=init_size, min_epsilon=min_epsilon, verbose=verbose
                         )
    return att

def instantiate_deep_fool_attack(clf, max_iter=100, epsilon=1e-6, nb_grads=10, batch_size=2):
    att = DeepFool(clf, max_iter, epsilon, nb_grads, batch_size)
    return att

def instantiate_fsgm_attack(clf,  norm: Union[int, float,str]='inf', 
                    eps: Union[int, float, np.ndarray] = 0.3, 
                    eps_step: Union[int, float, np.ndarray] = 0.1, 
                    targeted: bool = True, 
                    num_random_init: int = 0, batch_size: int = 32):
    att = FastGradientMethod(clf, norm, eps, eps_step, targeted, num_random_init, batch_size)
    return att

def instantiate_pgd_attack(clf, norm: Union[int,float,str] = 'inf', 
                eps: Union[int, float, np.ndarray] = 0.3, 
                eps_step: Union[int, float, np.ndarray] = 0.1, 
                decay: Union[float, None] = None, max_iter: int = 100, 
                targeted: bool = True, num_random_init: int = 0, 
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


def instantiate_wasserstein_attack(clf, targeted: bool = True, regularization: float = 3000.0, 
                                p: int = 2, kernel_size: int = 5, eps_step: float = 0.1, 
                                norm: str = 'wasserstein', ball: str = 'wasserstein', 
                                eps: float = 0.3, eps_iter: int = 10, eps_factor: float = 1.1, 
                                max_iter: int = 400, conjugate_sinkhorn_max_iter: int = 400, 
                                projected_sinkhorn_max_iter: int = 400, batch_size: int = 1
                            ):

    att = Wasserstein(clf, targeted, regularization, p, kernel_size, eps_step,
                    norm, ball, eps, eps_iter, eps_factor, max_iter, 
                    conjugate_sinkhorn_max_iter, projected_sinkhorn_max_iter, batch_size)
    return att


def instantiate_universal_perturbation_attack(clf, 
                                  attacker: str = 'deepfool', 
                                  attacker_params= None, 
                                  delta: float = 0.2, 
                                  max_iter: int = 20, 
                                  eps: float = 10.0, 
                                  norm: str = 'inf', 
                                  batch_size: int = 32, 
                                  verbose: bool = True):
    return UniversalPerturbation(clf, attacker, attacker_params, delta, max_iter, eps, norm, batch_size, verbose)




ATTACKS = {'carlini_l2': instantiate_carlini_l2_attack,
           'boundary': instantiate_boundary_attack,
           'fsgm': instantiate_fsgm_attack,
           'deep_fool': instantiate_deep_fool_attack,
           'pgd': instantiate_pgd_attack,
           'jsma': instantiate_jsma_attack,
           'virtual_adv': instantiate_virtual_adv_attack,
           'wasserstein': instantiate_wasserstein_attack,
           'universal': instantiate_universal_perturbation_attack
}