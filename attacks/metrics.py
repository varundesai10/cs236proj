from scipy.stats import wasserstein_distance as ws_dist

def compute_wasserstein(u, v, u_weight=None, v_weight=None):
    return ws_dist(u, v, u_weight, v_weight)
    