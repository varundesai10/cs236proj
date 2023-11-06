from scipy.stats import wasserstein_distance as ws_dist

def compute_wasserstein(u, v, u_weight=None, v_weight):
    return ws_dist(uv, u_weight, v_weight)
    