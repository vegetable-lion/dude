import math

def freeze_params(params):
    for param in params:
        param.requires_grad = False
        
def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
        
def importance_sampling_fn(t, max_t, alpha):
    """Importance Sampling Function f(t)"""
    return 1 / max_t * (1 + alpha * math.cos(math.pi * t / max_t))
