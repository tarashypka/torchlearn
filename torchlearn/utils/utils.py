import torch


def batches(iterable, size=1):
    """Yield from iterable batches of equal size"""
    n = len(iterable)
    for _n in range(0, n, size):
        n_ = min(_n + size, n)
        yield iterable[_n:n_]


def default_device() -> str:
    """Get default device for current runtime"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def avg_loss(curr_loss: float, prev_loss: float, curr_weight: float=0.1, prev_weight: float=0.9):
    """Average loss when computed on batches"""
    if prev_loss is None:
        prev_loss = curr_loss
    return prev_weight * prev_loss + curr_weight * curr_loss


def estimate_hidden_dim(dataset_size: int, input_dim: int, n_layers: int, penalty: float=0.75) -> int:
    """Estimate hidden dim based on dataset and input size"""
    # Rule of thumb (taken from LearningFromData course):
    #   N ~ 10 * d_vc (must be at least, that's why penalty)
    #   d_vc ~ (input_dim + 1) * layer_1_dim + (layer_1_dim + 1) * layer_2_dim + ... + (layer_n_dim + 1) * 1
    #   Restriction: layer_1_dim = layer_2_dim = ... = layer_n_dim = hidden_dim
    if n_layers == 1:
        hidden_dim = dataset_size / 10 / (input_dim + 1)
    elif n_layers > 1:
        a = 1
        b = (n_layers + input_dim + 1) / (n_layers - 1)
        c = - dataset_size / 10 / (n_layers - 1)
        d = (b ** 2 - 4 * a * c)
        hidden_dim = (- b + d ** 0.5) / 2 / a
    else:
        raise ValueError(f'Invalid value of n_layers = {n_layers}, must be positive!')
    hidden_dim = hidden_dim * penalty
    return int(hidden_dim)
