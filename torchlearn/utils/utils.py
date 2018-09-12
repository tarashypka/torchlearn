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
