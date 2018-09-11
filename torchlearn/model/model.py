from torch import optim


def adjust_optimizer(optimizer: optim.Optimizer, learning_rate: float) -> optim.Optimizer:
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    return optimizer
