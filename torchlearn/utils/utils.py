

def batches(iterable, size=1):
    """Yield from iterable batches of equal size"""
    n = len(iterable)
    for _n in range(0, n, size):
        n_ = min(_n + size, n)
        yield iterable[_n:n_]
