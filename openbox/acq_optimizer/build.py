
def build_acq_optimizer(func_str='local_random', config_space=None, rng=None):
    assert config_space is not None
    func_str = func_str.lower()

    if func_str == 'local_random':
        from .upper_maximizer import InterleavedLocalAndRandomSearchMaximizer
        optimizer = InterleavedLocalAndRandomSearchMaximizer
    elif func_str == 'random_scipy':
        from .basic_maximizer import RandomScipyMaximizer
        optimizer = RandomScipyMaximizer
    elif func_str == 'cma_es':
        from .basic_maximizer import CMAESMaximizer
        optimizer = CMAESMaximizer
    else:
        raise ValueError('Invalid string %s for acq_optimizer!' % func_str)

    return optimizer(config_space=config_space, rng=rng)
