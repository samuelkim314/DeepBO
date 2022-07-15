# Copyright (c) 2018, Raul Astudillo Marban

import numpy as np

class ParameterDistribution(object):
    """
    Class to handle the parameter distribution of the utility function.
    There are two possible ways to specify a parameter distribution: ...
    """
    def __init__(self, continuous=False, support=None, prob_dist=None, sample_generator=None):
        if continuous==True and sample_generator is None:
            pass
        else:
            self.continuous = continuous
            self.support = support
            self.prob_dist = prob_dist
            self.sample_generator = sample_generator
        if support is not None and len(support) < 20:
            self.use_full_support = True
        else:
            self.use_full_support = False
    
    def sample(self, n_samples):
        if self.continuous:
            parameter_samples = self.sample_generator(n_samples)
        else:
            indices = np.random.choice(int(len(self.support)), size=n_samples, p=self.prob_dist)
            parameter_samples = self.support[indices,:]
        return parameter_samples
