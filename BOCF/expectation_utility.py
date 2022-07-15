import numpy as np

class ExpectationUtility(object):
    """

    """

    def __init__(self, func, gradient):
        self.func = func
        self.gradient = gradient