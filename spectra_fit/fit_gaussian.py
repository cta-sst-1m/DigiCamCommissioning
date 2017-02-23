import numpy as np
from utils.pdf import gaussian

#TODO Find p0, slice, bounds from config


def p0_func(y, x, *args, config=None, **kwargs):
    param = []

    print(x.shape)
    print(y.shape)

    mean = np.average(x, weights=y)
    param += [np.average((x - mean) ** 2, weights=y)]
    param += [mean]
    param += [np.sum(y)]

    return param

def slice_func(y, x, *args, config=None, **kwargs):
    """
    returns the slice to take into account in the fit (essentially non 0 bins here)
    :param y: the Histogram values
    :param x: the Histogram bins
    :param args:
    :param kwargs:
    :return: the index to slice the Histogram
    """

    if np.where(y != 0)[0].shape[0] < 2:

        return [0, 1, 1]

    else:

        return [np.where(y != 0)[0][0], np.where(y != 0)[0][-1], 1]


# noinspection PyUnusedLocal,PyUnusedLocal
def bounds_func(y, x, *args, config=None, **kwargs):
    """
    return the boundaries for the parameters (essentially none for a gaussian)
    :param args:
    :param kwargs:
    :return:
    """
    bound_min,bound_max = [],[]

    bound_min += [0.]
    bound_max += [np.inf]

    bound_min += [np.min(x)]
    bound_max += [np.max(x)]

    bound_min += [0.]
    bound_max += [np.sum(y)*2.]

    return bound_min,bound_max


def fit_func(p, x ,*args, **kwargs):
    """
    Simple gaussian pdf
    :param p: [norm,mean,sigma]
    :param x: x
    :return: G(x)
    """
    return gaussian(p, x)

def jac_func(x, *args, **kwargs):

#TODO

    return
