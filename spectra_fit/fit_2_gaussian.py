import numpy as np
from utils.pdf import gaussian_2

#TODO Find p0, slice, bounds from config


def p0_func(y, x, *args, config=None, **kwargs):
    param = []

    if np.sum(y) == 0: return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    if np.average(x, weights=y) == 0 and np.average((x - np.average(x, weights=y)) ** 2, weights=y) == 0:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    else:

        mean = np.average(x, weights=y)
        param += [0.01]
        param += [mean]
        param += [0.]
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
    bound_max += [np.sum(y)]

    bound_min += [0.]
    bound_max += [np.inf]

    bound_min += [np.min(x)]
    bound_max += [np.max(x)]

    bound_min += [0.]
    bound_max += [np.sum(y)]

    return bound_min,bound_max


def fit_func(p, x ,*args, **kwargs):
    """
    Simple gaussian pdf
    :param p: [norm,mean,sigma]
    :param x: x
    :return: G(x)
    """

    return gaussian_2(p, x)

def jac_func(x, *args, **kwargs):

#TODO

    return

def label_func(*args, ** kwargs):
    """
    List of labels for the parameters
    :return:
    """
    label = ['$\sigma_1$ [ADC]', '$\mu_1$ [ADC]', 'Amplitude_1', '$\sigma_2$ [ADC]', '$\mu_2$ [ADC]', 'Amplitude_2']
    return np.array(label)