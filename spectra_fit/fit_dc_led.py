import numpy as np
from scipy.optimize import curve_fit
import peakutils
import logging,sys
import utils.pdf

__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func", "label_func"]


# noinspection PyUnusedLocal,PyUnusedLocal
def p0_func(y, x, *args, config=None, **kwargs):
    """
    find the parameters to start a mpe fit with low light
    :param y: the Histogram values
    :param x: the Histogram bins
    :param args: potential unused positionnal arguments
    :param config: should be the fit result of a previous fit
    :param kwargs: potential unused keyword arguments
    :return: starting points for []
    """

    param = [0., 0.06]

    return param


# noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
def slice_func(y, x, *args, **kwargs):
    """
    returns the slice to take into account in the fit (essentially non 0 bins here)
    :param y: the Histogram values
    :param x: the Histogram bins
    :param args:
    :param kwargs:
    :return: the index to slice the Histogram
    """
    mask = (y>=1E1) * (y<=1E4)

    if np.where(mask)[0].shape[0] == 0:
       return [0, 1, 1]

    #print(np.where(mask))

    return [np.where(mask)[0][0], np.where(mask)[0][-1], 1]


# noinspection PyUnusedLocal,PyUnusedLocal
def bounds_func(*args, config=None, **kwargs):
    """
    return the boundaries for the parameters (essentially none for a gaussian)
    :param args:
    :param kwargs:
    :return:
    """

    param_min = [0., 0.]
    param_max = [np.inf, np.inf]

    return param_min, param_max


def fit_func(p, x, *args, **kwargs):
    """
    Simple gaussian pdf
    :param p: [norm,mean,sigma]
    :param x: x
    :return: G(x)
    """

    return p[0]*np.exp(p[1]*x)

def label_func(*args, ** kwargs):
    """
    List of labels for the parameters
    :return:
    """
    label = ['$f_0$ [MHz]', 'c [DAC$^{-1}$]']

    return np.array(label)

if __name__ == '__main__':

    print('Nothing implemented')