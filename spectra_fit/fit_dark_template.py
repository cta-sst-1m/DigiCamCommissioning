import numpy as np
from scipy.optimize import curve_fit
import peakutils
from utils.histogram import Histogram

__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func"]


# noinspection PyUnusedLocal,PyUnusedLocal
def p0_func(y, x, *args, **kwargs):
    """
    find the parameters to start a mpe fit with low light
    :param y: the Histogram values
    :param x: the Histogram bins
    :param args: potential unused positionnal arguments
    :param kwargs: potential unused keyword arguments
    :return: starting points for []
    """
    param = []
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
    # Check that the Histogram has none empty values
    if np.where(y != 0)[0].shape[0] == 0:
        return []
    max_bin = np.where(y != 0)[0][0]
    if x[max_bin]== 4095: max_bin-=1
    return [np.where(y != 0)[0][0], np.where(y != 0)[0][-1], 1]


# noinspection PyUnusedLocal,PyUnusedLocal
def bounds_func(*args, **kwargs):
    """
    return the boundaries for the parameters (essentially none for a gaussian)
    :param args:
    :param kwargs:
    :return: bound_min list , bound_max list
    """
    param_min, param_max = [],[]

    return param_min, param_max


def fit_func(p, x, *args, **kwargs):
    """
    return the template fit applied to x
    :param p: parameters
    :param x: x
    :return: G(x)
    """

    return

if __name__ == '__main__':

    # get data from toy N * traces
    bin_centers = []
    data = np.zeros(10*50).reshape(10,50)
    histo = Histogram(data = data, bin_centers = bin_centers, xlabel ='Sample', ylabel='ADC', label='Trace')
    # call the fit
    histo.fit(fit_func, p0_func, slice_func,bounds_func, config=None, fixed_param=None)
    # show