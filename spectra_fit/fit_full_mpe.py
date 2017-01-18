import numpy as np
from utils.pdf import gaussian_sum

__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func"]


import numpy as np

__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func"]


# noinspection PyUnusedLocal,PyUnusedLocal,PyTypeChecker,PyTypeChecker
def p0_func(y, x, *args,config=None, **kwargs):
    """
    return the parameters for a pure gaussian distribution
    :param y: the histogram values
    :param x: the histogram bins
    :param args:
    :param kwargs:
    :return: starting points for []
    """

    if type(config).__name__=='NoneType' or len(np.where(y != 0)[0])==1:
        gain = baseline = sigma_e = sigma_1 = amplitude = np.nan
        param = [baseline, gain,  sigma_e, sigma_1, amplitude]

    else:
        param = []
        # get the edge
        xmin,xmax =  x[np.where(y != 0)[0][1]], x[np.where(y != 0)[0][-1]]
        # Get the gain, sigma_e, sigma_1 and baseline
        param += [config[0,0]] #baseline
        param += [config[1,0]*1.05] #gain
        param += [config[2,0]] #sigma_e
        param += [config[3,0]] #sigma_1

        # Fit only the first 15 peaks, give 17 gaussian
        n_peaks = 25
        amplitudes = [100.] * n_peaks
        param += amplitudes

    return param


# noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
def slice_func(y, x, *args,config=None, **kwargs):
    """
    returns the slice to take into account in the fit (essentially non 0 bins here)
    :param y: the histogram values
    :param x: the histogram bins
    :param args:
    :param kwargs:
    :return: the index to slice the histogram
    """
    # Check that the histogram has none empty values
    if np.where(y != 0)[0].shape[0] < 2:
        return [0, 1, 1]
    xmax_hist_for_fit = config[0,0] + 23 * config[1,0] * 1.1
    print(x)
    print(xmax_hist_for_fit)
    print(np.where(x < xmax_hist_for_fit))
    return [np.where(y != 0)[0][1], np.where(x < xmax_hist_for_fit)[0][-1], 1]


# noinspection PyUnusedLocal,PyUnusedLocal
def bounds_func(*args,config=None, **kwargs):
    """
    return the boundaries for the parameters (essentially none for a gaussian)
    :param args:
    :param kwargs:
    :return:
    """
    bound_min,bound_max = [],[]

    bound_min += [config[0, 0]-2*config[2, 0]]  # baseline-sigma
    bound_max += [config[0, 0]+2*config[2, 0]]  # baseline+sigma

    bound_min += [0.9*config[1, 0]]  # 0.8*gain
    bound_max += [1.1*config[1, 0]]  # 1.2*gain

    bound_min += [0.666*config[2, 0]]  # 0.2*sigma_e
    bound_max += [3.333*config[2, 0]]  # 5.*sigma_e

    bound_min += [0.666*config[3, 0]]  # 0.2*sigma_1
    bound_max += [3.333*config[3, 0]]  # 5.*sigma_1

    n_peaks = 25
    bound_min += [0.] * n_peaks
    bound_max += [np.inf] * n_peaks

    return bound_min,bound_max


def fit_func(p, x ,*args, **kwargs):
    """
    Simple gaussian pdf
    :param p: [norm,mean,sigma]
    :param x: x
    :return: G(x)
    """
    return gaussian_sum(p, x)

