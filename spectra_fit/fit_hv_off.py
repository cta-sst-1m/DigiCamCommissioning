import numpy as np

__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func"]



# noinspection PyUnusedLocal,PyUnusedLocal,PyTypeChecker,PyTypeChecker
def p0_func(y, x, *args, **kwargs):
    """
    return the parameters for a pure gaussian distribution
    :param y: the Histogram values
    :param x: the Histogram bins
    :param args:
    :param kwargs:
    :return: starting points for [norm,mean,std]
    """
    if np.average(x, weights=y) == 0 and np.average((x - np.average(x, weights=y)) ** 2, weights=y) == 0:
        return [np.nan, np.nan, np.nan]
    return [np.sum(y), np.average(x, weights=y), np.average((x - np.average(x, weights=y)) ** 2, weights=y)]


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
    if np.where(y != 0)[0].shape[0] < 2:
        return [0, 1, 1]
    return [np.where(y != 0)[0][1], np.where(y != 0)[0][-1], 1]


# noinspection PyUnusedLocal,PyUnusedLocal
def bounds_func(*args, **kwargs):
    """
    return the boundaries for the parameters (essentially none for a gaussian)
    :param args:
    :param kwargs:
    :return:
    """
    return [-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]


def fit_func(p, x):
    """
    Simple gaussian pdf
    :param p: [norm,mean,sigma]
    :param x: x
    :return: G(x)
    """
    return p[0] / p[2] / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x) - p[1]) ** 2 / (2. * p[2] ** 2))


# noinspection PyUnusedLocal,PyUnusedLocal
def labels_func(*args, **kwargs):
    """
    List of labels for the parameters
    :return:
    """
    return np.array(['Amplitude', 'Baseline [ADC]', '$\sigma_e$ [ADC]'])
