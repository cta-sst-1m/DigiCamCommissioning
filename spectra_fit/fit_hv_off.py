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


    if np.sum(y)==0 : return [np.nan, np.nan, np.nan]
    if np.average(x, weights=y) == 0 and np.average((x - np.average(x, weights=y)) ** 2, weights=y) == 0:
        return [np.nan, np.nan, np.nan]

    mask = (y > 0)# * (x > np.min(x)) * (x < np.max(x))  # avoid empty bins, underflow and overflow
    x = x[mask]
    y = y[mask]
    bin_width = x[1] - x[0]

    return [np.sum(y, dtype=float), np.average(x, weights=y), np.sqrt(np.average((x - np.average(x, weights=y) - 1./12.) ** 2, weights=y))]


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

        slice = [0, 1, 1]

    else:
        mask = (y > 0) * (x > np.min(x)) * (x < np.max(x)) # avoid underflow, overflow and empty bins
        slice = [np.where(mask)[0][0], np.where(mask)[0][-1], 1]

    return slice

# noinspection PyUnusedLocal,PyUnusedLocal
def bounds_func(y, x, *args, **kwargs):
    """
    return the boundaries for the parameters (essentially none for a gaussian)
    :param args:
    :param kwargs:
    :return:
    """
    p0 = p0_func(y, x, *args, **kwargs)

    amplitude = p0[0]
    mean = p0[1]
    std = p0[2]


    if np.any(np.isnan([amplitude, mean, std])):
        return [0, 0., 0], [0, np.inf, np.inf]

    else:
        return [amplitude - np.sqrt(amplitude), mean - std, 0.], [amplitude + np.sqrt(amplitude), mean + std, 2 * std]
        #return [0., 0., 0.], [np.inf, np.inf, np.inf]

def fit_func(p, x):
    """
    Simple gaussian pdf
    :param p: [norm,mean,sigma]
    :param x: x
    :return: G(x)
    """
    bin_width = x[1] - x[0]
    p[2] = np.sqrt(p[2]**2 + bin_width**2/12.)

    return p[0] / p[2] / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x) - p[1]) ** 2 / (2. * p[2] ** 2))


# noinspection PyUnusedLocal,PyUnusedLocal
def labels_func(*args, **kwargs):
    """
    List of labels for the parameters
    :return:
    """
    return np.array(['Amplitude', 'Baseline [LSB]', '$\sigma_e$ [LSB]'])
