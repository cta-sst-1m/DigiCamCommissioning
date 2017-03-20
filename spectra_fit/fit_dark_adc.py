import numpy as np
from utils.pdf import multi_gaussian_with0

__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func"]

#TODO Find p0, slice, bounds, from args=(y,x) if config==None

# noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal
def p0_func(y,x,*args, config=None, **kwargs):

    if np.where(y != 0)[0].shape[0] == 0:
        return [np.nan*3]
    param = []
    if np.sum(y)==0:
        return [np.nan]*3
    mean = np.average(x[np.where(y != 0)[0][0]: np.argmax(y)+3: 1], weights=y[np.where(y != 0)[0][0]: np.argmax(y)+3: 1])
    param += [np.sum(y[np.where(y != 0)[0][0]: np.argmax(y)+3: 1])]
    param += [mean]
    param += [np.average((x[np.where(y != 0)[0][0]: np.argmax(y)+3: 1] - mean) ** 2, weights=y[np.where(y != 0)[0][0]:np.argmax(y)+3: 1])]
    return param

# noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
def bounds_func(y,x, *args, config=None, **kwargs):

    bound_min,bound_max = [],[]
    bound_min += [0.]
    bound_max += [np.sum(y)*2.]
    bound_min += [np.min(x)]
    bound_max += [np.max(x)]
    bound_min += [0.]
    bound_max += [np.inf]

    return bound_min,bound_max


# noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal
def slice_func(y, x, *args, **kwargs):
    if np.where(y != 0)[0].shape[0] == 0:
        return [0, 1, 1]
    #if  np.argmax(y)-2>-0.5:
    #    return [np.argmax(y)-2, np.where(y != 0)[0][-1], 1]
    return [np.where(y != 0)[0][0], np.argmax(y)+3, 1]


# noinspection PyUnusedLocal
def fit_func(p, x, *args,config = None, **kwargs):
    mean = p[1]
    amplitude = p[0]
    sigma=p[2]
    return amplitude / np.sqrt(2 * sigma ** 2 * np.pi) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))



# noinspection PyUnusedLocal,PyUnusedLocal
def labels_func(*args, **kwargs):
    """
    List of labels for the parameters
    :return:
    """
    return np.array(['Amplitude', 'Baseline [ADC]', '$\sigma_e$ [ADC]'])
