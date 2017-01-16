import numpy as np
from utils.fitting import multi_gaussian_with0

__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func"]


# noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal
def p0_func(*args, config=None, **kwargs):
    return [config[2][0], 0.7, 5.6, 10000., 1000., 100.,  0., 100., 10.]


# noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
def bounds_func(x, *args, config=None, **kwargs):
    param_min = [config[2][0] * 0.1, 0.01, 0., 100., 1., 0.,  -100., 0., 0.]
    param_max = [config[2][0] * 10., 5., 100., np.inf, np.inf, np.inf,  100., np.inf,
                 np.inf]
    # param_min = [0.01, 0. , 100.   , 1.    , 0.  ,-10., 0.    ,0.]
    # param_max = [5. , 100., np.inf, np.inf, np.inf,10. , np.inf,np.inf]
    return param_min, param_max


# noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal
def slice_func(y, x, *args, **kwargs):
    if np.where(y != 0)[0].shape[0] == 0:
        return [0, 1, 1]
    if  np.argmax(y)-2>-0.5:
        return [np.argmax(y)-2, np.where(y != 0)[0][-1], 1]
    return [np.where(y != 0)[0][0], np.where(y != 0)[0][-1], 1]

'''
gaus0 = p[0] / (p[9]) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x) - (p[7])) ** 2 / (2. * (p[9] ** 2)))
gaus1 = p[4] / (np.sqrt(p[1] ** 2 + 1 * (p[2]) ** 2)) / np.sqrt(2. * np.pi) * np.exp(
    -(np.asfarray(x) - (p[3] + p[7]+p[8])) ** 2 / (2. * (p[1] ** 2 + 1 * (p[2]) ** 2)))
gaus2 = p[5] / (np.sqrt(p[1] ** 2 + 2 * (p[2]) ** 2)) / np.sqrt(2. * np.pi) * np.exp(
    -(np.asfarray(x) - (p[3] * 2 + p[7] + p[8])) ** 2 / (2. * (p[1] ** 2 + 2 * (p[2]) ** 2)))
gaus3 = p[6] / (np.sqrt(p[1] ** 2 + 3 * (p[2]) ** 2)) / np.sqrt(2. * np.pi) * np.exp(
    -(np.asfarray(x) - (p[3] * 3 + p[7] + [p[8]])) ** 2 / (2. * (p[1] ** 2 + 3 * (p[2]) ** 2)))
gaus4 = p[10] / (np.sqrt(p[1] ** 2 + 4 * (p[2]) ** 2)) / np.sqrt(2. * np.pi) * np.exp(
    -(np.asfarray(x) - (p[3] * 4 + p[7] + [p[8]])) ** 2 / (2. * (p[1] ** 2 + 4 * (p[2]) ** 2)))
gaus5 = p[11] / (np.sqrt(p[1] ** 2 + 5 * (p[2]) ** 2)) / np.sqrt(2. * np.pi) * np.exp(
    -(np.asfarray(x) - (p[3] * 5 + p[7] + [p[8]])) ** 2 / (2. * (p[1] ** 2 + 5 * (p[2]) ** 2)))
'''

# noinspection PyUnusedLocal
def fit_func(p, x, *args,config = None, **kwargs):
    p_new = [0.] * 12
    p_new[0] = 0.   ## amplitude peak 0
    p_new[1] = p[0] ## amplitude peak 0
    p_new[2] = p[1]
    p_new[3] = p[2]
    p_new[4] = p[3]
    p_new[5] = p[4]
    p_new[6] = p[5]
    p_new[7] = config[1][0]
    p_new[8] = p[6]
    p_new[9] = 1.
    p_new[10] = p[7]
    p_new[11] = p[8]

    return multi_gaussian_with0(p_new, x)
