import numpy as np
from utils.pdf import gaussian_sum

__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func"]

'''
# noinspection PyUnusedLocal,PyUnusedLocal,PyTypeChecker,PyTypeChecker
def p0_func(y, x, *args,n_peaks = 5,config=None, **kwargs):
    """
    return the parameters for a pure gaussian distribution
    :param y: the Histogram values
    :param x: the Histogram bins
    :param args:
    :param kwargs:
    :return: starting points for []
    """
    # TODO update with auto determination of the peaks

    if type(config).__name__=='NoneType' or len(np.where(y != 0)[0])<2:
        gain = baseline = sigma_e = sigma_1 = amplitude = np.nan
        param = [baseline, gain,  sigma_e, sigma_1, amplitude]

    else:
        param = []
        # get the edge
        xmin,xmax =  x[np.where(y != 0)[0][1]], x[np.where(y != 0)[0][-1]]
        # Get the gain, sigma_e, sigma_1 and baseline
        param += [config[1,0]] #baseline
        param += [5.6] #gain
        param += [config[2,0]] #sigma_e
        param += [config[2,0]/2.] #sigma_1
        amplitudes = [100.] * 5
        param += amplitudes
    return param


# noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
def slice_func(y, x, *args,config=None, **kwargs):
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
    xmax_hist_for_fit = config[0,0] + 20 * config[1,0] * 1.1
    if np.where(x <xmax_hist_for_fit)[0].shape[0] <2 :
        return [0, 1, 1]
    return [np.where(y != 0)[0][1], np.where(x < xmax_hist_for_fit)[0][-1], 1]


# noinspection PyUnusedLocal,PyUnusedLocal
def bounds_func(*args,n_peaks = 5, config=None, **kwargs):
    """
    return the boundaries for the parameters (essentially none for a gaussian)
    :param args:
    :param kwargs:
    :return:
    """
    bound_min,bound_max = [],[]

    bound_min += [config[1, 0]-2*config[2, 0]]  # baseline-sigma
    bound_max += [config[1, 0]+2*config[2, 0]]  # baseline+sigma

    bound_min += [0.7*5.6]  # 0.8*gain
    bound_max += [2.*5.6]  # 1.2*gain

    bound_min += [0.2*config[2, 0]]  # 0.2*sigma_e
    bound_max += [3.333*config[2, 0]]  # 5.*sigma_e

    bound_min += [0.2*0.5*config[2, 0]]  # 0.2*sigma_1
    bound_max += [3.333*0.5*config[2, 0]]  # 5.*sigma_1

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

# noinspection PyUnusedLocal,PyUnusedLocal
def labels_func(*args,n_peaks = 5, **kwargs):
    """
    List of labels for the parameters
    :return:
    """
    label = ['Baseline [ADC]', 'Gain [ADC / p.e.]', '$\sigma_e$ [ADC]', '$\sigma_1$ [ADC]']
    for p in range(n_peaks):
        label+=['Amplitude_%d'%p]
    return np.array(label)



'''
import numpy as np
from utils.pdf import multi_gaussian_with0

__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func"]


# noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal
def p0_func(*args, config=None, **kwargs):
    return [config[1][0], 0.7, 5.6, 10000., 1000., 100.,  0., 100., 10.]


# noinspection PyShadowingNames,PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
def bounds_func(x, *args, config=None, **kwargs):
    param_min = [config[1][0] * 0.1, 0.01, 0., 100., 1., 0.,  -100., 0., 0.]
    param_max = [config[1][0] * 10., 5., 100., np.inf, np.inf, np.inf,  100., np.inf,
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

"""
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
"""

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
