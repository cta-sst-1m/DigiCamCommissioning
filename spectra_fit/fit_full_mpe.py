import numpy as np
import peakutils
from utils.pdf import gaussian_sum
from scipy.optimize import curve_fit

__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func"]


import numpy as np

__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func"]

#TODO Find p0, slice, bounds, from args=(y,x) if config==None  ADD Jac


# noinspection PyUnusedLocal,PyUnusedLocal,PyTypeChecker,PyTypeChecker
def p0_func(y, x, *args, n_peaks=22, config=None, **kwargs):
    """
    return the parameters for a pure gaussian distribution
    :param y: the Histogram values
    :param x: the Histogram bins
    :param args:
    :param kwargs:
    :return: starting points for []
    """
    # TODO update with auto determination of the peaks

    param = []

    if config is None:

        amplitudes = [np.sum(y)/n_peaks]*n_peaks
        threshold = 0.3
        min_dist = 3.
        peak_index = peakutils.indexes(y, threshold, min_dist)
        photo_peak = np.arange(0, len(peak_index), 1)

        if len(peak_index) <= 2:

            gain = np.max(x[y > 0]) - np.min(x[y > 0])
            baseline = np.min(x[y > 0]) + gain / 2.

        else:

            gain = np.mean(np.diff(x[peak_index]))
            baseline = np.min(x[y > 0]) + gain / 2.

        sigma = np.zeros(peak_index.shape[-1])

        for i in range(sigma.shape[-1]):

            start = max(int(peak_index[i] - gain / 2.), 0)
            end = min(int(peak_index[i] + gain / 2.), len(x))

            try:

                temp = np.average(x[start:end], weights=y[start:end])
                sigma[i] = np.sqrt(np.average((x[start:end] - temp) ** 2, weights=y[start:end]))

            except Exception as inst:
                log.error('Could not compute weights for sigma !!!')
                sigma[i] = gain / 2.

        sigma_n = lambda sigma_e, sigma_1, n: np.sqrt(sigma_e ** 2 + n * sigma_1 ** 2 + 1./12.)
        sigmas, sigma_error = curve_fit(sigma_n, photo_peak, sigma, bounds=[0., np.inf])

        sigma_e = sigmas[0]
        sigma_1 = sigmas[1]

        param += [baseline]
        param += [gain]
        param += [sigma_e]
        param += [sigma_1]
        param += amplitudes


    else:
        if config.shape[-2] == 3:

            param += [-10.]  # baseline
            param += [5.6*4.3]  # gain
            #param += [config[2, 0]]  # sigma_e
            #param += [config[2, 0]*0.5]  # sigma_1
            param += [3.5]  # sigma_e
            param += [1.77]  # sigma_1
            amplitudes = [np.sum(y)/n_peaks] * n_peaks
            param += amplitudes
    return param


# noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
def slice_func(y, x, *args,n_peaks=22,config=None, **kwargs):
    """
    returns the slice to take into account in the fit (essentially non 0 bins here)
    :param y: the Histogram values
    :param x: the Histogram bins
    :param args:
    :param kwargs:
    :return: the index to slice the Histogram
    """

    if config is None:
        if np.where(y != 0)[0].shape[0] < 2:
            return [0, 1, 1]

        else:

            return [np.where(y != 0)[0][0], np.where(y != 0)[0][-1], 1]
        #return [2030, 2100, 1] #### ATENTNENT ###

    # Check that the Histogram has none empty values

    else:

        if np.where(y != 0)[0].shape[0] < 2:
            return [0, 1, 1]

        xmax_hist_for_fit = config[1,0] + (n_peaks-3) * 5.6 * 4.3

        if np.where(x <xmax_hist_for_fit)[0].shape[0] <2 :
            return [0, 1, 1]

        else:
            return [np.where(y != 0)[0][0], np.where(x < xmax_hist_for_fit)[0][-1], 1]


# noinspection PyUnusedLocal,PyUnusedLocal
def bounds_func(y,*args,n_peaks = 22, config=None, **kwargs):
    """
    return the boundaries for the parameters (essentially none for a gaussian)
    :param args:
    :param kwargs:
    :return:
    """
    bound_min,bound_max = [],[]

    if config is None:

        bound_min += [300]  # baseline-sigma
        bound_max += [700]  # baseline+sigma

        bound_min += [2.]
        bound_max += [10.*4.3]

        bound_min += [0.]
        bound_max += [6.]

        bound_min += [0.]
        bound_max += [3.]

        bound_min += [0.] * n_peaks
        bound_max += [np.sum(y)] * n_peaks

    else:

        if config.shape[-2] == 3:

            bound_min += [config[1, 0] - 30]  # baseline-sigma
            bound_max += [3.5]  # baseline+sigma
            '''
            bound_min += [0.9*5.6]  # 0.8*gain
            bound_max += [1.1*5.6]  # 1.2*gain

            bound_min += [0.7]  # 0.2*sigma_e
            bound_max += [1.3]  # 5.*sigma_e

            bound_min += [0.4]  # 0.2*sigma_1
            bound_max += [1.3]  # 5.*sigma_1
            '''
            bound_min += [0.7 * 5.6 * 4.3]  # 0.8*gain
            bound_max += [1.5 * 5.6 * 4.3]  # 1.2*gain

            bound_min += [0.2]  # 0.2*sigma_e
            bound_max += [6.]  # 5.*sigma_e

            bound_min += [0.2]  # 0.2*sigma_1
            bound_max += [3.]  # 5.*sigma_1

            bound_min += [0.] * n_peaks
            bound_max += [np.inf] * n_peaks
        else:

            bound_min += [config[1, 0] - 30]  # baseline-sigma
            bound_max += [config[1, 0] + 30]  # baseline+sigma
            #bound_min += [config[0, 0] - 2 * config[2, 0]]  # baseline-sigma
            #bound_max += [config[0, 0] + 2 * config[2, 0]]  # baseline+sigma
            '''
            bound_min += [0.9*5.6]  # 0.8*gain
            bound_max += [1.1*5.6]  # 1.2*gain

            bound_min += [0.7]  # 0.2*sigma_e
            bound_max += [1.3]  # 5.*sigma_e

            bound_min += [0.4]  # 0.2*sigma_1
            bound_max += [1.3]  # 5.*sigma_1
            '''
            bound_min += [0.7 * 5.6 * 4.3]  # 0.8*gain
            bound_max += [1.5 * 5.6 * 4.3]  # 1.2*gain

            bound_min += [0.2 * config[2, 0]]  # 0.2*sigma_e
            bound_max += [6.]  # 5.*sigma_e

            bound_min += [0.2 * config[3, 0]]  # 0.2*sigma_1
            bound_max += [3.]  # 5.*sigma_1

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
def label_func(*args,n_peaks = 22, **kwargs):
    """
    List of labels for the parameters
    :return:
    """
    label = ['Baseline [ADC]', 'Gain [ADC / p.e.]', '$\sigma_e$ [ADC]', '$\sigma_1$ [ADC]']
    for p in range(n_peaks):
        label += ['Amplitude_' + str(p)]
    return np.array(label)

