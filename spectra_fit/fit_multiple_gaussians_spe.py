import numpy as np
import peakutils
from utils.pdf import gaussian_sum
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func", "labels_func"]


def p0_func(y, x, *args, n_peaks=7, config=None, **kwargs):
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

    if True == True:
        #plt.ion()
        amplitudes = [float(np.sum(y)*10**(-i)) for i in range(n_peaks)]
        threshold = 0.01
        min_dist = 10.
        min_val = 17 if x[np.where(y > 0)[0][0]] < 20 else 34
        y_peak = y[np.where(x < min_val)[0][-1]:np.where(y != 0)[0][-1]]
        x_peak = x[np.where(x < min_val)[0][-1]:np.where(y != 0)[0][-1]]
        if y_peak.shape[0]<1:
            peak_index = np.zeros(1)
        else:
            n = np.argmax(y_peak)+np.where(x < min_val)[0][-1]

            zero_bin = np.argmin(np.abs(x))
            if min_val < 20:
                peak_index = np.arange(1,n_peaks+1) * (np.argmax(y_peak)+np.where(x < min_val)[0][-1]-zero_bin)+zero_bin
            else:
                peak_index = np.arange(1,n_peaks+1) * int((np.argmax(y_peak)+np.where(x < min_val)[0][-1]-zero_bin)/2)+zero_bin
        photo_peak = np.arange(0, len(peak_index), 1)
        '''

        #plt.plot(np.arange(x.shape[0]),y)
        #plt.plot(np.arange(x.shape[0]-1),np.diff(y))
        y_peak = y[np.where(x < min_val)[0][-1], np.where(y != 0)[0][-1], 1]
        x_peak =

        peak_index = peakutils.indexes(y_peak, threshold, min_dist)
        if len(peak_index) > 1 and y[peak_index[0]]<y[peak_index[1]]:
            peak_index = peak_index[1:]
        #plt.plot(peak_index,y[peak_index])
        #print(peak_index)
        photo_peak = np.arange(0, len(peak_index), 1)

        #if abs(np.argmax(y) - peak_index[0]) > 4:
        peak_index=np.arange(1,peak_index.shape[0]+1)*np.argmax(y_peak)
            #plt.show()
        #print('len(peak_index)',len(peak_index))
        #input('press')
        '''
        '''
        if np.mean(np.diff(x[peak_index]))> 30:
            plt.ion()
            plt.clf()
            plt.plot(x, y)
            plt.plot(x[peak_index], y[peak_index])
            plt.show()
            input('press')
        '''

        if len(peak_index) <= 2:

            try:

                gain = np.max(x[y > 0]) - np.min(x[y > 0])
                baseline = np.min(x[y > 0]) + gain / 2.

            except:

                gain = np.nan
                baseline = np.nan

        else:
            gain = x[peak_index[0]]#np.mean(np.diff(x[peak_index]))
            baseline = 0.#np.min(x[y > 0]) + gain / 2.

        sigma = np.zeros(peak_index.shape[-1])

        for i in range(sigma.shape[-1]):

            start = max(int(peak_index[i] - gain / 2.), 0)
            end = min(int(peak_index[i] + gain / 2.), len(x))

            try:

                temp = np.average(x[start:end], weights=y[start:end])
                sigma[i] = np.sqrt(np.average((x[start:end] - temp) ** 2, weights=y[start:end]))

            except Exception as inst:
                # log.error('Could not compute weights for sigma !!!')
                sigma[i] = gain / 2.

        sigma_n = lambda sigma_e, sigma_1, n: np.sqrt(sigma_e ** 2 + n * sigma_1 ** 2 + 1./12.)

        try:

            sigmas, sigma_error = curve_fit(sigma_n, photo_peak, sigma, bounds=[0., np.inf])

        except ValueError:

            sigmas = [0.5 * gain, 0.5 * gain]



        sigma_e = sigmas[0]
        sigma_1 = sigmas[1]

        param += [baseline]
        param += [float(gain)]
        param += [sigma_e]
        param += [sigma_1]
        param += amplitudes
    return param


def slice_func(y, x, *args, n_peaks=6, config=None, **kwargs):
    """
    returns the slice to take into account in the fit (essentially non 0 bins here)
    :param y: the Histogram values
    :param x: the Histogram bins
    :param args:
    :param kwargs:
    :return: the index to slice the Histogram
    """

    #init_parameters = p0_func(y, x, *args, n_peaks, config, **kwargs)

    #if config is None:

    if np.where(y != 0)[0].shape[0] < 2:
        return [0, 1, 1]

    else:
        #min_val = 17 if x[0] < 20 else 34
        return [np.where(y != 0)[0][0]+3, np.where(y != 0)[0][-1], 1]


def bounds_func(y, x, *args, n_peaks = 7, config=None, **kwargs):
    """
    return the boundaries for the parameters (essentially none for a gaussian)
    :param args:
    :param kwargs:
    :return:
    """
    bound_min, bound_max = [], []

    init_parameters = p0_func(y, x, *args, n_peaks, config, **kwargs)

    baseline = init_parameters[0]
    gain = init_parameters[1]
    sigma_e = init_parameters[2]
    sigma_1 = init_parameters[3]
    amplitudes = init_parameters[4:]

    #if config is None:

    bound_min += [baseline - 3*sigma_e]  # baseline-sigma
    bound_max += [baseline + 3*sigma_e]  # baseline+sigma

    bound_min += [0.6*gain]
    bound_max += [1.4 * gain]

    bound_min += [0.]
    bound_max += [2. * sigma_e]

    bound_min += [0.]
    bound_max += [ 5. * sigma_1]

    bound_min += [0.] * n_peaks
    bound_max += [np.inf] * n_peaks

    return bound_min, bound_max


def fit_func(p, x, *args, **kwargs):
    """
    Simple gaussian pdf
    :param p: [norm,mean,sigma]
    :param x: x
    :return: G(x)
    """
    return gaussian_sum(p, x, skip_first_peak = True)

def labels_func(*args, n_peaks = 7, **kwargs):
    """
    List of labels for the parameters
    :return:
    """
    label = ['Baseline [LSB]', 'Gain [LSB / p.e.]', '$\sigma_e$ [LSB]', '$\sigma_1$ [LSB]']
    for p in range(n_peaks):
        label += ['Amplitude_' + str(p)]

    return np.array(label)

