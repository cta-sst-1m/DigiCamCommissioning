import numpy as np
from scipy.optimize import curve_fit
import peakutils
import utils.pdf

__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func"]


# noinspection PyUnusedLocal,PyUnusedLocal
def p0_func(y, x, *args, config=None, **kwargs):
    """
    find the parameters to start a mpe fit with low light
    :param y: the histogram values
    :param x: the histogram bins
    :param args: potential unused positionnal arguments
    :param config: should be the fit result of a previous fit
    :param kwargs: potential unused keyword arguments
    :return: starting points for []
    """

    if config==None:

        mu = mu_xt = gain = baseline = sigma_e = sigma_1 = amplitude = offset = np.nan
        param = [mu, mu_xt, gain, baseline, sigma_e, sigma_1, amplitude, offset]

    else:

        mu = np.nan
        mu_xt = config[1, 0]
        gain = config[2, 0]
        baseline = config[3, 0]
        sigma_e = config[4, 0]
        sigma_1 = np.nan
        amplitude = np.nan
        offset = config[7, 0]
        #variance = config[8, 0]
        param = [mu, mu_xt, gain, baseline, sigma_e, sigma_1, amplitude, offset]

    # Get a primary amplitude to consider
    param[6] = np.sum(y)

#    param[8] = np.sqrt(np.average((x - np.average(x, weights=y))**2, weights=y))

    # Get the list of peaks in the histogram
    threshold = 0.05
    min_dist = param[2] // 2

    peak_index = peakutils.indexes(y, threshold, min_dist)

    if len(peak_index) == 0:
        return param

    else:

        photo_peak = np.arange(0, peak_index.shape[-1], 1)
        param[2] = np.polynomial.polynomial.polyfit(photo_peak, x[peak_index], deg=1)[1]

        sigma = np.zeros(peak_index.shape[-1])
        for i in range(sigma.shape[-1]):

            start = max(int(peak_index[i] - param[2] // 2), 0)
            end = min(int(peak_index[i] + param[2] // 2), len(x))

            if i == 0:

                param[0] = - np.log(np.sum(y[start:end]) / param[6])

            try:

                temp = np.average(x[start:end], weights=y[start:end])
                sigma[i] = np.sqrt(np.average((x[start:end] - temp) ** 2, weights=y[start:end]))

            except Exception as inst:
                print('Could not compute weights for sigma !!!')
                sigma[i] = param[4]

        sigma_n = lambda sigma_1, n: np.sqrt(param[4] ** 2 + n * sigma_1 ** 2)
        sigma, sigma_error = curve_fit(sigma_n, photo_peak, sigma, bounds=[0., np.inf])
        param[5] = sigma / param[2]

        return param



# noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
def slice_func(y, x, *args, **kwargs):
    """
    returns the slice to take into account in the fit (essentially non 0 bins here)
    :param y: the histogram values
    :param x: the histogram bins
    :param args:
    :param kwargs:
    :return: the index to slice the histogram
    """
    # Check that the histogram has none empty values
    if np.where(y != 0)[0].shape[0] == 0:
        return []
    return [np.where(y != 0)[0][0], np.where(y != 0)[0][-1], 1]


# noinspection PyUnusedLocal,PyUnusedLocal
def bounds_func(*args, config=None, **kwargs):
    """
    return the boundaries for the parameters (essentially none for a gaussian)
    :param args:
    :param kwargs:
    :return:
    """


    if True:

        param_min = [0., 0., 0., -np.inf, 0., 0., 0.,-np.inf]
        param_max = [np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf,np.inf]


    else:

        mu = config[0]
        mu_xt = config[1]
        gain = config[2]
        baseline = config[3]
        sigma_e = config[4]
        sigma_1 = config[5]
        amplitude = config[6]
        offset = config[7]

        param_min = [0.    , 0., 0                   , -np.inf                  , 0.                       , 0.     ,0.    ,-np.inf]
        param_max = [np.inf, 1 , gain[0] + 10*gain[1], baseline[0]+5*baseline[1], sigma_e[0] + 5*sigma_e[1], np.inf ,np.inf, np.inf]

    return param_min, param_max


def fit_func(p, x):
    """
    Simple gaussian pdf
    :param p: [norm,mean,sigma]
    :param x: x
    :return: G(x)
    """
    #mu, mu_xt, gain, baseline, sigma_e, sigma_1, amplitude, offset, variance = p
    mu, mu_xt, gain, baseline, sigma_e, sigma_1, amplitude, offset = p
    temp = np.zeros(x.shape)
    x = x - baseline
    n_peak = 30
    for n in range(0, n_peak, 1):

        sigma_n = np.sqrt(sigma_e ** 2 + n * sigma_1 ** 2)  * gain


        temp += utils.pdf.generalized_poisson(n, mu, mu_xt) * utils.pdf.gaussian(x , sigma_n, n * gain)
        #temp += utils.pdf.generalized_poisson(n, mu, mu_xt) * utils.pdf.gaussian(x, sigma_n, n * gain + (offset if n!=0 else 0))

    return temp * amplitude

if __name__ == '__main__':

    print('Hello')