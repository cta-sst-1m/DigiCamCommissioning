import numpy as np
from scipy.optimize import curve_fit
import scipy.interpolate
from scipy.interpolate import splev
import logging,sys
import utils.pdf

__all__ = ["p0_func", "slice_func", "bounds_func", "fit_func"]


# noinspection PyUnusedLocal,PyUnusedLocal
def p0_func(y, x, config=None, *args, **kwargs):
    """
    find the parameters to start a mpe fit with low light
    :param y: the Histogram values
    :param x: the Histogram bins
    :param args: potential unused positionnal arguments
    :param config: should be the fit result of a previous fit
    :param kwargs: potential unused keyword arguments
    :return: starting points for []
    """
    if config is None:

        max_position = np.argmax(y)
        t_0 = (max_position - 4) * 4.

        left_width = max_position - 4
        right_width = x.shape[0] - (max_position + 12)

        baseline_side = np.argmax([left_width, right_width])

        if baseline_side == 0:

            baseline = np.mean(y[0:left_width])

        elif baseline_side == 1:

            baseline = np.mean(y[-right_width + x.shape[0]:-1])

        amplitude = np.max(y - baseline)

        param = [t_0, amplitude, baseline]

    return param

def slice_func(y, x, *args, **kwargs):
    """
    returns the slice to take into account in the fit (essentially non 0 bins here)
    :param y: the Histogram values
    :param x: the Histogram bins
    :param args:
    :param kwargs:
    :return: the index to slice the Histogram
    """

    x_start = max(np.argmax(y) - 7, 0)
    x_end = min(np.argmax(y) + 12, x.shape[0])

    #return [0, x.shape[0], 1]
    return [x_start, x_end, 1] # not tested yet

def bounds_func(y, x, config=None, *args, **kwargs):
    """
    return the boundaries for the parameters (essentially none for a gaussian)
    :param args:
    :param kwargs:
    :return:
    """

    if config is None:

        param_min = [0, 0., np.min(y)]
        param_max = [x.shape[0]*4., np.max(y), np.max(y)]
        return param_min, param_max



def fit_func_mc(parameter, time, *args, **kwargs):
    """
    Simple gaussian pdf
    :param p: [norm,mean,sigma]
    :param x: x
    :return: G(x)
    """

    t_0 = parameter[0]
    amplitude = parameter[1]
    baseline = parameter[2]

    filename_pulse_shape = 'utils/pulse_SST-1M_AfterPreampLowGain.dat'  # pulse shape template file

    time_steps, amplitudes = np.loadtxt(filename_pulse_shape, unpack=True, skiprows=1)
    amplitudes = amplitudes / min(amplitudes)
    f = scipy.interpolate.interp1d(time_steps, amplitudes, kind='cubic', bounds_error=False, fill_value=0., assume_sorted=True)

    return amplitude * f(time - t_0) + baseline

def fit_func(parameter, time, pixel_id, *args, **kwargs):
    """
    Simple gaussian pdf
    :param p: [norm,mean,sigma]
    :param x: x
    :return: G(x)
    """

    t_0 = parameter[0]
    amplitude = parameter[1]
    baseline = parameter[2]

    #filename_pulse_shape = 'utils/pulse_SST-1M_AfterPreampLowGain.dat'  # pulse shape template file
    filename_pulse_shape = 'pulse_template/pulse_shape.npz'  # pulse shape template file

    data = np.load(filename_pulse_shape)
    pixel = data['pixel_id']

    spline = data['spline'][np.argwhere(pixel == pixel_id)[0][0]]

    return amplitude * splev(time - t_0 + 210, spline) + baseline

def label_func(*args, ** kwargs):
    """
    List of labels for the parameters
    :return:
    """
    label = ['$t_0$ [ns]', 'amplitude [LSB]', 'Baseline [LSB]']
    return np.array(label)

if __name__ == '__main__':

    print('Nothing implemented')