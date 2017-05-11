#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import pulse_shape
from utils import display, histogram, geometry
import logging,sys
import numpy as np
import logging
from tqdm import tqdm
from utils.logger import TqdmToLogger
import matplotlib.pyplot as plt

from ctapipe import visualization
import scipy
__all__ = ["create_histo", "perform_analysis", "display_results"]


def create_histo(options):
    """
    Create a list of ADC histograms and fill it with data

    :param options: a dictionary containing at least the following keys:
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram      (str)
        - 'synch_histo_filename'   : the name of the file containing the histogram      (str)
        - 'file_basename'    : the base name of the input files                   (str)
        - 'directory'        : the path of the directory containing input files   (str)
        - 'file_list'        : the list of base filename modifiers for the input files
                                                                                  (list(str))
        - 'evt_max'          : the maximal number of events to process            (int)
        - 'n_evt_per_batch'  : the number of event per fill batch. It can be
                               optimised to improve the speed vs. memory          (int)
        - 'n_pixels'         : the number of pixels to consider                   (int)
        - 'scan_level'       : number of unique poisson dataset                   (int)
        - 'adcs_min'         : the minimum adc value in histo                     (int)
        - 'adcs_max'         : the maximum adc value in histo                     (int)
        - 'adcs_binwidth'    : the bin width for the adcs histo                   (int)

    :return:
    """
    pulse_shape.load(filename = options.output_directory + options.pulse_histo_filename)
    print(pulse_shape.data.shape)
    window_width = options.window_width
    window_start = options.window_start

    def integrate_trace(d):
        return np.convolve(d, np.ones((window_width), dtype=int), 'valid')

    def contiguous_regions(data):
        """Finds contiguous True regions of the boolean array "condition". Returns
        a 2D array where the first column is the start index of the region and the
        second column is the end index."""
        condition = data > 0
        # Find the indicies of changes in "condition"
        d = np.diff(condition)
        idx, = d.nonzero()

        # We need to start things after the change in "condition". Therefore,
        # we'll shift the index by 1 to the right.
        idx += 1

        if condition[0]:
            # If the start of condition is True prepend a 0
            idx = np.r_[0, idx]

        if condition[-1]:
            # If the end of condition is True, append the length of the array
            idx = np.r_[idx, condition.size]  # Edit

        # Reshape the result into two columns
        idx.shape = (-1, 2)
        val = 0.
        for start, stop in idx:
            sum_tmp = np.sum(data[start:stop])
            if val < sum_tmp: val = sum_tmp
        return val


    integrate_trace_n = lambda x: integrate_trace(x, options.window_width)

    ## Create fake peak position and max window
    peak_positions = np.zeros((pulse_shape.data[20,...].shape[0],pulse_shape.data[20,...].shape[1]+1),dtype = float)
    for k in range(peak_positions.shape[1]):
        peak_positions[:,k] = pulse_shape.data[20,:,k]/np.sum(pulse_shape.data[20,...],axis=-1)
    peak = np.argmax(peak_positions, axis=1)
    mask = (peak_positions.T / np.sum(peak_positions, axis=1)).T > 1e-3
    mask_window = mask + np.append(mask[..., 1:], np.zeros((peak_positions.shape[0], 1), dtype=bool), axis=1) + \
                  np.append(np.zeros((peak_positions.shape[0], 1), dtype=bool), mask[..., :-1], axis=1)
    mask_windows_edge = mask_window * ~mask
    mask_window = mask_window[..., :-1]
    mask_windows_edge = mask_windows_edge[..., :-1]

    shift = window_start  # window_width - int(np.floor(window_width/2))+window_start
    missing = mask_window.shape[1] - (window_width - 1)
    mask_window = mask_window[..., shift:]
    # print(mask_window.shape[1], missing)
    missing = mask_window.shape[1] - missing
    mask_window = mask_window[..., :-missing]
    # print(mask_window.shape[1], missing)
    mask_windows_edge = mask_windows_edge[..., shift:]
    mask_windows_edge = mask_windows_edge[..., :-missing]

    for i,level in enumerate(options.levels):
        # first deal with normal values
        data = pulse_shape.data[i]
        max_idx = (np.arange(0, data.shape[0]), np.argmax(data, axis=1),)
        data_int = np.copy(data)
        data_sat = np.copy(data)
        data_sat[data_sat[max_idx] < options.threshold_sat] = 0.
        data_int[data_int[max_idx] >= options.threshold_sat] = 0.

        integration = np.apply_along_axis(integrate_trace, 1, data_int)
        local_max = np.argmax(np.multiply(integration, mask_window), axis=1)
        local_max_edge = np.argmax(np.multiply(integration, mask_windows_edge), axis=1)
        ind_max_at_edge = (local_max == local_max_edge)
        local_max[ind_max_at_edge] = peak[ind_max_at_edge] - window_start
        index_max = (np.arange(0, data_int.shape[0]), local_max,)
        ind_with_lt_th = integration[index_max] < 10.
        local_max[ind_with_lt_th] = peak[ind_with_lt_th] - window_start
        index_max = (np.arange(0, data_int.shape[0]), local_max,)
        full_integration = integration[index_max]
        # now deal with saturated ones:
        sat_integration = np.apply_along_axis(contiguous_regions, 1, data_sat)
        full_integration = full_integration + sat_integration #- baseline[level, :]
        hist.fill(full_integration, indices=(level,))


    return


def perform_analysis(options):

    pulse_shape = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    def integrate_trace(d, window_width):

        integrated_trace = np.convolve(d, np.ones((window_width), dtype=int), 'same')

        return integrated_trace

    integrate_trace_n = lambda x: integrate_trace(x, options.window_width)

    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    pbar = tqdm(total=pulse_shape.data.shape[0])
    tqdm_out = TqdmToLogger(log, level=logging.INFO)

    for level in range(pulse_shape.data.shape[0]):

        log.debug('--|> Moving to AC level %d [DAC]' %options.scan_level[level])

        for pixel in range(pulse_shape.data.shape[1]):

            pulse_shape.data[level, pixel] = np.apply_along_axis(integrate_trace_n, -1, pulse_shape.data[level, pixel])
            pulse_shape.errors[level, pixel] = np.sqrt(np.apply_along_axis(integrate_trace_n, -1, pulse_shape.errors[level, pixel]**2))
            pulse_shape.fit_result[level, pixel, 0, 0] = np.max(pulse_shape.data[level, pixel])
            pulse_shape.fit_result[level, pixel, 0, 1] = pulse_shape.errors[level, pixel, np.argmax(pulse_shape.data[level, pixel])]
            pulse_shape.fit_result[level, pixel, 1, 0] = np.sum(pulse_shape.data[level, pixel])
            pulse_shape.fit_result[level, pixel, 1, 1] = np.sqrt(np.sum(pulse_shape.errors[level, pixel]**2))
            pulse_shape.fit_result[level, pixel, 2, 0] = np.sum(pulse_shape.data[level, pixel]**2)
            pulse_shape.fit_result[level, pixel, 2, 1] = 0.
            pulse_shape.fit_result[level, pixel, 3, 0] = np.sum(pulse_shape.data[level, pixel] / pulse_shape.fit_result[level, pixel, 0, 0])
            pulse_shape.fit_result[level, pixel, 3, 1] = 0.
            pulse_shape.fit_result[level, pixel, 4, 0] = np.sum((pulse_shape.data[level, pixel] / pulse_shape.fit_result[level, pixel, 0, 0])**2)
            pulse_shape.fit_result[level, pixel, 4, 1] = 0.

        pbar.update(1)

    pulse_shape.save(options.output_directory + options.histo_filename)

    return


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    pulse_shape = histogram.Histogram(filename=options.output_directory + options.histo_filename)


    display.display_hist(pulse_shape, options=options, scale='linear')
    display.display_fit_result_level(pulse_shape, options=options, scale='linear')

    '''
    data = np.load(options.output_directory + options.pulse_shape_filename)['pulse_shape']
    data_substracted = np.load(options.output_directory + options.pulse_shape_filename.split('.')[0] + '_substracted.npz')['pulse_shape']
    pulse_integrals = np.load(options.output_directory + options.pulse_shape_filename.split('.')[0] + '_integrals.npz')['pulse_integrals']


    geom = geometry.generate_geometry_0(pixel_list=options.pixel_list)

    print(data_substracted.shape)


    #display.display_pulse_shape(data, options=options, geom=geom)
    display.display_pulse_shape(data_substracted, options=options)

    import matplotlib.pyplot as plt


    pixel_id = 9
    pixel_index = np.where(np.array(options.pixel_list)==pixel_id)

    plt.figure()
    plt.errorbar(np.array(options.scan_level), pulse_integrals[:, pixel_index, 0], yerr=pulse_integrals[:,pixel_index,1], label='pixel : %d' %pixel_id, fmt='ok')
    plt.xlabel('AC level [DAC]')
    plt.ylabel('integral [ns] ($n_{bins}=$ %d)' %options.window_width)
    plt.legend(loc='best')
    plt.show()

    print (np.max(data[:, pixel_index, :, 0], axis=-2).shape)
    print (data[:, pixel_index, :, 0].shape)
    print (np.array(options.scan_level).shape)

    plt.figure()
    plt.errorbar(np.array(options.scan_level), np.max(data[:, pixel_index, :, 0], axis=-2), label='pixel : %d' %pixel_id, fmt='ok')
    plt.xlabel('AC level [DAC]')
    plt.ylabel('amplitude [ADC] ($n_{bins}=$ %d)' %options.window_width)
    plt.legend(loc='best')
    plt.show()
    '''

    return
