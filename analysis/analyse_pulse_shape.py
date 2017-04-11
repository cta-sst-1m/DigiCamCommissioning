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

    pulse_shape_temp = np.zeros((len(options.scan_level), len(options.pixel_list), options.n_bins, 2))
    data_treatement.pulse_shape.run(pulse_shape_temp, options=options)

    pulse_shape = histogram.Histogram(data=pulse_shape_temp[..., 0],\
                                      bin_centers=np.arange(0, options.n_bins, 1) * options.sampling_time, \
                                      label='Pulse shape', xlabel='t [ns]', ylabel='amplitude [ADC]')

    pulse_shape.errors = pulse_shape_temp[..., 1]
    pulse_shape.fit_result_label = ['amplitude', 'full_integral [ADC]', 'full_square_integral [ADC$^2$]', 'full_integral_normalized [ns]', 'full_square_integral_normalized [ns$^2$]']
    pulse_shape.fit_result = np.zeros(pulse_shape.data.shape[0:-1] + (len(pulse_shape.fit_result_label),2))


    pulse_shape.save(options.output_directory + options.histo_filename)

    del pulse_shape

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
