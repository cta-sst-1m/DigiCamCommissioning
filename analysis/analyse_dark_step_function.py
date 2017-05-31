#!/usr/bin/env python3

# external modules

# internal modules
from utils import display, histogram, geometry
import matplotlib.pyplot as plt
from data_treatement import adc_hist
import logging,sys
import numpy as np
import peakutils
from scipy.interpolate import interp1d


__all__ = ["create_histo", "perform_analysis", "display_results"]


def create_histo(options):
    """
    Create a list histograms and fill it with data
    :param options: a dictionary containing at least the following keys:
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram      (str)
        - 'file_basename'    : the base name of the input files                   (str)
        - 'directory'        : the path of the directory containing input files   (str)
        - 'file_list'        : the list of base filename modifiers for the input files
                                                                                  (list(str))
        - 'evt_max'          : the maximal number of events to process            (int)
        - 'n_evt_per_batch'  : the number of event per fill batch. It can be
                               optimised to improve the speed vs. memory          (int)
        - 'n_pixels'         : the number of pixels to consider                   (int)
        - 'adcs_min'         : the minimum adc value in histo                     (int)
        - 'adcs_max'         : the maximum adc value in histo                     (int)
        - 'adcs_binwidth'    : the bin width for the adcs histo                   (int)
    :return:
    """
    dark = None

    if options.analysis_type == 'step_function':
        dark = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                                bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list),),
                                label='Dark step function', xlabel='threshold LSB', ylabel='entries')

        # Get the adcs
        adc_hist.run(dark, options, h_type='STEPFUNCTION')

    elif options.analysis_type == 'single_photo_electron':
        dark = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                                   bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list),),
                                   label='Pixel SPE', xlabel='Pixel ADC', ylabel='Count / ADC')
        # Get the adcs
        adc_hist.run(dark, options, 'SPE')

    elif options.analysis_type == 'adc_template':
        dark = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                                   bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list),),
                                   label='Dark LSB', xlabel='LSB', ylabel='entries')
        # Get the adcs
        adc_hist.run(dark, options, 'ADC')


    # Save the histogram
    dark.save(options.output_directory + options.histo_filename)

    del dark

    return


def perform_analysis(options):
    """
    Extract the dark rate and cross talk, knowing baseline gain sigmas from previous
    runs
    :param options: a dictionary containing at least the following keys:
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram      (str)
    :return:
    """

    dark_step_function = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    log = logging.getLogger(sys.modules['__main__'].__name__ + __name__)

    y = dark_step_function.data
    x = np.tile(dark_step_function.bin_centers, (y.shape[0], 1))

    y = - np.diff(np.log(y)) / np.diff(x)
    x = x[..., :-1] + np.diff(x) / 2.

    y[np.isnan(y)] = 0
    y[np.isinf(y)] = 0
    y[y < 0] = 0

    threshold = 0.15
    width = 3

    dark_count = np.zeros(y.shape[0])
    cross_talk = np.zeros(y.shape[0])

    n_samples = options.n_samples - options.baseline_per_event_limit - options.window_width + 1
    time = (options.max_event - options.min_event) * 4 * n_samples

    for pixel in range(y.shape[0]):
        max_indices = peakutils.indexes(y[pixel], thres=threshold, min_dist=options.min_distance)

        try:
            max_x = peakutils.interpolate(x[pixel], y[pixel], ind=max_indices, width=width)

        except RuntimeError:

            log.warning('Could not interpolate for pixel %d taking max as indices' % options.pixel_list[pixel])
            max_x = x[pixel][max_indices]

        gain = np.mean(np.diff(max_x))
        min_x = max_x + 0.5 * gain
        f = interp1d(dark_step_function.bin_centers, dark_step_function.data[pixel], kind='cubic')
        counts = [f(min_x[0]), f(min_x[1])]
        dark_count[pixel] = counts[0] / time
        cross_talk[pixel] = counts[1] / counts[0]

        if options.verbose:

            plt.figure()
            plt.semilogy(dark_step_function.bin_centers, dark_step_function.data[pixel])
            plt.axvline(min_x[0])
            plt.axvline(min_x[1])
            plt.show()

    np.savez(options.output_directory + options.analysis_result_filename, dark_count_rate=dark_count, cross_talk=cross_talk)

    return


def display_results(options):
    """
    Display the analysis results
    :param options:
    :return:
    """

    # Load the data
    dark_step_function = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    data = np.load(options.output_directory + options.analysis_result_filename)
    dark_count_rate = data['dark_count_rate']
    cross_talk = data['cross_talk']

    # Display step function
    display.display_hist(dark_step_function, options=options)

    # Display histograms of dark count rate and cross talk
    plt.figure()
    plt.hist(dark_count_rate * 1E3, bins='auto')
    plt.xlabel('$f_{dark}$ [MHz]')

    plt.figure()
    plt.hist(cross_talk, bins='auto')
    plt.xlabel('XT')

    plt.show()

    return

