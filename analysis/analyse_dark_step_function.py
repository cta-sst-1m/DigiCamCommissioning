#!/usr/bin/env python3

# external modules

# internal modules
from utils import display, histogram, geometry
import matplotlib.pyplot as plt
from data_treatement import adc_hist
import logging,sys
import numpy as np
import peakutils
from scipy.interpolate import barycentric_interpolate, interp1d

from scipy import stats

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
    dark_step_function = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                                   bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list),),
                                   label='Dark step function', xlabel='threshold LSB', ylabel='entries')

    # Get the adcs

    adc_hist.run(dark_step_function, options, h_type='STEPFUNCTION')


    # Save the histogram
    dark_step_function.save(options.output_directory + options.histo_filename)

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

    y = dark_step_function.data
    x = np.tile(dark_step_function.bin_centers, (y.shape[0], 1))

    y = - np.diff(np.log(y)) / np.diff(x)
    x = x[..., :-1] + np.diff(x) / 2.

    y[np.isnan(y)] = 0
    y[np.isinf(y)] = 0
    y[y < 0] = 0

    threshold = 0.8
    width = 2

    dark_count = np.zeros(y.shape[0])
    cross_talk = np.zeros(y.shape[0])

    time = (options.event_max - options.event_min) * 4 * options.n_bins

    for pixel in range(y.shape[0]):
        max_indices = peakutils.indexes(y[pixel], thres=threshold, min_dist=options.min_distance)
        max_x = peakutils.interpolate(x[pixel], y[pixel], ind=max_indices, width=width)
        gain = np.mean(np.diff(max_x))
        min_x = max_x + 0.5 * gain
        f = interp1d(dark_step_function.bin_centers, dark_step_function.data[pixel], kind='cubic')
        counts = [f(min_x[0]), f(min_x[1])]
        dark_count[pixel] = counts[0] / time
        cross_talk[pixel] = counts[1] / counts[0]

        if options.debug:

            plt.figure()
            plt.semilogy(dark_step_function.bin_centers, dark_step_function.data[pixel])
            plt.axvline(min_x[0])
            plt.axvline(min_x[1])
            plt.show()

    print(cross_talk)
    print(dark_count)
    return

def display_results(options):
    """
    Display the analysis results
    :param options:
    :return:
    """

    # Load the histogram
    adcs = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    display.display_hist(adcs, options=options)

    return