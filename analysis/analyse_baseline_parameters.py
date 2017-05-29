#!/usr/bin/env python3

# external modules

# internal modules
import logging
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as col
import numpy as np
from scipy import stats

from data_treatement import adc_hist
from utils import histogram


__all__ = ["create_histo", "perform_analysis", "display_results"]


def create_histo(options):
    """
    Fill

    :param options: a dictionary containing at least the following keys:
        - 'pixel_list'       : a list of the pixel id to consider                 (list(int))
        - 'min_event'          : the minimal event id to process                  (int)
        - 'max_event'          : the maximal event id to process                  (int)
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram      (str)
        other options might be needed from the adc_hist.run function

    :return:
    """

    # Create an histogram of shape (len(options.pixel_list), options.max_event, 2) to hold
    # per pixel and per event the baseline mean and rms as obtained from options.baseline_per_event_limit
    # events
    adcs = histogram.Histogram(bin_center_min=options.min_event, bin_center_max=options.max_event-1,
                               bin_width=1, data_shape=(2, len(options.pixel_list),),
                               label='Mean or RMS of the baseline', xlabel='Event ID', ylabel='Baseline Mean/RMS',
                               dtype = float)

    # Fill the histogram from events
    adc_hist.run(adcs, options, 'MEANRMS')

    # Save the histogram
    adcs.save(options.output_directory + options.histo_filename)

    # Delete the histograms
    del adcs

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
    log = logging.getLogger(sys.modules['__main__'].__name__ + __name__)
    log.info('Get the distribution of mean and RMS of the baseline')

    # Load the histogram
    adcs = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    # Compute the mean and stddev of the baseline mean and stddev, in all pixel
    adcs.fit_result = np.zeros((len(options.pixel_list), 4), dtype=float)
    adcs.data = adcs.data.astype(float)
    adcs.data[0][adcs.data[0] == 0] = np.nan
    adcs.data[1][adcs.data[1] == 0] = np.nan
    adcs.fit_result[..., 0] = stats.mode(adcs.data[0], axis=-1, nan_policy='omit')[0][...,0]
    adcs.fit_result[..., 1] = np.nanstd(adcs.data[0],axis=-1)
    adcs.fit_result[..., 2] = stats.mode(adcs.data[1], axis=-1, nan_policy='omit')[0][...,0]
    adcs.fit_result[..., 3] = np.nanstd(adcs.data[1],axis=-1)
    adcs.fit_result_label = np.array(
        ['Baseline_mean', 'Baseline_stddev', 'BaselineVariation_mode', 'BaselineVariation_stddev'])
    adcs.save(options.output_directory + options.histo_filename)


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    # Load the histogram
    adcs = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    pixel_idx = 0
    # Define Geometry
    plt.subplots(2, 2,figsize=(14,12))
    plt.subplot(2, 2, 1)
    plt.hist(adcs.fit_result[...,0][~np.isnan(adcs.fit_result[...,1])], bins=100, label='<Baseline mean>')
    plt.xlabel('$mode(<BL>_{%d})$'%options.baseline_per_event_limit)
    plt.subplot(2, 2, 2)
    plt.hist(adcs.fit_result[...,1][~np.isnan(adcs.fit_result[...,1])], bins=100, label='<Baseline Variation>')
    plt.xlabel('$\sigma(<BL>_{%d})$'%options.baseline_per_event_limit)
    plt.subplot(2, 2, 3)
    plt.hist(adcs.fit_result[...,2][~np.isnan(adcs.fit_result[...,1])], bins=100, label='\sigma(Baseline mean)')
    plt.xlabel('$mode(\sigma(BL)_{%d})$'%options.baseline_per_event_limit)
    plt.subplot(2, 2, 4)
    plt.hist(adcs.fit_result[...,3][~np.isnan(adcs.fit_result[...,1])], bins=100, label='\sigma(Baseline Variation)')
    plt.xlabel('$\sigma(\sigma(BL)_{%d})$'%options.baseline_per_event_limit)
    plt.show()

    log = logging.getLogger(sys.modules['__main__'].__name__ + __name__)
    log.info('Pixels not responding: ')
    for p in np.where(np.isnan(adcs.fit_result[...,1]))[0]:
        log.info(options.pixel_list[p])

    means = adcs.data[0]
    rmses = adcs.data[1]
    plt.subplots(3,1,figsize=(20,7))
    plt.subplot(1,3,1)
    plt.hist((means[pixel_idx].astype(float) - adcs.fit_result[pixel_idx, 0]) / adcs.fit_result[pixel_idx, 1],bins=100)
    plt.xlabel('$ \\frac{<BL>_{%d}-mode(<BL>_{%d})}{\sigma(<BL>_{%d})}$'%
               (options.baseline_per_event_limit,options.baseline_per_event_limit,options.baseline_per_event_limit))
    plt.subplot(1,3,2)
    plt.hist( (rmses[pixel_idx].astype(float) - adcs.fit_result[pixel_idx, 2 ])/adcs.fit_result[pixel_idx, 3],bins=200)
    plt.xlabel('$ \\frac{\sigma(BL)_{%d}-mode(\sigma(BL)_{%d})}{\sigma(\sigma(BL)_{%d})}$' %
               (options.baseline_per_event_limit,options.baseline_per_event_limit,options.baseline_per_event_limit))
    plt.subplot(1,3,3)
    plt.hist2d((means[pixel_idx].astype(float) - adcs.fit_result[pixel_idx, 0])/adcs.fit_result[pixel_idx, 1] ,
               (rmses[pixel_idx].astype(float) - adcs.fit_result[pixel_idx, 2 ])/adcs.fit_result[pixel_idx, 3],bins=50)
    plt.xlabel('$ \\frac{<BL>_{%d}-mode(<BL>_{%d})}{\sigma(<BL>_{%d})}$' %
               (options.baseline_per_event_limit,options.baseline_per_event_limit,options.baseline_per_event_limit))
    plt.ylabel('$ \\frac{\sigma(BL)_{%d}-mode(\sigma(BL)_{%d})}{\sigma(\sigma(BL)_{%d})}$' %
               (options.baseline_per_event_limit,options.baseline_per_event_limit,options.baseline_per_event_limit))
    plt.show()
    input('type to leave')
    return


def create_report(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    return
