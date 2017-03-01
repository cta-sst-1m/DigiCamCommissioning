#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import dark_hist
from spectra_fit import fit_hv_off
from utils import display, histogram, geometry
from analysis import analyse_dark
import logging,sys
import scipy.stats
import numpy as np

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

    # Define the histograms

    hv_off_hist = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth, data_shape=(options.n_pixels,),
                               label='Dark ADC',xlabel='ADC',ylabel = 'entries')

     = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth, data_shape=(options.n_pixels,),
                               label='Dark ADC',xlabel='ADC',ylabel = 'entries')

    # Get the adcs
    dark_hist.run(adcs, options)



    # Save the histogram
    adcs.save(options.output_directory + options.histo_filename)

    # Delete the histograms
    del adcs

    return


def perform_analysis(options):
    """
    Perform a simple gaussian fit of the ADC histograms

    :param options: a dictionary containing at least the following keys:
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram      (str)

    :return:
    """
    log = logging.getLogger(sys.modules['__main__'].__name__+__name__)
    log.info('No analysis is implemented for ADC distribution in dark conditions')

    dark_hist = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    x = dark_hist.bin_centers


    if options.hist_type == 'raw':

        dark_hist.fit_result_label = ['baseline [ADC]', 'mean [ADC]', '$f_{dark}$ [MHz]', '$\mu_{XT}$']
        dark_hist.fit_result = np.ones((options.n_pixels, len(dark_hist.fit_result_label), 2))*np.nan

        for pixel in range(options.n_pixels):

            #if pixel not in options.pixel_list:

            #    continue

            if pixel in options.dead_pixel:

                dark_hist.fit_result[pixel, :, 0] = 0
                dark_hist.fit_result[pixel, :, 1] = np.nan

                continue

            y = dark_hist.data[pixel]
            baseline = x[np.argmax(y)]
            mean = np.average(x, weights=y)
            mean_error = np.sqrt(np.average((x-mean)**2, weights=y))/np.sqrt(np.sum(y))
            mu_xt = 0.06 + np.random.normal(0, 0.00001)
            mu_xt_error = 0.
            f_dark = (mean-baseline)/5.6/15.6/(1.+mu_xt) * 1E3
            f_dark_error = 0.

            dark_hist.fit_result[pixel, 0, 0] = baseline
            dark_hist.fit_result[pixel, 0, 1] = 0
            dark_hist.fit_result[pixel, 1, 0] = mean
            dark_hist.fit_result[pixel, 1, 1] = mean_error
            dark_hist.fit_result[pixel, 2, 0] = f_dark
            dark_hist.fit_result[pixel, 2, 1] = f_dark_error
            dark_hist.fit_result[pixel, 3, 0] = mu_xt
            dark_hist.fit_result[pixel, 3, 1] = mu_xt_error

    dark_hist.save(options.output_directory + options.histo_filename)


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    # Load the histogram
    adcs = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    # Define Geometry
    geom = geometry.generate_geometry_0()

    # Perform some plots
    display.display_hist(adcs, index=(options.pixel_list[0],))
    display.display_hist(adcs, geom=geom, index=(options.pixel_list[0],))
    display.display_hist(adcs, geom=geom, param_to_display=0, index=(options.pixel_list[0],), draw_fit=False)
    display.display_hist(adcs, geom=geom, param_to_display=1, index=(options.pixel_list[0],), draw_fit=False)
    display.display_hist(adcs, geom=geom, param_to_display=2, index=(options.pixel_list[0],), draw_fit=False)
    display.display_hist(adcs, geom=geom, param_to_display=3, index=(options.pixel_list[0],), draw_fit=False)

    input('press button to quit')

    return
