#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import adc_hist
from spectra_fit import fit_hv_off
from utils import display, histogram, geometry

__all__ = ["create_histo", "perform_analysis", "display_results"]


def create_histo(options):
    """
    Create a list of ADC histograms and fill it with data

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
    adcs = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth, data_shape=(options.n_pixels,))

    # Get the adcs
    adc_hist.run(adcs, options, 'ADC')

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
        - 'fit_filename'     : the name of the file containing the fit results    (str)

    :return:
    """

    # Load the histogram
    adcs = histogram.Histogram(options.output_directory + options.histo_filename)

    # Fit the baseline and sigma_e of all pixels
    adcs.fit(fit_hv_off.fit_func, fit_hv_off.p0_func, fit_hv_off.slice_func, fit_hv_off.bounds_func,
             labels_func=fit_hv_off.labels_func)

    # Save the fit
    adcs.save(options.output_directory + options.fit_filename)

    # Delete the histograms
    del adcs


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    # Load the histogram
    adcs = histogram.Histogram(options.output_directory + options.histo_filename)

    # Define Geometry
    geom = geometry.generate_geometry_0()

    # Perform some plots
    display.display_fit_result(adcs, geom, index_var=2, limits=[0., 2.], bin_width=0.05)
    display.display_fit_result(adcs, geom, index_var=1, limits=[1950., 2050.], bin_width=10.)

    # display([adcs], geom, fit_hv_off.slice_func, norm='linear')
    input('press button to quit')

    return
