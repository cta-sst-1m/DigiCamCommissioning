#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import adc_hist
from spectra_fit import fit_hv_off
from utils import display, histogram, geometry
from analysis import analyse_hvoff
import logging

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

    # Call it from analyse_hvoff
    analyse_hvoff.create_histo(options)

    return


def perform_analysis(options):
    """
    Perform a simple gaussian fit of the ADC histograms

    :param options: a dictionary containing at least the following keys:
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram      (str)

    :return:
    """

    # Load the histogram
    adcs = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    # Fit the baseline and sigma_e of all pixels
    log = logging.getLogger(sys.modules['__main__'].__name__+__name__)

    log.error('No analysis is implemented for ADC distribution in dark conditions')

    # Delete the histograms
    del adcs


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
    display.display_hist(adcs, geom, index_default=(700,), param_to_display=-1, limits=[1900., 2100.])

    input('press button to quit')

    return
