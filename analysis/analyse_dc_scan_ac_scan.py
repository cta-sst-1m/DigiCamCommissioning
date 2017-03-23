#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import mpe_hist
from spectra_fit import fit_hv_off
from utils import display, histogram, geometry
import logging,sys
import numpy as np
import logging
from tqdm import tqdm
from utils.logger import TqdmToLogger

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

    # Define the histograms
    mpes = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth, data_shape=(len(options.scan_level),len(options.pixel_list),),
                               label='MPE',xlabel='Peak ADC',ylabel = '$\mathrm{N_{entries}}$')

    # Get the reference sampling time
    peaks = histogram.Histogram(filename = options.output_directory + options.synch_histo_filename)

    # Construct the histogram
    baseline = histogram.Histogram(fit_only=True, filename=options.directory + options.baseline_filename)
    print(baseline.fit_result.shape)
    mpe_hist.run(mpes, options, peak_positions=peaks.data, baseline=baseline.fit_result[:,:,1,0])


    # Save the histogram
    mpes.save(options.output_directory + options.histo_filename)

    # Delete the histograms
    del mpes,peaks, baseline

    return


def perform_analysis(options):
    """
    Perform a simple gaussian fit of the ADC histograms

    :param options: a dictionary containing at least the following keys:
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram
                                                 whose fit contains the gain,sigmas etc...(str)

    :return:
    """
    # Fit the baseline and sigma_e of all pixels

    mpes = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    mpes.fit(fit_hv_off.fit_func, fit_hv_off.p0_func, fit_hv_off.slice_func, fit_hv_off.bounds_func, \
            labels_func=fit_hv_off.labels_func)

    mpes.save(options.output_directory + options.histo_filename)


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    # Load the histogram
    mpes = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    geom = geometry.generate_geometry_0(pixel_list=options.pixel_list)

    try:

        display.display_hist(mpes, geom=geom, options=options, display_parameter=True, draw_fit=True)

    except:

        display.display_hist(mpes, geom=geom, options=options)



    display.display_fit_result_level(mpes, options=options)

    mpes.fit_result[:,:, 1, 0] = mpes.fit_result[:,:, 1, 0]/mpes.fit_result[0,:, 1, 0]
    mpes.fit_result_label[1] = 'Gain/Gain$_{dark}$'

    display.display_fit_result_level(mpes, options=options)

    input('press button to quit')

    return
