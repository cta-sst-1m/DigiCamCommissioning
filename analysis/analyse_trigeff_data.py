#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import trigger_eff_hist
from spectra_fit import fit_low_light,fit_high_light
from utils import display, histogram, geometry
import logging,sys
import numpy as np
import logging
import matplotlib.pyplot as plt
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


    triggers = histogram.Histogram(bin_center_min=0, bin_center_max=431,
                               bin_width=1, data_shape=(len(options.scan_level),),
                               label='Triggers',xlabel='Pixels',ylabel = '$\mathrm{N_{triggers}}$')


    # Construct the histogram
    trigger_eff_hist.run(triggers, options)

    # Save the histogram
    triggers.save(options.output_directory + options.histo_filename)

    # Delete the histograms
    del triggers

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
    return


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    # Load the histogram
    triggers = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    for i,level in enumerate(triggers.data):
        print(i,level)
    print(triggers.data.shape)
    test = np.argmax(np.sum(triggers.data,axis=0))
    trig = triggers.data[:,test]
    print(test)

    # Define Geometry
    p = trig /  options.events_per_level
    efficiencies = p
    q = 1. - p
    # in data[3], put the rate in Hz
    # in data[4], put the binomial error on rate in Hz
    errors = np.sqrt(options.events_per_level * p * q)/options.events_per_level

    plt.ion()
    plt.errorbar(options.scan_level,p*100.,yerr=errors*100.,fmt='ok')
    plt.fill_between(options.scan_level,p*100.-errors*100.,p*100.+errors*100.,alpha= 0.3, edgecolor='k', facecolor='k')
    plt.xscale('log')
    plt.xlabel('N(p.e.) measured in patch 192')
    plt.ylabel('Trigger Efficiency')
    plt.title()


    return
