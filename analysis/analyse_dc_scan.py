#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import mpe_hist, adc_hist
from analysis import analyse_hvoff
from utils import display, histogram, geometry
from spectra_fit import fit_hv_off
import logging,sys
import numpy as np
import logging
from tqdm import tqdm
from utils.logger import TqdmToLogger
import matplotlib.pyplot as plt
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

    #baseline = histogram.Histogram(filename=options.directory + options.baseline_filename)



    nsb = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth,
                               data_shape=(len(options.scan_level), len(options.pixel_list), ),
                               label='DC LED', xlabel='ADC', ylabel='$\mathrm{N_{entries}}$')

    mpe_hist.run(nsb, options)
    nsb.save(options.output_directory + options.histo_filename)

    del nsb

    return


def perform_analysis(options):

    nsb = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    nsb.fit_result_label = ['mean [ADC]']
    nsb.fit_result = np.zeros((nsb.data.shape[:-1])+(len(nsb.fit_result_label),2,))

    log = logging.getLogger(sys.modules['__main__'].__name__ + '.' + __name__)
    pbar = tqdm(total=nsb.data.shape[0])
    tqdm_out = TqdmToLogger(log, level=logging.INFO)

    for level in range(nsb.data.shape[0]):

        log.debug('--|> Moving to  level %d [DAC]' %options.scan_level[level])

        for pixel in range(nsb.data.shape[1]):

            n_entries = np.sum(nsb.data[level, pixel])
            nsb.fit_result[level, pixel, 0, 0] = np.average(nsb.bin_centers, weights=nsb.data[level, pixel])
            std = np.sqrt(np.average((nsb.bin_centers-nsb.fit_result[level, pixel, 0, 0])**2, weights=nsb.data[level, pixel]))
            nsb.fit_result[level, pixel, 0, 1] = std/ np.sqrt(n_entries)

        pbar.update(1)

    nsb.save(options.output_directory + options.histo_filename)

    return


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    nsb = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    geom = geometry.generate_geometry_0(pixel_list=options.pixel_list)

    try:

        display.display_hist(nsb, geom=geom, options=options, display_parameter=True)

    except:

        display.display_hist(nsb, geom=geom, options=options)


    display.display_fit_result_level(nsb, options=options)



    return
