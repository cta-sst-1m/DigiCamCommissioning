#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import adc_hist
from utils import display, histogram, geometry
from spectra_fit import fit_multiple_gaussians_full_mpe
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

    # Fix options
    if not hasattr(options,'dc_level') and not hasattr(options,'ac_level'):
        raise Exception('No level specified')
    elif not hasattr(options,'ac_level'):
        setattr(options,'ac_level',[0])
    elif not hasattr(options,'dc_level'):
        setattr(options,'dc_level',[0])
    else:
        pass

    # Define the histograms
    mpes = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth, data_shape=(len(options.dc_level),
                                                                            len(options.ac_level),
                                                                            len(options.pixel_list),),
                               label='MPE',xlabel='LSB',ylabel = '$\mathrm{N_{entries}}$')

    # Get the baseline if needed
    if hasattr(options,'histo_dark_adc'):
        baseline_fit = histogram.Histogram(filename=options.output_directory + options.histo_dark_adc, fit_only=True).fit_result
        if baseline_fit.shape[0]==1296:
            baseline_fit = np.take(baseline_fit,options.pixel_list)
    # Get the reference sampling time
    peaks = None
    if hasattr(options,'synch_histo_filename'):
        peaks = histogram.Histogram(filename = options.output_directory + options.synch_histo_filename)

        if peaks.shape[0]==1296:
            peaks = np.take(peaks,options.pixel_list)


    # Construct the histogram
    adc_hist.run(mpes, options, 'CHARGE_PER_LEVEL',peak_position=peaks.data if hasattr(options,'synch_histo_filename') else None,
                 baseline= baseline_fit.fit_result[...,1,0] if hasattr(options,'histo_dark_adc') else None,
                 prev_fit_result = baseline_fit.fit_result if hasattr(options,'histo_dark_adc') else None)

    # Save the histogram
    mpes.save(options.output_directory + options.histo_filename)

    # Delete the histograms
    del mpes,baseline_fit,peaks

    return


def perform_analysis(options):

    # Load mpes
    mpes = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    # Create the full mpes
    full_mpes = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth, data_shape=(len(options.dc_level), len(options.pixel_list), ),
                               label='Full MPE',xlabel='LSB',ylabel = '$\mathrm{N_{entries}}$')
    full_mpes.data = np.sum(mpes.data, axis=1)
    full_mpes.errors = np.sqrt(full_mpes.data)

    if True:
        options.scan_level = options.dc_level
        display.display_hist(full_mpes, options=options)
        import matplotlib.pyplot as plt
        plt.show()

    # Fit the full mpes

    full_mpes.fit(fit_multiple_gaussians_full_mpe.fit_func, fit_multiple_gaussians_full_mpe.p0_func, fit_multiple_gaussians_full_mpe.slice_func, fit_multiple_gaussians_full_mpe.bounds_func, \
         labels_func=fit_multiple_gaussians_full_mpe.labels_func, config=None)

    full_mpes.save(options.output_directory + options.full_histo_filename)

    del mpes, full_mpes

    return





def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    # Load the histograms
    #mpes = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    full_mpes = histogram.Histogram(filename=options.output_directory + options.full_histo_filename)

    print(full_mpes.fit_result.shape)
    """
    geom,pixlist = geometry.generate_geometry(options.cts)
    mpes.data = mpes.data[0]
    mpes.errors = mpes.errors[0]
    options.scan_level = options.ac_level
    display.display_hist(mpes,  options=options)
    """

    #full_mpes.data = full_mpes.data[0]
    #full_mpes.errors = full_mpes.errors[0]

    options.scan_level = options.dc_level
    display.display_hist(full_mpes, options=options, draw_fit=True)

    axes = fig.get_axes()
    axis_histo = axes[0]


    input('press a key')



    '''
    try:

        display.display_hist(mpes, geom=geom, options=options, display_parameter=True, draw_fit=True)

    except:

        display.display_hist(mpes, geom=geom, options=options)



    display.display_fit_result_level(mpes, options=options, scale='log')
    display.display_fit_result_level(mpes, options=options, scale='linear')
    '''

    return
