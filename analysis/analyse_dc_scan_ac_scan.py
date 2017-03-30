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
                               label='MPE',xlabel='ADC',ylabel = '$\mathrm{N_{entries}}$')

    # Get the reference sampling time
    peaks = histogram.Histogram(filename = options.output_directory + options.synch_histo_filename)

    # Construct the histogram
    nsb = histogram.Histogram(fit_only=True, filename=options.directory + options.baseline_filename)
    mpe_hist.run(mpes, options, peak_positions=peaks.data, baseline=nsb.fit_result[:,:,0,0])


    # Save the histogram
    mpes.save(options.output_directory + options.histo_filename)

    # Delete the histograms
    del mpes,peaks

    return


def perform_analysis(options):

    mpes = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    mpes.fit_result_label = ['mean [ADC]', 'G/G$_{dark}$ []']
    mpes.fit_result = np.zeros((mpes.data.shape[:-1])+(len(mpes.fit_result_label),2,))

    #nsb = histogram.Histogram(filename=options.directory + options.nsb_filename)
    #pulse_shape = histogram.Histogram(filename=options.directory + options.pulse_shape_filename)
    #full_mpe = histogram.Histogram(filename=options.directory + options.full_mpe_filename)
    #gain = np.ones((mpes.data.shape[0], mpes.data.shape[1], 2)) * 23 #full_mpe.fit_result[..., 2, 0]
    #gain[...,1] = 0.


    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    pbar = tqdm(total=mpes.data.shape[0])
    tqdm_out = TqdmToLogger(log, level=logging.INFO)

    for level in range(mpes.data.shape[0]):

        log.debug('--|> Moving to DC level %d [DAC]' %options.scan_level[level])

        for pixel in range(mpes.data.shape[1]):

            n_entries = np.sum(mpes.data[level, pixel])

            if n_entries!=0:

                mpes.fit_result[level, pixel, 0, 0] = np.average(mpes.bin_centers, weights=mpes.data[level, pixel])
                std = np.sqrt(np.average((mpes.bin_centers-mpes.fit_result[level, pixel, 0, 0])**2, weights=mpes.data[level, pixel]))
                mpes.fit_result[level, pixel, 0, 1] = std / np.sqrt(n_entries)
                mpes.fit_result[level, pixel, 1, 0] = mpes.fit_result[level, pixel, 0, 0] / mpes.fit_result[0, pixel, 0, 0]
                mpes.fit_result[level, pixel, 1, 1] = np.abs(mpes.fit_result[level, pixel, 1, 0]) * np.sqrt((mpes.fit_result[level, pixel, 0, 1]/mpes.fit_result[level, pixel, 0, 0])**2 + (mpes.fit_result[0, pixel, 0, 1]/mpes.fit_result[0, pixel, 0, 0])**2)

            else:
                log.debug('level %d, pixel %d is empty' %(options.scan_level[level], options.pixel_list[pixel]))

        pbar.update(1)

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



    display.display_fit_result_level(mpes, options=options, scale='log')
    display.display_fit_result_level(mpes, options=options, scale='linear')


    return
