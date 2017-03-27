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
    mpe_hist.run(mpes, options, peak_positions=peaks.data, baseline=baseline.fit_result[:,:,1,0])


    # Save the histogram
    mpes.save(options.output_directory + options.histo_filename)

    # Delete the histograms
    del mpes,peaks, baseline

    return


def perform_analysis(options):

    mpes = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    mpes.fit_result = np.zeros((mpes.data.shape[:-1])+(5,2,))
    mpes.fit_result_label = ['amplitude []', 'mean [ADC]', 'sigma [ADC]', 'G/G$_{dark}$ []', r'$\frac{f_{nsb}}{1-\mu_{XT}}$ [GHz]']

    dc_led_histogram = histogram.Histogram(filename=options.directory + options.baseline_filename)
    pulse_integrals = np.load(options.output_directory + options.pulse_shape_filename.split('.')[0] + '_integrals.npz')['pulse_integrals']
    #full_mpe = histogram.Histogram(filename=options.directory + options.full_mpe_filename)
    gain = np.ones((mpes.data.shape[0], mpes.data.shape[1], 2)) * 23 #full_mpe.fit_result[..., 2, 0]
    gain[...,1] = 0.

    level_for_pulse = 20

    for i in range(mpes.data.shape[0]):
        for j in range(mpes.data.shape[1]):

            if np.sum(mpes.data[i,j])!=0:

                mpes.fit_result[i, j, 0, 0] = np.sum(mpes.data[i, j])
                mpes.fit_result[i, j, 0, 1] = np.sqrt(mpes.fit_result[i, j, 0, 0])
                mpes.fit_result[i, j, 1, 0] = np.average(mpes.bin_centers, weights=mpes.data[i, j])
                mpes.fit_result[i, j, 2, 0] = np.sqrt(np.average((mpes.bin_centers-mpes.fit_result[i, j, 0, 0])**2, weights=mpes.data[i, j]))
                mpes.fit_result[i, j, 1, 1] = mpes.fit_result[i, j, 2, 0]/ np.sqrt(mpes.fit_result[i, j, 0, 0])
                mpes.fit_result[i, j, 3, 0] = mpes.fit_result[i, j, 1, 0]/mpes.fit_result[0, j, 1, 0]
                mpes.fit_result[i, j, 3, 1] = mpes.fit_result[i, j, 3, 0] * (mpes.fit_result[i, j, 1, 1]/mpes.fit_result[i, j, 1, 0] + mpes.fit_result[0, j, 1, 1]/mpes.fit_result[0, j, 1, 0])
                mpes.fit_result[i, j, 4, 0] = (dc_led_histogram.fit_result[i, j, 1, 0] - dc_led_histogram.fit_result[0, j, 1, 0])
                temp_1  = mpes.fit_result[i, j, 4, 0]
                temp_1_error  = np.sqrt(dc_led_histogram.fit_result[i, j, 1, 1]**2 + dc_led_histogram.fit_result[0, j, 1, 1]**2)
                temp_2 = (gain[i ,j, 0] * mpes.fit_result[i, j, 3, 0] * pulse_integrals[level_for_pulse, j, 0])
                temp_2_error =  (gain[i, j, 1]/gain[i, j, 0] + mpes.fit_result[i, j, 3, 1]/mpes.fit_result[i, j, 3, 0] + pulse_integrals[level_for_pulse, j, 1]/pulse_integrals[level_for_pulse, j, 0])
                mpes.fit_result[i, j, 4, 0] /= temp_2
                mpes.fit_result[i, j, 4, 1] = mpes.fit_result[i, j, 4, 0] * (temp_1_error/temp_1 + temp_2_error/temp_2)


                #mpes.fit_result[i, j, 4, 1] = mpes.fit_result[i,j, 4, 0] * ((dc_led_histogram.fit_result[i, j, 1, 1]/dc_led_histogram.fit_result[i, j, 1, 0])**2 + (pulse_integrals[i, j, 1]/pulse_integrals[i, j, 0])**2)
            else:
                print('level %d, pixel %d not ok' %(options.scan_level[i], options.pixel_list[j]))

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


    input('press button to quit')

    return
