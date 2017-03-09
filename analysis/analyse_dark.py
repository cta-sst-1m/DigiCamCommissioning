#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import adc_hist
from spectra_fit import fit_hv_off
from utils import display, histogram, geometry
from analysis import analyse_dark
import logging,sys
import scipy.stats
import numpy as np

__all__ = ["create_histo", "perform_analysis", "display_results", "compute_dark_parameters"]


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
    adcs = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list  ),),
                               label='Dark ADC',xlabel='ADC',ylabel = 'entries')

    # Get the adcs
    adc_hist.run(adcs, options,'ADC')



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
    log = logging.getLogger(sys.modules['__main__'].__name__+__name__)
    log.info('No analysis is implemented for ADC distribution in dark conditions')

    dark_hist = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    x = dark_hist.bin_centers

    dark_hist.fit_result_label = ['baseline [ADC]', '$f_{dark}$ [MHz]', '$\mu_{XT}$']
    dark_hist.fit_result = np.ones((len(options.pixel_list), len(dark_hist.fit_result_label), 2))*np.nan

    for pixel in range(len(options.pixel_list)):

        y = dark_hist.data[pixel]

        if options.mc:

            baseline = 2010
            gain = 5.6
            sigma_1 = 0.48
            sigma_e = np.sqrt(0.86**2.)

        dark_parameters =  compute_dark_parameters(x, y, baseline, gain, sigma_1, sigma_e)

        dark_hist.fit_result[pixel, 0, 0] = baseline
        dark_hist.fit_result[pixel, 0, 1] = 0
        dark_hist.fit_result[pixel, 1, 0] = dark_parameters[0,0]
        dark_hist.fit_result[pixel, 1, 1] = dark_parameters[0,1]
        dark_hist.fit_result[pixel, 2, 0] = dark_parameters[1,0]
        dark_hist.fit_result[pixel, 2, 1] = dark_parameters[1,1]

    dark_hist.save(options.output_directory + options.histo_filename)
    del dark_hist


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    # Load the histogram
    adcs = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    # Define Geometry
    geom = geometry.generate_geometry_0(pixel_list=options.pixel_list)

    #. Perform some plots
    display.display_hist(adcs, options=options, geom=geom)
    #display.display_fit_result(adcs, geom=geom, display_fit=True)
    display.display_fit_result(adcs, display_fit=True)
    input('press button to quit')

    return

def compute_dark_parameters(x, y, baseline, gain, sigma_1, sigma_e):
    '''
    In developement
    :param x:
    :param y:
    :param baseline:
    :param gain:
    :param sigma_1:
    :param sigma_e:
    :return:
    '''


    x = x - baseline
    sigma_1 = sigma_1/gain

    mean_adc = np.average(x, weights=y)
    sigma_2_adc = np.average((x - mean_adc) ** 2, weights=y) - 1./12.
    pulse_shape_area = 15.11851125 * gain
    pulse_shape_2_area = 9.25845231 * gain**2
    alpha = (mean_adc * pulse_shape_2_area)/((sigma_2_adc - sigma_e**2)*pulse_shape_area)

    if (1./alpha - sigma_1**2)<0 or np.isnan(1./alpha - sigma_1**2):
        mu_borel = np.nan
        mu_xt_dark = np.nan
        f_dark = np.nan

    elif np.sqrt(1./alpha - sigma_1**2)<1:

        mu_xt_dark = 0.
        f_dark = mean_adc / pulse_shape_area



    else:

        mu_borel = np.sqrt(1./alpha - sigma_1**2)
#        mu_borel = 1./(1.-0.06)
        mu_xt_dark = 1. - 1./mu_borel
        f_dark = mean_adc / mu_borel / pulse_shape_area

    f_dark_error = np.nan
    mu_xt_dark_error = np.nan

    """
    print('gain [ADC/p.e.]: %0.4f'%gain)
    print('baseline [ADC]: %0.4f'%baseline)
    print('sigma_e [ADC]: %0.4f'%sigma_e)
    print('sigma_1 [ADC]: %0.4f'%(sigma_1*gain))
    print('mean adc [ADC]: %0.4f' % mean_adc)
    print('sigma_2 adc [ADC]: %0.4f' % sigma_2_adc)
    print ('mu_borel : %0.4f [p.e.]'%mu_borel)
    print('f_dark %0.4f [MHz]' %(f_dark*1E3))
    print('dark XT : %0.4f [p.e.]' %mu_xt_dark)
    """

    return np.array([[f_dark*1E3, f_dark_error*1E3], [mu_xt_dark, mu_xt_dark_error]])