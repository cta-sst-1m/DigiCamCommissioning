#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import dark_hist, synch_hist, mpe_hist
from spectra_fit import fit_hv_off, fit_low_light, fit_high_light
from utils import display, histogram, geometry
from analysis import analyse_dark
import logging,sys
import matplotlib.pyplot as plt
import scipy.stats
from analysis.analyse_dark import compute_dark_parameters
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

    log = logging.getLogger(sys.modules['__main__'].__name__+__name__)


    log.info('Creating the histograms for : HV off, Dark, Low light, Medium light, High light')
    hv_off_histogram = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list),),
                               label='HV Off',xlabel='ADC',ylabel = 'entries')

    dark_histogram = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list),),
                               label='Dark',xlabel='ADC',ylabel = 'entries')

    low_light_histogram = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth, data_shape=(len(options.scan_level), len(options.pixel_list),),
                               label='Low light',xlabel='ADC',ylabel = 'entries')

    medium_light_histogram = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth, data_shape=(len(options.scan_level), len(options.pixel_list),),
                               label='Medium light',xlabel='ADC',ylabel = 'entries')

    high_light_histogram = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth, data_shape=(len(options.scan_level), len(options.pixel_list),),
                               label='High light',xlabel='ADC',ylabel = 'entries')

    synch_histogram = histogram.Histogram(bin_center_min=0, bin_center_max=options.sample_max,
                               bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list),),
                               label='Synch',xlabel='time sample',ylabel = 'entries')

    # Fill HV_Off and Dark histo
    log.info('Filling HV off')
    options.file_basename = 'hv_low.%s.fits.fz'
    dark_hist.run(hist=hv_off_histogram, options=options, hist_type='raw')
    hv_off_histogram.save(options.output_directory + 'hv_low.npz')
    del hv_off_histogram
    log.info('Filling Dark')
    options.file_basename = 'dark.%s.fits.fz'
    dark_hist.run(hist=dark_histogram, options=options, hist_type='raw')
    dark_histogram.save(options.output_directory + 'dark.npz')
    del dark_histogram

    # Fill the synch hist
    log.info('Filling Synch')
    options.file_basename = 'low_light.%s.fits.fz'
    synch_hist.run(hist=synch_histogram, options=options, min_evt=0, max_evt=options.max_event)
    synch_histogram.save(options.output_directory + 'synch.npz')

    # Fill the light hist
    log.info('Filling Low light')
    options.file_basename = 'low_light.%s.fits.fz'
    mpe_hist.run(hist=low_light_histogram, options=options, peak_positions=synch_histogram.data)
    low_light_histogram.save(options.output_directory + 'low_light.npz')
    del low_light_histogram

    log.info('Filling Medium light')
    options.file_basename = 'medium_light.%s.fits.fz'
    mpe_hist.run(hist=medium_light_histogram, options=options, peak_positions=synch_histogram.data)
    medium_light_histogram.save(options.output_directory + 'medium_light.npz')
    del medium_light_histogram

    log.info('Filling High light')
    options.file_basename = 'high_light.%s.fits.fz'
    mpe_hist.run(hist=high_light_histogram, options=options, peak_positions=synch_histogram.data)
    high_light_histogram.save(options.output_directory + 'high_light.npz')
    del high_light_histogram
    del synch_histogram


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

    # Analyse hv_off

    hv_off_histogram = histogram.Histogram(filename = options.output_directory + 'hv_low.npz')
    hv_off_histogram.fit(fit_hv_off.fit_func, fit_hv_off.p0_func, fit_hv_off.slice_func, fit_hv_off.bounds_func,
             labels_func=fit_hv_off.labels_func)
    hv_off_histogram.save(options.output_directory + 'hv_low.npz')

    # Analyse low_light

    low_light_histogram = histogram.Histogram(filename = options.output_directory + 'low_light.npz')
    low_light_histogram.fit(fit_low_light.fit_func, fit_low_light.p0_func, fit_low_light.slice_func,
             fit_low_light.bounds_func, config=None, labels_func=fit_low_light.label_func)
    low_light_histogram.save(options.output_directory + 'low_light.npz')

    # Analyse dark

    dark_histogram = histogram.Histogram(filename = options.output_directory + 'dark.npz')
    x = dark_histogram.bin_centers
    dark_histogram.fit_result_label = ['$f_{dark}$ [MHz]', '$\mu_{XT} [p.e]$']
    dark_histogram.fit_result = np.ones((len(options.pixel_list), len(dark_histogram.fit_result_label), 2)) * np.nan

    for i in range(len(options.pixel_list)):

        y = dark_histogram.data[i]
        dark_parameter = compute_dark_parameters(x, y, baseline=hv_off_histogram.fit_result[i,1,0], gain=low_light_histogram.fit_result[0,i,2,0], sigma_e=hv_off_histogram.fit_result[i,2,0], sigma_1=low_light_histogram.fit_result[0,i,5,0])
        dark_histogram.fit_result[i, 0, 0] = dark_parameter[0, 0]
        dark_histogram.fit_result[i, 1, 0] = dark_parameter[1, 0]
        dark_histogram.fit_result[i, 0, 1] = dark_parameter[0, 1]
        dark_histogram.fit_result[i, 1, 1] = dark_parameter[1, 1]

    dark_histogram.save(options.output_directory + 'dark.npz')

    del hv_off_histogram
    del low_light_histogram
    del dark_histogram

def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    # Load the histograms
    hv_off_histogram = histogram.Histogram(filename = options.output_directory + 'hv_low.npz')
    dark_histogram = histogram.Histogram(filename = options.output_directory + 'dark.npz')
    synch_histogram = histogram.Histogram(filename = options.output_directory + 'synch.npz')
    low_light_histogram = histogram.Histogram(filename = options.output_directory + 'low_light.npz')
    medium_light_histogram = histogram.Histogram(filename = options.output_directory + 'medium_light.npz')
    high_light_histogram = histogram.Histogram(filename = options.output_directory + 'high_light.npz')

    # Define Geometry
    geom = geometry.generate_geometry_0(pixel_list=options.pixel_list)

    # Perform some plots

    display.display_hist(hv_off_histogram, options, geom=geom, display_parameter=True, draw_fit=True)
    display.display_hist(dark_histogram, options, geom=geom, display_parameter=True)
    display.display_hist(synch_histogram, options, geom=geom)
    display.display_hist(low_light_histogram, options, geom=geom, display_parameter=True, draw_fit=True)
    display.display_hist(medium_light_histogram, options, geom=geom)
    display.display_hist(high_light_histogram, options, geom=geom)
    display.display_fit_result(low_light_histogram, options)

    print('press + to go to next pixel')
    print('press - to go to previous pixel')
    print('press * to go to next level')
    print('press / to go to previous level')
    print('press . to go to next param')
    input('press button enter to quit \n')

    return

def save(options):

    import pandas as pd

    hv_off_histogram = histogram.Histogram(filename = options.output_directory + 'hv_low.npz')
    dark_histogram = histogram.Histogram(filename = options.output_directory + 'dark.npz')
    #synch_histogram = histogram.Histogram(filename = options.output_directory + 'synch.npz')
    low_light_histogram = histogram.Histogram(filename = options.output_directory + 'low_light.npz')
    module_list = np.repeat(options.module_list, 12)

    data = {'pix_id': options.pixel_list, \
            hv_off_histogram.fit_result_label[0] + ' (hv low)': hv_off_histogram.fit_result[:,0,0], \
            hv_off_histogram.fit_result_label[1] + ' (hv low)': hv_off_histogram.fit_result[:,1,0], \
            hv_off_histogram.fit_result_label[2] + ' (hv low)': hv_off_histogram.fit_result[:,2,0], \
            dark_histogram.fit_result_label[0] + ' (dark)': dark_histogram.fit_result[:,0,0], \
            dark_histogram.fit_result_label[1] + ' (dark)': dark_histogram.fit_result[:, 1, 0], \
            low_light_histogram.fit_result_label[0] + ' (low light)': low_light_histogram.fit_result[0,:, 0, 0], \
            low_light_histogram.fit_result_label[1] + ' (low light)': low_light_histogram.fit_result[0,:, 1, 0], \
            low_light_histogram.fit_result_label[2] + ' (low light)': low_light_histogram.fit_result[0,:, 2, 0], \
            low_light_histogram.fit_result_label[3] + ' (low light)': low_light_histogram.fit_result[0,:, 3, 0], \
            low_light_histogram.fit_result_label[4] + ' (low light)': low_light_histogram.fit_result[0,:, 4, 0], \
            low_light_histogram.fit_result_label[5] + ' (low light)': low_light_histogram.fit_result[0,:, 5, 0], \
            low_light_histogram.fit_result_label[6] + ' (low light)': low_light_histogram.fit_result[0,:, 6, 0], \
            low_light_histogram.fit_result_label[7] + ' (low light)': low_light_histogram.fit_result[0,:, 7, 0], \
            '$\chi^2$ /ndf (hv low)': hv_off_histogram.fit_chi2_ndof[:,0]/hv_off_histogram.fit_chi2_ndof[:,1],\
            #'$\chi^2$ /ndf (dark)': dark_histogram.fit_chi2_ndof[:,0]/dark_histogram.fit_chi2_ndof[:,1],\
            '$\chi^2$ /ndf (low light)': low_light_histogram.fit_chi2_ndof[0,:,0]/low_light_histogram.fit_chi2_ndof[0,:,1], \
            'module': module_list \
            }

    df = pd.DataFrame(data)
    df.to_csv(options.output_directory + 'module_analysis_results.csv')

    return