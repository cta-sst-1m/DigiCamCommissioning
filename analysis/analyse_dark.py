#!/usr/bin/env python3

# external modules

# internal modules
from spectra_fit import fit_dark_adc
from utils import display, histogram, geometry

from data_treatement import adc_hist
import logging,sys
import scipy.stats
import numpy as np

from scipy import stats

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
    adcs = None
    # Define the histograms
    adcs = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list),),
                               label='Dark LSB', xlabel='LSB', ylabel='entries')

    # Get the adcs
    adc_hist.run(adcs, options, 'ADC')

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
    if options.dark_analysis == 'fit_baseline':
        log = logging.getLogger(sys.modules['__main__'].__name__ + __name__)
        log.info('Perform a gaussian fit of the left side of the dark (==> baseline and sigmae for full mpe)')

        # Load the histogram
        adcs = histogram.Histogram(filename=options.output_directory + options.histo_filename)
        # Fit the baseline and sigma_e of all pixels
        adcs.fit(fit_dark_adc.fit_func, fit_dark_adc.p0_func, fit_dark_adc.slice_func, fit_dark_adc.bounds_func, \
                 labels_func=fit_dark_adc.labels_func)  # , limited_indices=tuple(options.pixel_list))
        # adcs.fit(fit_2_gaussian.fit_func, fit_2_gaussian.p0_func, fit_2_gaussian.slice_func, fit_2_gaussian.bounds_func, \
        #         labels_func=fit_2_gaussian.label_func)#, limited_indices=tuple(options.pixel_list))

        # Save the fit
        adcs.save(options.output_directory + options.histo_filename)

        # Delete the histograms
        del adcs

    elif options.dark_analysis == 'analytic':

        log = logging.getLogger(sys.modules['__main__'].__name__ + __name__)
        log.info('Perform an analytic extraction of mu_XT ( ==> baseline and sigmae for full mpe )')

        mpes_full = histogram.Histogram(filename=options.output_directory + options.full_histo_filename, fit_only=True)
        mpes_full_fit_result = np.copy(mpes_full.fit_result)
        del mpes_full


        # Load the histogram
        dark_hist = histogram.Histogram(filename=options.output_directory + options.histo_filename)
        dark_hist_for_baseline = histogram.Histogram(filename=options.output_directory + options.histo_filename)
        dark_hist_for_baseline.fit(fit_dark_adc.fit_func, fit_dark_adc.p0_func, fit_dark_adc.slice_func, fit_dark_adc.bounds_func, \
                 labels_func=fit_dark_adc.labels_func)

        dark_hist.fit_result = np.zeros((len(options.pixel_list), 3, 2))
        dark_hist.fit_result_label = np.array(['baseline [LSB]', '$f_{dark}$ [MHz]', '$\mu_{XT}$'])
        
        
        x = dark_hist.bin_centers

        #print(mpes_full_fit_result.shape)
        #print(mpes_full_fit_result[1 , 0, 0])
        #print(mpes_full_fit_result[1 , 1, 0])
        #print(mpes_full_fit_result[1 , 2, 0])
        #print(mpes_full_fit_result[1 , 3, 0])
        baseline = dark_hist.fit_result[..., 1, 0]
        baseline_error = dark_hist.fit_result[..., 1, 1]
        gain = mpes_full_fit_result[...,1,0]
        gain_error = mpes_full_fit_result[...,1,1]
        sigma_e = mpes_full_fit_result[...,2,0]
        sigma_e_error = mpes_full_fit_result[...,2,1]
        sigma_1 = mpes_full_fit_result[...,3,0]
        sigma_1_error = mpes_full_fit_result[...,3,1]

        print(baseline)

        integ = np.load(options.output_directory + options.pulse_shape_filename)
        integral = integ['integrals']
        integral_square = integ['integrals_square']

        for pixel in range(len(options.pixel_list)):

            y = dark_hist.data[pixel]

            if options.mc:
                baseline = 2010
                gain = 5.6
                sigma_1 = 0.48
                sigma_e = np.sqrt(0.86 ** 2.)

            dark_parameters = compute_dark_parameters(x, y, baseline[pixel], gain[pixel], sigma_1[pixel], sigma_e[pixel],integral[pixel],integral_square[pixel])

            #print(pixel)
            #print(baseline)
            #print(dark_hist.fit_result.shape)

            dark_hist.fit_result[pixel, 1, 0] = dark_parameters[0, 0]
            dark_hist.fit_result[pixel, 1, 1] = dark_parameters[0, 1]
            dark_hist.fit_result[pixel, 2, 0] = dark_parameters[1, 0]
            dark_hist.fit_result[pixel, 2, 1] = dark_parameters[1, 1]

        dark_hist.fit_result[:, 0, 0] = baseline
        dark_hist.fit_result[:, 0, 1] = baseline_error

        #dark_hist.save(options.output_directory + options.histo_filename.split('.npz')[0]+'_xt.npz')
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
    geom,pix_list = geometry.generate_geometry(options.cts,all_camera=True)

    #. Perform some plots

    print(adcs.data.shape)
    print(adcs.data[0])
    print(np.max(adcs.data[1]))

    display.display_hist(adcs, options=options, geom=geom, draw_fit=False)
    display.display_fit_result(adcs, geom=geom, options=options, display_fit=True)
    #display.display_fit_result(adcs, display_fit=True)
    input('press button to quit')

    return

def compute_dark_parameters(x, y, baseline, gain, sigma_1, sigma_e, integral,integral_square):
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
    pulse_shape_area = integral * gain
    pulse_shape_2_area = integral_square * gain**2
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
    print('baseline [LSB]: %0.4f'%baseline)
    print('sigma_e [LSB]: %0.4f'%sigma_e)
    print('sigma_1 [LSB]: %0.4f'%(sigma_1*gain))
    print('mean adc [LSB]: %0.4f' % mean_adc)
    print('sigma_2 adc [LSB]: %0.4f' % sigma_2_adc)
    print ('mu_borel : %0.4f [p.e.]'%mu_borel)
    print('f_dark %0.4f [MHz]' %(f_dark*1E3))
    print('dark XT : %0.4f [p.e.]' %mu_xt_dark)
    """

    return np.array([[f_dark*1E3, f_dark_error*1E3], [mu_xt_dark, mu_xt_dark_error]])