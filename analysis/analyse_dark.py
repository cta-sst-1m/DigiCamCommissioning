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
    if options.dark_analysis == 'mean_rms':
        adcs = [np.zeros((len(options.pixel_list), options.evt_max)),
                np.zeros((len(options.pixel_list), options.evt_max))]
        # Get the adcs
        adc_hist.run(adcs, options,'MEANRMS')
        np.savez_compressed(options.output_directory + options.histo_filename, mean = adcs[0], rms = adcs[1])
    else:

        adcs = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                                   bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list),),
                                   label='Dark ADC', xlabel='ADC', ylabel='entries')

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
    elif  options.dark_analysis == 'mean_rms':
        log = logging.getLogger(sys.modules['__main__'].__name__ + __name__)
        log.info('Get the distribution of mean and RMS in the dark')

        # Load the histogram
        adcs = np.load(options.output_directory + options.histo_filename)
        params = np.zeros((len(options.pixel_list),4),dtype=float)
        #print(stats.mode(adcs['mean'], axis=-1 ))
        tmp = stats.mode(adcs['mean'], axis=-1 )[0]
        print(tmp.shape)
        params[...,0]  = tmp[0]
        params[...,1]  = np.std( adcs['mean'], axis=-1 )
        params[...,2]  = stats.mode(adcs['rms'], axis=-1 )[0][0]
        params[...,3]  = np.std( adcs['rms'], axis=-1 )
        np.savez_compressed(options.output_directory + options.histo_filename, mean = adcs['mean'], rms = adcs['rms'], params = params)




    elif  options.dark_analysis == 'analytic':

        log = logging.getLogger(sys.modules['__main__'].__name__ + __name__)
        log.info('Perform an analytic extraction of mu_XT ( ==> baseline and sigmae for full mpe )')

        mpes_full = histogram.Histogram(filename=options.output_directory + options.full_histo_filename, fit_only=True)
        mpes_full_fit_result = np.copy(mpes_full.fit_result)
        del mpes_full


        # Load the histogram
        dark_hist = histogram.Histogram(filename=options.output_directory + options.histo_filename)
        x = dark_hist.bin_centers

        baseline = mpes_full_fit_result[...,0,0]
        gain = mpes_full_fit_result[...,1,0]
        sigma_e = mpes_full_fit_result[...,2,0]
        sigma_1 = mpes_full_fit_result[...,3,0]

        integ = np.load(options.output_directory + options.pulse_shape_filename)
        integral = integ['pulse_integrals']
        integral_square = integ['pulse_integrals_square']

        for pixel in range(len(options.pixel_list)):

            y = dark_hist.data[pixel]

            if options.mc:
                baseline = 2010
                gain = 5.6
                sigma_1 = 0.48
                sigma_e = np.sqrt(0.86 ** 2.)

            dark_parameters = compute_dark_parameters(x, y, baseline[pixel], gain[pixel], sigma_1[pixel], sigma_e[pixel],integral[pixel],integral_square[pixel])

            dark_hist.fit_result[pixel, 0, 0] = baseline[pixel]
            dark_hist.fit_result[pixel, 0, 1] = 0
            dark_hist.fit_result[pixel, 1, 0] = dark_parameters[0, 0]
            dark_hist.fit_result[pixel, 1, 1] = dark_parameters[0, 1]
            dark_hist.fit_result[pixel, 2, 0] = dark_parameters[1, 0]
            dark_hist.fit_result[pixel, 2, 1] = dark_parameters[1, 1]

        dark_hist.save(options.output_directory + options.histo_filename.split('.npz')[0]+'_xt.npz')
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
    display.display_hist(adcs, options=options, geom=geom,draw_fit=False)
    #display.display_fit_result(adcs, geom=geom, display_fit=True)
    display.display_fit_result(adcs, display_fit=True)
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