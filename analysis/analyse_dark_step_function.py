#!/usr/bin/env python3

# external modules

# internal modules
from utils import display, histogram, geometry
import matplotlib.pyplot as plt
from data_treatement import adc_hist
import logging,sys
import numpy as np
import peakutils
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep
from spectra_fit import fit_dark_adc


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
    dark = None

    if options.analysis_type == 'step_function':
        dark = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                                bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list),),
                                label='Dark step function', xlabel='threshold LSB', ylabel='entries')

        # Get the adcs
        adc_hist.run(dark, options, h_type='STEPFUNCTION')

    elif options.analysis_type == 'single_photo_electron':
        dark = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                                   bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list),),
                                   label='Pixel SPE', xlabel='Pixel ADC', ylabel='Count / ADC')
        # Get the adcs
        adc_hist.run(dark, options, 'SPE')

    elif options.analysis_type == 'adc_template':
        dark = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                                   bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list),),
                                   label='Dark LSB', xlabel='LSB', ylabel='entries')
        # Get the adcs
        adc_hist.run(dark, options, 'ADC')


    # Save the histogram
    dark.save(options.output_directory + options.histo_filename)

    del dark

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
    if options.analysis_type == 'step_function':
        step_function(options)
    elif options.analysis_type == 'adc_template':
        adc_template(options)
    elif options.analysis_type == 'single_photo_electron':
        single_photo_electron(options)


def display_results(options):
    """
    Display the analysis results
    :param options:
    :return:
    """

    # Load the data
    dark_step_function = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    dark_count_rate = dark_step_function.fit_result[:, 0, 0]
    cross_talk = dark_step_function.fit_result[:, 1, 0]

    mask = ~np.isnan(dark_count_rate)
    dark_count_rate = dark_count_rate[mask]
    cross_talk = cross_talk[mask]
    # Display step function
    display.display_hist(dark_step_function, options=options)

    # Display histograms of dark count rate and cross talk
    plt.figure()
    plt.hist(dark_count_rate * 1E3, bins='auto')
    plt.xlabel('$f_{dark}$ [MHz]')

    plt.figure()
    plt.hist(cross_talk, bins='auto')
    plt.xlabel('XT')

    plt.show()

    return


def step_function(options):

    dark_step_function = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    log = logging.getLogger(sys.modules['__main__'].__name__ + __name__)

    y = dark_step_function.data
    x = np.tile(dark_step_function.bin_centers, (y.shape[0], 1))

    y = - np.diff(np.log(y)) / np.diff(x)
    x = x[..., :-1] + np.diff(x) / 2.

    y[np.isnan(y)] = 0
    y[np.isinf(y)] = 0
    y[y < 0] = 0

    threshold = 0.15
    width = 3

    dark_count = np.zeros(y.shape[0])
    cross_talk = np.zeros(y.shape[0])

    n_samples = options.n_samples - options.baseline_per_event_limit - options.window_width + 1
    time = (options.max_event - options.min_event) * 4 * n_samples

    for pixel in range(y.shape[0]):
        max_indices = peakutils.indexes(y[pixel], thres=threshold, min_dist=options.min_distance)

        try:
            max_x = peakutils.interpolate(x[pixel], y[pixel], ind=max_indices, width=width)

        except RuntimeError:

            log.warning('Could not interpolate for pixel %d taking max as indices' % options.pixel_list[pixel])
            max_x = x[pixel][max_indices]

        gain = np.mean(np.diff(max_x))
        min_x = max_x + 0.5 * gain
        f = interp1d(dark_step_function.bin_centers, dark_step_function.data[pixel], kind='cubic')

        try:

            counts = [f(min_x[0]), f(min_x[1])]

        except IndexError:

            log.warning('Could not find 0.5 p.e. and 1.5 p.e. for pixel %d' % options.pixel_list[pixel])
            counts = [np.nan, np.nan]

        # spline step function method

        spline_step_function = splrep(dark_step_function.bin_centers, dark_step_function.data[pixel], k=3, w=1./np.sqrt(dark_step_function.data[pixel]), s=10)

        # x_around_minima = np.linspace(0, max_x[1] + 0.5 * gain, 100)
        # spline_step_function_second_derivative = splev(x_around_minima, tck=spline_step_function, der=2)


        dark_count[pixel] = counts[0] / time
        cross_talk[pixel] = counts[1] / counts[0]

        if options.verbose:
            x_spline = np.linspace(dark_step_function.bin_centers[0], dark_step_function.bin_centers[-1],
                                   num=len(dark_step_function.bin_centers) * 20)
            plt.figure()
            plt.semilogy(x_spline, splev(x_spline, spline_step_function), label='spline')
            plt.semilogy(dark_step_function.bin_centers, dark_step_function.data[pixel], label='data', linestyle='None', marker='o')
            plt.axvline(min_x[0])
            plt.axvline(min_x[1])
            plt.legend()

            plt.figure()
            plt.plot(x_spline, ((splev(x_spline, spline_step_function, der=2))), label='spline second der')
            plt.legend()
            plt.show()

    dark_step_function.fit_result = np.zeros((dark_step_function.data.shape[0], 4, 2))
    dark_step_function.fit_result[:, 0, 0] = dark_count
    dark_step_function.fit_result[:, 1, 0] = cross_talk

    dark_step_function.save(options.output_directory + options.histo_filename)
    return


def adc_template(options):
    log = logging.getLogger(sys.modules['__main__'].__name__ + __name__)
    log.info('Perform an analytic extraction of mu_XT ( ==> baseline and sigmae for full mpe )')

    mpes_full = histogram.Histogram(filename=options.output_directory + options.full_histo_filename, fit_only=True)
    mpes_full_fit_result = np.copy(mpes_full.fit_result)
    del mpes_full

    # Load the histogram
    dark_hist = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    dark_hist_for_baseline = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    dark_hist_for_baseline.fit(fit_dark_adc.fit_func, fit_dark_adc.p0_func, fit_dark_adc.slice_func,
                               fit_dark_adc.bounds_func, \
                               labels_func=fit_dark_adc.labels_func)

    dark_hist.fit_result = np.zeros((len(options.pixel_list), 3, 2))
    dark_hist.fit_result_label = np.array(['baseline [LSB]', '$f_{dark}$ [MHz]', '$\mu_{XT}$'])

    x = dark_hist.bin_centers

    # print(mpes_full_fit_result.shape)
    # print(mpes_full_fit_result[1 , 0, 0])
    # print(mpes_full_fit_result[1 , 1, 0])
    # print(mpes_full_fit_result[1 , 2, 0])
    # print(mpes_full_fit_result[1 , 3, 0])
    baseline = dark_hist.fit_result[..., 1, 0]
    baseline_error = dark_hist.fit_result[..., 1, 1]
    gain = mpes_full_fit_result[..., 1, 0]
    gain_error = mpes_full_fit_result[..., 1, 1]
    sigma_e = mpes_full_fit_result[..., 2, 0]
    sigma_e_error = mpes_full_fit_result[..., 2, 1]
    sigma_1 = mpes_full_fit_result[..., 3, 0]
    sigma_1_error = mpes_full_fit_result[..., 3, 1]

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

        dark_parameters = compute_dark_parameters(x, y, baseline[pixel], gain[pixel], sigma_1[pixel], sigma_e[pixel],
                                                  integral[pixel], integral_square[pixel])

        # print(pixel)
        # print(baseline)
        # print(dark_hist.fit_result.shape)

        dark_hist.fit_result[pixel, 1, 0] = dark_parameters[0, 0]
        dark_hist.fit_result[pixel, 1, 1] = dark_parameters[0, 1]
        dark_hist.fit_result[pixel, 2, 0] = dark_parameters[1, 0]
        dark_hist.fit_result[pixel, 2, 1] = dark_parameters[1, 1]

    dark_hist.fit_result[:, 0, 0] = baseline
    dark_hist.fit_result[:, 0, 1] = baseline_error

    # dark_hist.save(options.output_directory + options.histo_filename.split('.npz')[0]+'_xt.npz')
    dark_hist.save(options.output_directory + options.histo_filename)
    del dark_hist


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

def single_photo_electron(options):
    return