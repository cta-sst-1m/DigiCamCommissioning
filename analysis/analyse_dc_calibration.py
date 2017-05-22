#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import dc_calibration
import sys
import numpy as np
import logging
from tqdm import tqdm
from utils.logger import TqdmToLogger
from utils.histogram import Histogram
import utils.display as display
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev, splint

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


    dc = Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth,
                               data_shape=(len(options.scan_level), len(options.pixel_list), ),
                               label='DC LED raw', xlabel='[LSB]', ylabel='$\mathrm{N_{entries}}$')
    ac_dc = Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                    bin_width=options.adcs_binwidth,
                    data_shape=(len(options.scan_level), len(options.pixel_list),),
                    label='AC/DC LED raw', xlabel='[LSB]', ylabel='$\mathrm{N_{entries}}$')
    options.hist_type = 'nsb'
    options.file_basename = options.file_basename_dc
    dc_calibration.run(dc, options)
    dc.save(options.output_directory + options.dc_scan_filename)

    options.hist_type = 'nsb+signal'
    options.file_basename = options.file_basename_ac_dc
    pulse_shape = dc_calibration.run(ac_dc, options)
    ac_dc.save(options.output_directory + options.ac_dc_scan_filename)
    np.savez(options.output_directory + 'temp.npz', pulse_shape=pulse_shape)

    return


def perform_analysis(options):
    log = logging.getLogger(sys.modules['__main__'].__name__ + '.' + __name__)

    nsb = Histogram(filename=options.output_directory + options.dc_scan_filename)
    nsb_and_signal = Histogram(filename=options.output_directory + options.ac_dc_scan_filename)
    pulse_shape = np.load(options.output_directory + 'temp.npz')['pulse_shape']

    baseline = np.zeros((nsb.data.shape[0], nsb.data.shape[1]))
    baseline_shift = np.zeros((nsb.data.shape[0], nsb.data.shape[1]))
    baseline_gain = np.zeros((nsb.data.shape[0], nsb.data.shape[1]))
    peak_amplitude = np.zeros((nsb_and_signal.data.shape[0], nsb_and_signal.data.shape[1]))
    gain_drop = np.zeros((nsb_and_signal.data.shape[0], nsb_and_signal.data.shape[1]))
    nsb_rate = np.zeros((nsb_and_signal.data.shape[0], nsb_and_signal.data.shape[1]))
    delta_t = np.zeros(nsb_and_signal.data.shape[1])
    pulse_shape_spline = [None]*pulse_shape.shape[0]
    mu_xt = 0.07
    gain = 5.8
    t = np.arange(0, pulse_shape.shape[1], 1) * 4.


    for dc_level in range(nsb.data.shape[0]):

        for pixel in range(nsb.data.shape[1]):

            baseline[dc_level, pixel] = np.average(nsb.bin_centers, weights=nsb.data[dc_level, pixel])

            if dc_level == 0:

                pulse_shape[pixel] = pulse_shape[pixel] - baseline[0, pixel]
                pulse_shape_spline[pixel] = splrep(t, pulse_shape[pixel])
                max_index = np.argmax(pulse_shape[pixel])
                t_temp = np.linspace(t[max_index-1], t[max_index+1], 100)
                data_max = splev(t_temp, pulse_shape_spline[pixel])
                data_max = np.max(data_max)
                pulse_shape[pixel] = pulse_shape[pixel] / data_max
                #data_range = [np.where(pulse_shape[pixel] > 0.05)[0][0] - 1, pulse_shape.shape[1]]
                #t_temp = np.arange(0, data_range[1] - data_range[0], 1) * 4
                #y_temp = pulse_shape[pixel][data_range[0]:data_range[1]]

                #print(t_temp)
                #print(y_temp)

                #pulse_shape_spline[pixel] = splrep(t_temp, y_temp)
                pulse_shape_spline[pixel] = splrep(t, pulse_shape[pixel])
                delta_t[pixel] = splint(t[0], t[-1], pulse_shape_spline[pixel])

                #delta_t[pixel] = np.trapz(pulse_shape[pixel], x=np.arange(0, pulse_shape.shape[1], 1)*4)

            baseline_shift[dc_level, pixel] = baseline[dc_level, pixel] - baseline[0, pixel]
            baseline_gain[dc_level, pixel] = baseline_shift[dc_level, pixel] / baseline[0, pixel]

            peak_amplitude[dc_level, pixel] = np.average(nsb_and_signal.bin_centers, weights=nsb_and_signal.data[dc_level, pixel])
            peak_amplitude[dc_level, pixel] = peak_amplitude[dc_level, pixel] - baseline[dc_level, pixel]
            gain_drop[dc_level, pixel] = peak_amplitude[dc_level, pixel] / peak_amplitude[0, pixel]
            nsb_rate[dc_level, pixel] = baseline_shift[dc_level, pixel] / (delta_t[pixel] * (1. / (1 - mu_xt)) * gain * gain_drop[dc_level, pixel])

    parameter = np.empty((nsb.data.shape[1], 3))
    options.scan_level = np.array(options.scan_level)
    mask = (options.scan_level > 0)

    for pixel in range(nsb.data.shape[1]):
        xdata = options.scan_level[mask]
        ydata = nsb_rate[mask][..., pixel]
        parameter[pixel, :], covariance = curve_fit(dc_led_fit_function, xdata=xdata, ydata=ydata, p0=[0, 0.0000003, 0])



    np.savez(options.output_directory + options.dc_calibration_filename, \
             pixel_id=options.pixel_list, \
             dc_level=options.scan_level,
             baseline=baseline,
             baseline_shift=baseline_shift,
             peak_amplitude=peak_amplitude,
             gain_drop=gain_drop,
             nsb_rate=nsb_rate,
             fit_parameters=parameter,
             pulse_shape=pulse_shape,
             pulse_shape_spline=pulse_shape_spline,
             baseline_gain=baseline_gain)

    np.savez(options.output_directory + options.pulse_shape_filename, pixel_id=options.pixel_list, spline=pulse_shape_spline)

    plt.show()


    return


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    nsb = Histogram(filename=options.output_directory + options.dc_scan_filename)
    nsb_and_signal = Histogram(filename=options.output_directory + options.ac_dc_scan_filename)

    display.display_hist(nsb, options=options)
    display.display_hist(nsb_and_signal, options=options)

    data = np.load(options.output_directory + options.dc_calibration_filename)
    mask = data['dc_level'] >0

    pixel_id = data['pixel_id']
    dc_level = data['dc_level'][mask]
    baseline = data['baseline'][mask]
    baseline_shift = data['baseline_shift'][mask]
    peak_amplitude = data['peak_amplitude'][mask]
    gain_drop = data['gain_drop'][mask]
    nsb_rate = data['nsb_rate'][mask]
    fit_parameters = data['fit_parameters']
    baseline_gain = data['baseline_gain'][mask]
    pulse_shape = data['pulse_shape']
    pulse_shape_spline = data['pulse_shape_spline']

    fig = plt.figure()
    axis = fig.add_subplot(111)
    for pixel in range(len(pixel_id)):
        axis.semilogy(dc_level, nsb_rate[..., pixel] * 1E3, marker='x', label='pixel : %d' % pixel_id[pixel])

    axis.set_xlabel('DC DAC')
    axis.set_ylabel('$f_{nsb}$ [MHz]')
    axis.legend(loc='best',prop={'size': 6})

    fig = plt.figure()
    axis = fig.add_subplot(111)
    for pixel in range(len(pixel_id)):
        axis.semilogy(dc_level, gain_drop[..., pixel], marker='x',
                      label='pixel : %d' % pixel_id[pixel])

    axis.set_xlabel('DC DAC')
    axis.set_ylabel('Gain drop')
    axis.legend(loc='best', prop={'size': 6})

    fig = plt.figure()
    axis = fig.add_subplot(111)
    for pixel in range(len(pixel_id)):
        axis.loglog(nsb_rate[..., pixel] * 1E3, gain_drop[..., pixel], marker='x', linestyle='None',
                      label='pixel : %d' % pixel_id[pixel])
    x = np.linspace(np.min(nsb_rate), np.max(nsb_rate), 1000)
    axis.loglog(x * 1E3, gain_drop_function(x), label='model', linestyle='-')
    axis.set_xlabel('$f_{nsb}$ [MHz]')
    axis.set_ylabel('Gain drop')
    axis.legend(loc='best', prop={'size': 6})

    fig = plt.figure()
    axis = fig.add_subplot(111)

    pixel = 0

    axis.semilogy(dc_level, nsb_rate[..., 1] * 1E3, linestyle='None', marker='o',
                      label='pixel : %d' % pixel_id[pixel], color='k')

    axis.semilogy(dc_level, dc_led_fit_function(dc_level, fit_parameters[1, 0], fit_parameters[1, 1], fit_parameters[1, 2]) * 1E3, linestyle='-', color='r',
                      label='fit : %d' % pixel_id[pixel])
    axis.set_xlabel('DC DAC')
    axis.set_ylabel('$f_{nsb}$ [MHz]')
    axis.legend(loc='best', prop={'size': 10})


    fig = plt.figure()
    axis = fig.add_subplot(111)


    axis.semilogy(dc_level, baseline_shift[...,1],  linestyle='None', marker='x',
                      label='pixel : %d' % pixel_id[pixel], color='k')

    axis.set_xlabel('DC DAC')
    axis.set_ylabel('Baseline shift')
    axis.legend(loc='best', prop={'size': 10})


    fig = plt.figure()
    axis = fig.add_subplot(111)


    axis.semilogy(dc_level, baseline_gain[...,1],  linestyle='None', marker='x',
                      label='pixel : %d' % pixel_id[pixel], color='k')

    axis.set_xlabel('DC DAC')
    axis.set_ylabel('Baseline gain')
    axis.legend(loc='best', prop={'size': 10})

    fig = plt.figure()
    axis = fig.add_subplot(111)

    t_clock = np.arange(0, pulse_shape.shape[1], 1) * 4
    t_spline = np.linspace(t_clock[0], t_clock[-1], 1000)
    axis.step(t_clock, pulse_shape[pixel], label='pixel : %d' % pixel_id[pixel], color='k', where='post')
    axis.plot(t_spline, splev(t_spline, pulse_shape_spline[pixel]), label='spline pixel : %d' % pixel_id[pixel], color='r')
    axis.set_xlabel('t [ns]')
    axis.set_ylabel('u.a.')
    axis.legend(loc='best', prop={'size': 10})

    if options.mc:

        import h5py
        simulation_parameters = h5py.File(options.output_directory + options.file_basename_dc % 1)['simulation_parameters']

        true_nsb_rate = simulation_parameters['nsb_rate']

        fig = plt.figure()
        axis = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2)
        axis_down = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        for pixel in range(len(pixel_id)):

            #print(true_nsb_rate[pixel, :])
            #print(nsb_rate[..., pixel])

            axis.loglog(true_nsb_rate[pixel, :][mask], nsb_rate[..., pixel] * 1E3, marker='x', linestyle='None',
                        label='pixel : %d' % pixel_id[pixel])

        axis_down.loglog(true_nsb_rate[0, :][mask], np.abs(np.mean(nsb_rate, axis=1) * 1E3 - true_nsb_rate[0, :][mask]) / true_nsb_rate[0, :][mask])
        x = np.linspace(np.min(nsb_rate), np.max(nsb_rate), 1000)
        axis.loglog(true_nsb_rate[pixel, :], true_nsb_rate[pixel, :], label='True', linestyle='-')
        axis_down.set_xlabel('true $f_{nsb}$ [MHz]')
        axis.set_ylabel('measured $f_{nsb}$ [MHz]')
        axis.legend(loc='best', prop={'size': 6})

    plt.show()
    return


def dc_led_fit_function(x, a, b, c):

    return a * np.exp(b * x) + c

def gain_drop_function(x) :

    return 1. /  (1 + 1E4*85*1E-15*x*1E9)