#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import pulse_shape
from utils import display, histogram, geometry
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

    pulse_shapes = np.zeros((len(options.scan_level), len(options.pixel_list), options.n_bins, 2))
    pulse_shapes = pulse_shape.run(pulse_shapes, options=options)
    np.savez(options.output_directory + options.pulse_shape_filename, pulse_shape=pulse_shapes)
    return

def integrate_trace(d,window_width):

    integrated_trace = np.convolve(d, np.ones((window_width), dtype=int), 'same')

    return integrated_trace

def perform_analysis(options):

    #mpes_full = histogram.Histogram(filename=options.output_directory + options.full_histo_filename, fit_only=True)
    #mpes_full_fit_result = np.copy(mpes_full.fit_result)
    pulses = np.load(options.output_directory + options.pulse_shape_filename)['pulse_shape']
    pulse_integrals = np.zeros((pulses.shape[0], pulses.shape[1], 2))


    pulse_shape = pulses[:, :, : ,0]
    pulse_shape_error = pulses[:, :, : ,1]

    integrate_trace_n = lambda x: integrate_trace(x, options.window_width)


    print(pulse_shape_error[0,6, 8])

    n_bins = 4
    first_bins = pulse_shape[...,0:n_bins]
    baseline = np.mean(first_bins, axis=-1)
    baseline_error = np.std(first_bins, axis=-1)/np.sqrt(n_bins)
    pulse_shape = pulse_shape - baseline[...,None]
    pulse_shape_error = np.sqrt(pulse_shape_error**2 + baseline_error[...,None]**2)
    pulse_shape = np.apply_along_axis(integrate_trace_n, -1, pulse_shape)
    pulse_shape_error = np.sqrt(np.apply_along_axis(integrate_trace_n, -1, pulse_shape_error**2))
    pulse_shape_max = np.max(pulse_shape, axis=-1)

    argmax = np.argmax(pulse_shape, axis=-1)
    for i in range(pulse_shape.shape[0]):
        for j in range(pulse_shape.shape[1]):

            pulse_shape_max_error = pulse_shape_error[i,j, argmax[i,j]]

    pulse_shape = np.divide(pulse_shape, pulse_shape_max[...,None])
    pulse_shape_error = pulse_shape * (pulse_shape_error/np.multiply(pulse_shape, pulse_shape_max[...,None]) + (pulse_shape_max_error/pulse_shape_max)[...,None])

    pulse = np.zeros(pulse_shape.shape + (2,))
    pulse[..., 0] = pulse_shape
    pulse[..., 1] = pulse_shape_error

    pulse_integrals[..., 0] = np.sum(pulse_shape, axis=-1) * options.sampling_time
    pulse_integrals[..., 1] = np.sqrt(np.sum(pulse_shape_error**2, axis=-1)) * options.sampling_time


    '''
    print (pulses[...,1])
    ps = pulses[..., 0]
    ps = ps - np.mean(ps[..., 0:5], axis=-1).reshape(np.mean(ps[..., 0:5], axis=-1).shape + (1,))
    integrate_trace_7 = lambda x: integrate_trace(x, options.window_width)
    ps = np.apply_along_axis(integrate_trace_7, -1, ps)
    ps = np.divide(ps, np.max(ps, axis=-1).reshape(np.max(ps, axis=-1).shape + (1,)))
    pulses[..., 1] = pulses[...,0] * ( pulses)
        np.divide(pulses[...,1], np.max(ps, axis=-1).reshape(np.max(ps, axis=-1).shape + (1,))) #TODO compute this !
    pulses[..., 0][..., 0:ps.shape[-1]] = ps
    pulses[..., 0][..., ps.shape[-1]:-1] = 0.
    pulses[..., 0][..., -1] = 0.
    #pulse_integrals[...,0] = np.sum(pulses[...,0] ,axis=-1)
    #print(pulses[35, 0, :, 0])

'''
    np.savez(options.output_directory + options.pulse_shape_filename.split('.')[0] + '_substracted.npz', pulse_shape=pulse)
    np.savez(options.output_directory + options.pulse_shape_filename.split('.')[0] + '_integrals.npz', pulse_integrals=pulse_integrals)


    return


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """
    #data = np.load(options.output_directory + options.pulse_shape_filename)['pulse_shape']
    data_substracted = np.load(options.output_directory + options.pulse_shape_filename.split('.')[0] + '_substracted.npz')['pulse_shape']
    pulse_integrals = np.load(options.output_directory + options.pulse_shape_filename.split('.')[0] + '_integrals.npz')['pulse_integrals']


    geom = geometry.generate_geometry_0(pixel_list=options.pixel_list)

    print(data_substracted.shape)


    #display.display_pulse_shape(data, options=options, geom=geom)
    display.display_pulse_shape(data_substracted, options=options, geom=geom)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.errorbar(np.array(options.scan_level), pulse_integrals[:, 18, 0], yerr=pulse_integrals[:,18,1])



    plt.show()
    return
