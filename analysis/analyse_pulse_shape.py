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

from ctapipe import visualization
import scipy
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
    pulse_shapes = pulse_shape.run(pulse_shapes, options=options, compute_errors=True)
    np.savez(options.output_directory + options.pulse_shape_filename, pulse_shapes=pulse_shapes)
    return


def integrate_trace(d,window_width):
    return np.convolve(d, np.ones((window_width), dtype=int), 'valid')

def perform_analysis(options):

    mpes_full = histogram.Histogram(filename=options.output_directory + options.full_histo_filename, fit_only= True)
    mpes_full_fit_result = np.copy(mpes_full.fit_result)
    pulses = np.load(options.output_directory + options.pulse_shape_filename)['pulse_shapes']
    ps = pulses[...,0]
    ps = ps- np.mean(ps[...,0:5],axis=-1).reshape(np.mean(ps[...,0:5],axis=-1).shape+(1,))
    integrate_trace_7 = lambda x : integrate_trace(x,7)
    ps =  np.apply_along_axis(integrate_trace_7,-1,ps)
    ps = np.divide(ps ,np.max(ps,axis=-1).reshape(np.max(ps,axis=-1).shape+(1,)))
    pulses[...,1]=pulses[...,0]
    pulses[...,0][...,0:ps.shape[-1]]=ps
    pulses[..., 0][...,ps.shape[-1]:-1]=0.
    pulses[...,0][...,-1]=0.
    print(pulses[35,0,:,0])
    np.savez(options.output_directory + options.pulse_shape_filename.split('.')[0]+'subtracted.npz', pulse_shapes=pulses)
    return


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """
    data = np.load(options.output_directory + options.pulse_shape_filename.split('.')[0]+'subtracted.npz')
    data = data['pulse_shapes']
    geom = geometry.generate_geometry_0(pixel_list=options.pixel_list)

    print('there',data.shape)
    #display.display_pulse_shape(data, options=options, geom=geom)
    good_levels=[0]*data.shape[1]
    for l,level in enumerate(data):
        for p,pixel in enumerate(level):
            if np.max(data[l,p,:,1])<3500:
                good_levels[p]=l
    good_template = np.zeros((data.shape[1],data.shape[2]),dtype=float)
    integrals = np.ones((data.shape[1],),dtype=float)
    for p in range(data.shape[1]):
        good_template[p]=data[good_levels[p],p,:,0]
        integrals[p]=np.trapz(good_template[p], dx=4)

    plt.figure()

    plt.plot(np.arange(92),good_template[197])
    plt.show()
    plt.figure()
    integrals2 = np.copy(integrals)
    plt.plot(np.arange(data.shape[1]),integrals)
    plt.show()
    fig,ax=plt.subplots(1,1)
    camera_visu = visualization.CameraDisplay(geom, ax=ax, title='', norm='lin', cmap='viridis',
                                              allow_pick=True)
    camera_visu.image = integrals2
    print(camera_visu.image[0])
    camera_visu.add_colorbar()
    camera_visu.axes.set_xlabel('x [mm]')
    camera_visu.axes.set_ylabel('y [mm]')
    plt.show()
    np.savez(options.output_directory + options.integrals_filename, integrals=integrals)
    print('here')
    h = input('press a key')
    return
