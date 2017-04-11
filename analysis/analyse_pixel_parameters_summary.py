#!/usr/bin/env python3

# external modules

# internal modules
from spectra_fit import fit_dark_adc
from utils import display, histogram, geometry
from ctapipe import visualization
from data_treatement import adc_hist
import logging,sys
import scipy.stats
import numpy as np
from ctapipe.io import zfits
import matplotlib.pyplot as plt
from tqdm import tqdm

__all__ = ["create_histo", "perform_analysis", "display_results"]


def create_histo(options):
    """
    :return:
    """

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
    # baseline --> taken from dark with subtraction
    # sigma_e --> taken from dark with subtraction

    # gain --> taken from mpe_full
    # sigma_1 --> taken from mpe_full
    # cross talk --> taken from dark or mpe full

    return


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    # Load the histogram
    baseline = np.load(options.output_directory + options.histo_filename)['baseline']
    # Define Geometry
    geom = geometry.generate_geometry_0(pixel_list=options.pixel_list)
    fig,ax = plt.subplots(1,2)
    camera_visu = visualization.CameraDisplay(geom, ax=ax[0], title='', norm='lin', cmap='viridis',
                                              allow_pick=True)
    image = np.var(baseline - np.mean(baseline, axis=-1)[:, None], axis=-1)
    image2 = np.var(baseline - np.mean(baseline, axis=-1)[:, None], axis=-1)
    image2[image2>1]=1
    image[image > 0.2] = 0.2
    camera_visu.image = image
    camera_visu.add_colorbar()
    camera_visu.axes.set_xlabel('x [mm]')
    camera_visu.axes.set_ylabel('y [mm]')

    plt.subplot(1,2,2)
    plt.hist(image,bins=100)


    display.display_by_pixel(image2,options)

    #. Perform some plots
    input('press button to quit')

    return
