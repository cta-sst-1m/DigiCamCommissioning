#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import mpe_hist
from spectra_fit import fit_full_mpe
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

    # recover previous fit
    spes_fit_result = histogram.Histogram(options.output_directory + options.input_dark_filename,fit_only=True)
    adcs_fit_result = histogram.Histogram(options.output_directory + options.input_hvoff_filename,fit_only=True)

    # Now build a fake fit result for stating point
    prev_fit_result = np.expand_dims(adcs_fit_result[:, 1] + spes_fit_result[:, 6], axis=1)
    prev_fit_result = np.append(prev_fit_result, np.expand_dims(spes_fit_result[:, 2], axis=1), axis=1)
    prev_fit_result = np.append(prev_fit_result, np.expand_dims(spes_fit_result[:, 0], axis=1), axis=1)
    prev_fit_result = np.append(prev_fit_result, np.expand_dims(spes_fit_result[:, 1], axis=1), axis=1)

    # Define the histograms
    mpes = histogram.Histogram(options.output_directory + options.mpes_histo_filename)
    # Get the reference sampling time
    mpe_tmp = np.copy(mpes.data)
    mpe_tmp[mpe_tmp == 0] = 1e-6
    mpe_mean = np.average(np.repeat(
        np.repeat(
            mpes.bin_centers[1:-1:1].reshape(1, 1, -1), mpe_tmp.shape[0], axis=0), mpe_tmp.shape[1], axis=1),
        weights=mpe_tmp[..., 1:-1:1], axis=2)
    del mpe_tmp
    # subtract the baseline
    mpe_mean = np.subtract(mpe_mean, np.repeat(prev_fit_result[:, 0, 0].reshape((1,) + prev_fit_result[:, 0, 0].shape),
                                               mpe_mean.shape[0], axis=0))
    mpe_tmp = np.copy(mpes.data)
    for i in range(mpe_tmp.shape[0]):
        for j in range(mpe_tmp.shape[1]):
            # TODO parametrise this
            if mpe_mean[i, j] < 5 or np.where(mpe_tmp[i, j] != 0)[0].shape[0] < 2: mpe_tmp[i, j, :] = 0
    mpe_tmp = np.sum(mpe_tmp, axis=0)
    mpes_full = histogram.Histogram(data=np.copy(mpe_tmp), bin_centers=mpes.bin_centers, xlabel='ADC',
                          ylabel='$\mathrm{N_{trigger}}$', label='Summed MPE')
    del mpe_tmp,mpe_mean
    # Save the histogram
    mpes_full.save(options.output_directory + options.histo_filename)

    # Delete the histograms
    del mpes,mpes_full,mpe_tmp

    return


def perform_analysis(options):
    """
    Perform a simple gaussian fit of the ADC histograms

    :param options: a dictionary containing at least the following keys:
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram
                                                 whose fit contains the gain,sigmas etc...(str)

    :return:
    """
    # Fit the baseline and sigma_e of all pixels
    mpes_full = histogram.Histogram(options.output_directory + options.histo_filename)

    # recover previous fit
    spes_fit_result = histogram.Histogram(options.output_directory + options.input_dark_filename, fit_only=True)
    adcs_fit_result = histogram.Histogram(options.output_directory + options.input_hvoff_filename, fit_only=True)

    # Now build a fake fit result for stating point
    prev_fit_result = np.expand_dims(adcs_fit_result[:, 1] + spes_fit_result[:, 6], axis=1)
    prev_fit_result = np.append(prev_fit_result, np.expand_dims(spes_fit_result[:, 2], axis=1), axis=1)
    prev_fit_result = np.append(prev_fit_result, np.expand_dims(spes_fit_result[:, 0], axis=1), axis=1)
    prev_fit_result = np.append(prev_fit_result, np.expand_dims(spes_fit_result[:, 1], axis=1), axis=1)

    reduced_bounds = lambda *args,config=None, **kwargs: fit_full_mpe.bounds_func(*args,n_peaks = 22, config=config, **kwargs)
    reduced_p0 = lambda *args,config=None, **kwargs: fit_full_mpe.p0_func(*args,n_peaks = 22, config=config, **kwargs)
    mpes_full.fit(fit_full_mpe.fit_func, reduced_p0, fit_full_mpe.slice_func,
                  reduced_bounds, config=prev_fit_result)
    # get the bad fits
    print('Try to correct the pixels with wrong fit results')
    for pix,pix_fit_result in enumerate(mpes_full.fit_result):
        if np.isnan(pix_fit_result[0,1]) and not np.isnan(pix_fit_result[0,0]):
            print('Pixel %d refit',pix)
            i = 22
            while  np.isnan(mpes_full.fit_result[pix,0,1]) and i > 15:
                reduced_bounds = lambda *args, config=None, **kwargs: fit_full_mpe.bounds_func(*args, n_peaks = i ,
                                                                                             config=config, **kwargs)
                reduced_p0 = lambda *args, config=None, **kwargs: fit_full_mpe.p0_func(*args, n_peaks = i , config=config,
                                                                                       **kwargs)
                mpes_full.fit(fit_full_mpe.fit_func, reduced_p0, fit_full_mpe.slice_func,
                              reduced_bounds, config=prev_fit_result ,limited_indices=(pix,),force_quiet=True)
                i-=1

    for pix,pix_fit_result in enumerate(mpes_full.fit_result):
        if np.isnan(pix_fit_result[0,1]) and not np.isnan(pix_fit_result[0,0]):
            print('-----|> Pixel %d is still badly fitted'%pix)

    mpes_full.save(options.output_directory + options.fit_filename)


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    # Load the histogram
    adcs = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    # Define Geometry
    geom = geometry.generate_geometry_0()

    # Perform some plots
    display.display_hist(adcs, geom, index_default=(700,), param_to_display=-1, limits=[1900., 2100.])

    input('press button to quit')

    return
