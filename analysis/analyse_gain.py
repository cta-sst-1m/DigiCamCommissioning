#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import mpe_hist
from spectra_fit import fit_full_mpe
from utils import display, histogram, geometry
import logging,sys
import numpy as np
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

    log = logging.getLogger(sys.modules['__main__'].__name__ + '.' +  __name__)
    log.info('\t-|> Get various inputs')

    # recover previous fit
    if options.mc:

        prev_fit_result = np.ones((options.n_pixels, 8, 2))

    else:

        spes_fit = histogram.Histogram(filename=options.output_directory + options.input_dark_filename, fit_only=True)
        prev_fit_result = np.copy(spes_fit.fit_result)
        del spes_fit

    # Define the histograms
    mpes = histogram.Histogram(filename = options.output_directory + options.mpes_histo_filename)

    mpes_full = histogram.Histogram(data=np.zeros(mpes.data[0].shape),bin_centers=mpes.bin_centers, xlabel='ADC',
                          ylabel='$\mathrm{N_{trigger}}$', label='Summed MPE')

    # Add an Histogram corresponding to the sum of all other only if the mu is above a certain threshold

    pbar = tqdm(total=mpes.data.shape[0]*mpes.data.shape[1])
    #print(pbar.total)
    tqdm_out = TqdmToLogger(log, level=logging.INFO)

    log.info('\t-|> Summing the MPEs:')
    for i in range(mpes.data.shape[0]):
        for j in range(mpes.data.shape[1]):

            if pbar.total<=1000:
                pbar.update(1)
            else:
                if (i*mpes.data.shape[1]+j) %int(pbar.total/1000)==0: pbar.update(pbar.total/1000)
            # put the slice or remove empty bins

            if not options.mc:

                s = [np.where(mpes.data[i,j] != 0)[0][0], np.where(mpes.data[i,j] != 0)[0][-1]]
                if s[0]==s[1]:continue
                mpe_tmp = mpes.data[i,j]
                mean = np.average(mpes.bin_centers[np.nonzero(mpe_tmp)],
                                  weights=mpe_tmp[np.nonzero(mpe_tmp)]) -prev_fit_result[j,0,0]

                if mean < 10 : continue
                mpes_full.data[j]=mpes_full.data[j]+mpes.data[i,j]

            else:

                mpes_full.data[j]=mpes_full.data[j]+mpes.data[i,j]


    mpes_full._compute_errors()

    # Save the histogram
    mpes_full.save(options.output_directory + options.histo_filename)

    # Delete the histograms
    del mpes,mpes_full

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

    # recover previous fit
    mpes_full = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    if options.mc:

        prev_fit_result = None

     # recover previous fit
    else:

        spes_fit = histogram.Histogram(filename=options.output_directory + options.input_dark_filename,fit_only=True)
        prev_fit_result = np.copy(spes_fit.fit_result)
        del spes_fit


    n_peak = 15
    reduced_bounds = lambda *args,config=None, **kwargs: fit_full_mpe.bounds_func(*args,n_peaks = n_peak, config=config, **kwargs)
    reduced_p0 = lambda *args,config=None, **kwargs: fit_full_mpe.p0_func(*args,n_peaks = n_peak, config=config, **kwargs)
    reduced_slice = lambda *args, config=None, **kwargs: fit_full_mpe.slice_func(*args, n_peaks=n_peak, config=config, **kwargs)
    mpes_full.fit(fit_full_mpe.fit_func, reduced_p0, reduced_slice,
                  reduced_bounds, config=prev_fit_result, labels_func=fit_full_mpe.labels_func)#,limited_indices=(4,))


    # get the bad fits
    log = logging.getLogger(sys.modules['__main__'].__name__ + '.' +  __name__)
    log.info('\t-|> Try to correct the pixels with wrong fit results')
    for pix,pix_fit_result in enumerate(mpes_full.fit_result):
        if np.isnan(pix_fit_result[0,1]) and not np.isnan(pix_fit_result[0,0]):
            i = 6
            while  np.isnan(mpes_full.fit_result[pix,0,1]) and i > 2:
                reduced_bounds = lambda *args, config=None, **kwargs: fit_full_mpe.bounds_func(*args, n_peaks = i ,
                                                                                             config=config, **kwargs)
                reduced_p0 = lambda *args, config=None, **kwargs: fit_full_mpe.p0_func(*args, n_peaks = i , config=config,
                                                                                       **kwargs)
                reduced_slice = lambda *args, config=None, **kwargs: fit_full_mpe.slice_func(*args, n_peaks=i,
                                                                                             config=config, **kwargs)
                mpes_full.fit(fit_full_mpe.fit_func, reduced_p0, reduced_slice,
                              reduced_bounds, config=prev_fit_result ,limited_indices=(pix,),force_quiet=True,
                              labels_func=fit_full_mpe.labels_func)
                i-=1

        if np.isnan(pix_fit_result[0,1]) and not np.isnan(pix_fit_result[0,0]):
            i = n_peak-1
            while  np.isnan(mpes_full.fit_result[pix,0,1]) and i > 15:
                reduced_bounds = lambda *args, config=None, **kwargs: fit_full_mpe.bounds_func(*args, n_peaks = i ,
                                                                                             config=config, **kwargs)
                reduced_p0 = lambda *args, config=None, **kwargs: fit_full_mpe.p0_func(*args, n_peaks = i , config=config,
                                                                                       **kwargs)
                reduced_slice = lambda *args, config=None, **kwargs: fit_full_mpe.slice_func(*args, n_peaks=i,
                                                                                             config=config, **kwargs)
                mpes_full.fit(fit_full_mpe.fit_func, reduced_p0, reduced_slice,
                              reduced_bounds, config=prev_fit_result ,limited_indices=(pix,),force_quiet=True,
                              labels_func=fit_full_mpe.labels_func)
                i-=1


    for pix,pix_fit_result in enumerate(mpes_full.fit_result):
        if np.isnan(pix_fit_result[0,1]) and not np.isnan(pix_fit_result[0,0]):
            log.info('\t-|> Pixel %s still badly fitted'%pix)

    mpes_full.save(options.output_directory + options.histo_filename)


def display_results(options, param_to_display=1):
    """
    Display the analysis results

    :param options:

    :return:
    """

    # Load the histogram
    adcs = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    # Define Geometry
    geom = geometry.generate_geometry_0(options.n_pixels)

    if options.mc:

        index_default = (0,)

    else:

        index_default = (4,)


    # Perform some plots
    display.display_hist(adcs, geom, index_default=index_default, param_to_display=param_to_display,limitsCam=[4.,6.], draw_fit = True)
    display.display_fit_result(adcs, geom, index_var=1)

    input('press button to quit')

    return
