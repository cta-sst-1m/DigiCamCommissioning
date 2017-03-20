#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import synch_hist
from utils import display, histogram, geometry
from spectra_fit import fit_synch
import logging,sys
import numpy as np

__all__ = ["create_histo", "perform_analysis", "display_results"]


def create_histo(options):
    """
    Create a list of ADC histograms and fill it with data

    :param options: a dictionary containing at least the following keys:
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram      (str)
        - 'file_basename'    : the base name of the input files                   (str)
        - 'directory'        : the path of the directory containing input files   (str)
        - 'file_list'        : the list of base filename modifiers for the input files
                                                                                  (list(str))
        - 'evt_min'          : the minimal event number to process                (int)
        - 'evt_max'          : the maximal event number to process                (int)
        - 'n_pixels'         : the number of pixels to consider                   (int)
        - 'sample_max'       : the maximum number of sample                       (int)

    :return:
    """
    # Define the histograms
    peaks = histogram.Histogram(bin_center_min=0, bin_center_max=options.sample_max,
                               bin_width=1, data_shape=(len(options.pixel_list),),
                               label='Position of the peak',xlabel='Sample [/ 4 ns]',ylabel = 'Events / sample')

    # Get the adcs
    dark = histogram.Histogram(filename=options.output_directory + options.dark_histo_filename, fit_only=True)
    options.prev_fit_result = np.copy(dark.fit_result)
    del dark

    synch_hist.run(peaks, options,min_evt = options.evt_min , max_evt=options.evt_max)

    # Save the histogram

    peaks.save(options.output_directory + options.histo_filename)


    print(peaks.data.shape)
    # Delete the histograms
    del peaks


    return


def perform_analysis(options):
    """
    Perform a simple gaussian fit of the ADC histograms

    :param options: a dictionary containing at least the following keys:
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram      (str)
        - 'hv_off_histo_filename' : the name of the hv_off fit results            (str)

    :return:
    """
    # Fit the baseline and sigma_e of all pixels
    log = logging.getLogger(sys.modules['__main__'].__name__+__name__)

    peaks = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    peaks.data[...,0]=1e-8
    peaks.data[...,1]=1e-8
    peaks.data[...,2]=1e-8
    peaks.data[...,3]=1e-8
    peaks.data[...,4]=1e-8
    peaks.data[...,5]=1e-8
    peaks.data[...,-1]=1e-8
    sum = np.sum(peaks.data, axis=1)[:,None]
    peaks.data = peaks.data.astype(dtype=float)/sum
    peaks.errors = peaks.errors.astype(dtype=float)/sum*5

    '''

    # Fit the baseline and sigma_e of all pixels
    peaks.fit(fit_synch.fit_func, fit_synch.p0_func, fit_synch.slice_func, fit_synch.bounds_func, \
            labels_func=fit_synch.labels_func)  # , limited_indices=tuple(options.pixel_list))

    f  = lambda x, pix: peaks.fit_result[pix,3,0]*(np.exp(-peaks.fit_result[pix,4,0]*(x))+peaks.fit_result[pix,5,0])+\
         peaks.fit_result[pix,6,0]*(np.exp(-peaks.fit_result[pix,7,0]*(x))+peaks.fit_result[pix,8,0])

    for pix in range(peaks.data.shape[0]):
        slice = fit_synch.slice_func(peaks.data[pix],peaks.bin_centers)
        for i,x in enumerate(peaks.bin_centers):#[slice[0]:slice[1]:1]):
            #print(i,x)
            #print(peaks.data[pix][slice[0]:slice[1]:1])
            #print(peaks.errors[pix][slice[0]:slice[1]:1])
            #print(f(x,pix))
            if peaks.data[pix][i]<f(x,pix)+peaks.errors[pix][i]:
                peaks.data[pix][i] = 1e-8
        '''
    peaks.data[...,-2]=1e-8
    peaks.data[...,-3]=1e-8
    peaks.data[...,-4]=1e-8

    peaks.save(options.output_directory + options.histo_filename)
    del peaks



def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    # Load the histogram
    peaks = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    # Define Geometry
    geom = geometry.generate_geometry_0(pixel_list=options.pixel_list)

    # Perform some plots

    display.display_hist(peaks, options=options, geom=geom,draw_fit=False, scale='linear')

    #display.display_hist(peaks,  geom)

    input('press button to quit')

    return
