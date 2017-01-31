#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import adc_hist
from spectra_fit import fit_full_mpe
from utils import display, histogram, geometry

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
        - 'evt_max'          : the maximal number of events to process            (int)
        - 'n_evt_per_batch'  : the number of event per fill batch. It can be
                               optimised to improve the speed vs. memory          (int)
        - 'n_pixels'         : the number of pixels to consider                   (int)
        - 'adcs_min'         : the minimum adc value in histo                     (int)
        - 'adcs_max'         : the maximum adc value in histo                     (int)
        - 'adcs_binwidth'    : the bin width for the adcs histo                   (int)

    :return:
    """
    # Define the histograms
    adcs = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth, data_shape=(options.n_pixels,),
                               label='Pixel SPE',xlabel='Pixel ADC',ylabel = 'Count / ADC')

    hv_off_fit = histogram.Histogram(filename=options.output_directory + options.hv_off_histo_filename,fit_only=True)

    # Get the adcs
    adc_hist.run(adcs, options, 'SPE',prev_fit_result=hv_off_fit.fit_result )

    # Save the histogram
    adcs.save(options.output_directory + options.histo_filename)

    # Delete the histograms
    del adcs,hv_off_fit


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

    # Load the histogram
    adcs = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    # load previous fit result
    hv_off_fit = histogram.Histogram(filename=options.output_directory + options.hv_off_histo_filename,fit_only=True)


    # Fit the gain and sigma's
    adcs.fit(fit_full_mpe.fit_func, fit_full_mpe.p0_func, fit_full_mpe.slice_func,
                fit_full_mpe.bounds_func, config=hv_off_fit.fit_result, labels_func=fit_full_mpe.labels_func)

    # Save the fit
    adcs.save(options.output_directory + options.histo_filename)

    # Delete the histograms
    del adcs,hv_off_fit


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
    display.display_fit_result(adcs, geom, index_var=0, limits=[0., 1.], bin_width=0.05)
    display.display_fit_result(adcs, geom, index_var=1, limits=[0., 1.], bin_width=0.05)
    display.display_fit_result(adcs, geom, index_var=2, limits=[4., 6.], bin_width=0.05)

    display.display_hist(adcs,  geom, index_default=(700,),param_to_display=1,limits = [1900.,2100.])

    input('press button to quit')

    return
