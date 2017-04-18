#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import n_pe_hist
from spectra_fit import fit_low_light,fit_high_light
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

    # Define the histograms
    npes = histogram.Histogram(bin_center_min=0, bin_center_max=len(options.pixel_list)-1,
                               bin_width=1, data_shape=(len(options.shower_ids),),
                               label='Shower',xlabel='Pixel',ylabel = '$\mathrm{N_{entries}}$')

    # Get the reference sampling time
    peaks = histogram.Histogram(filename = options.output_directory + options.synch_histo_filename)

    # Construct the histogram
    n_pe_hist.run(npes, options, peak_positions=peaks.data)

    # Save the histogram
    npes.save(options.output_directory + options.histo_filename)

    # Delete the histograms
    del npes,peaks

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
    npes = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    mpes_full = histogram.Histogram(filename=options.output_directory + options.full_histo_filename, fit_only= True)
    mpes_full_fit_result = np.copy(mpes_full.fit_result)
    pe_values = np.zeros(npes.data.shape,dtype=float)
    for s,shower in enumerate(npes.data):
        for pix,adc in enumerate(shower):
            print(pix,adc)
            pe_values[s,pix]=float(adc)/mpes_full.fit_result[pix,1,0]*(1.-0.082)

    pe_values_patch = np.zeros(npes.data.shape,dtype=float)
    for s,shower in enumerate(npes.data):
        for patch in options.cts.LED_patches:
            total_pe = 0.
            for p in patch.leds_camera_pixel_id:
                pix = options.pixel_list.index(p)
                total_pe+=pe_values[s,pix]
            for p in patch.leds_camera_pixel_id:
                pix = options.pixel_list.index(p)
                pe_values_patch[s,pix]=total_pe

    np.savez_compressed(options.output_directory + options.shower_filename, patch=pe_values_patch, pixel = pe_values)




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

    import matplotlib.pyplot as plt

    pixel_start = options.pixel_list[2]

    # Perform some plots
    if options.mc:

        for level in options.scan_level:

            fig = plt.figure()
            axis = fig.add_subplot(111)
            display.draw_hist(axis, adcs, index=(level, pixel_start,), limits=[2005, 2150], draw_fit=True, label='Pixel %s')

        x = np.array(options.scan_level)*5.
        y = adcs.fit_result[:,int(options.n_pixels-1),0,0]
        yerr = adcs.fit_result[:,int(options.n_pixels-1),0,1]
        mask = np.isfinite(x)*np.isfinite(y)*np.isfinite(yerr)

        param = np.polyfit(x[mask], y[mask], 4, w=1./yerr[mask])
        text_param = ''
        for i in range(len(param)):
            text_param += 'p_%d = %0.9f  \n' %(i, param[i])

        true_param = np.array([11 * 1E-8, 0., 0., 0., 0.])

        fig = plt.figure()
        ax_up = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
        ax_down = plt.subplot2grid((4,4), (3,0), colspan=4, sharex=ax_up)
        #ax_down_2 = plt.subplot2grid((4,4), (3,0), colspan=4, sharex=ax_up)
        ax_up.plot(x, np.polyval(param, x), label='MC observed best fit p_0 = %0.4f [p.e.]' %param[-1])
        ax_up.plot(x, np.polyval(true_param, x), label='MC generated')
        ax_up.errorbar(x[mask], y[mask], yerr=yerr[mask], label='MC observed', linestyle='None', barsabove=True, markersize=12, marker='o')
        ax_down.plot(x[mask], np.abs(np.polyval(param, x[mask])-np.polyval(true_param, x[mask]))/np.polyval(param, x[mask]), label='bias polynomial')
        ax_down.plot(x[mask], np.abs(y[mask]-np.polyval(true_param, x[mask]))/y[mask], label='bias measurements')
        #ax_down_2.plot(x[mask], np.abs(y[mask]-np.polyval(true_param, x[mask]))/yerr[mask], label='pull')
        #ax_up.text(x[-3], y[-3], text_param)
        ax_down.set_xlabel('DAC')
        ax_up.set_ylabel('$\mu$ [p.e.]')
        #ax_down.set_ylabel('$\\frac{\mu_{t}- \mu_{m}}{\sigma_{m}}$')
        fig.subplots_adjust(hspace=0.1)
        plt.setp(ax_up.get_xticklabels(), visible=False)
        #plt.setp(ax_down.get_xticklabels(), visible=False)
        ax_up.set_yscale('log')
        ax_down.set_yscale('log')
        #ax_down_2.set_yscale('log')
        ax_up.legend()
        ax_down.legend()
        #ax_down_2.legend()


    else:

        display.display_hist(adcs, options=options, geom=geom,draw_fit=True,scale='log')
    input('press button to quit')

    return
