#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import dark_hist
from utils import display,  geometry
from utils.histogram import Histogram
import logging,sys
import numpy as np
import logging
from tqdm import tqdm
from utils.logger import TqdmToLogger

__all__ = ["create_histo", "perform_analysis", "display_results"]


def create_histo(options):


    # Define the histograms
    histogram = Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list),),
                               label=' ',xlabel='[LSB]',ylabel = '$\mathrm{N_{entries}}$')

    # Construct the histogram
    dark_hist.run(histogram, options, hist_type='raw')
     # Save the histogram
    histogram.save(options.output_directory + options.histo_filename)


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

    print('Nothing here !')


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """



    histogram_0 = Histogram(filename=options.output_directory + 'histogram_dac_0_data.npz')
    histogram_1 = Histogram(filename=options.output_directory + 'histogram_dac_277_data.npz')
    histogram_2 = Histogram(filename=options.output_directory + 'histogram_dac_288_data.npz')
    histogram_3 = Histogram(filename=options.output_directory + 'histogram_dac_295_data.npz')
    histogram_4 = Histogram(filename=options.output_directory + 'histogram_dac_306_data.npz')
    histogram_5 = Histogram(filename=options.output_directory + 'histogram_dac_323_data.npz')
    histogram_6 = Histogram(filename=options.output_directory + 'histogram_dac_333_data.npz')
    histogram_7 = Histogram(filename=options.output_directory + 'histogram_dac_343_data.npz')
    histogram_8 = Histogram(filename=options.output_directory + 'histogram_dac_353_data.npz')
    histogram_0_mc = Histogram(filename=options.output_directory + 'histogram_dark_mc.npz')
    histogram_1_mc = Histogram(filename=options.output_directory + 'histogram_40_mc.npz')
    histogram_2_mc = Histogram(filename=options.output_directory + 'histogram_80_mc.npz')
    histogram_3_mc = Histogram(filename=options.output_directory + 'histogram_125_mc.npz')
    histogram_4_mc = Histogram(filename=options.output_directory + 'histogram_660_mc.npz')
    histogram_0_care = Histogram(filename=options.output_directory + 'histogram_dark_care.npz')
    histogram_1_care = Histogram(filename=options.output_directory + 'histogram_40_care.npz')
    histogram_2_care = Histogram(filename=options.output_directory + 'histogram_80_care.npz')
    histogram_3_care = Histogram(filename=options.output_directory + 'histogram_125_care.npz')
    histogram_4_care = Histogram(filename=options.output_directory + 'histogram_660_care.npz')


    """
    display.display_hist(histogram_dark, options)
    display.display_hist(histogram_1, options)
    display.display_hist(histogram_2, options)
    display.display_hist(histogram_3, options)
    display.display_hist(histogram_dark_mc, options)
    display.display_hist(histogram_2_mc, options)
    """
    list_histo = [histogram_0, histogram_1, histogram_2, histogram_3, histogram_4, histogram_5, histogram_6, \
                  histogram_7, histogram_8, histogram_0_mc, histogram_1_mc, histogram_2_mc, histogram_3_mc, \
                  histogram_4_mc, histogram_0_care, histogram_1_care, histogram_2_care, histogram_3_care, \
                  histogram_4_care]

    labels = ['DATA DAC 0', 'DATA DAC 277', 'DATA DAC 288', 'DATA DAC 295', 'DATA DAC 306', 'DATA DAC 323', \
              'DATA DAC 333', 'DATA DAC 343', 'DATA DAC 353', 'TOY Dark', 'TOY 40 MHz', 'TOY 80 MHz', \
              'TOY 125 MHz', 'TOY 660 MHz', 'CARE Dark', 'CARE 40 MHz', 'CARE 80 MHz', 'CARE 125 MHz',\
              'CARE 660 MHz']

    #colors = ['b', 'r', 'g', 'k', 'b', 'r', 'g', 'k', 'y', 'b', 'r', 'g', 'k', 'y']
    #line_styles = ['-', '-', '-', '-', ':', ':', ':', ':', ':', ':', ':', ':', ':', ':']
    #markers = ['None', 'None', 'None', 'None', 'o', 'o', 'o', 'o', 'o', 'x', 'x', 'x', 'x', 'x']
    import matplotlib.pyplot as plt

    for pixel_id in range(21):
        if pixel_id != 1:
            continue
        fig = plt.figure(figsize=(10,5))
        plt.title('pixel %d' % pixel_id)
        axis = fig.add_subplot(111)
        for i, histo in enumerate(list_histo):

            if 'DATA' not in labels[i]: #i not in [8, 13, 18]:
                continue

            if 'DATA' in labels[i]:
                color = 'k'
            elif 'TOY' in labels[i]:
                color = 'r'
            elif 'CARE' in labels[i]:
                color = 'g'

            x = histo.bin_centers #- np.average(histo.bin_centers, weights=histo.data[pixel_id])
            y = histo.data[pixel_id] / np.sum(histo.data[pixel_id])
            yerr = histo.errors[pixel_id] / np.sum(histo.data[pixel_id])

            #print(np.average(histo.bin_centers, weights=histo.data[pixel_id]))

            mask = y>0

            x = x[mask]
            y = y[mask]
            yerr = yerr[mask]

            axis.errorbar(x, y, yerr=yerr, label=labels[i], color=color, linestyle='-', marker='o')

        axis.legend(loc='best')
        axis.set_yscale('log')
        axis.set_xlabel('[LSB]')

    plt.show()


    return
