
#!/usr/bin/env python3

# external modules
import logging,sys
import numpy as np
from tqdm import tqdm
from utils.logger import TqdmToLogger


# internal modules
from data_treatement import trigger
from utils import display, histogram, geometry

__all__ = ["create_histo", "perform_analysis", "display_results", "save"]


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

    #temp = np.load(options.cts_directory + 'config/cluster.p')

    #camera.Clusters = temp['']
    #camera.Clusters.ID = np.arange(0, len(camera.Clusters), 1)
    #print(temp)
    #Clusters_patch_list = temp['patches_in_cluster'][0:options.n_clusters]

    triggers = histogram.Histogram(data=np.zeros((len(options.scan_level), options.n_clusters, len(options.threshold))), \
                                 bin_centers=np.array(options.threshold), label='Trigger', \
                                 xlabel='Threshold [ADC]', ylabel='trigger rate [Hz]')



    trigger_spectrum = trigger.run(triggers, options=options)
    triggers.save(options.output_directory + options.histo_filename)

    np.save(arr=trigger_spectrum.ravel(), file=options.output_directory + options.trigger_spectrum_filename)

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

    print('Nothing implemented')

    return


def display_results(options):
    """
    Display the analysis results
    :param options:
    :return:
    """

    # Load the histogram
    triggers = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    # Define Geometry
    #geom = geometry.generate_geometry_0(pixel_list=np.arange(0, 432, 1))

    #display.display_hist(dc_led, options, geom=geom, scale='linear')
    #display.display_hist(triggers, options, scale='log')

    trigger_spectrum = np.load(options.output_directory + options.trigger_spectrum_filename + '.npy')
    import matplotlib.pyplot as plt


    fig_1 = plt.figure()
    axis_1 = fig_1.add_subplot(111)
    axis_1.set_title('window width : %d samples' %options.window_width)
    for level in reversed(options.scan_level):

        print(triggers.errors.shape)
        print(triggers.data.shape)

        axis_1.errorbar(x=triggers.bin_centers, y=triggers.data[level, 0], yerr=triggers.errors[level, 0], fmt='o', label='$f_{nsb} = $ %0.1f [MHz]' %(options.nsb_level[level]*1E3))

    axis_1.axhline(y=500, color = 'k', label='safe threshold', linestyle='-.')
    axis_1.set_xlabel(triggers.xlabel)
    plt.legend(loc='best')
    axis_1.set_ylabel(triggers.ylabel)
    axis_1.set_yscale('log')
    plt.show()


    fig = plt.figure()
    axis = fig.add_subplot(111)
    axis.hist(trigger_spectrum, bins=np.arange(int(np.min(trigger_spectrum)), int(np.max(trigger_spectrum))+1, 1), normed=True, label='spectrum cluster, 3 [MHz]')
    plt.xlabel('Threshold [ADC]')
    plt.legend(loc='best')
    plt.ylabel('P')
    axis.set_yscale('log')
    plt.show()


    return


def save(options):

    print('Nothing implemented')

    return




