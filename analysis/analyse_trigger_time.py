
#!/usr/bin/env python3

# external modules
import logging,sys
import numpy as np
from tqdm import tqdm
from utils.logger import TqdmToLogger
from scipy.stats import expon




# internal modules
from data_treatement import trigger_time
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

    time = []
    trigger_mask = []


    trigger_time.run(time_list=time, options=options, trigger_mask=trigger_mask)
    time = np.array(time)
    time_interval = np.diff(time)
    np.savez(options.output_directory + options.histo_filename, time=time, time_interval=time_interval, trigger_mask=trigger_mask)

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
    import matplotlib.pyplot as plt
    data = np.load(options.output_directory + options.histo_filename)
    time = data['time']
    time_interval = data['time_interval']
    trigger_mask = data['trigger_mask']
    time_interval = time_interval[(time_interval<options.max_time) * (time_interval>=0)]

    param = expon.fit(time_interval, floc=0)
    np.savez(options.output_directory + options.histo_filename, time=time, time_interval=time_interval, trigger_mask=trigger_mask, param=param)

    return

def display_results(options):

    data = np.load(options.output_directory + options.histo_filename)
    time = data['time']
    time_interval = data['time_interval']
    trigger_mask = data['trigger_mask']

    try:

        param = data['param']

    except:

        param = None

    import matplotlib.pyplot as plt
    plt.figure()
    hist = plt.hist(time_interval, bins=np.linspace(0, options.max_time,100), log=True, normed=False)
    n_entries = np.sum(hist[0])
    bin_width = hist[1][1] - hist[1][0]

    if param is not None:

        pdf_fit = expon(loc=param[0], scale=param[1])
        plt.plot(hist[1], n_entries*bin_width*pdf_fit.pdf(hist[1]), label='$f_{trigger}$ = %0.2f [Hz]' %(1E9/param[1]))
        plt.xlabel('$\Delta t$ [ns]')
        plt.legend(loc='best')

    plt.figure()
    plt.hist(time/1E9, bins=np.linspace(np.min(time), np.max(time), 40)/1E9)
    plt.xlabel('t [s]')

    plt.figure()
    plt.bar(time[trigger_mask]/1E9, np.ones(time[trigger_mask].shape), width=param[1]/4/1E9, align='center', label='Above threshold', color='g')
    plt.xlabel('t [s]')
    plt.legend(loc='best')

    plt.figure()
    plt.bar(time[~trigger_mask]/1E9, np.ones(time[~trigger_mask].shape), width=param[1]/4/1E9, align='center', label='Bellow threshold', color='r')
    plt.xlabel('t [s]')
    plt.legend(loc='best')

    return




