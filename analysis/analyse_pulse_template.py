#!/usr/bin/env python3


# external modules
import random
import matplotlib.pyplot as plt
import numpy as np

# internal modules
from utils.pulse_template import NPulseTemplate
from data_treatement import pulse_shape

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

    # Create the pulse templates
    pulse_templates = NPulseTemplate(shape=(len(options.ac_level), len(options.dc_level)))
    average_pulses = pulse_shape.run(options=options)

    # Interpolate data points to get pulse template (spline)
    pulse_templates.interpolate(pulse_data=average_pulses, pixels_id=options.pixel_list)

    # Save the pulse template
    pulse_templates.save(options.output_directory + options.histo_filename)

    return


def perform_analysis(options):

    print('Nothing implemented')
    return


def display_results(options):

    pulse_templates = NPulseTemplate(filename=options.output_directory + options.histo_filename)

    cluster = [482, 516, 517, 518, 519, 552, 553, 554, 555, 556, 588, 589, 590, 591, 624, 625, 626, 627, 628, 661, 662]

    for i in range(len(options.ac_level)):
        for j in range(len(options.dc_level)):

            fig = plt.figure(figsize=(10, 10))
            axis = fig.add_subplot(111)
            axis.set_title('AC : %d, DC : %d' % (options.ac_level[i], options.dc_level[j]))
            pulse_templates.display(cluster, indices=(i, j), axis=axis)
            integral = pulse_templates.pulse_template[i][j].integral
            integral_square = pulse_templates.pulse_template[i][j].integral_square

            plt.figure(figsize=(10, 10))
            plt.hist(integral[cluster], bins='auto', label='AC : %d, DC : %d' % (options.ac_level[i], options.dc_level[j]))
            plt.xlabel('pulse integral [ns]')
            plt.legend()

            plt.figure(figsize=(10, 10))
            plt.hist(integral_square[cluster], bins='auto', label='AC : %d, DC : %d' % (options.ac_level[i], options.dc_level[j]))
            plt.xlabel('pulse square integral [ns]')
            plt.legend()

    return
