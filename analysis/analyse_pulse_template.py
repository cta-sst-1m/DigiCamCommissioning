#!/usr/bin/env python3


# external modules
import random

# internal modules
from utils.pulse_template import PulseTemplate
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
    pulse_templates = PulseTemplate(n_pixels=len(options.pixel_list))
    average_pulses = pulse_shape.run(options=options)

    # Interpolate data points to get pulse template (spline)
    pulse_templates.interpolate(pulse_data=average_pulses, pixels_id=options.pixel_list)

    # Save pulse templates
    pulse_templates.save(options.output_directory + options.histo_filename)

    return


def perform_analysis(options):

    pulse_templates = PulseTemplate(filename=options.output_directory + options.histo_filename)

    pulse_templates.save(options.output_directory + options.histo_filename)

    return


def display_results(options):

    pulse_templates = PulseTemplate(filename=options.output_directory + options.histo_filename)

    pixels_id = random.sample(options.pixel_list, 5)

    pulse_templates.display(pixels_id, derivative=0, moment=1)
    pulse_templates.display(pixels_id, derivative=1, moment=1)
    pulse_templates.display(pixels_id, derivative=0, moment=2)
    pulse_templates.display(pixels_id, derivative=1, moment=2)

    return 