#!/usr/bin/env python3


# external modules
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

# internal modules
from utils import display, histogram, geometry
import logging,sys
import numpy as np
from utils.logger import TqdmToLogger
from utils.pulse_template import PulseTemplate


from ctapipe import visualization
import scipy
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

    pulse_templates = PulseTemplate(n_pixels=len(options.pixel_list))



    average_pulses = pulse_shape.run(options=options)

    print(average_pulses.shape)

    pulse_templates.interpolate(pulse_data=average_pulses, pixels_id=options.pixel_list)

    pulse_templates.save(options.output_directory + options.histo_filename)


    return