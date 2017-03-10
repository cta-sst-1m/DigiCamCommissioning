#!/usr/bin/env python3

# external modules
from optparse import OptionParser
from  yaml import load,dump
import matplotlib.pyplot as plt
import logging,sys
from utils.geometry import generate_geometry_0, generate_geometry
#internal modules
from utils import logger
import numpy as np
from cts_core.cameratestsetup import CTS

if __name__ == '__main__':
    """
    The main call

    """
    parser = OptionParser()

    # Job configuration (the only mandatory option)
    parser.add_option("-y", "--yaml_config", dest="yaml_config",
                      help="full path of the yaml configuration function",
                      default='options/module.yaml')

    # Other options allows to overwrite the yaml_config interactively

    # Output level
    parser.add_option("-v", "--verbose",
                      action="store_false", dest="verbose", default=True,
                      help="move to debug")

    # Steering of the passes
    parser.add_option("-c", "--create_histo", dest="create_histo", action="store_true",
                      help="create the main histogram")

    parser.add_option("-a", "--perform_analysis", dest="perform_analysis", action="store_true",
                      help="perform the analysis of ADC with HV OFF")

    parser.add_option("-d", "--display_results", dest="display_results", action="store_true",
                      help="display the result of the analysis")

    # Logfile basename
    parser.add_option("-l", "--log_file_basename", dest="log_file_basename",
                      help="string to appear in the log file name")

    parser.add_option("-s", "--save_to_csv", dest="save", action="store_true",
                      help="create a csv file for the result of the analysis")

    # Parse the options
    (options, args) = parser.parse_args()

    # Load the YAML configuration
    options_yaml = {}
    with open(options.yaml_config) as f:
        options_yaml.update(load(f))


    # Update with interactive options
    for key,val in options_yaml.items():
        if not (key in options.__dict__.keys()): # and (options.__dict__[key])):
            options.__dict__[key]=val
        else:
            options_yaml[key]=options.__dict__[key]

    __name__ = options.analysis_module
    # Start the loggers
    logger.initialise_logger( options )
    # load the analysis module
    analysis_module = __import__('analysis.%s'%options.analysis_module,
                                 locals=None,
                                 globals=None,
                                 fromlist=[None],
                                 level=0)

    if not hasattr(options,'pixel_list') or options.pixel_list is None or len(options.pixel_list)==0:
        # TODO add usage of digicam and cts geometry to define the list
        options.pixel_list = np.arange(0, options.n_pixels, 1)

    if options.angle_cts is not None:
        cts = CTS('/home/alispach/Documents/PhD/ctasoft/CTS/config/cts_config_' + str(int(options.angle_cts)) + '.cfg', '/home/alispach/Documents/PhD/ctasoft/CTS/config/camera_config.cfg', angle=options.angle_cts, connected=True)
        options.pixel_list = generate_geometry(cts, available_board=None)[1]

    # Some logging
    log = logging.getLogger(sys.modules['__main__'].__name__)
    log.info('\t\t-|> Will run %s with the following configuration:'%options.analysis_module)
    for key,val in options_yaml.items():
        log.info('\t\t |--|> %s : \t %s'%(key,val))
    log.info('-|')

    # Histogram creation
    if options.create_histo:
        # Call the histogram creation function
        log.info('\t\t-|> Create the analysis histogram')
        analysis_module.create_histo(options)

    # Analysis of the histogram
    if options.perform_analysis:
        # Call the histogram creation function
        log.info('\t\t-|> Perform the analysis')
        analysis_module.perform_analysis(options)

    # Display the results of the analysis
    if options.display_results:
        # make the plots non blocking
        plt.ion()
        # Call the histogram creation function
        log.info('-|> Display the analysis results')

        analysis_module.display_results(options)

    if options.save:
        # make the plots non blocking
        plt.ion()
        if hasattr(analysis_module,'save'):
            # Call the histogram creation function
            log.info('-|> Save the analysis results')

            analysis_module.save(options)
        else:
            log.warning('-|> Save function does not exist')
