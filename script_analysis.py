#!/usr/bin/env python3

# external modules
import logging
import sys
from optparse import OptionParser
import matplotlib
import numpy as np
from cts_core.camera import Camera
from cts_core.cameratestsetup import CTS
from utils.geometry import generate_geometry
from  yaml import load
from utils import logger


if __name__ == '__main__':
    """
    The main call

    """
    parser = OptionParser()

    # Job configuration (the only mandatory option)
    parser.add_option("-y", "--yaml_config", dest="yaml_config",
                      help="full path of the yaml configuration",
                      default='options/cts_victor/3_0_gain_evaluation.yaml')

    # Other options allows to overwrite the yaml_config interactively
    parser.add_option("-i", "--yaml_data_config", dest="yaml_data_config",
                      help="full path of the yaml data configuration",
                      default='options/cts_victor/0_generic_config.yaml')

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

    parser.add_option("-s", "--save_to_csv", dest="save", action="store_true",
                      help="create a csv file for the result of the analysis")

    parser.add_option("-r", "--create_report", dest="create_report", action="store_true",
                      help="create a PDF report for the analysis")

    # Logfile basename
    parser.add_option("-l", "--log_file_basename", dest="log_file_basename",
                      help="string to appear in the log file name")


    # Parse the options
    (options, args) = parser.parse_args()


    # Load the YAML configuration
    options_yaml = {}
    options_data_yaml = {}

    with open(options.yaml_data_config) as f:
        options_data_yaml.update(load(f))


    with open(options.yaml_config) as f:
        options_yaml.update(load(f))


    # Update with interactive options
    for key,val in options_data_yaml.items():
        if not (key in options.__dict__.keys()): # and (options.__dict__[key])):
            options.__dict__[key]=val
        else:
            options_data_yaml[key]=options.__dict__[key]

    for key,val in options_yaml.items():
        if not (key in options.__dict__.keys()): # and (options.__dict__[key])):
            options.__dict__[key]=val
        else:
            options_yaml[key]=options.__dict__[key]

    __name__ = options.analysis_module
    # Start the loggers
    logger.initialise_logger( options, options.analysis_module )
    # load the analysis module
    print('--------------------------',options.analysis_module)
    analysis_module = __import__('analysis.%s'%options.analysis_module,
                                 locals=None,
                                 globals=None,
                                 fromlist=[None],
                                 level=0)

    cts_path = '/data/software/CTS/'

    # Configure the CTS angle
    if hasattr(options, 'angle_cts'):
        options.cts = CTS(cts_path + 'config/cts_config_' + str(int(options.angle_cts)) + '.cfg', cts_path + 'config/camera_config.cfg', angle=options.angle_cts, connected=True)

        if not hasattr(options, 'pixel_list'):
            options.pixel_list = generate_geometry(options.cts, available_board=None)[1]

        elif options.pixel_list == 'all':
            options.pixel_list = np.arange(0, 1296, 1)
    else :
        if not hasattr(options, 'pixel_list') or options.pixel_list == 'all':
            options.pixel_list = np.arange(0, 1296, 1)


            
    # Some logging
    log = logging.getLogger(sys.modules['__main__'].__name__)
    log.info('\t\t-|> Will run %s with the following configuration:'%options.analysis_module)
    for key,val in options_data_yaml.items():
        log.info('\t\t |--|> %s : \t %s'%(key,val))
    log.info('-|')
    for key,val in options_yaml.items():
        log.info('\t\t |--|> %s : \t %s'%(key,val))
    log.info('-|')

    print('options.create_histo',options.create_histo)
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
        import matplotlib.pyplot as plt
        plt.ion()
        # Call the histogram creation function
        log.info('-|> Display the analysis results')
        analysis_module.display_results(options)

    if options.save:
        # make the plots non blocking
        import matplotlib.pyplot as plt
        plt.ion()
        if hasattr(analysis_module,'save'):
            # Call the histogram creation function
            log.info('-|> Save the analysis results')
            analysis_module.save(options)
        else:
            log.warning('-|> Save function does not exist')

    if options.create_report:
        # create a report for the analysis
        matplotlib.use('Agg')
        analysis_module.create_report(options)