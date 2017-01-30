#!/usr/bin/env python3

# external modules
from optparse import OptionParser
from  yaml import load

#internal modules
from utils import logger
from analysis.analyse_hvoff import display_results,create_histo,perform_analysis

if __name__ == '__main__':
    """
    The main call

    """
    parser = OptionParser()
    # Job configuration
    parser.add_option("-q", "--quiet",
                      action="store_false", dest="verbose", default=True,
                      help="don't print status messages to stdout")

    parser.add_option("-c", "--create_histo", dest="create_histo", action="store_true",
                      help="create the main histogram", default=False)

    parser.add_option("-a", "--perform_analysis", dest="perform_analysis", action="store_true",
                      help="perform the analysis of ADC with HV OFF", default=False)

    parser.add_option("-d", "--display", dest="display", action="store_true",
                      help="display the result of the analysis", default=False)

    parser.add_option("-y", "--yaml_config", dest="yaml_config",
                      help="full path of the yaml configuration function",
                      default='/data/software/DigiCamCommissioning/options/hv_off.yaml')

    parser.add_option("-l", "--log_file_basename", dest="log_file_basename",
                      help="string to appear in the log file name",
                      default='test')

    (options, args) = parser.parse_args()

    # Load the YAML configuration
    with open(options.yaml_config) as f: options.update(load(f))

    # Start the loggers
    logger.initialise_logger( options , __name__)

    # Histogram creation
    if options.create_histo:
        # Call the histogram creation function
        create_histo(options)
    # Analysis of the histogram
    if options.perform_analysis:
        # Call the histogram creation function
        perform_analysis(options)
    # Display the results of the analysis
    if options.display:
        # Call the histogram creation function
        display_results(options)
