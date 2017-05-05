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
from cts_core.camera import Camera

if __name__ == '__main__':
    """
    The main call

    """
    parser = OptionParser()

    # Job configuration (the only mandatory option)
    parser.add_option("-y", "--yaml_config", dest="yaml_config",
                      help="full path of the yaml configuration function",
                      default='options/module.yaml')

    # Output level
    parser.add_option("-v", "--verbose",
                      action="store_false", dest="verbose", default=True,
                      help="move to debug")

    # Logfile basename
    parser.add_option("-l", "--log_file_basename", dest="log_file_basename",
                      help="string to appear in the log file name")
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

    __name__ = options.render_module
    # Start the loggers
    logger.initialise_logger( options, options.render_module  )
    # load the analysis module
    print('--------------------------',options.render_module)
    render_module = __import__('render.%s'%options.render_module,
                                 locals=None,
                                 globals=None,
                                 fromlist=[None],
                                 level=0)


    if hasattr(options,'angle_cts'):
        cts_path = '/data/software/CTS/'
        #cts_path = '/home/alispach/Documents/PhD/ctasoft/CTS/'
        options.cts = CTS(cts_path + 'config/cts_config_' + str(int(options.angle_cts)) + '.cfg', cts_path + 'config/camera_config.cfg', angle=options.angle_cts, connected=True)
        options.pixel_list = generate_geometry(options.cts, available_board=None)[1]

    if not hasattr(options,'pixel_list') and hasattr(options,'n_pixels'):
        # TODO add usage of digicam and cts geometry to define the list
        options.pixel_list = np.arange(0, options.n_pixels, 1)

    #if not hasattr(options, 'threshold'):
    #    options.threshold = np.arange(options.threshold_min, options.threshold_max, options.threshold_step)

    # Some logging
    log = logging.getLogger(sys.modules['__main__'].__name__)
    log.info('\t\t-|> Will run %s with the following configuration:'%options.render_module)
    for key,val in options_yaml.items():
        log.info('\t\t |--|> %s : \t %s'%(key,val))
    log.info('-|')

    render_module.plot(options)
