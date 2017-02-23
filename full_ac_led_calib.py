
## First run analyse_hvoff script


if __name__ == '__main__':
    """
    The main call

    """
    parser = OptionParser()

    # Job configuration (the only mandatory option)
    parser.add_option("-y", "--yaml_config", dest="yaml_config",
                      help="full path of the yaml configuration function",
                      default='/data/software/DigiCamCommissioning/options/full_gain.yaml')

    # Other options allows to overwrite the yaml_config interactively

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
        if not ((key in options.__dict__.keys()) and (options.__dict__[key])):
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

    # Some logging
    log = logging.getLogger(sys.modules['__main__'].__name__)
    log.info('\t\t-|> Will run %s with the following configuration:'%options.analysis_module)
    for key,val in options_yaml.items():
        log.info('\t\t |--|> %s : \t %s'%(key,val))
    log.info('-|')

    analysis_hvoff.create_histo(options['hvoff'])
    analysis_hvoff.perform_analysis(options['hvoff'])
    analysis_spe.create_histo(options['spe'])
    analysis_spe.perform_analysis(options['spe'])
    analysis_mpe.create_histo(options['mpe'])
    analysis_gain.create_histo(options['gain'])
    analysis_gain.perform_analysis(options['gain'])


