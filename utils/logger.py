import logging,sys

__all__ = ['initialise_logger']

def initialise_logger(options):
    """
    Setting up the main logger

    :param options:
    :return:
    """
    # define base logger
    logging.basicConfig(level= logging.INFO if options.verbose else logging.DEBUG)
    logger = logging.getLogger(sys.modules['__main__'].__name__)
    # define file handler and stream handler
    fh = logging.FileHandler('%s_%s.log' % (options.analysis_module,options.log_file_basename))
    fh.setLevel(logging.INFO if options.verbose else logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(level= logging.INFO if options.verbose else logging.DEBUG)
    # define format
    formatter_fh = logging.Formatter('%(asctime)s \t %(name)s \t %(levelname)s : \t %(message)s')
    formatter_ch = logging.Formatter('%(name)s.%(levelname)s :BLA \t %(message)s')
    fh.setFormatter(formatter_fh)
    ch.setFormatter(formatter_ch)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return
