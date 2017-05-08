import logging,sys
from tqdm import tqdm
import io

__all__ = ['initialise_logger']

def initialise_logger(options,module):
    """
    Setting up the main logger

    :param options:
    :return:
    """

    print(' ')

    # define base logger
    logger = logging.getLogger(sys.modules['__main__'].__name__)
    logger.setLevel(logging.INFO if options.verbose else logging.DEBUG)
    # define file handler and stream handler
    fh = logging.FileHandler('%s_%s.log' % (module,options.log_file_basename))
    fh.setLevel(logging.INFO if options.verbose else logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(level= logging.INFO if options.verbose else logging.DEBUG)
    # define format
    formatter_fh = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s : \t %(message)s')
    formatter_ch = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
    fh.setFormatter(formatter_fh)
    ch.setFormatter(formatter_ch)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)

