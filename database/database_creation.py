import pymysql
import os,sys
import logging

# ssh -f cocov@dpnc.unige.ch -L 3307:localhost:3306 -N

try:
    from IPython import embed
except ImportError:
    import code


    def embed():
        vars = globals()
        vars.update(locals())
        shell = code.InteractiveConsole(vars)
        shell.interact()


def initialise_log( filename):
    logger = logging.getLogger(sys.modules['__main__'].__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def connect_db(logname):
    logging.getLogger(sys.modules['__main__'].__name__)
    logging.info('Connect to CTA_SST1M_UNIGE_DB')
    return pymysql.connect(host='localhost', user='ctaunige', passwd='ctaUniGe_SST1M',
                                 db='CTA_SST1M_UNIGE_DB',port=3307)


def add_pixel():


if __name__ == '__main__':
    logger = initialise_log('database.log')
    connection = connect_db()
    cursor = connection.cursor()
    try:
        embed()
    finally:
        cursor.close()
        connection.close()
        sys.exit()

