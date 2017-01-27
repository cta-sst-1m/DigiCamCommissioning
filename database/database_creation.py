import sqlite3
import os
import logging

def create_main_database(dbname, dblocation,logger):
    """
    Create a new database
    :param dbname: the name of the database (str)
    :param dblocation: the path to its location
    :return:
    """

    if os.path.isfile( dblocation + dbname ):
        answer = input('--|> A database already exist under this name. Do you really want to delete it (yes|no)?')
        if answer != 'yes':
            logger.debug('User did not wanted to delete the %s database'%(dblocation + dbname))
            return
        else:
            logger.warning('User overwrote the %s database'%(dblocation + dbname))

    conn = sqlite3.connect(dblocation+dbname)
    logger.info('%s database created' % (dblocation + dbname))
    conn.close()
    return

def create_table():
    # Create table
    c.execute('''CREATE TABLE
                 (date text, trans text, symbol text, qty real, price real)''')


def initialise_log(logname , filename):
    logger = logging.getLogger(logname)
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

if __name__ == '__main__':
    logger = initialise_log('database_log','database.log')
    create_main_database('test.db','/data/software/DigiCamCommissioning/',logger)