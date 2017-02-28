import pymysql
import os,sys
import logging
import csv
import re
import cts.cameratestsetup as camtestsetup
import pymongo
from pymongo import MongoClient

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

def create_database():
    client = MongoClient()
    db = client.test_sst1M
    '''
    PIXELS
    '''
    sipms = db.sipm
    sipm_calibs = db.sipm_calibs
    lightguides = db.lightguide

    pixels = db.pixel
    pixel_calibs = db.pixel_calib
    '''
    MODULES
    '''
    scbds = db.scbd
    scbd_calibs = db.scbd_calibs
    preampbds = db.preampbd
    preampbd_calibs = db.preampbd_calib

    modules = db.module

    '''
    PDP
    '''
    pdps = db.pdp

    '''
    DIGICAM
    '''
    fadcs = db.fadc
    fadc_calibs = db.fadc_calib
    trgbds = db.trgbd
    trgbd_calibs = db.trgbd_calib
    backplanes = db.backplane
    mcrs = db.mcr
    digicam = db.digicam

    '''
    CAMERA
    '''

    cameras = db.camera

    '''
    TELESCOPE
    '''
    telescope = db.telescope


def load_sipm_dict(pathes=['/home/cocov/CTA/Database/161128_S10943-5477InspectionData(1-375).csv']):
    sipm_dict = {}
    for path in pathes:
        with open(path, 'r') as csvfile:
            sipm_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            line = next(sipm_reader)
            i_type = int(re.sub('[^0-9]', '', line[1]))
            # Skip one line
            line = next(sipm_reader);
            line = next(sipm_reader)
            f_test_temperature = float(line[1].split('Ta=')[1].split('degC')[0])
            # Skip few lines
            line = next(sipm_reader);
            line = next(sipm_reader);
            line = next(sipm_reader)
            line = next(sipm_reader);
            line = next(sipm_reader);
            line = next(sipm_reader)
            line = next(sipm_reader)
            sipm = {}
            i = -1
            for row in sipm_reader:
                if len(row) == 0 or row[2] == '' or row[0] == 'Serial No.': continue
                if row[0] != '':
                    i = int(row[0])
                    sipm_dict[(i_type << 11) | i] = {'i_SN': int(row[0]), 'f_VA1': float(row[2]), \
                                                   'f_Vvariation': float(row[3]), 'f_iA1': float(row[4]), \
                                                   'i_type': i_type, 'f_test_temperature': f_test_temperature}
                else:
                    sipm_dict[(i_type << 11) | i]['f_V%s' % row[1]] = float(row[2])
                    sipm_dict[(i_type << 11) | i]['f_i%s' % row[1]] = float(row[4])
    return sipm_dict

def load_pixel_dict(cam_id,path):
    pixel_dict = {}
    f = open(path)
    l = f.readline()
    while l :
        l = f.readline()
        if l=='':continue
        values = l.split('\n')[0].split('\t')
        sipm_id = (int(re.sub('[^0-9]', '', values[0]))<<11) | int(values[1])
        tmp_pixid =(cam_id<< 18) | (int(values[2])<< 11)  | int(values[3])
        pixel_dict[tmp_pixid]={'sipm_id':sipm_id,'lightguide_id':tmp_pixid,'module_id':(cam_id << 7) | int(values[2])}
    return pixel_dict

def load_scb_dict(path):
    scb_dict = {}
    f = open(path)
    l = f.readline()
    while l :
        l = f.readline()
        if l=='':continue
        values = l.split('\n')[0].split('\t')
        scb_dict[(int(re.sub('[^0-9]', '', values[4]))<<7)|int(values[5])]={'scb_id':(int(re.sub(r'[^0-9]', '', values[4]))<<7)|int(values[5])}
    return scb_dict


def load_preamp_dict(path):
    preamp_dict = {}
    f = open(path)
    l = f.readline()
    while l:
        l = f.readline()
        if l == '': continue
        values = l.split('\n')[0].split('\t')
        preamp_dict[(int(re.sub('[^0-9]', '', values[6])) << 7) | int(values[7])] = {
            'preamp_id': (int(re.sub('[^0-9]', '', values[6])) << 7) | int(values[7])}
    return preamp_dict

def load_module_dict(cam_id,path):
    module_dict = {}
    f = open(path)
    l = f.readline()
    while l:
        l = f.readline()
        if l=='':continue
        values = l.split('\n')[0].split('\t')
        module_id = (cam_id << 7) | int(values[2])
        module_dict[module_id]={'scb_id':(int(re.sub('[^0-9]', '', values[4])) << 7) | int(values[5]), \
                                'preamp_id': (int(re.sub('[^0-9]', '', values[6])) << 7) | int(values[7]),\
                                }
    return module_dict
'''
def load_module_dict(cam_id,path):
    f = open(path)
    l = f.readline()
    module_dict = []
    while l :
        l = f.readline()
        values = l.split('\n')[0].split('\t')
        if not (int(values[2]) in module_dict.keys()):
            module_dict[int( values[2] )]={}
            # now fill it
            module_dict[int( values[2] )]['module_ID'] = int( values[2] )
            module_dict[int( values[2] )]['scbd_ID']   = ( int( values[4].split('A')[0] ) << 7 ) | int( values[5] )
            module_dict[int( values[2] )]['preampbd_ID'] = ( int( values[6] ) << 7 ) | int( values[7] )
            module_dict[int(values[2])]['pixels']=[]

        module_dict[int(values[2])]['pixels'].append({})
        module_dict[int(values[2])]['pixels'][-1]['pixel_id']= cam_id << 11 | int(values[3])
        module_dict[int(values[2])]['pixels'][-1]['hvref']= float( values[8])
        module_dict[int(values[2])]['pixels'][-1]['gain']= float( values[9])
        module_dict[int(values[2])]['pixels'][-1]['sigma_e']= float( values[10])
        module_dict[int(values[2])]['pixels'][-1]['sigma_1']= float( values[11])
        module_dict[int(values[2])]['pixels'][-1]['XT']= float( values[12])
        module_dict[int(values[2])]['pixels'][-1]['dark_count_rate']= float( values[13])
        module_dict[int(values[2])]['pixels'][-1]['mu']= float( values[14])
        module_dict[int(values[2])]['pixels'][-1]['avg_temp']= float( values[15])
        module_dict[int(values[2])]['pixels'][-1]['sensor_ID']= ( int( values[0].split('A')[0] ) << 11 ) | int( values[1] )
        module_dict[int(values[2])]['pixels'][-1]['lightguide_ID']= cam_id << 11 | int(values[3])
    return module_dict
'''



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

def connect_db():
    logging.getLogger(sys.modules['__main__'].__name__)
    logging.info('Connect to CTA_SST1M_UNIGE_DB')
    return pymysql.connect(host='localhost', user='ctaunige', passwd='ctaUniGe_SST1M',
                                 db='CTA_SST1M_UNIGE_DB',port=3307)

def add_scb(cursor,connection,scb_dict):
    log=logging.getLogger(sys.modules['__main__'].__name__)
    try:
        if not cursor.execute("SELECT * FROM  SCBD  WHERE `SCBD_ID`='%s'"%(scb_dict['scbd_ID'])):
            cursor.execute("INSERT INTO SCBD VALUES (%s)"%(scb_dict['scbd_ID']))
            log.info("Inserted SCB %s"%scb_dict['scbd_ID'])
        else:
            log.info("SCB %s exist already"%scb_dict['scbd_ID'])
            # Check the parameters
        connection.commit()
    except:
        connection.rollback()

def add_preamp(cursor,connection,preamp_dict):
    log=logging.getLogger(sys.modules['__main__'].__name__)
    try:
        if not cursor.execute("SELECT * FROM  PREAMPBD  WHERE `PREAMPBD_ID`='%s'"%(scb_dict['scbd_ID'])):
            cursor.execute("INSERT INTO PREAMPBD VALUES (%s)"%(preamp_dict['preampbd_ID']))
            log.info("Inserted PREAMPBD %s"%preamp_dict['preampbd_ID'])
        else:
            log.info("PREAMPBD %s exist already"%preamp_dict['preampbd_ID'])
        connection.commit()
    except:
        connection.rollback()


def add_pixel(cursor,connection,pixel_dict):
    log=logging.getLogger(sys.modules['__main__'].__name__)

    try:
        if not cursor.execute("SELECT * FROM  PIXEL  WHERE `PIXEL_ID`='%s'"%(pixel_dict['pixel_ID'])):
            cursor.execute("INSERT INTO PIXEL VALUES (%s,%s,%s,%s,%s)"%(pixel_dict['pixel_ID'],pixel_dict['lightguide_ID'],
                                                                pixel_dict['sipm_ID'],pixel_dict['module_ID'],
                                                                pixel_dict['slot_ID']))
            log.info("Inserted pixel %s"%pixel_dict['pixel_ID'])
        else:
            log.info("Pixel %s exist already"%pixel_dict['pixel_ID'])
            # Check the parameters

        connection.commit()
    except:
        connection.rollback()


def add_module(cursor,connection,module_dict):
    log=logging.getLogger(sys.modules['__main__'].__name__)
    try:
        if not cursor.execute("SELECT * FROM  MODULE  WHERE `MODULE_ID`='%s'"%(module_dict['module_ID'])):
            cursor.execute("INSERT INTO MODULE VALUES (%s,%s,%s,%s,%s)"%(module_dict['module_ID'],module_dict['scbd_ID'],
                                                                module_dict['preampbd_ID'],module_dict['fadc_ID'],
                                                                module_dict['slot_ID']))

            log.info("Inserted module %s"%module_dict['module_ID'])
        for pix in module_dict['pixels']:
            add_pixel(cursor,connection,pix)

        add_scb(cursor,connection,{'scbd_ID':module_dict['scbd_ID']})
        add_preamp(cursor,connection,{'preampbd_ID':module_dict['preampbd_ID']})

        connection.commit()
    except:
        connection.rollback()


def add_mounting_info(cam_id,dict_db):
    cts0 = camtestsetup.CTS('config/cts_config_0.cfg',
                           'config/camera_config.cfg', angle = 0., connected = False)
    cts120 = camtestsetup.CTS('config/cts_config_120.cfg',
                           'config/camera_config.cfg', angle = 120., connected = False)
    cts240 = camtestsetup.CTS('config/cts_config_240.cfg',
                           'config/camera_config.cfg', angle = 240., connected = False)


    camera = cts0.camera

    for module in range(camera.Modules):
        dict_db['modules'][(cam_id << 7) | module.ID]['fadc_ID']= (cam_id << 5) | module.fadc
        dict_db['modules'][(cam_id << 7) | module.ID]['mcr_ID']= (cam_id << 2) | module.sector
        dict_db['modules'][(cam_id << 7) | module.ID]['camera_ID']= cam_id
        dict_db['modules'][(cam_id << 7) | module.ID]['digicam_ID']= cam_id

    dict_db['led_pixel'] = {}
    '''
    for pixel in range(camera.Pixels):
        pixel_id = (cam_id<< 18) | (pixel.module<< 11)  | pixel.ID
        for ctsStr,ctsIt in {'CTS0':cts0,'CTS120':cts120,}
        dict_db['pixel'][pixel_id]['CTS0']['dc_led_id']=cts0.LEDs['DC'][cts0.pixel_to_led[pixel.ID]]
        dict_db['pixel'][pixel_id]['CTS120']['dc_led_id']=cts120.LEDs['DC'][cts120.pixel_to_led[pixel.ID]]
        dict_db['pixel'][pixel_id]['CTS240']['dc_led_id']=cts240.LEDs['DC'][cts240.pixel_to_led[pixel.ID]]
    '''


    try:
        dict_db['camera'][cam_id] = {}
    except KeyError:
        dict[key] = 1
    dict_db['camera'][cam_id]={}



if __name__ == '__main__':
    sipm_dict = load_sipm_dict(['/home/cocov/CTA/Database/141020_S10943-3739(X)InspectionData(1-1217).csv',
                                '/home/cocov/CTA/Database/150608_S10943-3739(X)InspectionData(1281-1308).csv',
                                '/home/cocov/CTA/Database/141110_S10943-3739(X)InspectionData(1218-1250).csv',
                                '/home/cocov/CTA/Database/161128_S10943-5477InspectionData(1-375).csv',
                                '/home/cocov/CTA/Database/170213_S10943-5477InspectionData(971-1555).csv',
                                '/home/cocov/CTA/Database/150323_S10943-3739(X)InspectionData(1253-1277).csv'])

    f = open('dump_sipm.txt','w')

    for k in ['i_type', 'i_SN','f_mean']:
        f.write('%s\t'%k)
    f.write('\n')
    for si in sipm_dict.keys():
        for k in ['i_type', 'i_SN']:
            f.write('%d\t' %sipm_dict[si][k] )
        mean = (float(sipm_dict[si]['f_VA1'])+float(sipm_dict[si]['f_VA2'])+float(sipm_dict[si]['f_VB1'])+float(sipm_dict[si]['f_VB2']))/4
        f.write('%f\n' % mean)

    f.close()

    #pixel_dict = load_pixel_dict(0,'/home/cocov/CTA/Database/OpticalCalib_DB.txt')
    #f = open('pixel.txt')
    #for sipm in
    '''
    logger = initialise_log('database.log')
    connection = connect_db()
    cursor = connection.cursor()
    modules_dict = load_module_dict(0,'/home/cocov/CTA/Database/OpticalCalib_DB.txt')
    try:
        embed()
    finally:
        cursor.close()
        connection.close()
        sys.exit()
    '''
