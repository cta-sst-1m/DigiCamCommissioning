import numpy as np
from ctapipe.io import zfits
from utils.peakdetect import spe_peaks_in_event_list
from utils.toy_reader import ToyReader
import logging
import sys
from utils.logger import TqdmToLogger

from tqdm import tqdm


def visualise(options, view_type='camera'):
    """
    Fill the adcs Histogram out of darkrun/baseline runs
    :param h_type: type of Histogram to produce: ADC for all samples adcs or SPE for only peaks
    :param hist: the Histogram to fill
    :param options: see analyse_spe.py
    :param prev_fit_result: fit result of a previous step needed for the calculations
    :return:
    """
    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    event_number= options.event_min

    if not options.mc:
        log.info('Viewing on DigiCam data')
    else:
        log.info('Viewing on MC data')


    tqdm_out = TqdmToLogger(log, level=logging.INFO)

    geometry =

    for file in options.file_list:
        # Open the file
        _url = options.directory + options.file_basename % file

        if not options.mc:

            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=options.event_max)

        else:

            inputfile_reader = ToyReader(filename=_url, id_list=[0],
                                         max_events=options.event_max,
                                         )

        log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file

        for event in inputfile_reader:

            event_number += 1

            if event_number > options.event_max:

                break

            for telid in event.r0.tels_with_data:

                data = np.array(list(event.r0.tel[telid].adc_samples.values()))

                camera_values = np.sum(data, axis=1)




    return
