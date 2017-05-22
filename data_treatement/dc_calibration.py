import numpy as np
from ctapipe.io import zfits
from utils.mc_events_reader import hdf5_mc_event_source
import logging
import sys
import peakutils
from utils.logger import TqdmToLogger

from tqdm import tqdm


def run(hist, options):
    """
    Fill the adcs Histogram out of darkrun/baseline runs
    :param h_type: type of Histogram to produce: ADC for all samples adcs or SPE for only peaks
    :param hist: the Histogram to fill
    :param options: see analyse_spe.py
    :param prev_fit_result: fit result of a previous step needed for the calculations
    :return:
    """
    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    # Reading the file
    event_number = 0
    dark_event_number = 0
    progress_bar = tqdm(total=options.events_per_level * len(options.scan_level))
    batch = []
    pulse_shape = np.zeros((len(options.pixel_list), options.n_bins, ))
    level = 0

    for file in options.file_list:
        # Open the file
        _url = options.directory + options.file_basename % file

        if options.mc:
            log.info('Running on MC data')
            inputfile_reader = hdf5_mc_event_source(url=_url, events_per_dc_level=options.dc_step, events_per_ac_level=options.ac_step, dc_start=options.dc_start, ac_start=options.ac_start, max_events=options.max_event)

        else :
            log.info('Running on DigiCam data')
            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=options.max_event)  #TODO data_type arg does not exist anymore

        log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file
        for event in inputfile_reader:

            if event_number > options.max_event:
                break

            for telid in event.r0.tels_with_data:

                # Take data from zfits
                data = np.array(list(event.r0.tel[telid].adc_samples.values()))
                data = data[options.pixel_list]

                if options.hist_type == 'nsb':
                    #data = data
                    #if event_number % options.events_per_level == 0:
                    #    batch = data
                    #batch = np.concatenate((batch, data), axis=1)
                    hist.fill_with_batch(data, indices=(level, ))

                elif options.hist_type == 'nsb+signal':

                    if level==0:
                        pulse_shape += data/options.events_per_level
                        dark_event_number += 1
                    hist.fill(np.max(data, axis=1), indices=(level, ))
                    #data = np.max(data, axis=1).reshape(data.shape[0], 1)
                    #if event_number == 0:
                    #    batch = data
                    #batch = np.append(batch, data, axis=1)

                else:

                    log.error('Unknown hist_type = %s' % options.hist_type)

                event_number += 1
                if event_number % options.events_per_level == 0:
                    log.debug('Completed level index : %d, dc level : %d' % (level, options.scan_level[level]))
                    level += 1

                #if event_number % options.events_per_batch == 0:
                #    log.debug('Appending batch')
                #    hist.fill_with_batch(batch, indices=(level,))
                #    batch = []

                #print(pulse_shape[0])

                progress_bar.update(1)

    if options.hist_type == 'nsb+signal':

        return pulse_shape

