import numpy as np
from ctapipe.io import zfits
from utils.mc_events_reader import hdf5_mc_event_source
from spectra_fit.fit_pulse_shape import p0_func, slice_func, bounds_func, fit_func, fit_func_mc
import logging
import sys
import scipy.optimize
from utils.logger import TqdmToLogger
from iminuit import Minuit
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.event_iterator import EventCounter
from data_treatement.timing import compute_time


def run(options):
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
    event_counter = EventCounter(options.min_event, options.max_event, options.scan_level[0], options.scan_level[-1], options.events_per_level, options.events_per_level_in_file, log)

    cosmic_info = []

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
        for event, counter in zip(inputfile_reader, event_counter):
            if counter.continuing:
                continue

            for telid in event.r0.tels_with_data:

                # Take data from zfits
                data = np.array(list(event.r0.tel[telid].adc_samples.values()))

                min_data = np.min(data, axis=1)
                max_data = np.max(data, axis=1)

                std = np.std(data, axis=1)
                # mask = (std > options.cut)
                mask = ((max_data - min_data) > options.cut)

                # data = data - np.mean(data[:, 0:-1], axis=1)[:, np.newaxis]

                # print (np.sum(data))
                # mask = (np.sum(data) > options.cut)
                pixel_list = np.arange(0, data.shape[0], 1)[mask]
                data = data[mask]

                time = np.arange(0, data.shape[-1], 1) * 4
                errors = np.ones(data.shape[-1]) / np.sqrt(12)

                event_info = {'event_id': counter.event_id, 'time': np.zeros(len(pixel_list)), 'charge': np.zeros(len(pixel_list)), 'pixel': pixel_list}

                for pixel_id, pixel_soft_id in enumerate(pixel_list):
                    fit_result = compute_time(data[pixel_id], time, errors, options, pixel_soft_id)
                    event_info['time'][pixel_id] = fit_result[0]
                    event_info['charge'][pixel_id] = fit_result[1]

                cosmic_info.append(event_info)

    return cosmic_info









