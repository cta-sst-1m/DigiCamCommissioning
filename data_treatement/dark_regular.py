import numpy as np
from ctapipe.io import zfits
from utils.mc_events_reader import hdf5_mc_event_source
import logging
import sys
from utils.logger import TqdmToLogger
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.event_iterator import EventCounter
import peakutils


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
    event_counter = EventCounter(options.event_min, options.event_max, level_dc_min=0, level_dc_max=0, level_ac_min=0, level_ac_max=0, event_per_level=options.event_max, event_per_level_in_file=options.event_max, log=log, batch_size=1000)

    thresholds = np.arange(hist.bin_centers[0], hist.bin_centers[-1] + hist.bin_width, hist.bin_width)

    for file in options.file_list:
        # Open the file
        _url = options.directory + options.file_basename % file

        if options.mc:
            log.info('Running on MC data')
            inputfile_reader = hdf5_mc_event_source(url=_url, events_per_dc_level=options.dc_step, events_per_ac_level=options.ac_step, dc_start=options.dc_start, ac_start=options.ac_start, max_events=options.max_event)

        else:
            log.info('Running on DigiCam data')
            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=options.event_max)  #TODO data_type arg does not exist anymore

        log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file
        for event, counter in zip(inputfile_reader, event_counter):
            if counter.continuing:
                continue

            for telid in event.r0.tels_with_data:

                # Take data from zfits
                data = np.array(list(event.r0.tel[telid].adc_samples.values()))
                data = data[options.pixel_list]
                n_peaks = compute_n_peaks(data, thresholds=thresholds, min_distance=options.min_distance)

                if options.debug:

                    plt.figure()
                    plt.step(np.arange(data.shape[1]), data[0], label='%s' % n_peaks[0])
                    plt.legend()
                    plt.show()
                #print(n_peaks)
                hist.data += n_peaks

    return


def compute_n_peaks(data, thresholds, min_distance):

    n_peaks = np.zeros((data.shape[0], len(thresholds)), dtype='int64')
    for i, threshold in enumerate(thresholds):
        n_peaks[..., i] = np.apply_along_axis(compute_peaks, axis=1, arr=data, threshold=threshold, min_distance=min_distance)

    return n_peaks


def compute_peaks(y, threshold, min_distance):

    threshold = threshold / np.max(y)
    n_peaks = len(peakutils.indexes(y, thres=threshold, min_dist=min_distance))

    return n_peaks


