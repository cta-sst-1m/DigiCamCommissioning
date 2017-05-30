import logging
import sys

import numpy as np
from ctapipe.io import zfits

from data_treatement.generic import subtract_baseline,integrate

from utils.event_iterator import EventCounter
from utils.histogram import Histogram
from utils.peakdetect import spe_peaks_in_event_list,compute_n_peaks

def batch_reset(n_batch, shape, options):
    """
    Create/reset the batches to be filled
    :param n_batch: number of events in a batch    (int)
    :param shape: data shape                       (tuple)
    :param options: configuration container        (yaml container)
    :return: the batch                             (ndarray)
    """
    batch = np.zeros((shape[0], n_batch, shape[1] - options.window_width + 1), dtype=int)
    if hasattr(options, 'baseline_per_event_limit'):
        batch = np.zeros(
            (shape[0], n_batch, shape[1] - options.window_width - options.baseline_per_event_limit),
            dtype=float)
    return batch


def run(hist, options, h_type='ADC', prev_fit_result=None, baseline=None):
    """
    Fill the adcs Histogram for all pixels
    :param hist: the Histogram to fill                            (histogram.Histogram)
    :param options:  configuration container                      (yaml container)
    :param h_type: type of Histogram to produce:                  (str)
               ADC for all samples adcs or
               SPE for peak finding
               MEANRMS for mean and rms of the samples
               in the baseline samples
               STEPFUNCTION for peakfinding with thresholds
    :param prev_fit_result: fit result of a previous step
           needed for the calculations                            (ndarray)
    :param baseline: baseline of each pixels                      (ndarray)
    :return:
    """

    # Initialisation phase *********************************************************************************************

    # If no integration window has been specified, then set it to 1
    if not hasattr(options, 'window_width'):
        options.window_width = 1

    # Get the baseline parameters if running in per event baseline subtraction mode
    params = None
    if hasattr(options, 'baseline_per_event_limit') and not h_type == 'MEANRMS':
        params = Histogram(filename=options.output_directory + options.baseline_param_data, fit_only=True)
        # Initialise the baseline holder
        baseline = np.zeros((options.pixel_list,), dtype = float)

    # Start the logging
    log = logging.getLogger(sys.modules['__main__'].__name__ + '.' + __name__)

    # Initialise the event counter
    # TODO check the options of the event counter and build-in batch
    event_counter = EventCounter(options.min_event, options.max_event, log)
    setattr(event_counter,'batch',None)
    setattr(event_counter,'batch_num',0)
    setattr(event_counter,'n_batch', options.n_evt_per_batch if hasattr(options,'n_evt_per_batch') else -1)


    # Loop over the events *********************************************************************************************

    for file in options.file_list:
        # Open the file
        _url = options.directory + options.file_basename % file
        if options.mc:
            log.info('Running on MC data')
            # TODO Update (Cyril)
            inputfile_reader = None
            # ToyReader(filename=_url, id_list=[0],max_events=options.evt_max,n_pixel=options.n_pixels)
        else:
            log.info('Running on DigiCam data')
            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=options.max_event)

        log.debug('--|> Moving to file %s' % _url)

        # Loop over event in this file
        for event, counter in zip(inputfile_reader, event_counter):
            if counter.continuing: continue
            # Loop over the telescopes
            for telid in event.r0.tels_with_data:

                # Data handeling *************************************

                data = np.array(list(event.r0.tel[telid].adc_samples.values()))
                # Subtract the baseline and get rid of unwanted pixels
                data = subtract_baseline(data[options.pixel_list], options, params, baseline)
                # Perform integration
                data = integrate(data, options)

                # Batch data treatement ******************************

                if counter.n_batch > 0 and counter.event_id % counter.n_batch == 0:
                    if counter.event_id == 0:
                        # TODO Check if it should not be 1
                        batch = batch_reset(counter.n_batch, data.shape, options)
                    else:
                        log.debug('Treating the batch #%d of %d events' % (counter.batch_num, counter.n_batch))
                        # Fill the necessary histo with batch
                        if h_type == 'ADC':
                            hist.fill_with_batch(batch.reshape(batch.shape[0], batch.shape[1] * batch.shape[2]))
                        elif h_type == 'SPE':
                            hist.fill_with_batch(
                                spe_peaks_in_event_list(batch, prev_fit_result[:, 1, 0], prev_fit_result[:, 2, 0]))
                        else:
                            pass
                        # Reset the batch
                        batch = batch_reset(counter.n_batch, data.shape, options)
                        # Increment the batch count
                        counter.batch_num += 1
                        log.debug('Reading  the batch #%d of %d events' % (counter.batch_num, counter.n_batch))

                # Data treatement ************************************

                if h_type == 'MEANRMS' and hasattr(options, 'baseline_per_event_limit') and baseline is None:
                    # Get the mean and std deviations
                    _baseline = np.mean(data[..., 0:options.baseline_per_event_limit], axis=-1)
                    _rms = np.std(data[..., 0:options.baseline_per_event_limit], axis=-1)
                    hist.data[0, ..., counter.event_id] = _baseline
                    hist.data[1, ..., counter.event_id] = _rms
                elif h_type == 'STEPFUNCTION':
                    # Get the number of peak above threshold
                    # TODO check if this can be batched + threshold
                    hist.data += compute_n_peaks(data, thresholds=thresholds, min_distance=options.min_distance)
                elif h_type == 'RAW':
                    # Fill the full trace
                    hist.fill_with_batch(data)
                else:
                    # Store in batch
                    batch[:, counter.event_id % n_batch, :] = data


    return

