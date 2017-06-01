import logging
import sys

import numpy as np
from ctapipe.io import zfits

from data_treatement.generic import subtract_baseline,integrate

from utils.event_iterator import EventCounter
from utils.histogram import Histogram
from utils.peakdetect import spe_peaks_in_event_list,compute_n_peaks
from utils.mc_reader import hdf5_mc_event_source

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
    if not hasattr(options, 'window_width') or h_type=='MEANRMS':
        options.window_width = 1

    # Get the baseline parameters if running in per event baseline subtraction mode
    params = None
    if hasattr(options, 'baseline_per_event_limit') and not h_type == 'MEANRMS':
        params = Histogram(filename=options.output_directory + options.baseline_param_data, fit_only=True).fit_result
        # Initialise the baseline holder
        baseline = np.zeros((len(options.pixel_list),), dtype = float)

    # Start the logging
    log = logging.getLogger(sys.modules['__main__'].__name__ + '.' + __name__)

    # Initialise the event counter
    event_counter = EventCounter(options.min_event, options.max_event, log, batch_size=options.n_evt_per_batch if hasattr(options,'n_evt_per_batch') else -1)

    # Loop over the events *********************************************************************************************

    for file in options.file_list:
        # Open the file
        _url = options.directory + options.file_basename % file
        if options.mc:
            log.info('Running on MC data')
            inputfile_reader = hdf5_mc_event_source(url=_url, max_event=options.max_event, events_per_dc_level=options.max_event, events_per_ac_level=0, dc_start=0, ac_start=0)
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
                data,baseline = subtract_baseline(data[options.pixel_list], counter.event_id,options, params, baseline)
                # Perform integration
                data = integrate(data, options)

                # Batch data treatement ******************************

                # fisrt batch creation
                if counter.event_id == 0 and counter.batch_size > 0:
                    batch = np.zeros((data.shape[0], counter.batch_size, data.shape[1]), dtype=int)



                # Data treatement ************************************

                if h_type == 'MEANRMS' and hasattr(options, 'baseline_per_event_limit') and baseline is None:
                    # Get the mean and std deviations
                    _baseline = np.mean(data[..., 0:options.baseline_per_event_limit], axis=-1)
                    _rms = np.std(data[..., 0:options.baseline_per_event_limit], axis=-1)
                    hist.data[0, ..., counter.event_id] = _baseline
                    hist.data[1, ..., counter.event_id] = _rms
                #elif h_type == 'STEPFUNCTION':
                    # Get the number of peak above threshold
                #    hist.data += compute_n_peaks(data, thresholds=hist.bin_centers, min_distance=options.min_distance)
                elif h_type == 'RAW':
                    # Fill the full trace
                    hist.fill_with_batch(data)
                else:
                    # Store in batch
                    batch[:, counter.event_id % counter.batch_size , :] = data

                if counter.fill_batch:
                    log.debug('Treating the batch #%d of %d events' % (counter.batch_id, counter.batch_size))
                    # Fill the necessary histo with batch
                    if h_type == 'ADC':
                        hist.fill_with_batch(batch.reshape(batch.shape[0], batch.shape[1] * batch.shape[2]))
                    elif h_type == 'SPE':
                        if type(prev_fit_result).__name__=='ndarray' :
                            hist.fill_with_batch(
                                spe_peaks_in_event_list(batch, prev_fit_result[:, 1, 0], prev_fit_result[:, 2, 0]))
                        elif hasattr(options, 'baseline_per_event_limit'):
                            hist.fill_with_batch(
                                spe_peaks_in_event_list(batch, np.zeros((len(options.pixel_list),)),
                                                        np.ones((len(options.pixel_list),)) *0.7 * options.window_width))
                        else:
                            raise Exception

                    elif h_type == 'STEPFUNCTION':
                        hist.data+= compute_n_peaks(batch.reshape(batch.shape[0],-1), thresholds=hist.bin_centers,min_distance=options.min_distance)
                    else:
                        pass
                    # Reset the batch
                    batch = np.zeros((data.shape[0], counter.batch_size , data.shape[1]), dtype=int)
                    log.debug('Reading  the batch #%d of %d events' % (counter.batch_id, counter.batch_size))



    return

