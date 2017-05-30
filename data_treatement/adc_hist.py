import numpy as np
from ctapipe.io import zfits
from utils.peakdetect import spe_peaks_in_event_list
from utils.toy_reader import ToyReader
import logging
import sys
from utils.logger import TqdmToLogger

from tqdm import tqdm

import logging
import sys

import numpy as np
from ctapipe.io import zfits

from utils.event_iterator import EventCounter
from utils.histogram import Histogram
from utils.peakdetect import spe_peaks_in_event_list


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
    :param prev_fit_result: fit result of a previous step
           needed for the calculations                            (ndarray)
    :param baseline: baseline of each pixels                      (ndarray)
    :return:
    """

    # If no integration window has been specified, then set it to 1
    if not hasattr(options, 'window_width'):
        options.window_width = 1

    # Get the baseline parameters if running in per event baseline subtraction mode
    params = None
    if hasattr(options, 'baseline_per_event_limit') and not h_type == 'MEANRMS':
        params = Histogram(filename=options.output_directory + options.baseline_param_data, fit_only=True)

    # Define the integration function
    def integrate_trace(d):
        return np.convolve(d, np.ones(options.window_width, dtype=int), 'valid')

    # Start the logging
    log = logging.getLogger(sys.modules['__main__'].__name__ + '.' + __name__)

    # Initialise the event counter
    # TODO check the options of the event counter
    event_counter = EventCounter(options.min_event, options.max_event, log)

    # Batch counters
    # TODO fold it in the event counter
    # n_batch, batch_num, batch = options.n_evt_per_batch if hasattr(options,'n_evt_per_batch') else -1, 0, None
    _tmp_baseline = None

    # Loop over the input files
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
                # Get the data
                data = np.array(list(event.r0.tel[telid].adc_samples.values()))
                # Get ride off unwanted pixels
                data = data[options.pixel_list]
                # Update whenever batch size is filled
                if counter.batch_fill:
                    log.debug('Treating the batch #%d' % counter.batch_id)
                    # Fill the necessary histo with batch
                    if h_type == 'ADC':
                        hist.fill_with_batch(batch.reshape(batch.shape[0], batch.shape[1] * batch.shape[2]))
                    elif h_type == 'SPE':
                        hist.fill_with_batch(
                            spe_peaks_in_event_list(batch, prev_fit_result[:, 1, 0], prev_fit_result[:, 2, 0]))
                    else :
                        pass
                    # Reset the batch
                    batch = batch_reset(counter.batch_size, data.shape, options)

                # First batch reset
                if counter.event_id == 0:  # TODO Check if it should not be 1
                    batch = batch_reset(counter.batch_size, data.shape, options)

                # Treat the case where the baseline is computed from the event itself and is not known
                if hasattr(options, 'baseline_per_event_limit') and baseline is None:
                    # Get the mean and std deviations
                    _baseline = np.mean(data[..., 0:options.baseline_per_event_limit], axis=-1)
                    _rms = np.std(data[..., 0:options.baseline_per_event_limit], axis=-1)
                    if h_type == 'MEANRMS':
                        # Case where we want to evaluate the baseline parameters
                        hist.data[0, ..., counter.event_id] = _baseline
                        hist.data[1, ..., counter.event_id] = _rms

                    elif params is not None:
                        # Case where the baseline parameters have been evaluated already

                        # Get the pixel for which the rms is good, ie. there have been no huge fluctuation in the
                        # samples in which it was evaluated
                        ind_good_baseline = (_rms - params[:, 2]) / params[:, 3] < 0.5
                        # If at least one event was computed, only update the previous baseline for the pixel with no
                        # large fluctuations
                        if counter.event_id > 0:
                            _tmp_baseline[ind_good_baseline] = _baseline[ind_good_baseline]
                        else:
                            _tmp_baseline = _baseline
                        # Subtract the baseline
                        data = data - _tmp_baseline[:, None]
                # Treat the case where the baseline has been specified
                elif baseline is not None:
                    data = data - baseline[:, None]

                # For all h_type except MEANRMS, update the batch
                if not h_type == 'MEANRMS':
                    batch[:, counter.event_id % n_batch, :] = np.apply_along_axis(integrate_trace, -1,
                                                                                  data[...,
                                                                                  options.baseline_per_event_limit:-1])
                '''
                # TODO Check what was the use for this :
                else:
                    hist.fill_with_batch(data)
                '''

    return


def run(hist, options, h_type='ADC', prev_fit_result=None, baseline=None):
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
    n_evt, n_batch, batch_num, max_evt = 0, options.n_evt_per_batch, 0, options.evt_max
    _tmp_baseline = None
    batch = None

    if not options.mc:
        log.info('Running on DigiCam data')
    else:
        log.info('Running on MC data')
    params=None
    if hasattr(options, 'baseline_per_event_limit') and not h_type=='MEANRMS':

        params = np.load(options.output_directory + options.baseline_param_data)['params']
        #print(params)
        #print(params.keys())
        #print(params[:, 1])
        #print(params[:, 2])
        #print(params[:, 3])


    def integrate_trace(d):
        return np.convolve(d, np.ones((options.window_width), dtype=int), 'valid')

    pbar = tqdm(total=max_evt)
    tqdm_out = TqdmToLogger(log, level=logging.INFO)

    log.debug('Treating the batch #%d of %d events' % (batch_num, n_batch))
    for file in options.file_list:
        # Open the file
        _url = options.directory + options.file_basename % file
        if not options.mc:
            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=options.evt_max)
        else:
            inputfile_reader = ToyReader(filename=_url, id_list=[0],
                                         max_events=options.evt_max,
                                         n_pixel=options.n_pixels)

        log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file
        for event in inputfile_reader:
            n_evt += 1
            if n_evt > max_evt:
                break

            pbar.update(1)

            #print(event.dl0)
            #break

            for telid in event.r0.tels_with_data:
                #print('hello')

                data = np.array(list(event.r0.tel[telid].adc_samples.values()))
                # Get ride off unwanted pixels
                data = data[options.pixel_list]

                if n_evt % n_batch == 0:
                    log.debug('Treating the batch #%d of %d events' % (batch_num, n_batch))
                    # Update adc histo
                    if h_type == 'ADC':
                        #print(batch[0,0])
                        pass
                        #hist.fill_with_batch(batch.reshape(batch.shape[0], batch.shape[1] * batch.shape[2]))
                    elif h_type == 'SPE':
                        hist.fill_with_batch(
                            spe_peaks_in_event_list(batch, prev_fit_result[:, 1, 0], prev_fit_result[:, 2, 0]))
                    # Reset the batch
                    if hasattr(options, 'window_width'):
                        batch = np.zeros((data.shape[0], n_batch, data.shape[1] - options.window_width + 1), dtype=int)
                        if hasattr(options, 'baseline_per_event_limit'):
                            batch = np.zeros((data.shape[0], n_batch, data.shape[1]-options.window_width-options.baseline_per_event_limit),dtype=float)
                    else:
                        pass
                        #batch = np.zeros((data.shape[0], n_batch, data.shape[1]), dtype=int)
                    batch_num += 1
                    log.debug('Reading  the batch #%d of %d events' % (batch_num, n_batch))
                # Get the data

                #print(data)
                if n_evt==1:
                    if hasattr(options,'window_width'):
                        batch = np.zeros((data.shape[0], n_batch, data.shape[1]-options.window_width+1),dtype=int)
                        if hasattr(options, 'baseline_per_event_limit'):

                            batch = np.zeros((data.shape[0], n_batch, data.shape[1]-options.window_width-options.baseline_per_event_limit),dtype=float)
                    else:
                        batch = np.zeros((data.shape[0], n_batch, data.shape[1]),dtype=int)
                if hasattr(options,'window_width'):
                    #print('hello')
                    if hasattr(options, 'baseline_per_event_limit') and baseline is None:
                        #print('hello')
                        _baseline = np.mean(data[...,0:options.baseline_per_event_limit], axis=-1)
                        _rms = np.std(data[...,0:options.baseline_per_event_limit], axis=-1)
                        if h_type == 'MEANRMS':
                            hist[0][...,n_evt-1]=_baseline
                            hist[1][...,n_evt-1]=_rms
                        # get the indices where baseline is good
                        #print('hello')

                        if params is not None:

                            ind_good_baseline = (_rms - params[:,2])/params[:,3] < 0.5
                            #print(params[:,2])
                            if n_evt > 1:
                                _tmp_baseline[ind_good_baseline] = _baseline[ind_good_baseline]
                                #_tmp_baseline[~ind_good_baseline] = 10000

                            else:
                                _tmp_baseline = _baseline
                            #_tmp_baseline = baseline
                            data = data - _tmp_baseline[:, None]
                    elif baseline is not None:
                        data = data - baseline[:, None]
                    if not  h_type == 'MEANRMS':
                        batch[:,n_evt%n_batch,:]=np.apply_along_axis(integrate_trace,-1,data[...,options.baseline_per_event_limit:-1])
                        #print(batch[:,n_evt%n_batch][1])
                        #print(batch.shape,n_evt%n_batch)

                else:
                    #print(np.sum(np.sum(data, axis=0), axis=0))
                    #print(hist.data.shape)
                    #print(data.shape)
                    hist.fill_with_batch(data)

    return
