# external modules
import numpy as np
from ctapipe.io import zfits
import sys
import logging

# internal modules
from utils.event_iterator import EventCounter
from utils.logger import TqdmToLogger
from data_treatement.generic import subtract_baseline, integrate
from utils.histogram import Histogram
from utils.mc_reader import hdf5_mc_event_source


def run(options):


    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    tqdm_out = TqdmToLogger(log, level=logging.INFO)
    event_counter = EventCounter(options.min_event, options.max_event, log)

    # Get the baseline parameters if running in per event baseline subtraction mode
    params = None
    if hasattr(options, 'baseline_per_event_limit'):
        params = Histogram(filename=options.output_directory + options.baseline_param_data, fit_only=True).fit_result
        # Initialise the baseline holder
        baseline = np.zeros((len(options.pixel_list),), dtype=float)

    for file in options.file_list:

        # Get the file
        _url = options.directory + options.file_basename % file

        if options.mc:

            inputfile_reader = hdf5_mc_event_source(url=_url, max_event=options.max_event, events_per_dc_level=options.max_event, events_per_ac_level=0, dc_start=0, ac_start=0)

        else:

            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=options.max_event)

        if options.verbose:

            log.debug('--|> Moving to file %s' % _url)

        # Loop over event in this file
        for event, counter in zip(inputfile_reader, event_counter):

            if counter.continuing:
                continue

            for telid in event.r0.tels_with_data:

                # Get the data
                data = np.array(list(event.r0.tel[telid].adc_samples.values()), dtype=float)
                # Get data of selected pixels
                data = data[options.pixel_list]
                # Subtract the baseline and get rid of unwanted pixels
                data, baseline = subtract_baseline(data, counter.event_id, options, params, baseline)
                # Perform integration
                data = integrate(data, options)
                # Initialise pulse_shape
                if counter.event_id == 0:
                    pulse_shape = np.zeros(data.shape)

                pulse_shape += data

    return pulse_shape / event_counter.event_count
