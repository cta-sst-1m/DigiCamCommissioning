# external modules
import numpy as np
from ctapipe.io import zfits

# internal modules
from utils.toy_reader import ToyReader
from utils.event_iterator import EventCounter
from utils.logger import TqdmToLogger


def run(options):


    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    tqdm_out = TqdmToLogger(log, level=logging.INFO)
    event_counter = EventCounter(options.min_event, options.max_event, log)


    for file in options.file_list:

        # Get the file
        _url = options.directory + options.file_basename % file

        if not options.mc:
            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=len(options.scan_level)*options.events_per_level)
        else:
            seed = 0
            inputfile_reader = ToyReader(filename=_url, id_list=[0], seed=seed, max_events=len(options.scan_level)*options.events_per_level, n_pixel=options.n_pixels, events_per_level=options.events_per_level, level_start=options.scan_level[0])

        if options.verbose:
            log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file
        for event, counter in zip(inputfile_reader, event_counter):
            if counter.continuing:
                continue

            for telid in event.r0.tels_with_data:

                # get the data
                data = np.array(list(event.r0.tel[telid].adc_samples.values()), dtype=float)

                # get data of selected pixels
                data = data[options.pixel_list]

                # subtract the pedestals
                data = data - np.mean(data[:,0:options.n_bins_before_signal], axis=-1)[...,None]


                pulse_shapes[level, :, :, 0] += data/options.events_per_level
                pulse_shapes[level, :, :, 1] += (data*data)/options.events_per_level

    pulse_shapes[:, :, :, 1] = np.sqrt((pulse_shapes[:, :, :, 1] - pulse_shapes[:,:,:,0]**2) /options.events_per_level)

    return
