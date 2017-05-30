import numpy as np
from ctapipe.io import zfits
import logging,sys
from tqdm import tqdm
from utils.logger import TqdmToLogger
from utils.toy_reader import ToyReader
from data_treatement.trigger import compute_cluster_trace


def run(time_list, options, trigger_mask=None):


    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    pbar = tqdm(total=options.max_events)
    event_count = 0

    for file in options.file_list:

        # read the file
        _url = options.directory + options.file_basename % file

        if not options.mc:
            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=options.max_events)

        else:
            seed = 0
            inputfile_reader = ToyReader(filename=_url, id_list=[0], n_pixel=len(options.pixel_list), \
                                         events_per_level=options.events_per_level, seed=seed, \
                                         max_events=len(options.nsb_rate)*options.events_per_level, level_start=0)

        if options.verbose:
            log.debug('--|> Moving to file %s' % _url)

        for event in inputfile_reader:

            for telescope_id in event.r0.tels_with_data:

                time = event.r0.tel[telescope_id].local_camera_clock
                time_list.append(time)

                if trigger_mask is not None:

                    data = np.array(list(event.r0.tel[telescope_id].adc_samples.values()))
                    baseline = np.mean(data[..., data.shape[-1]-options.baseline_window_width:-1], axis=-1)
                    data = data - baseline[:, np.newaxis]

                    cluster_trace, patch_trace = compute_cluster_trace(data, options)
                    cluster_max = np.max(cluster_trace)
                    trigger_mask.append(cluster_max > options.threshold)

                event_count += 1

            pbar.update(1)

    return trigger_mask
