import numpy as np
from ctapipe.io import zfits
import logging, sys
from tqdm import tqdm
from utils.logger import TqdmToLogger
from utils.toy_reader import ToyReader
from utils.mc_events_reader import hdf5_mc_event_source
from cts_core.camera import Camera


def run(trigger_rate_camera, options, min_evt=0, cluster_hist=None, patch_hist=None, max_cluster_hist=None,
        time_cluster_hist=None):
    # Few counters

    def integrate_trace(d):

        return np.convolve(d, np.ones((options.baseline_window_width), dtype=int), 'valid')

    event_number, event_min, event_max = 0, options.evt_min, options.evt_max
    level = 0
    time = np.zeros(len(options.nsb_rate))
    baseline = []
    baseline_computed = False
    baseline_counter = 0

    log = logging.getLogger(sys.modules['__main__'].__name__ + '.' + __name__)
    progress_bar = tqdm(total=options.events_per_level * len(options.nsb_rate))
    tqdm_out = TqdmToLogger(log, level=logging.INFO)

    for file in options.file_list:

        # read the file
        _url = options.directory + options.file_basename % file

        if not options.mc:
            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=options.evt_max)

        else:
            """
            inputfile_reader = ToyReader(filename=_url, id_list=[0], n_pixel=len(options.pixel_list),
                                         events_per_level=options.events_per_level, seed=seed,
                                         max_events=len(options.nsb_rate) * options.events_per_level, level_start=0)
            """
            inputfile_reader = hdf5_mc_event_source(url=_url, events_per_ac_level=0, events_per_dc_level=options.events_per_level, dc_start=0, ac_start=0, max_events=options.evt_max)

        if options.verbose:
            log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file

        for event in inputfile_reader:
            if event_number > event_max:
                break

            elif event_number < event_min:
                event_number += 1
                continue

            for telid in event.r0.tels_with_data:

                data = np.array(list(event.r0.tel[telid].adc_samples.values()))

                if options.mc:

                    baseline = np.mean(data, axis=1)

                    """
                    if baseline_counter < options.baseline_window_width :
                        baseline.append(data)
                        baseline_counter += data.shape[-1]
                        event_number += 1
                        progress_bar.update(1)
                        break

                    elif baseline_counter >= options.baseline_window_width and not baseline_computed:
                        log.debug('Computing baseline for level % d with %d bins' % (level, options.baseline_window_width))
                        baseline = np.array(baseline)
                        baseline = np.mean(np.mean(np.array(baseline), axis=0), axis=-1).astype(int)
                        baseline_computed = True
                        log.debug('Baseline recomputed for level %d : = %d [LSB]' % (level, np.mean(baseline)))
                    """

                    data = data[options.pixel_list, :] - baseline[:, np.newaxis]

                    """
                    for i in range(len(options.crate)):

                        if not options.crate[i]:

                            data[pixel_in_sector[i]] = np.zeros((len(pixel_in_sector[i]), data.shape[-1]))

                        if not options.pdp[i] and options.crate[i]:

                            data[pixel_in_sector[i]] = np.random.normal(options.baseline_mc, options.sigma_e, size=(len(pixel_in_sector[i]), data.shape[-1]))
                    """

                else:

                    if baseline_counter < options.baseline_window_width:
                        baseline.append(data)
                        baseline_counter += data.shape[-1]
                        event_number += 1
                        progress_bar.update(1)
                        break

                    elif baseline_counter >= options.baseline_window_width and not baseline_computed:
                        log.debug('Computing baseline for level % d with %d bins' % (level, options.baseline_window_width))
                        baseline = np.array(baseline)
                        baseline = np.mean(np.mean(np.array(baseline), axis=0), axis=-1).astype(int)
                        baseline_computed = True
                        log.debug('Baseline recomputed for level %d : = %d [LSB]' % (level, np.mean(baseline)))


                    data = data[options.pixel_list, :] - baseline[:, np.newaxis]



                if options.blinding:

                    if data.shape[-1] < options.window_width:
                        time[level] += data.shape[-1]

                    else:
                        time[level] += data.shape[-1] - data.shape[-1] % options.window_width
                        print(data.shape)

                    """
                    elif options.window_width % data.shape[-1] == options.window_width:

                        print (options.window_width)
                        print (data.shape[-1])
                        print (options.window_width % data.shape[-1])
                        print(level)
                        time[level] += data.shape[-1]
                        print('hello')
                    """


                else:

                    time[level] += data.shape[-1]
                cluster_trace, patch_trace = compute_cluster_trace(data, options, log)
                cluster_max_sector, cluster_max, cluster_max_time, cluster_max_id = compute_trigger_info(cluster_trace,
                                                                                                         options)
                trigger_count = compute_trigger_count(cluster_trace, options, log)

                cluster_hist.fill_with_batch(cluster_trace, indices=(level,))
                patch_hist.fill_with_batch(patch_trace, indices=(level,))
                max_cluster_hist[cluster_max_sector].append(cluster_max)
                time_cluster_hist[cluster_max_sector].append(cluster_max_time)
                trigger_rate_camera.data[level] += trigger_count

                progress_bar.update(1)
                event_number += 1
                if event_number % options.events_per_level == 0:
                    log.debug('Going to level %d' % level)
                    level += 1
                    baseline = []
                    baseline_computed = False
                    baseline_counter = 0

    trigger_rate_camera.errors = np.sqrt(trigger_rate_camera.data) / (time[:, np.newaxis] * 4.) * 1E9
    trigger_rate_camera.data = trigger_rate_camera.data / (time[:, np.newaxis] * 4.) * 1E9

    return


def compute_cluster_trace(data, options, log=None):
    camera = options.cts.camera

    cluster_trace = np.zeros((len(camera.Clusters_7), data.shape[-1]))
    patch_trace = np.zeros((len(camera.Patches), data.shape[-1]))

    for cluster_index, cluster in enumerate(camera.Clusters_7):

        for patch_index, patch in enumerate(cluster.patches):

            for pixel in patch.pixels:
                patch_trace[patch.ID] += data[pixel.ID]

            patch_trace[patch.ID] /= options.compression_factor
            patch_trace[patch.ID] = np.clip(patch_trace[patch.ID], 0., options.clipping_patch).astype(int)
            cluster_trace[cluster.ID] += patch_trace[patch.ID]

    return cluster_trace, patch_trace


def compute_trigger_info(cluster_trace, options):

    camera = options.cts.camera

    cluster_max = np.max(cluster_trace)
    index_max = np.argmax(cluster_trace)
    index_max = np.unravel_index(index_max, cluster_trace.shape)

    cluster_max_id = index_max[0]
    # cluster_max_time = index_max[1]
    try:

        cluster_max_time = np.min(np.where(cluster_trace[cluster_max_id] > options.threshold[0])[0][0])

    except:

        cluster_max_time = -1

    cluster_max_sector = camera.Clusters_7[cluster_max_id].patches[0].sector - 1

    return cluster_max_sector, cluster_max, cluster_max_time, cluster_max_id


def compute_trigger_count(cluster_trace, options, log):
    trigger_count = np.zeros(len(options.threshold))

    for threshold_index, threshold in enumerate(options.threshold):

        t = 0

        while t < cluster_trace.shape[-1] - (cluster_trace.shape[-1] % options.window_width):

            if np.any(cluster_trace[:, t] > threshold):

                trigger_count[threshold_index] += 1
                log.debug('Trigger at time %d [ns] for threshold %d [ADC]' % (t * 4, threshold))

                if options.blinding:
                    t += options.window_width + 1

            t += 1

    return trigger_count
