import numpy as np
from ctapipe.io import zfits
import logging,sys
from tqdm import tqdm
from utils.logger import TqdmToLogger
from utils.toy_reader import ToyReader
from cts_core.camera import Camera



def run(trigger_rate_camera, options, min_evt=0, cluster_hist=None, patch_hist=None, max_cluster_hist=None):
    # Few counters

    camera = Camera(options.cts_directory + 'config/camera_config.cfg', options.cts_directory + 'config/cluster.p')
    cluster_adc = []

    pixel_in_sector_1 = []
    pixel_in_sector_2 = []
    pixel_in_sector_3 = []

    for pix in camera.Pixels:

        if pix.sector == 1:

            pixel_in_sector_1.append(pix.ID)

        elif pix.sector == 2:

            pixel_in_sector_2.append(pix.ID)

        elif pix.sector == 3:

            pixel_in_sector_3.append(pix.ID)

    pixel_in_sector = [pixel_in_sector_1, pixel_in_sector_2, pixel_in_sector_3]

    def integrate_trace(d):

        return np.convolve(d, np.ones((options.baseline_window_width), dtype=int), 'valid')

    evt_num, first_evt, first_evt_num = 0, True, 0

    n_evt, n_batch, batch_num, max_evt = (options.evt_max - options.evt_min), options.n_evt_per_batch, 0, options.evt_max
    batch = None
    level = 0
    time = np.zeros(len(options.nsb_rate))

    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    pbar = tqdm(total=options.events_per_level*len(options.nsb_rate))
    tqdm_out = TqdmToLogger(log, level=logging.INFO)

    for file in options.file_list:

        if evt_num > max_evt: break
        # read the file
        _url = options.directory + options.file_basename % file

        inputfile_reader = None
        if not options.mc:
            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=max_evt)

        else:

            seed = 0
            inputfile_reader = ToyReader(filename=_url, id_list=[0], n_pixel=len(options.pixel_list), events_per_level=options.events_per_level, seed=seed, max_events=len(options.nsb_rate)*options.events_per_level, level_start=0)


        if options.verbose:
            log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file

        batch_index = 0

        for event in inputfile_reader:
            if evt_num < min_evt:
                evt_num += 1
                pbar.update(1)
                continue
            else:
                # progress bar logging

                if max_evt-min_evt<=1000:

                    if evt_num!=0:
                        pbar.update(1)
                else:

                    if evt_num % int((max_evt-min_evt)/1000)==0: #TODO make this work properly
                        pbar.update(int((max_evt-min_evt)/1000))
            if evt_num > max_evt: break

            for telid in event.dl0.tels_with_data:
                evt_num += 1
                if evt_num % n_batch == 0:
                    log.debug('Treating the batch #%d of %d events' % (batch_num, n_batch))
                    # Update adc trigger_rate_camerao
                    #print(batch)
                    trigger_rate_camera.fill_with_batch(batch.reshape(batch.shape[0], batch.shape[1] ))
                    # Reset the batch
                    batch = np.zeros((data.shape[0], n_batch),dtype=int)
                    batch_num += 1
                    log.debug('Reading  the batch #%d of %d events' % (batch_num, n_batch))

                if evt_num > max_evt: break
                # get the data
                data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
                # get rid of unwanted pixels

                if not options.mc:
                    data = data[options.pixel_list, :]

                #print(baseline_window.shape)

                if options.mc:

                    #print(data[1])
                    baseline = np.apply_along_axis(integrate_trace, 1, data) / options.baseline_window_width
                    #print(baseline[1])
                    data = data[:, 0:-options.baseline_window_width + 1] - baseline.astype(int)
                    #print(data[1])

                    for i in range(len(options.crate)):

                        if not options.crate[i]:

                            data[pixel_in_sector[i]] = np.zeros((len(pixel_in_sector[i]), data.shape[-1]))

                        if not options.pdp[i] and options.crate[i]:

                            data[pixel_in_sector[i]] = np.random.normal(options.baseline_mc, options.sigma_e, size=(len(pixel_in_sector[i]), data.shape[-1]))



                else:

                    baseline = np.mean(data[..., 0:options.baseline_window_width], axis=1)
                    #print(baseline[700])
                    data = data - baseline.astype(int)[:, np.newaxis]
                    #print('baseline not implemented for data !!!')

                if options.blinding:

                    time[level] += data.shape[-1] - data.shape[-1] % options.window_width

                else:

                    time[level] += data.shape[-1]

                trigger_count, cluster_trace, patch_trace, max_cluster = compute_trigger_v2(data, camera, options, log)
                #cluster_trace, patch_trace = compute_cluster_trace(data, camera, options, log)
                cluster_hist.fill_with_batch(cluster_trace, indices=(level, ))
                patch_hist.fill_with_batch(patch_trace, indices=(level, ))
                #print(max_cluster)
                #print(max_cluster_hist.data.shape)

                max_cluster_hist.append(max_cluster)
                #max_cluster_hist.fill(max_cluster, indices=(level, ))
                trigger_rate_camera.data[level] += trigger_count

                #print(trigger_rate_camera.data[level])
                if evt_num%options.events_per_level==0:
                    level += 1

    trigger_rate_camera.errors = np.sqrt(trigger_rate_camera.data) / (time[:, np.newaxis] * 4.) * 1E9
    trigger_rate_camera.data = trigger_rate_camera.data / (time[:, np.newaxis] * 4.) * 1E9

    return

def compute_cluster_trace(data, camera, options, log=None):

    cluster_trace = np.zeros((len(camera.Clusters_7), data.shape[-1]))
    patch_trace = np.zeros((len(camera.Patches), data.shape[-1]))


    for cluster_index, cluster in enumerate(camera.Clusters_7):

        if cluster_index in options.clusters:

            for patch_index, patch in enumerate(cluster.patches):

                #print(len(cluster.patches))

                for pixel in patch.pixels:

                    #print(patch.ID, pixel.ID)
                    patch_trace[patch.ID] += data[pixel.ID]


                patch_trace[patch.ID] /= options.compression_factor
                patch_trace[patch.ID] = np.clip(patch_trace[patch.ID], 0., options.clipping_patch).astype(int)

                cluster_trace[cluster.ID] += patch_trace[patch.ID]



    return cluster_trace, patch_trace


def compute_trigger_v2(data, camera, options, log):

    trigger_count = np.zeros(len(options.threshold))


    cluster_trace, patch_trace = compute_cluster_trace(data, camera, options, log)

    max_cluster = np.max(np.max(cluster_trace, axis=1), axis=0).astype(int)

    for threshold_index, threshold in enumerate(options.threshold):

        t = 0

        while(t<data.shape[-1]):

            if np.any(cluster_trace[:, t] > threshold):

                #print(cluster_trace[:, t])

                #print(cluster_trace[np.where(cluster_trace[:, t] > threshold)[0][0]])
                trigger_count[threshold_index] += 1
                log.debug('Trigger at time %d [ns] for threshold %d [ADC]' % (t * 4, threshold))

                if options.blinding:
                    t += options.window_width + 1

            t += 1

    return trigger_count, cluster_trace, patch_trace, max_cluster



def compute_trigger(data, camera, options, log):


    trigger_count = np.zeros(len(options.threshold))

    ### Compute Clusters traces and trigger


    for threshold_index, threshold in enumerate(options.threshold):

        t = 0 # time

        while(t<data.shape[-1]):

            for cluster_index, cluster in enumerate(camera.Clusters_7):

                cluster_trace = 0.

                for patch_index, patch in enumerate(cluster.patches):

                    patch_trace = 0.

                    for pixel in patch.pixels:

                        patch_trace += data[pixel.ID, t]
                        #print(t)

                    patch_trace /= options.compression_factor
                    patch_trace = np.clip(patch_trace, 0., options.clipping_patch)

                    cluster_trace += patch_trace

                if cluster_trace > threshold:

                    trigger_count[threshold_index] += 1


                    if options.blinding:

                        t += options.window_width + 1

                    log.debug('Cluster %d trigged at time %0.1f [ns] for threshold %d [ADC]' %(cluster_index, t*4, threshold))
                    break
        #    print(t)
            t += 1

        #if trigger_count[threshold_index] == 0:
        #    break

    return trigger_count