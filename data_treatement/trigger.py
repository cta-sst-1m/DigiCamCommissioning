import numpy as np
from ctapipe.io import zfits
import logging,sys
from tqdm import tqdm
from utils.logger import TqdmToLogger
from utils.toy_reader import ToyReader
from cts_core.camera import Camera



def run(hist, options, min_evt = 0):
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
                    # Update adc histo
                    #print(batch)
                    hist.fill_with_batch(batch.reshape(batch.shape[0], batch.shape[1] ))
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

                    baseline = np.apply_along_axis(integrate_trace, 1, data) / options.baseline_window_width

                    for i in range(len(options.crate)):

                        if not options.crate[i]:

                            data[pixel_in_sector[i]] = np.zeros((len(pixel_in_sector[i]), data.shape[-1]))


                        if not options.pdp[i] and options.crate[i]:

                            data[pixel_in_sector[i]] = np.random.normal(options.baseline_mc, options.sigma_e, size=(len(pixel_in_sector[i]), data.shape[-1]))




                else:

                    print('baseline not implemented for data !!!')
                t = data.shape[-1]  - options.baseline_window_width

                time[level] += t - t % options.window_width
                trigger_count = compute_trigger_v2(data, baseline, camera, options, log)
                #hist.data[level, :] += trigger_count
                hist.data[level] += trigger_count
                #cluster_adc.append(test[1].ravel())

                if evt_num%options.events_per_level==0:
                    level += 1

    hist.errors = np.sqrt(hist.data) / (time[:, np.newaxis] * 4.) * 1E9
    hist.data = hist.data / (time[:, np.newaxis] * 4.) * 1E9

    return #np.array(cluster_adc).ravel()




def compute_trigger(data, baseline, camera, options, output_type='bool'):

    data = data[:,0:-options.baseline_window_width+1] - baseline.astype(int)


    temp = np.load(options.cts_directory + 'config/cluster.p')

    temp = temp['patches_in_cluster']
    temp[options.cluster_list[0]].append(options.cluster_list[0])
    Clusters_patch_list = []

    for cluster in options.cluster_list:

        Clusters_patch_list.append(temp[cluster])

    #Clusters_patch_list.append(options.cluster_list[0])

    #print(Clusters_patch_list)


    trigger = np.zeros((len(Clusters_patch_list), len(options.threshold), data.shape[-1]))
    trigger_count = np.zeros((len(Clusters_patch_list), len(options.threshold)))
    patch_sum = np.zeros((len(camera.Patches), data.shape[-1]))
    cluster_sum = np.zeros((len(Clusters_patch_list), data.shape[-1]))

    #print(options.pixel_list)

    for i, pixel in enumerate([camera.Pixels[index] for index in options.pixel_list]):

        patch_sum[pixel.patch] += data[i]

    patch_sum /= options.compression_factor
    patch_sum = np.clip(patch_sum, 0., options.clipping_patch)


    for i, cluster_list in enumerate(Clusters_patch_list):

        cluster_sum[i] += np.sum(patch_sum[cluster_list, :], axis=0)

    for i, cluster in enumerate(Clusters_patch_list):
        for j, threshold in enumerate(options.threshold):

            if options.blinding:

                k = 0
                while k<trigger.shape[-1]:

                    if cluster_sum[i, k]>threshold:

                        trigger_count[i, j] += 1
                        k += options.window_width + 1
                    else:

                        k += 1

            else:

                trigger[i, j] = (cluster_sum[i]>threshold)



    trigger.astype(bool)

    #if options.blinding:

    #    new_size = trigger.shape[-1] - trigger.shape[-1]%options.window_width
    #    trigger = trigger[..., 0:new_size]
    #    trigger = np.split(trigger, options.window_width, axis=-1)
    #    trigger = np.sum(trigger, axis=0)
    #    trigger[trigger>0] = 1

    if options.blinding:

        return trigger_count, cluster_sum

    else:

        return np.sum(trigger, axis=-1), cluster_sum
"""
    if output_type=='bool':

        pass
        #return np.any(trigger, axis=0)

    elif output_type=='all':

        return trigger

    elif output_type=='trigger_per_cluster':

        return np.sum(trigger, axis=-1), cluster_sum

    elif output_type=='debug':

        return trigger, cluster_sum

"""


def compute_trigger_v2(data, baseline, camera, options, log):



    ### Substract baseline and init

    data = data[:,0:-options.baseline_window_width+1] - baseline.astype(int)

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
                    t += options.window_width + 1
                    log.debug('Cluster %d trigged at time %0.1f [ns] for threshold %d [ADC]' %(cluster_index, t*4, threshold))
                    break
        #    print(t)
            t += 1

        #if trigger_count[threshold_index] == 0:
        #    break

    return trigger_count