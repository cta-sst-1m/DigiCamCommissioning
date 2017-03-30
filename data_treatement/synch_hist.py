import numpy as np
from ctapipe.io import zfits
import logging,sys
from tqdm import tqdm
from utils.logger import TqdmToLogger
from utils.toy_reader import ToyReader


def run(hist, options, min_evt = 0):
    # Few counters
    evt_num, first_evt, first_evt_num = 0, True, 0

    n_evt, n_batch, batch_num, max_evt = (options.evt_max - options.evt_min), options.n_evt_per_batch, 0, options.evt_max
    batch = None

    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    pbar = tqdm(total=max_evt)
    tqdm_out = TqdmToLogger(log, level=logging.INFO)
    for file in options.file_list:

        if evt_num > max_evt: break
        # read the file
        _url = options.directory + options.file_basename % file

        inputfile_reader = None
        if not options.mc:
            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=max_evt)

        else:
            inputfile_reader = ToyReader(filename=_url, id_list=[0], max_events=options.evt_max, n_pixel=options.n_pixels, events_per_level=options.evt_max/2, level_start=7)
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

                if options.prev_fit_result is not None:

                    data = data[options.pixel_list]-options.prev_fit_result[...,1,0][:,None]/options.window_width

                else:

                    data = data[options.pixel_list]

                if evt_num==min_evt + 1:
                    batch = np.zeros((data.shape[0], n_batch),dtype=int)

                # subtract the pedestals

                data_max = np.argmax(data, axis=1)

                if options.prev_fit_result is not None:

                    data_max[data[(np.arange(0,data.shape[0]),data_max)] < 40] = 0
                    data_max[data[(np.arange(0,data.shape[0]),data_max)] > 3000] = 0 #TODO need to adapt this more generic
                #if (data_max-np.argmin(data, axis=1))/data_max>0.2:
                batch[:,batch_index]=data_max
                batch_index += 1
                if batch_index%n_batch==0:
                    batch_index = 0
    # Update the errors
    # noinspection PyProtectedMember
    hist._compute_errors()
    # Save the histo in a file
    hist.save(options.output_directory + options.histo_filename) #TODO check for double saving

"""
def run(hist, options):

    # Few counters
    level, evt_num, first_evt, first_evt_num = 0, 0, True, 0

    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    pbar = tqdm(total=len(options.scan_level)*options.events_per_level)
    tqdm_out = TqdmToLogger(log, level=logging.INFO)

    for file in options.file_list:
        if level > len(options.scan_level) - 1:
            break
        # Get the file
        _url = options.directory + options.file_basename % file
        inputfile_reader = None
        if not options.mc:
            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=len(options.scan_level)*options.events_per_level)
        else:

            seed = 0
            inputfile_reader = ToyReader(filename=_url, id_list=[0], seed=seed, max_events=len(options.scan_level)*options.events_per_level, n_pixel=options.n_pixels, events_per_level=options.events_per_level, level_start=options.scan_level[0])


        if options.verbose:
            log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file
        for event in inputfile_reader:
            if level > len(options.scan_level) - 1:
                break
            for telid in event.dl0.tels_with_data:
                if first_evt:
                    first_evt_num = event.dl0.tel[telid].event_number
                    first_evt = False
                evt_num = event.dl0.tel[telid].event_number - first_evt_num
                if evt_num % options.events_per_level == 0:
                    level = int(evt_num / options.events_per_level)
                    if level > len(options.scan_level) - 1:
                        break
                    if options.verbose:

                        log.debug('--|> Moving to DAC Level %d' % (options.scan_level[level]))
                pbar.update(1)

                # get the data
                data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
                # subtract the pedestals
                data = data[options.pixel_list]
                # extract max
                data_max = np.argmax(data, axis=1)

                hist.fill(data_max, indices=(level,))

    # Update the errors
    hist._compute_errors()

"""