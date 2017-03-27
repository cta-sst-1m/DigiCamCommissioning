import numpy as np
from ctapipe.io import zfits
import logging,sys
from tqdm import tqdm
from utils.logger import TqdmToLogger
from utils.toy_reader import ToyReader


def run(hist, options, min_evt = 0 , max_evt=50000):
    # Few counters
    evt_num, first_evt, first_evt_num = 0, True, 0

    n_evt, n_batch, batch_num, max_evt = 0, options.n_evt_per_batch, 0, options.evt_max
    batch = None
    _tmp_baseline=None

    params=None
    if hasattr(options, 'baseline_per_event_limit'):
        params = np.load(options.output_directory + options.baseline_param_data)['params']
    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    pbar = tqdm(total=max_evt-min_evt)
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
        for event in inputfile_reader:
            if evt_num < min_evt:
                evt_num += 1
                continue
            else:
                # progress bar logging
                if evt_num % int((max_evt-min_evt)/1000)==0: #TODO make this work properly
                    pbar.update(int((max_evt-min_evt)/1000))
            if evt_num > max_evt: break
            for telid in event.r0.tels_with_data:
                evt_num += 1
                if evt_num % n_batch == 0:
                    log.debug('Treating the batch #%d of %d events' % (batch_num, n_batch))
                    # Update adc histo
                    hist.fill_with_batch(batch.reshape(batch.shape[0], batch.shape[1] ))
                    # Reset the batch
                    batch = np.zeros((data.shape[0], n_batch),dtype=int)
                    batch_num += 1
                    log.debug('Reading  the batch #%d of %d events' % (batch_num, n_batch))

                if evt_num > max_evt: break
                # get the data
                data = np.array(list(event.r0.tel[telid].adc_samples.values()))
                # get rid of unwanted pixels
                data = data[options.pixel_list]

                if hasattr(options, 'baseline_per_event_limit'):
                    baseline = np.mean(data[..., 0:options.baseline_per_event_limit], axis=-1)
                    rms = np.std(data[..., 0:options.baseline_per_event_limit], axis=-1)
                    ind_good_baseline = (rms - params[:, 2]) / params[:, 3] < 0.5
                    if n_evt > 1:
                        _tmp_baseline[ind_good_baseline] = baseline[ind_good_baseline]
                    else:
                        _tmp_baseline = baseline
                    # _tmp_baseline = baseline
                    data = data - _tmp_baseline[:, None]
                else:
                    data = data-options.prev_fit_result[...,1,0][:,None]/options.window_width

                if evt_num==1:
                    batch = np.zeros((data.shape[0], n_batch),dtype=int)

                # subtract the pedestals
                data_max = np.argmax(data, axis=1)

                data_max[data[(np.arange(0,data.shape[0]),data_max)] < 40]=0
                data_max[data[(np.arange(0,data.shape[0]),data_max)] >3000]=0
                #if (data_max-np.argmin(data, axis=1))/data_max>0.2:
                batch[:,n_evt%n_batch-1]=data_max
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
            for telid in event.r0.tels_with_data:
                if first_evt:
                    first_evt_num = event.r0.tel[telid].event_number
                    first_evt = False
                evt_num = event.r0.tel[telid].event_number - first_evt_num
                if evt_num % options.events_per_level == 0:
                    level = int(evt_num / options.events_per_level)
                    if level > len(options.scan_level) - 1:
                        break
                    if options.verbose:

                        log.debug('--|> Moving to DAC Level %d' % (options.scan_level[level]))
                pbar.update(1)

                # get the data
                data = np.array(list(event.r0.tel[telid].adc_samples.values()))
                # subtract the pedestals
                data = data[options.pixel_list]
                # extract max
                data_max = np.argmax(data, axis=1)

                hist.fill(data_max, indices=(level,))

    # Update the errors
    hist._compute_errors()

"""