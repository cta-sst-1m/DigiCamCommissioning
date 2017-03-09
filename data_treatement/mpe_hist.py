import numpy as np
#from ctapipe.calib.camera import integrators fix import with updated cta
from ctapipe.io import zfits
import logging,sys
from tqdm import tqdm
from utils.logger import TqdmToLogger
from utils.toy_reader import ToyReader

# noinspection PyProtectedMember
def run(hist, options, peak_positions=None, charge_extraction = 'amplitude'):

    # Few counters
    level, evt_num, first_evt, first_evt_num = 0, 0, True, 0

    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    pbar = tqdm(total=len(options.scan_level)*options.events_per_level)
    tqdm_out = TqdmToLogger(log, level=logging.INFO)

    peak = None
    if type(peak_positions).__name__ == 'ndarray':
        peak = np.argmax(peak_positions, axis=1)

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
                #if evt_num % int(options.events_per_level/1000)== 0:
                pbar.update(1)

                # get the data
                data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
                #print(np.sum(data))
                # subtract the pedestals
                data = data[options.pixel_list]
                # put in proper format
                data = data.reshape((1,) + data.shape)
                #### TODO put the new charge extraction
                # charge extraction type
                if charge_extraction == 'amplitude':
                    if peak is None:
                        peak = np.argmax(data[0], axis=1)
                    index_max = (np.arange(0, data[0].shape[0]), peak,)
                    hist.fill(data[0][index_max], indices=(level,))
                elif charge_extraction == 'integration':
                    if not isinstance(peak,None):
                        integrator.window_start = np.mean(peak,axis=0)[0] - 2
                        hist.fill(integrator.extract_charge(data[0])/5, indices=(level,))

    # Update the errors
    hist._compute_errors()