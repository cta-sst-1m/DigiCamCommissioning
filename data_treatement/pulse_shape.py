import numpy as np
#from ctapipe.calib.camera import integrators fix import with updated cta
from ctapipe.io import zfits
import logging,sys
from tqdm import tqdm
from utils.logger import TqdmToLogger
from utils.toy_reader import ToyReader

def run(pulse_shapes, options):

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
                    #print(event.r0.tel[telid].camera_event_number)
                    first_evt_num = event.r0.tel[telid].camera_event_number
                    first_evt = False
                evt_num = event.r0.tel[telid].camera_event_number - first_evt_num
                if evt_num % options.events_per_level == 0:
                    level = int(evt_num / options.events_per_level)
                    if level > len(options.scan_level) - 1:
                        break
                    if options.verbose:

                        log.debug('--|> Moving to DAC Level %d' % (options.scan_level[level]))
                #if evt_num % int(options.events_per_level/1000)== 0:
                pbar.update(1)

                # get the data
                data = np.array(list(event.r0.tel[telid].adc_samples.values()), dtype=float)
                #print(np.sum(data))
                # subtract the pedestals
                data = data[options.pixel_list]

                pulse_shapes[level, :, :, 0] += data/options.events_per_level
                pulse_shapes[level, :, :, 1] += (data*data)/options.events_per_level

    pulse_shapes[:, :, :, 1] = np.sqrt((pulse_shapes[:, :, :, 1] - pulse_shapes[:,:,:,0]**2) /options.events_per_level)


    return pulse_shapes
