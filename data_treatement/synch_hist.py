import numpy as np
from ctapipe.io import zfits
import logging,sys
from tqdm import tqdm
from utils.logger import TqdmToLogger
from utils.toy_reader import ToyReader

def run(hist, options, min_evt = 5000.*3 , max_evt=5000*10):
    # Few counters
    evt_num, first_evt, first_evt_num = 0, True, 0

    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    pbar = tqdm(total=max_evt-min_evt)
    tqdm_out = TqdmToLogger(log, level=logging.INFO)
    for file in options.file_list:
        if evt_num > max_evt: break
        # read the file
        _url = options.directory + options.file_basename % file
        inputfile_reader = None
        if not options.mc:
            inputfile_reader = zfits.zfits_event_source(url=_url, data_type='r1', max_events=max_evt)

        else:
            inputfile_reader = ToyReader(filename=_url, id_list=[0], max_events=max_evt, n_pixel=options.n_pixels)

        if options.verbose:
            log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file
        for event in inputfile_reader:
            if evt_num < min_evt:
                evt_num += 1
                continue
            else:
                # progress bar logging
                if evt_num % int((max_evt-min_evt)/1000)==0:
                    pbar.update(int((max_evt-min_evt)/1000))
            if evt_num > max_evt: break
            for telid in event.r1.tels_with_data:
                evt_num += 1
                if evt_num > max_evt: break
                # get the data
                data = np.array(list(event.r1.tel[telid].adc_samples.values()))
                # subtract the pedestals
                data = data
                hist.fill(np.argmax(data, axis=1))

    # Update the errors
    # noinspection PyProtectedMember
    hist._compute_errors()
    # Save the histo in a file
    hist.save(options.output_directory + options.histo_filename)
