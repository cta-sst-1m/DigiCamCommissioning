import numpy as np
from ctapipe.io import zfits
from utils.toy_reader import ToyReader


def run(hist, options, min_evt = 5000.*3 , max_evt=5000*10):
    # Few counters
    evt_num, first_evt, first_evt_num = 0, True, 0
    for index_file, file in enumerate(options.file_list):
        if evt_num > max_evt: break
        # read the file
        _url = options.directory + options.file_basename % file

        if not options.toy_test:

            inputfile_reader = zfits.zfits_event_source(url=_url, data_type='r1', max_events=100000)

        else:

            weight = options.weights[index_file] / np.sum(options.weights)
            inputfile_reader = ToyReader(filename=_url, id_list=[0], max_events=options.evt_max, n_pixel=options.n_pixels, weights=weight)


        if options.verbose:
            print('--|> Moving to file %s' % _url)
        # Loop over event in this file
        for event in inputfile_reader:
            if evt_num < min_evt:
                evt_num += 1
                continue
            if evt_num > max_evt: break
            for telid in event.r1.tels_with_data:
                evt_num += 1
                if evt_num > max_evt: break
                if options.verbose and event.r1.event_id % 1000 == 0:
                    print("Progress {:2.1%}".format(
                        evt_num / max_evt), end="\r")
                # get the data
                data = np.array(list(event.r1.tel[telid].adc_samples.values()))
                # subtract the pedestals
                data_max = np.argmax(data, axis=1) #TODO Skip level 0 to avoid biais on position finding
                #if (data_max-np.argmin(data, axis=1))/data_max>0.2:
                hist.fill(data_max)

    # Update the errors
    # noinspection PyProtectedMember
    hist._compute_errors()
    # Save the MPE histos in a file

    if options.verbose:
        print('--|> Save the data in %s' % (options.output_directory + options.peak_histo_filename))
    np.savez_compressed(options.output_directory + options.peak_histo_filename,
                        peaks=hist.data, peaks_bin_centers=hist.bin_centers
                        )
