import numpy as np
from ctapipe.calib.camera import integrators
from ctapipe.io import zfits


# noinspection PyProtectedMember
def run(hists, options, peak_positions=None):
    # Few counters
    level, evt_num, first_evt, first_evt_num = 0, 0, True, 0
    for file in options.file_list:
        if level > len(options.scan_level) - 1:
            break
        # Get the file
        _url = options.directory + options.file_basename % file
        inputfile_reader = zfits.zfits_event_source(url=_url, data_type='r1', max_events=100000)
        if options.verbose:
            print('--|> Moving to file %s' % _url)
        # Loop over event in this file
        for event in inputfile_reader:
            if level > len(options.scan_level) - 1:
                break
            for telid in event.r1.tels_with_data:
                if first_evt:
                    first_evt_num = event.r1.tel[telid].eventNumber
                    first_evt = False
                evt_num = event.r1.tel[telid].eventNumber - first_evt_num
                if evt_num % options.events_per_level == 0:
                    level = int(evt_num / options.events_per_level)
                    if level > len(options.scan_level) - 1:
                        break
                    if options.verbose:
                        print('--|> Moving to DAC Level %d' % (options.scan_level[level]))
                if options.verbose and event.r1.event_id % 100 == 0:
                    print("Progress {:2.1%}".format(
                        (evt_num - level * options.events_per_level) / options.events_per_level), end="\r")
                # get the data
                data = np.array(list(event.r1.tel[telid].adc_samples.values()))
                # subtract the pedestals
                data = data
                # put in proper format
                data = data.reshape((1,) + data.shape)
                # integration parameter
                params = {"integrator": "nb_peak_integration", "integration_window": [8, 4],
                          "integration_sigamp": [2, 4], "integration_lwt": 0}
                # now integrate
                #integration, window, peakpos = integrators.simple_integration(data, params)
                # try with the max instead
                peak = np.argmax(data[0], axis=1)
                if type(peak_positions).__name__ == 'ndarray' :
                    peak = np.argmax(peak_positions,axis=1)

                index_max = (np.arange(0, data[0].shape[0]), peak,)
                '''
                peak_m1 =  peak - 1
                peak_m1[peak_m1<0]=0
                peak_p1 =  peak + 1
                peak_p1[peak_p1>49]=49

                index_max_m1 = (np.arange(0, data[0].shape[0]), peak_m1,)
                index_max_p1 = (np.arange(0, data[0].shape[0]), peak_p1,)
                h = np.append(data[0][index_max].reshape(data[0][index_max].shape+(1,)),
                              data[0][index_max_m1].reshape(data[0][index_max_m1].shape+(1,)),axis=1)
                h = np.append(h,
                              data[0][index_max_p1].reshape(data[0][index_max_p1].shape + (1,)),axis=1)

                max_value = np.max(h,axis=1)
                '''
                hists[0].fill(data[0][index_max], indices=(level,))
                # and fill the histos
                #if hists[0] : hists[0].fill(integration[0], indices=(level,))
                #if hists[1]: hists[1].fill(max_value, indices=(level,))

    # Update the errors
    for hist in hists:
        hist._compute_errors()
    # Save the MPE histos in a file
    if options.verbose:
        print('--|> Save the data in %s' % (options.output_directory + options.histo_filename))
    np.savez_compressed(options.output_directory + options.histo_filename, mpes=hists[0].data,
                        mpes_bin_centers=hists[0].bin_centers)
