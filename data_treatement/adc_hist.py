import numpy as np
from ctapipe.io import zfits
from utils.peakdetect import spe_peaks_in_event_list


def run(hist, options, h_type='ADC', prev_fit_result=None):
    """
    Fill the adcs Histogram out of darkrun/baseline runs
    :param h_type: type of Histogram to produce: ADC for all samples adcs or SPE for only peaks
    :param hist: the Histogram to fill
    :param options: see analyse_spe.py
    :param prev_fit_result: fit result of a previous step needed for the calculations
    :return:
    """
    # Reading the file
    n_evt, n_batch, batch_num, max_evt = 0, options.n_evt_per_batch, 0, options.evt_max
    batch = None

    print('--|> Treating the batch #%d of %d events' % (batch_num, n_batch))
    for file in options.file_list:
        # Open the file
        _url = options.directory + options.file_basename % file
        inputfile_reader = zfits.zfits_event_source(
            url=_url
            , data_type='r1', max_events=options.evt_max)

        if options.verbose:
            print('--|> Moving to file %s' % _url)
        # Loop over event in this file
        for event in inputfile_reader:
            n_evt += 1
            if n_evt > max_evt:
                break
            if (n_evt - n_batch * batch_num) % n_batch / 100 == 0:
                print("Progress {:2.1%}".format(float(n_evt - batch_num * n_batch) / n_batch), end="\r")
            for telid in event.r1.tels_with_data:
                if n_evt % n_batch == 0:
                    print('--|> Treating the batch #%d of %d events' % (batch_num, n_batch))
                    # Update adc histo
                    if h_type == 'ADC':
                        hist.fill_with_batch(batch.reshape(batch.shape[0], batch.shape[1] * batch.shape[2]))
                    elif h_type == 'SPE':
                        hist.fill_with_batch(
                            spe_peaks_in_event_list(batch, prev_fit_result[:, 1, 0], prev_fit_result[:, 2, 0]))
                    # Reset the batch
                    batch = None
                    batch_num += 1
                    print('--|> Reading  the batch #%d of %d events' % (batch_num, n_batch))
                # Get the data
                data = np.array(list(event.r1.tel[telid].adc_samples.values()))
                # Append the data to the batch
                if type(batch).__name__ != 'ndarray':
                    batch = data.reshape(data.shape[0], 1, data.shape[1])
                else:
                    batch = np.append(batch, data.reshape(data.shape[0], 1, data.shape[1]), axis=1)

    if options.verbose:
        print('--|> Save the data in %s' % (options.output_directory + options.histo_filename))
    np.savez_compressed(options.output_directory + options.histo_filename,
                        adcs=hist.data, adcs_bin_centers=hist.bin_centers)
