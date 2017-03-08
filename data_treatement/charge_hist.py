import numpy as np
from ctapipe.io import zfits
from ctapipe.calib.camera.charge_extractors import SimpleIntegrator
import logging,sys
from tqdm import tqdm
from utils.logger import TqdmToLogger
from utils.toy_reader import ToyReader
import matplotlib.pyplot as plt

# noinspection PyProtectedMember
def run(hist, options, peak_positions=None, prev_fit_result= None):

    # Few counters
    level, evt_num, first_evt, first_evt_num = 0, 0, True, 0

    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    pbar = tqdm(total=options.evt_max)
    tqdm_out = TqdmToLogger(log, level=logging.INFO)
    charge_extraction = options.integration_method
    if charge_extraction == 'integration' or charge_extraction == 'integration_sat':
        window_width = options.window_width
        # WARNING: START WRT MAX
        window_start = options.window_start
        normalisation = 1.
        if window_width == 7:
            if window_start == 1:
                normalisation = 1.
            elif window_start == 2:
                normalisation = 1.
            elif window_start == 3:
                normalisation = 1#3.60670943412 # 1./7#4.2740160339
        if window_width == 6:
            if window_start == 1:
                normalisation = 1.
            elif window_start == 2:
                normalisation = 1.
            elif window_start == 3:
                normalisation = 1.
        if window_width == 5:
            if window_start == 1:
                normalisation = 1.
            elif window_start == 2:
                normalisation = 1.

    peak = None
    if type(peak_positions).__name__ == 'ndarray':
        peak = np.argmax(peak_positions, axis=1)
        mask = (peak_positions.T/np.sum(peak_positions,axis=1)).T>0.01
        mask_window = mask+np.append(mask[...,1:],np.zeros((peak_positions.shape[0],1),dtype=bool),axis=1)+\
                 np.append(np.zeros((peak_positions.shape[0],1),dtype=bool),mask[...,:-1],axis=1)
        mask_windows_edge = mask_window*~mask
        mask_window=mask_window[...,:-1]
        mask_windows_edge=mask_windows_edge[...,:-1]
        if charge_extraction == 'integration'or charge_extraction == 'integration_sat':
            shift = window_start#window_width - int(np.floor(window_width/2))+window_start
            missing = mask_window.shape[1]-(window_width-1)
            mask_window = mask_window[...,shift:]
            print(mask_window.shape[1],missing)
            missing = mask_window.shape[1]-missing
            mask_window = mask_window[...,:-missing]
            print (mask_window.shape[1],missing)
            mask_windows_edge = mask_windows_edge[...,shift:]
            mask_windows_edge = mask_windows_edge[...,:-missing]
            #mask_window = np.append(mask_window,np.zeros((mask_window.shape[0],missing),dtype=bool),axis=1)
            #mask_windows_edge = np.append(mask_windows_edge,np.zeros((mask_windows_edge.shape[0],missing),dtype=bool),axis=1)
            print(shift)
    def integrate_trace(d):
        return np.convolve(d, np.ones((window_width), dtype=int), 'valid')

    def contiguous_regions(data):
        """Finds contiguous True regions of the boolean array "condition". Returns
        a 2D array where the first column is the start index of the region and the
        second column is the end index."""
        condition = data>0
        # Find the indicies of changes in "condition"
        d = np.diff(condition)
        idx, = d.nonzero()

        # We need to start things after the change in "condition". Therefore,
        # we'll shift the index by 1 to the right.
        idx += 1

        if condition[0]:
            # If the start of condition is True prepend a 0
            idx = np.r_[0, idx]

        if condition[-1]:
            # If the end of condition is True, append the length of the array
            idx = np.r_[idx, condition.size]  # Edit

        # Reshape the result into two columns
        idx.shape = (-1, 2)
        val = 0.
        for start, stop in idx:
            sum_tmp = np.sum( data[start:stop])
            if val < sum_tmp: val = sum_tmp
        return val

    plt.figure()
    plt.ion()
    for file in options.file_list:
        if evt_num > options.evt_max :
            print(evt_num,options.evt_max)
            break
        # Get the file
        _url = options.directory + options.file_basename % file
        inputfile_reader = None
        if not options.mc:
            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=options.evt_max)
        else:
            seed = 0
            inputfile_reader = ToyReader(filename=_url, id_list=[0], seed=seed, max_events=options.evt_max, n_pixel=options.n_pixels, events_per_level=options.events_per_level, level_start=options.scan_level[0])

        if options.verbose:
            log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file
        for event in inputfile_reader:
            if evt_num > options.evt_max :
                print(evt_num,options.evt_max)
                break
            evt_num+=1
            for telid in event.dl0.tels_with_data:
                if evt_num % int(options.evt_max/1000)== 0:
                    pbar.update(int(options.evt_max/1000))

                # get the data
                data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
                #print(np.sum(data))
                # subtract the pedestals
                #data = (data.T - prev_fit_result[:,1,0]).T
                # put in proper format
                data = data.reshape((1,) + data.shape)
                # find the local maximum within the appropriate time window

                # charge extraction type
                if charge_extraction == 'global_max':
                    local_max = np.argmax(np.multiply(data[0], mask_window), axis=1)
                    local_max_edge = np.argmax(np.multiply(data[0], mask_windows_edge), axis=1)
                    ind_max_at_edge = (local_max == local_max_edge)
                    local_max[ind_max_at_edge] = peak[ind_max_at_edge]
                    index_max = (np.arange(0, data[0].shape[0]), local_max,)
                    ind_with_cumsum_lt_7 = data[0][index_max] < prev_fit_result[:,2,0]*3
                    local_max[ind_with_cumsum_lt_7] = peak[ind_with_cumsum_lt_7]
                    index_max = (np.arange(0, data[0].shape[0]), local_max,)
                    hist.fill(data[0][index_max])
                elif charge_extraction == 'fixed_max':
                    index_max = (np.arange(0, data[0].shape[0]), peak,)
                    hist.fill(data[0][index_max])
                elif charge_extraction == 'integration':
                    integration = np.apply_along_axis(integrate_trace,1,data[0])
                    local_max = np.argmax(np.multiply(integration, mask_window), axis=1)
                    local_max_edge = np.argmax(np.multiply(integration, mask_windows_edge), axis=1)
                    ind_max_at_edge = (local_max == local_max_edge)
                    local_max[ind_max_at_edge] = peak[ind_max_at_edge]-window_start
                    index_max = (np.arange(0, data[0].shape[0]), local_max,)
                    ind_with_lt_th = integration[index_max] < 10.
                    local_max[ind_with_lt_th] = peak[ind_with_lt_th]-window_start
                    index_max = (np.arange(0, data[0].shape[0]), local_max,)
                    hist.fill((integration[index_max]- prev_fit_result[:,1,0])/normalisation)
                    '''
                    plt.clf()
                    pix_2_inspect = 661
                    test_window_of_arrival = np.zeros(data[0].shape)
                    test_window_of_arrival[pix_2_inspect, local_max[pix_2_inspect]] = integration[pix_2_inspect, local_max[pix_2_inspect]]
                    print('ref timing', peak[pix_2_inspect], 'moving timing', local_max[pix_2_inspect])
                    print('INT', integration[pix_2_inspect])
                    print('DATA', data[0][pix_2_inspect])
                    print('MASK', mask_window[pix_2_inspect])
                    print('MAX_BIN', test_window_of_arrival[pix_2_inspect])
                    print('BEST_VAL', data[0][index_max][pix_2_inspect])
                    plt.step(np.arange(0, 50), data[0][pix_2_inspect])
                    plt.step(np.arange(0, integration.shape[1]), integration[pix_2_inspect])
                    plt.step(np.arange(0, 50), test_window_of_arrival[pix_2_inspect])
                    plt.step(np.arange(0, mask_window.shape[1]), mask_window[pix_2_inspect] * 10.)

                    plt.show()
                    input('press')
                    '''
                elif charge_extraction == 'local_max':
                    cum_sum_trace1 = data[0].cumsum(axis=-1)
                    cum_sum_trace1[..., 3:] = (cum_sum_trace1[..., 3:] - cum_sum_trace1[..., :-3])
                    cum_sum_trace1 = np.append(cum_sum_trace1[..., 1:], np.zeros((data[0].shape[0], 1,)), axis=1)
                    local_max = np.argmax(np.multiply(cum_sum_trace1, mask_window), axis=1)
                    local_max_edge = np.argmax(np.multiply(cum_sum_trace1, mask_windows_edge), axis=1)
                    ind_max_at_edge = (local_max == local_max_edge)
                    local_max[ind_max_at_edge] = peak[ind_max_at_edge]
                    index_max = (np.arange(0, data[0].shape[0]), local_max,)
                    ind_with_cumsum_lt_7 = cum_sum_trace1[index_max] < prev_fit_result[:,1,0]+ 7
                    local_max[ind_with_cumsum_lt_7] = peak[ind_with_cumsum_lt_7]
                    index_max = (np.arange(0, data[0].shape[0]), local_max,)
                    hist.fill(data[0][index_max]- prev_fit_result[:,1,0])

                elif charge_extraction == 'integration_sat':
                    # first deal with normal values
                    max_idx = (np.arange(0, data[0].shape[0]),np.argmax(data[0], axis=1) ,)
                    data_int = np.copy(data[0])
                    data_sat = np.copy(data[0])
                    data_sat[ data_sat[max_idx] < (options.threshold_sat - prev_fit_result[:,1,0])] = 0.
                    data_int[ data_int[max_idx] >= (options.threshold_sat - prev_fit_result[:,1,0])] = 0.

                    integration = np.apply_along_axis(integrate_trace,1,data_int)
                    local_max = np.argmax(np.multiply(integration, mask_window), axis=1)
                    local_max_edge = np.argmax(np.multiply(integration, mask_windows_edge), axis=1)
                    ind_max_at_edge = (local_max == local_max_edge)
                    local_max[ind_max_at_edge] = peak[ind_max_at_edge]-window_start
                    index_max = (np.arange(0, data_int.shape[0]), local_max,)
                    ind_with_lt_th = integration[index_max] < 10.
                    local_max[ind_with_lt_th] = peak[ind_with_lt_th]-window_start
                    index_max = (np.arange(0, data_int.shape[0]), local_max,)
                    full_integration = integration[index_max]
                    # now deal with saturated ones:
                    sat_integration = np.apply_along_axis(contiguous_regions, 1, data_sat)
                    full_integration = full_integration + sat_integration
                    hist.fill(full_integration)
                    '''
                    plt.clf()
                    pix_2_inspect = 661
                    test_window_of_arrival = np.zeros(data[0].shape)
                    test_window_of_arrival[pix_2_inspect, local_max[pix_2_inspect]] = integration[pix_2_inspect, local_max[pix_2_inspect]]
                    print('ref timing', peak[pix_2_inspect], 'moving timing', local_max[pix_2_inspect])
                    print('INT', integration[pix_2_inspect])
                    print('DATA', data[0][pix_2_inspect])
                    #print('MASK', mask_above_0[pix_2_inspect]*1000)
                    print('MAX_BIN', test_window_of_arrival[pix_2_inspect])
                    print('BEST_VAL', data[0][index_max][pix_2_inspect])
                    plt.step(np.arange(0, 50), data[0][pix_2_inspect])
                    plt.step(np.arange(0, integration.shape[1]), integration[pix_2_inspect])
                    plt.step(np.arange(0, 50), test_window_of_arrival[pix_2_inspect])
                    plt.step(np.arange(0, mask_window.shape[1]), mask_window[pix_2_inspect] * 10.)

                    plt.show()
                    input('press')
                    hist.fill(integration[index_max] / normalisation)
                    hist.fill(data[0][index_max])
                    '''


    # Update the errors
    hist._compute_errors()

    print(evt_num, options.evt_max)
    # Save the MPE histos in a file
    hist.save(options.output_directory + options.histo_filename)