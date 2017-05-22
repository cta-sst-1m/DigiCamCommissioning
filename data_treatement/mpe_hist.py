import numpy as np
#from ctapipe.calib.camera import integrators fix import with updated cta
from ctapipe.io import zfits
import logging,sys
from tqdm import tqdm
from utils.logger import TqdmToLogger
from utils.toy_reader import ToyReader
import matplotlib.pyplot as plt
# noinspection PyProtectedMember
def run(hist, options, peak_positions=None, charge_extraction = 'amplitude', baseline=0., trigger_output=None):
    # Few counters
    level, evt_num, first_evt, first_evt_num = 0, 0, True, 0

    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    pbar = tqdm(total=len(options.scan_level)*options.events_per_level)
    tqdm_out = TqdmToLogger(log, level=logging.INFO)

    params=None
    if hasattr(options, 'baseline_per_event_limit'):
        params = np.load(options.output_directory + options.baseline_param_data)['params']


    charge_extraction = options.integration_method
    if charge_extraction == 'integration' or charge_extraction == 'integration_sat' or charge_extraction == 'baseline':
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
                normalisation = 1  # 3.60670943412 # 1./7#4.2740160339
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
        mask = (peak_positions.T / np.sum(peak_positions, axis=1)).T > 1e-3
        mask_window = mask + np.append(mask[..., 1:], np.zeros((peak_positions.shape[0], 1), dtype=bool), axis=1) + \
                      np.append(np.zeros((peak_positions.shape[0], 1), dtype=bool), mask[..., :-1], axis=1)
        mask_windows_edge = mask_window * ~mask
        mask_window = mask_window[..., :-1]
        mask_windows_edge = mask_windows_edge[..., :-1]
        if charge_extraction == 'integration' or charge_extraction == 'integration_sat':
            shift = window_start  # window_width - int(np.floor(window_width/2))+window_start
            missing = mask_window.shape[1] - (window_width - 1)
            mask_window = mask_window[..., shift:]
            #print(mask_window.shape[1], missing)
            missing = mask_window.shape[1] - missing
            mask_window = mask_window[..., :-missing]
            #print(mask_window.shape[1], missing)
            mask_windows_edge = mask_windows_edge[..., shift:]
            mask_windows_edge = mask_windows_edge[..., :-missing]
            # mask_window = np.append(mask_window,np.zeros((mask_window.shape[0],missing),dtype=bool),axis=1)
            # mask_windows_edge = np.append(mask_windows_edge,np.zeros((mask_windows_edge.shape[0],missing),dtype=bool),axis=1)
            #print(shift)

    def integrate_trace(d):
        return np.convolve(d, np.ones((window_width), dtype=int), 'valid')

    def contiguous_regions(data):
        """Finds contiguous True regions of the boolean array "condition". Returns
        a 2D array where the first column is the start index of the region and the
        second column is the end index."""
        condition = data > 0
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
            sum_tmp = np.sum(data[start:stop])
            if val < sum_tmp: val = sum_tmp
        return val

    batch = None
    for file in options.file_list:
        if level > len(options.scan_level) - 1:
            break
        # Get the file
        _url = options.directory + options.file_basename % file
        inputfile_reader = None
        if not options.mc:
            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=len(options.scan_level)*options.events_per_level,expert_mode=type(trigger_output).__name__ == 'ndarray')
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
                    first_evt_num = event.r0.tel[telid].camera_event_number
                    batch_index = 0
                    batch = np.zeros((len(options.pixel_list), options.events_per_level),dtype=int)
                    if charge_extraction=='baseline':
                        pass
                        #batch = np.zeros((len(options.pixel_list*(1+options.n_bins-options.window_width)), options.events_per_level),dtype=int)

                    first_evt = False
                evt_num = event.r0.tel[telid].camera_event_number - first_evt_num
                if evt_num % options.events_per_level == 0:
                    batch_index = 0
                    if charge_extraction == 'integration':
                        #print(batch)
                        hist.fill_with_batch(batch, indices=(level,))
                        # Reset the batch
                        batch = np.zeros((len(options.pixel_list), options.events_per_level),dtype=int)
                    if charge_extraction == 'baseline':
                        pass
                        #batch = np.zeros((len(options.pixel_list*(1+ options.n_bins - options.window_width)), options.events_per_level), dtype=int)
                    level = int(evt_num / options.events_per_level)
                    if level > len(options.scan_level) - 1:
                        break
                    if options.verbose:

                        log.debug('--|> Moving to DAC Level %d' % (options.scan_level[level]))
                if options.events_per_level<=1000:
                    pbar.update(1)
                else:
                    if evt_num % int(options.events_per_level/1000)== 0:
                        pbar.update(int(options.events_per_level/1000))

                # get the data
                data = np.array(list(event.r0.tel[telid].adc_samples.values()))

                data = data[options.pixel_list]

                #print(np.sum(data))
                # subtract the pedestals
                # put in proper format
                #rdata = data.reshape((1,) + data.shape)
                # charge extraction type
                if type(trigger_output).__name__ == 'ndarray':
                    trig = event.r0.tel[telid].trigger_output_patch7
                    trigger_output[level,:,1]+=1
                    trigger_output[level,:,0]+=np.any(trig>0.5,axis=-2)*np.ones((trig.shape[0]),dtype=int)

                # charge extraction type
                if charge_extraction == 'global_max':
                    local_max = np.argmax(np.multiply(data, mask_window), axis=1)
                    local_max_edge = np.argmax(np.multiply(data, mask_windows_edge), axis=1)
                    ind_max_at_edge = (local_max == local_max_edge)
                    local_max[ind_max_at_edge] = peak[ind_max_at_edge]
                    index_max = (np.arange(0, data.shape[0]), local_max,)
                    ind_with_cumsum_lt_7 = data[index_max] < prev_fit_result[:,2,0]*3
                    local_max[ind_with_cumsum_lt_7] = peak[ind_with_cumsum_lt_7]
                    index_max = (np.arange(0, data.shape[0]), local_max,)
                    hist.fill(data[index_max] - baseline[level, :], indices=(level,))
                elif charge_extraction == 'fixed_max':
                    index_max = (np.arange(0, data.shape[0]), peak,)
                    hist.fill(data[index_max] - baseline[level, :], indices=(level,))
                elif charge_extraction == 'integration':
                    if hasattr(options, 'baseline_per_event_limit'):
                        baseline = np.mean(data[...,0:options.baseline_per_event_limit], axis=-1)
                        rms = np.std(data[...,0:options.baseline_per_event_limit], axis=-1)
                        # get the indices where baseline is good
                        ind_good_baseline = (rms - params[:,2])/params[:,3] < 0.5
                        if evt_num > 1:
                            _tmp_baseline[ind_good_baseline] = baseline[ind_good_baseline]
                        else:
                            _tmp_baseline = baseline
                        data = data - _tmp_baseline[:, None]
                        #if level>32 :

                    else:
                        baseline = np.zeros((len(options.scan_level), data.shape[0]))
                    integration = np.apply_along_axis(integrate_trace,1,data)
                    local_max = np.argmax(np.multiply(integration, mask_window), axis=1)
                    local_max_edge = np.argmax(np.multiply(integration, mask_windows_edge), axis=1)
                    ind_max_at_edge = (local_max == local_max_edge)
                    local_max[ind_max_at_edge] = peak[ind_max_at_edge]-window_start
                    index_max = (np.arange(0, data.shape[0]), local_max,)
                    ind_with_lt_th = integration[index_max] < 10.
                    local_max[ind_with_lt_th] = peak[ind_with_lt_th]-window_start
                    local_max[local_max<0]=0
                    index_max = (np.arange(0, data.shape[0]), local_max,)
                    '''
                    print('integrated value',integration[index_max][options.pixel_list.index(44)])
                    print('------------------------------------------')
                    if integration[index_max][options.pixel_list.index(44)]<-5:
                        plt.plot(np.arange(92),data[options.pixel_list.index(44)])
                        plt.plot(np.arange(92),data[options.pixel_list.index(29)])
                        plt.plot(np.arange(92),data[options.pixel_list.index(30)])
                        plt.plot(np.arange(92),data[options.pixel_list.index(45)])
                        plt.plot(np.arange(92),data[options.pixel_list.index(61)])
                        plt.plot(np.arange(92),data[options.pixel_list.index(62)])
                        plt.show()
                    '''
                    #print(baseline)
                    batch[...,batch_index]=integration[index_max] - baseline[level,:]
                    batch_index += 1
                    #hist.fill(integration[index_max],indices=(level,))

                elif charge_extraction == 'local_max':
                    cum_sum_trace1 = data.cumsum(axis=-1)
                    cum_sum_trace1[..., 3:] = (cum_sum_trace1[..., 3:] - cum_sum_trace1[..., :-3])
                    cum_sum_trace1 = np.append(cum_sum_trace1[..., 1:], np.zeros((data.shape[0], 1,)), axis=1)
                    local_max = np.argmax(np.multiply(cum_sum_trace1, mask_window), axis=1)
                    local_max_edge = np.argmax(np.multiply(cum_sum_trace1, mask_windows_edge), axis=1)
                    ind_max_at_edge = (local_max == local_max_edge)
                    local_max[ind_max_at_edge] = peak[ind_max_at_edge]
                    index_max = (np.arange(0, data.shape[0]), local_max,)
                    ind_with_cumsum_lt_7 = cum_sum_trace1[index_max] < prev_fit_result[:,1,0]+ 7
                    local_max[ind_with_cumsum_lt_7] = peak[ind_with_cumsum_lt_7]
                    index_max = (np.arange(0, data.shape[0]), local_max,)
                    hist.fill(data[index_max] - baseline[level,:],indices=(level,))

                elif charge_extraction == 'integration_sat':
                    # first deal with normal values
                    max_idx = (np.arange(0, data.shape[0]),np.argmax(data, axis=1) ,)
                    data_int = np.copy(data)
                    data_sat = np.copy(data)
                    data_sat[ data_sat[max_idx] < options.threshold_sat ] = 0.
                    data_int[ data_int[max_idx] >= options.threshold_sat ] = 0.

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
                    full_integration = full_integration + sat_integration - baseline[level, :]
                    hist.fill(full_integration,indices=(level,))

                elif charge_extraction == 'full':

                    temp = np.sum(data, axis=1) - baseline[level, :]
                    hist.fill(temp, indices=(level,))

                elif charge_extraction == 'baseline':

                    integral = np.sum(data[:,window_start:window_start+window_width], axis=1)
                    #batch[:,evt_num%(options.events_per_level*integral.shape[1]):evt_num%(options.events_per_level*integral.shape[1])+integral.shape[1]]=np.apply_along_axis(integrate_trace,-1,data)
                    hist.fill(integral,indices=(level,))

                if batch_index % options.events_per_level == 0:
                    batch_index = 0




    # Update the errors
    hist._compute_errors()