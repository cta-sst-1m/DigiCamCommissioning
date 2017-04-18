import numpy as np
#from ctapipe.calib.camera import integrators fix import with updated cta
from ctapipe.io import zfits
import logging,sys
from tqdm import tqdm
from utils.logger import TqdmToLogger
from utils.toy_reader import ToyReader
import matplotlib.pyplot as plt
# noinspection PyProtectedMember
def run(hist, options, peak_positions=None, charge_extraction = 'amplitude', baseline=0.):

    # Few counters
    level, evt_num, first_evt, first_evt_num = 0, 0, True, 0

    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    pbar = tqdm(total=len(options.shower_ids)*options.evt_per_shower)
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
        # temporary hack
        peak_positions=np.zeros(peak_positions.shape,dtype=float)
        peak_positions[:,67]=1.
        peak_positions[:,66]=0.8
        peak_positions[:,65]=0.4
        peak_positions[:,68]=0.8
        peak_positions[:,69]=0.4
        peak_positions[:,70]=0.2
        peak_positions[:,64]=0.2
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
    n_init = 0
    for file in options.file_list:
        # Get the file
        _url = options.directory + options.file_basename % file
        inputfile_reader = None
        if not options.mc:
            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=len(options.shower_ids)*options.evt_per_shower)
        else:

            seed = 0
            inputfile_reader = ToyReader(filename=_url, id_list=[0], seed=seed, max_events=len(options.shower_ids)*options.evt_per_shower, n_pixel=options.n_pixels, evt_per_shower=options.evt_per_shower, level_start=options.shower_ids[0])


        if options.verbose:
            log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file
        for event in inputfile_reader:
            for telid in event.r0.tels_with_data:
                if first_evt:
                    first_evt_num = event.r0.tel[telid].camera_event_number
                    batch_index = 0
                    batch = np.zeros((len(options.pixel_list), options.evt_per_shower),dtype=int)
                    if charge_extraction=='baseline':
                        pass
                        #batch = np.zeros((len(options.pixel_list*(1+options.n_bins-options.window_width)), options.evt_per_shower),dtype=int)

                    first_evt = False
                evt_num = event.r0.tel[telid].camera_event_number - first_evt_num

                # get the data
                data = np.array(list(event.r0.tel[telid].adc_samples.values()))

                data = data[options.pixel_list]

                #print(np.sum(data))
                # subtract the pedestals
                # put in proper format
                #rdata = data.reshape((1,) + data.shape)
                # charge extraction type


                if hasattr(options, 'baseline_per_event_limit'):
                    baseline = np.mean(data[..., 0:options.baseline_per_event_limit], axis=-1)
                    rms = np.std(data[..., 0:options.baseline_per_event_limit], axis=-1)
                    # get the indices where baseline is good
                    ind_good_baseline = (rms - params[:, 2]) / params[:, 3] < 0.5
                    if evt_num > 1:
                        _tmp_baseline[ind_good_baseline] = baseline[ind_good_baseline]
                    else:
                        _tmp_baseline = baseline
                    data = data - _tmp_baseline[:, None]
                    # if level>32 :
                integration = np.apply_along_axis(integrate_trace, 1, data)
                local_max = np.argmax(np.multiply(integration, mask_window), axis=1)
                local_max_edge = np.argmax(np.multiply(integration, mask_windows_edge), axis=1)
                ind_max_at_edge = (local_max == local_max_edge)
                local_max[ind_max_at_edge] = peak[ind_max_at_edge] - window_start
                index_max = (np.arange(0, data.shape[0]), local_max,)
                ind_with_lt_th = integration[index_max] < 10.
                local_max[ind_with_lt_th] = peak[ind_with_lt_th] - window_start
                local_max[local_max < 0] = 0
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
                hist.data[evt_num] = integration[index_max]  # - baseline[level,:]
                    #hist.fill(integration[index_max],indices=(level,))




    # Update the errors
    hist._compute_errors()