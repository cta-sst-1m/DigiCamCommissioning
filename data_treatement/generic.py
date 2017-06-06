import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def subtract_baseline(data,event_id,options,params,baseline=None):
    """

    :param data:
    :param options: YAML object from the config file. Relevant items:
                    - baseline_per_event_limit : Baseline is evaluated between 0 and N

    :param params: TODO
    :param baseline: TODO
    :return:
    """
    # Treat the case where the baseline is computed from the event itself and is not known
    if hasattr(options, 'baseline_per_event_limit'):
        # Get the mean and std deviations
        _baseline = np.mean(data[..., 0:options.baseline_per_event_limit], axis=-1)
        _rms = np.std(data[..., 0:options.baseline_per_event_limit], axis=-1)
        if params is not None:
            # Case where the baseline parameters have been evaluated already

            # Get the pixel for which the rms is good, ie. there have been no huge fluctuation in the
            # samples in which it was evaluated
            ind_good_baseline = np.abs(_rms - params[1, :, 1 , 0]) / params[1, : , 2 , 0] < 2
            # If at least one event was computed, only update the previous baseline for the pixel with no
            # large fluctuations
            if event_id > 0:
                baseline[ind_good_baseline] = _baseline[ind_good_baseline]
            else:
                baseline = _baseline

    if baseline is None:
        return data,baseline
    else :
        return np.round(data.astype(float) - baseline[:, None],0).astype(int)[..., options.baseline_per_event_limit:],baseline


# Define the integration function
def integrate(data,options):
    """
    Simple integration function over N samples

    :param data:
    :param options:
    :return:
    """
    if options.window_width == 1 : return data
    h = ndimage.convolve1d(data,np.ones(options.window_width, dtype=int),axis=-1,mode='constant',cval=-1.e8)
    return h[...,int(np.floor((options.window_width-1)/2)):-int(np.floor(options.window_width/2))]



def fake_timing_hist(options,n_samples):
    """
    Create a timing array based on options.central_sample and options.timing_width
    :param options:
    :param n_samples:
    :return:
    """
    timing = np.zeros((len(options.pixel_list),n_samples+1,),dtype=float)
    timing[...,int(options.central_sample-options.timing_width):int(options.central_sample+options.timing_width)]=1.
    return timing

def generate_timing_mask(options,peak_positions):
    """
    Generate mask arround the possible peak position
    :param peak_positions:
    :return:
    """
    peak = np.argmax(peak_positions, axis=1)
    mask = (peak_positions.T / np.sum(peak_positions, axis=1)).T > 1e-3
    mask_window = mask + np.append(mask[..., 1:], np.zeros((peak_positions.shape[0], 1), dtype=bool), axis=1) + \
                  np.append(np.zeros((peak_positions.shape[0], 1), dtype=bool), mask[..., :-1], axis=1)
    mask_windows_edge = mask_window * ~mask
    mask_window = mask_window[..., :-1]
    mask_windows_edge = mask_windows_edge[..., :-1]
    shift = options.window_start  # window_width - int(np.floor(window_width/2))+window_start
    missing = mask_window.shape[1] - (options.window_width - 1)
    mask_window = mask_window[..., shift:]
    missing = mask_window.shape[1] - missing
    mask_window = mask_window[..., :-missing]
    mask_windows_edge = mask_windows_edge[..., shift:]
    mask_windows_edge = mask_windows_edge[..., :-missing]
    return peak,mask_window,mask_windows_edge

def extract_charge(data,timing_mask,timing_mask_edge,peak,options,integration_type='integration_saturation'):
    """
    Extract the charge.
       - check which pixels are saturated
       - get the local maximum within the timing mask and check if it is not at the edge of the mask
       - move options.window_start from the maximum
    :param data:
    :param timing_mask:
    :param timing_mask_edge:
    :param peak_position:
    :param options:
    :param integration_type:
    :return:
    """
    is_saturated = np.max(data,axis=-1)>options.threshold_sat
    local_max = np.argmax(np.multiply(data, timing_mask), axis=1)
    local_max_edge = np.argmax(np.multiply(data, timing_mask_edge), axis=1)
    ind_max_at_edge = (local_max == local_max_edge)
    local_max[ind_max_at_edge] = peak[ind_max_at_edge] - options.window_start
    index_max = (np.arange(0, data.shape[0]), local_max,)
    ind_with_lt_th = data[index_max] < 10.
    local_max[ind_with_lt_th] = peak[ind_with_lt_th] - options.window_start
    local_max[local_max < 0] = 0
    index_max = (np.arange(0, data.shape[0]), local_max,)
    sat_integration = np.zeros(data.shape[0],dtype=int)
    if np.any(is_saturated) and integration_type == 'integration_saturation':
        sat_integration = np.apply_along_axis(contiguous_regions, 1, data)

    if True == False:
        plt.ion()
        pix_2_inspect = options.pixel_list.index(10)
        test_window_of_arrival = np.zeros(data.shape)
        test_window_of_arrival[pix_2_inspect, local_max[pix_2_inspect]] = sat_integration[pix_2_inspect]
        print(peak)
        print('ref timing', peak[pix_2_inspect], 'moving timing', local_max[pix_2_inspect])
        print('INT', sat_integration[pix_2_inspect])
        print('DATA', data[pix_2_inspect])
        print('MASK', timing_mask[pix_2_inspect])
        print('MASK_EDGE', timing_mask_edge[pix_2_inspect])
        print('MAX_BIN', test_window_of_arrival[pix_2_inspect])
        print('BEST_VAL', data[index_max][pix_2_inspect])
        plt.step(np.arange(0, data.shape[1]), data[pix_2_inspect])
        #plt.step(np.arange(0, data.shape[1]), data[pix_2_inspect])
        plt.step(np.arange(0, data.shape[1]), test_window_of_arrival[pix_2_inspect])
        plt.step(np.arange(0, timing_mask.shape[1]), timing_mask[pix_2_inspect] * 10.)

        plt.show()
        input('press')

        plt.clf()

    return data[index_max]*~is_saturated+sat_integration*is_saturated


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
