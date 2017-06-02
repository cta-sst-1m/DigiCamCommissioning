import numpy as np
import scipy.ndimage as ndimage


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



def extract_charge(data,timing_mask,timing_mask_edge,options,integration_type='integration_saturation'):
    local_max = np.argmax(np.multiply(data, timing_mask), axis=1)
    local_max_edge = np.argmax(np.multiply(data, timing_mask_edge), axis=1)
    ind_max_at_edge = (local_max == local_max_edge)
    local_max[ind_max_at_edge] = peak[ind_max_at_edge] - options.window_start
    index_max = (np.arange(0, data.shape[0]), local_max,)
    ind_with_lt_th = data[index_max] < 10.
    local_max[ind_with_lt_th] = peak[ind_with_lt_th] - options.window_start
    local_max[local_max < 0] = 0
    index_max = (np.arange(0, data.shape[0]), local_max,)
    return



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
