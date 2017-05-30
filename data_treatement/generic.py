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
            ind_good_baseline = (_rms - params[:, 2]) / params[:, 3] < 0.5
            # If at least one event was computed, only update the previous baseline for the pixel with no
            # large fluctuations
            if event_id > 0:
                baseline[ind_good_baseline] = _baseline[ind_good_baseline]
            else:
                baseline = _baseline
            # Subtract the baseline
            data = data - baseline[:, None]
    # Treat the case where the baseline has been specified

    if baseline is not None:
        return (data - baseline[:, None])[..., options.baseline_per_event_limit:]
    else :
        return data


# Define the integration function
def integrate(data,options):
    """
    Simple integration function over N samples

    :param data:
    :param options:
    :return:
    """
    if options.window_width == 1 : return data
    h = ndimage.convolve1d(data,np.ones(options.window_width, dtype=int),axis=-1,mode='constant',cval=-1e8)
    print(options.window_width)
    print(h[h>1e-6].shape[0]/data.shape[0])
    print(h[h>1e-6].reshape(data.shape[0],-1))

    return h[h>1e-6].reshape(data.shape[0],-1)