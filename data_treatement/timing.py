import numpy as np
from ctapipe.io import zfits
from utils.mc_events_reader import hdf5_mc_event_source
from spectra_fit.fit_pulse_shape import p0_func, slice_func, bounds_func, fit_func, fit_func_mc
import logging
import sys
import scipy.optimize
from utils.logger import TqdmToLogger
from iminuit import Minuit
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.event_iterator import EventCounter


def run(options):
    """
    Fill the adcs Histogram out of darkrun/baseline runs
    :param h_type: type of Histogram to produce: ADC for all samples adcs or SPE for only peaks
    :param hist: the Histogram to fill
    :param options: see analyse_spe.py
    :param prev_fit_result: fit result of a previous step needed for the calculations
    :return:
    """
    arrival_time = np.zeros((len(options.ac_level), len(options.dc_level), len(options.pixel_list), options.events_per_level))


    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    # Reading the file

    event_counter = EventCounter(event_min=options.min_event, event_max=options.max_event, log=log, batch_size=1, level_dc_min=options.dc_level[0], level_dc_max=options.dc_level[-1], level_ac_min=options.ac_level[0], level_ac_max=options.ac_level[-1], event_per_level=options.events_per_level, event_per_level_in_file=options.events_per_level_in_file)

    for file in options.file_list:
        # Open the file
        _url = options.directory + options.file_basename % file

        if options.mc:
            log.info('Running on MC data')
            inputfile_reader = hdf5_mc_event_source(url=_url, events_per_dc_level=options.dc_step, events_per_ac_level=options.ac_step, dc_start=options.dc_start, ac_start=options.ac_start, max_events=options.max_event)

        else :
            log.info('Running on DigiCam data')
            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=options.max_event)  #TODO data_type arg does not exist anymore

        log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file
        for event, counter in zip(inputfile_reader, event_counter):
            if counter.continuing:
                continue

            for telid in event.r0.tels_with_data:

                # Take data from zfits
                data = np.array(list(event.r0.tel[telid].adc_samples.values()))
                data = data[options.pixel_list]
                time = np.arange(0, data.shape[-1], 1) * 4
                errors = np.ones(data.shape[-1]) / np.sqrt(12)

                for pixel_id, pixel_soft_id in enumerate(options.pixel_list):

                    fit_result = compute_time(data[pixel_id], time, errors, options, pixel_soft_id)
                    # print(counter.level_ac, counter.level_dc, pixel_id, counter.event_id_in_level)
                    arrival_time[counter.level_ac, counter.level_dc, pixel_id, counter.event_count_in_level] = fit_result[0]

                    if options.debug:

                        t_temp = np.linspace(0, data.shape[-1], 1000) * 4
                        plt.figure(figsize=(10,10))
                        plt.step(np.arange(0, data.shape[-1], 1) * 4, data[pixel_id], where='post')
                        if options.mc:
                            plt.plot(t_temp, fit_func_mc(fit_result, t_temp, pixel_id=pixel_soft_id), label='$t_0 =$ %0.2f [ns]' % fit_result[0])
                        else:
                            plt.plot(t_temp, fit_func(fit_result, t_temp, pixel_id=pixel_soft_id), label='$t_0 =$ %0.2f [ns]' % fit_result[0])

                        plt.legend()
                        plt.show()


    return arrival_time


def residual_function(function, p, x, y, y_err, pixel_id):

    return (y - function(p, x, pixel_id)) / y_err


def compute_time(data, time, errors, options, pixel_soft_id):

    if options.mc:

        residual = lambda p, x, y, y_err: residual_function(fit_func_mc, p,
                                                            x, y, y_err, pixel_id=None)
    else:

        residual = lambda p, x, y, y_err: residual_function(fit_func, p,
                                                          x, y, y_err, pixel_id=482) #TODO change this
    starting_parameters = p0_func(data, time)
    bounds = bounds_func(data, time)
    #slice = slice_func(time, data)

    try:
        out = scipy.optimize.least_squares(residual, starting_parameters, args=(time, data, errors), bounds=bounds)

        return out.x

    except:

        if options.debug:

            plt.figure()
            plt.step(time, data)
            plt.show()
            print(starting_parameters)

        return starting_parameters








