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
    if charge_extraction == 'integration':
        integrator = SimpleIntegrator(None, None)
        integrator.window_width= 8
        integrator.window_start= 3

    peak = None
    if type(peak_positions).__name__ == 'ndarray':
        peak = np.argmax(peak_positions, axis=1)

    plt.figure()
    plt.ion()
    for file in options.file_list:
        if evt_num > options.evt_max :
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
                break
            evt_num+=1
            for telid in event.dl0.tels_with_data:
                if evt_num % int(options.evt_max/1000)== 0:
                    pbar.update(1)

                # get the data
                data = np.array(list(event.dl0.tel[telid].adc_samples.values()))
                #print(np.sum(data))
                # subtract the pedestals
                data = (data.T - prev_fit_result[:,1,0]).T
                # put in proper format
                data = data.reshape((1,) + data.shape)
                # charge extraction type
                if charge_extraction == 'amplitude':
                    if peak is None:
                        peak = np.argmax(data[0], axis=1)
                    index_max = (np.arange(0, data[0].shape[0]), peak,)
                    hist.fill(data[0][index_max])
                elif charge_extraction == 'integration':
                    if peak is not None:
                        #integrator.window_start = np.mean(peak) - 2
                        hist.fill(integrator.extract_charge(data)[0]/4.5*1.13)
                elif charge_extraction == 'local_max':
                    if peak is not None:
                        plt.clf()
                        cum_sum_trace1 = data[0].cumsum(axis=-1)
                        cum_sum_trace1[...,3:] = (cum_sum_trace1[...,3:] - cum_sum_trace1[...,:-3])
                        #cum_sum_trace1 = cum_sum_trace1[:, 2-data[0].shape[1]:]
                        #peak[peak==0]=2
                        all_peaks = (np.arange(0, data[0].shape[0]), peak,)
                        all_peaksp1 = (np.arange(0, data[0].shape[0]), peak+1,)
                        all_peaksp2 = (np.arange(0, data[0].shape[0]), peak+2,)
                        all_peaksp3 = (np.arange(0, data[0].shape[0]), peak+3,)
                        all_peaksm1 = (np.arange(0, data[0].shape[0]), peak-1,)
                        all_peaksm2 = (np.arange(0, data[0].shape[0]), peak-2,)
                        mask_window_of_arrival = np.zeros(data[0].shape)#[:, data[0].shape[1]]
                        test_window_of_arrival = np.zeros(data[0].shape)#[:, data[0].shape[1]]
                        #mask_window_of_arrival = np.zeros(data[0].shape)#[:, 2-data[0].shape[1]:]
                        mask_window_of_arrival[all_peaksm1]=1
                        mask_window_of_arrival[all_peaksm2]=1
                        mask_window_of_arrival[all_peaks]=1
                        mask_window_of_arrival[all_peaksp1]=1
                        mask_window_of_arrival[all_peaksp2]=1
                        mask_window_of_arrival[all_peaksp3]=1
                        #local_max = np.argmax(np.multiply(data[0],mask_window_of_arrival), axis=1)
                        local_max = np.argmax(np.multiply(cum_sum_trace1,mask_window_of_arrival), axis=1)
                        ind_with_localmax_m2 = (local_max-peak+2)==0
                        #ind_with_localmax_p2 = (local_max-peak+2)==0
                        ind_with_localmax_p3 = (local_max-peak-3)==0
                        local_max[ind_with_localmax_m2]=peak[ind_with_localmax_m2]
                        local_max[ind_with_localmax_p3]=peak[ind_with_localmax_p3]
                        index_max = (np.arange(0, data[0].shape[0]), local_max,)
                        #ind_with_cumsum_lt_7 = data[0][index_max]<2.5
                        ind_with_cumsum_lt_7 = cum_sum_trace1[index_max]<7
                        local_max[ind_with_cumsum_lt_7]= peak[ind_with_cumsum_lt_7]
                        index_max = (np.arange(0, data[0].shape[0]), local_max-1,)


                        hist.fill(data[0][index_max])
                        '''
                        test_window_of_arrival[661,local_max[661]]=data[0][661,local_max[661]-1]
                        print('ref timing',peak[661],'moving timing',local_max[661])
                        plt.errorbar(np.arange(0,50)+0.5,data[0][661],yerr=np.sqrt(data[0][661]))
                        plt.step(np.arange(0,50),cum_sum_trace1[661])
                        plt.step(np.arange(0,50),test_window_of_arrival[661])
                        plt.step(np.arange(0,51),peak_positions[661]/np.sum(peak_positions[661])*data[0][661,local_max[661]-1])

                        plt.show()
                        input('press')
                        '''

    # Update the errors
    hist._compute_errors()
    # Save the MPE histos in a file
    hist.save(options.output_directory + options.histo_filename)