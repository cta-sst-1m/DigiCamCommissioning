#!/usr/bin/env python3

# external modules

# internal modules
from spectra_fit import fit_dark_adc
from utils import display, histogram, geometry
from ctapipe import visualization
from data_treatement import adc_hist
import logging,sys
import scipy.stats
import numpy as np
from ctapipe.io import zfits
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from cts_core.cameratestsetup import CTS

__all__ = ["create_histo", "perform_analysis", "display_results"]


def create_histo(options):
    """
    :return:
    """

    # Define the array

    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    # Reading the file
    event_number = 0

    if not options.mc:
        log.info('Running on DigiCam data')
    else:
        log.info('Running on MC data')

    pbar = tqdm(total=options.max_event)
    # Get the adcs
    event_number = 0
    cnt = 0

    suprious_mask = np.zeros((432),dtype = bool)
    suprious_mask[[391,392,403,404,405,416,417]]= True

    n_int_base = 50000

    # the final arrays
    # testbase = np.zeros((1296,(options.max_event) * 50 ), dtype=int)
    full_baseline = np.zeros((1296,options.max_event // n_int_base * 50 ), dtype=float)
    full_std = np.zeros((1296,options.max_event // n_int_base * 50 ), dtype=float)
    full_time = np.zeros((options.max_event // n_int_base * 50), dtype=int)

    i_int = -1

    central_event = None
    prev_event = None
    next_event = None
    next_event_time = None
    

    mask_baseline = np.zeros((1296, n_int_base ), dtype=bool)
    baseline = np.zeros((1296, n_int_base ), dtype=int)
    time = np.zeros((n_int_base ), dtype=int)

    for file in options.file_list:
        if event_number > options.max_event :
            break
        if event_number%100000==0: print('################### event NUM %d'%event_number)
        # Open the file
        _url = options.directory + options.file_basename % file

        inputfile_reader = zfits.zfits_event_source(url=_url, max_events=options.max_event, expert_mode = True)

        log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file
        # Number to integrate

        for event in inputfile_reader:
            if event_number > options.max_event:
                break
            if event_number%int(1000) == 0 :
                pbar.update(int(1000))

            prev_event = central_event
            central_event = next_event
            central_event_time = next_event_time
            telid = event.r0.tels_with_data[0]
            trigger_out = np.array(list(event.r0.tel[telid].trigger_output_patch7.values()))


            # Check if the central event was ok:
            event_number+=1
            # Skip non-spurious events
            if np.sum(trigger_out[suprious_mask])<0.5 or np.sum(trigger_out[~suprious_mask])>0.5:
                continue

            next_event = np.array(list(event.r0.tel[telid].adc_samples.values()))
            next_event_time = event.r0.tel[telid].local_camera_clock


            if prev_event is None:
                continue

            if i_int < 0 :
                i_int = 0
            ## Save the value if enough events
            if i_int*50 > (n_int_base-1)  and event_number>0:
                # Save the value...
                #print(cnt,event_number,i_int*50 ,full_baseline.shape)
                baseline = baseline.astype(dtype=float)
                baseline[~mask_baseline]=np.nan
                full_baseline[:,cnt] = np.nanmean(baseline,axis=-1)
                full_std[:,cnt] = np.nanstd(baseline,axis=-1)
                full_time[cnt] = np.mean(time)

                mask_baseline = np.zeros((1296,n_int_base),dtype=bool)
                baseline = np.zeros((1296,n_int_base),dtype=int)
                time  = np.zeros((n_int_base),dtype=int)
                i_int = 0
                cnt+=1

            mean_prev = np.mean(prev_event, axis=-1)
            mean_central = np.mean(central_event, axis=-1)
            mean_next = np.mean(next_event, axis=-1)

            # are the pixel baseline of the central to be used in the baseline calculation?
            tmp_mask =  mean_central-np.minimum(mean_prev,mean_next)<50
            for ii in range(50):
                #print(mask_baseline.shape,i_int*50+ii,i_int)
                mask_baseline[:,i_int*50+ii] = mean_central-np.minimum(mean_prev,mean_next)<50
            baseline[:,i_int*50:(i_int+1)*50] = central_event
            #try:
            #    testbase[:,event_number*50:(event_number+1)*50] = central_event
            #except:
            #    print(event_number*50,(event_number+1)*50)
            time[i_int] = central_event_time
            i_int+=1
            event_number += 1



    # Save the histogram
    np.savez_compressed(options.output_directory + options.histo_filename,
                        baseline=full_baseline[:,:-1],rms=full_std[:,:-1],time=full_time[:-1])

    # Delete the histograms
    del full_baseline
    del full_std
    del full_time

    return


def perform_analysis(options):
    """
    Extract the dark rate and cross talk, knowing baseline gain sigmas from previous
    runs

    :param options: a dictionary containing at least the following keys:
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram      (str)

    :return:
    """
    return


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """
    cmap = matplotlib.cm.get_cmap('viridis')
    cmap_o = matplotlib.cm.get_cmap('Oranges')
    cmap_p = matplotlib.cm.get_cmap('Purples')
    cmap_g = matplotlib.cm.get_cmap('Greens')

    # Load the histogram
    baseline = np.load(options.output_directory + options.histo_filename)['baseline']
    #rms  = np.load(options.output_directory + options.histo_filename)['rms']
    time  = np.load(options.output_directory + options.histo_filename)['time']
    spurious  = np.load(options.output_directory + options.histo_filename)['spurious']

    # Get the baseline events
    baseline = baseline[:,spurious]
    time = time[spurious]



#    plt.hist(rms.reshape(-1),bins=100,range=(np.min(rms),np.max(rms)))
    cnt_shity_pix = 0

    min_baseline = []
    max_baseline = []
    min_rms = []
    max_rms = []

    import matplotlib.cm as cm

    max_diff = []
    all_ffts = []

    cnt_sector_1 = 0
    cnt_sector_2 = 0
    cnt_sector_3 = 0

    for pixel in range(len(baseline)):
        data_base_avg = np.mean(baseline[pixel].reshape(-1, 50), axis=1)
        time_step = 1./float(options.int_trig_rate)*50.
        plt.plot(time, data_base_avg, linestyle='-', linewidth=2)
    plt.show()
    plt.set_xlabel('time [s]')
    plt.set_ylabel('baseline [LSB]')
    plt.xaxis.get_label().set_ha('right')
    plt.xaxis.get_label().set_position((1, 0))
    plt.yaxis.get_label().set_ha('right')
    plt.yaxis.get_label().set_position((0, 1))

    #. Perform some plots
    input('press button to quit')

    return
