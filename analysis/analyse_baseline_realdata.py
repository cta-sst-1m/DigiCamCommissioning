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

from data_treatement.generic import subtract_baseline,integrate,extract_charge,fake_timing_hist,generate_timing_mask



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

    n_int_base = options.baseline_integration

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
    baseline_tmp = None
    std_tmp = None

    mask_baseline = np.zeros((1296, n_int_base ), dtype=bool)
    baseline = np.zeros((1296, n_int_base ), dtype=int)
    time = np.zeros((n_int_base ), dtype=int)



    ## data for charge extraction
    peak, mask, mask_edges = None, None , None
    peak_position = fake_timing_hist(options,options.n_samples-options.baseline_per_event_limit)
    peak, mask, mask_edges = generate_timing_mask(options,peak_position)

    hist = np.zeros((1296,1000),dtype = int)
    hist2 = np.zeros((1296,1000),dtype = int)
    batch = np.zeros((1296,10000),dtype = int)
    cnt_good = 0
    for file in options.file_list:
        if cnt_good>batch.shape[1]:continue
        if event_number > options.max_event :
            break
        # Open the file
        _url = options.directory + options.file_basename % file

        inputfile_reader = zfits.zfits_event_source(url=_url, max_events=options.max_event, expert_mode = True)

        log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file
        # Number to integrate

        for event in inputfile_reader:
            if event_number > options.max_event:
                break
            if event_number%int(float(options.max_event)/100) == 0 :
                pbar.update(int(float(options.max_event)/100))

            prev_event = central_event
            central_event = next_event
            central_event_time = next_event_time
            telid = event.r0.tels_with_data[0]
            trigger_out = np.array(list(event.r0.tel[telid].trigger_output_patch7.values()), dtype=int)
            # Skip non-spurious events
            if ( np.sum(trigger_out[suprious_mask])<0.5 or np.sum(trigger_out[~suprious_mask])>0.5 ) and not (baseline_tmp is None) and cnt_good<batch.shape[1]:
                # subtract the baseline
                data = np.array(list(event.r0.tel[telid].adc_samples.values()))
                data = data - baseline_tmp.reshape(-1,1)
                data_tmp1 = np.copy(data)
                # Perform integration
                data = integrate(data, options)
                data_tmp = np.copy(data)
                data_tmp[data_tmp<5*10]=0
                data_tmp1[data_tmp1<5*8]=0
                if cnt_good < 1000 :
                    hist[:,cnt_good]=np.argmax(data_tmp,axis=-1)
                    hist2[:,cnt_good]=np.argmax(data_tmp1,axis=-1)
                # integrate
                data = extract_charge(data, mask, mask_edges, peak, options, integration_type='integration_saturation')

                batch[..., cnt_good] = data
                cnt_good+=1
            elif ( np.sum(trigger_out[suprious_mask])>0.5 and np.sum(trigger_out[~suprious_mask])<0.5 ):
                next_event = np.array(list(event.r0.tel[telid].adc_samples.values()))
                next_event_time = event.r0.tel[telid].local_camera_clock

                # Check if the central event was ok:
                event_number += 1

                if prev_event is None:
                    continue

                if i_int < 0:
                    i_int = 0
                ## Save the value if enough events
                if i_int * 50 > (n_int_base - 1) and event_number > 0:
                    # Save the value...
                    # print(cnt,event_number,i_int*50 ,full_baseline.shape)
                    baseline = baseline.astype(dtype=float)
                    baseline[~mask_baseline] = np.nan
                    full_baseline[:, cnt] = np.nanmean(baseline, axis=-1)
                    full_std[:, cnt] = np.nanstd(baseline, axis=-1)
                    full_time[cnt] = np.mean(time)
                    baseline_tmp = full_baseline[:, cnt]
                    std_tmp = full_std[:, cnt]

                    mask_baseline = np.zeros((1296, n_int_base), dtype=bool)
                    baseline = np.zeros((1296, n_int_base), dtype=int)
                    time = np.zeros((n_int_base), dtype=int)
                    i_int = 0
                    cnt += 1

                mean_prev = np.mean(prev_event, axis=-1)
                mean_central = np.mean(central_event, axis=-1)
                mean_next = np.mean(next_event, axis=-1)

                # are the pixel baseline of the central to be used in the baseline calculation?
                tmp_mask = mean_central - np.minimum(mean_prev, mean_next) < 50
                for ii in range(50):
                    # print(mask_baseline.shape,i_int*50+ii,i_int)
                    mask_baseline[:, i_int * 50 + ii] = mean_central - np.minimum(mean_prev, mean_next) < 50
                baseline[:, i_int * 50:(i_int + 1) * 50] = central_event
                # try:
                #    testbase[:,event_number*50:(event_number+1)*50] = central_event
                # except:
                #    print(event_number*50,(event_number+1)*50)
                time[i_int] = central_event_time
                i_int += 1


            event_number += 1



    # Save the histogram
    np.savez_compressed(options.output_directory + options.histo_filename,
                        baseline=full_baseline[:,:-1],rms=full_std[:,:-1],time=full_time[:-1])
    np.savez_compressed(options.output_directory + 'signal_'+options.histo_filename,
                        data=batch,max_int=hist,max_nonint=hist2)

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

    data = np.load(options.output_directory + 'signal_'+options.histo_filename)['data']
    geom,pixid = geometry.generate_geometry(options.cts, all_camera= True)
    fig, ax = plt.subplots(figsize=(8,6))
    fig1, ax1 = plt.subplots(figsize=(8,6))
    camera_visu1 = visualization.CameraDisplay(geom, ax=ax , title='', norm='lin', cmap='viridis')
    camera_visu2 = visualization.CameraDisplay(geom, ax=ax1 , title='', norm='log', cmap='viridis')
    camera_visu1.add_colorbar()

    fig.show()
    fig1.show()
    i=0
    colbar=True
    while i < data.shape[1]:
        i+=1
        if np.sum(data[:,i])/22>5000:
            tmp = np.copy(data[:,i]/23)
            camera_visu1.image=data[:,i]/23
            tmp[tmp<0.1]=np.sqrt(0.1)
            tmp[tmp<np.sqrt(8*8*2)]=np.sqrt(8*8*2)
            camera_visu2.image=tmp
            if colbar :
                camera_visu2.add_colorbar()
                colbar=False
            input('print enter to go to next event')
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
