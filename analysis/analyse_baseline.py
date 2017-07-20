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
    print(len(options.pixel_list))
    baseline = np.zeros((len(options.pixel_list  ),options.evt_max+1),dtype=float)
    rms = np.zeros((len(options.pixel_list  ),options.evt_max+1),dtype=float)
    print(baseline.shape)

    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    # Reading the file
    event_number = 0

    if not options.mc:
        log.info('Running on DigiCam data')
    else:
        log.info('Running on MC data')

    pbar = tqdm(total=options.evt_max)
    # Get the adcs
    event_number = 0
    for file in options.file_list:
        # Open the file
        _url = options.directory + options.file_basename % file

        inputfile_reader = zfits.zfits_event_source(url=_url, max_events=options.evt_max)

        log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file
        for event in inputfile_reader:
            #print('bla',event_number,options.evt_max)
            if event_number > options.evt_max:
                break

            if event_number%int(float(options.evt_max)/100) == 0 :
                pbar.update(int(float(options.evt_max)/100))
            for telid in event.r0.tels_with_data:
                # Take data from zfits

                #print('evt_num',event_number)
                data = np.array(list(event.r0.tel[telid].adc_samples.values()))
                data = data[options.pixel_list]
                maxsample = -1
                if hasattr(options,'max_sample_baseline'):
                    maxsample=options.max_sample_baseline
                means = np.mean(data[...,0:maxsample],axis=-1)

                stddev = np.std(data[...,0:maxsample],axis=-1)
                #print(means[0])
                baseline[...,event_number]=means
                rms[...,event_number]=stddev
            event_number += 1



    # Save the histogram
    np.savez_compressed(options.output_directory + options.histo_filename,baseline=baseline,rms=rms)

    # Delete the histograms
    del baseline

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
    cts_path = '/data/software/DigiCamCommissioning/'
    cts = CTS(cts_path + 'config/cts_config_' + str(0) + '.cfg',
                      cts_path + 'config/camera_config.cfg', angle=0, connected=True)

    cmap = matplotlib.cm.get_cmap('viridis')
    cmap_o = matplotlib.cm.get_cmap('Oranges')
    cmap_p = matplotlib.cm.get_cmap('Purples')
    cmap_g = matplotlib.cm.get_cmap('Greens')

    # Load the histogram
    baseline = np.load(options.output_directory + options.histo_filename)['baseline']
    rms  = np.load(options.output_directory + options.histo_filename)['rms']

    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

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
        if (np.max(rms[pixel]) < 400):

            data_base = baseline[pixel][:options.evt_max-len(baseline[pixel])]
            data_base_avg = np.mean(data_base.reshape(-1, options.rescale_factor), axis=1)

            data_rms = rms[pixel][:options.evt_max - len(rms[pixel])]
            data_rms_avg = np.mean(data_rms.reshape(-1, options.rescale_factor), axis=1)

            #extract mean value of baseline over 10 first values
            base_mean = 0
            for s in range(10):
                base_mean+=baseline[pixel][s]
            base_mean/=10.

            if options.baseline_subtract:
                data_base_avg = data_base_avg - base_mean

            diff_base = np.abs(np.diff(data_base_avg))

            #all_ffts.append(np.abs(np.fft.fft(data_base_avg))**2)
            all_ffts.append(np.abs(np.fft.rfft(data_base_avg)))

            min_baseline.append(np.min(data_base_avg))
            max_baseline.append(np.max(data_base_avg))
            min_rms.append(np.min(data_rms_avg))
            max_rms.append(np.max(data_rms_avg))

            max_diff.append(np.max(diff_base))

            time_step = 1./float(options.int_trig_rate)*options.rescale_factor
            x_array = np.arange(0,int(len(data_base_avg)*time_step),time_step)

            pix_in_mod = cts.camera.Pixels[options.pixel_list[pixel]].id_inModule
            module = cts.camera.Pixels[options.pixel_list[pixel]].module
            fadc_mult = cts.camera.Pixels[options.pixel_list[pixel]].fadc
            sector = cts.camera.Pixels[options.pixel_list[pixel]].sector
            pixel_sw = options.pixel_list[pixel]

            if np.abs(max_baseline[-1]-min_baseline[-1]) > 5 and max_diff[-1] > 5:
                print(pixel_sw, sector, fadc_mult, pix_in_mod, module, np.abs(max_baseline[-1]-min_baseline[-1]),
                      max_diff[-1])

            if sector == 1:
                col_curve = cmap_o(float(cnt_sector_1) / 432)
                cnt_sector_1 += 1
            elif sector == 2:
                col_curve = cmap_p(float(cnt_sector_2) / 432)
                cnt_sector_2 += 1
            else:
                col_curve = cmap_g(float(cnt_sector_3) / 432)
                cnt_sector_3 += 1

            ax1.plot(x_array, data_base_avg, color=col_curve, linestyle='-', linewidth=2)
            ax2.plot(x_array, data_rms_avg, color=col_curve, linestyle='-', linewidth=2)

        else:
            cnt_shity_pix += 1

    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('baseline [LSB]')
    ax1.xaxis.get_label().set_ha('right')
    ax1.xaxis.get_label().set_position((1, 0))
    ax1.yaxis.get_label().set_ha('right')
    ax1.yaxis.get_label().set_position((0, 1))
    #ax1.xaxis.set_ticks(np.arange(0, 2400,1000))
    if np.min(min_baseline)<0:
        min_axis = 1.1*np.min(min_baseline)
    else:
        min_axis = 0.9 * np.min(min_baseline)
    ax1.set_ylim(min_axis,1.1*np.max(max_baseline))
    #ax1.set_ylim(400,600)

    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('RMS [LSB]')
    ax2.xaxis.get_label().set_ha('right')
    ax2.xaxis.get_label().set_position((1, 0))
    ax2.yaxis.get_label().set_ha('right')
    ax2.yaxis.get_label().set_position((0, 1))
    ax2.set_ylim(-0.1,1.1*np.max(max_rms))
    #ax2.xaxis.set_ticks(np.arange(0, 3500, 1000))
    #lgd = ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()

    plot_name = options.histo_filename.replace(".npz", ".png")
    fig.savefig(options.output_directory + plot_name, dpi=300, format='png')

    figolu,ax = plt.subplots(1)
    for pixel in range(len(all_ffts)):
       # if (np.abs(max_baseline[-1] - min_baseline[-1])) > 5:
        col_curve = cmap(float(pixel) / len(options.pixel_list))

        #time_step = 1. / 10000.
        #freqs = np.fft.fftfreq(data_base_avg[pixel].size, time_step)
        #idx = np.argsort(freqs)
        plt.plot(all_ffts[pixel], color=col_curve, linestyle='-', linewidth=2)

    ax.set_xlabel('frequence [Hz]')
    ax.set_ylabel('amplitude [a.u]')
    ax.xaxis.get_label().set_ha('right')
    ax.xaxis.get_label().set_position((1, 0))
    ax.yaxis.get_label().set_ha('right')
    ax.yaxis.get_label().set_position((0, 1))

    plt.show()

    print('Found %i desynchronised pixel' % cnt_shity_pix)
    '''
    # Define Geometry
    geom, pix_id = geometry.generate_geometry(cts, all_camera=True)
    figolu,ax = plt.subplots(1)
    camera_visu = visualization.CameraDisplay(geom, ax=ax, title='', norm='lin', cmap='Wistia', allow_pick=True)

    #h = np.log(np.abs(np.array(max_baseline)-np.array(min_baseline)))
    #h[h<-1.] =-1.
    #h[h>0] = 0
    h = np.array(max_diff)
    h[h<1.] = 1.
    camera_visu.image = np.log10(h)
    camera_visu.axes.set_xlabel('x [mm]')
    camera_visu.axes.set_ylabel('y [mm]')
    camera_visu.add_colorbar()

    #hist_base = histogram.Histogram(data = baseline,bin_centers=xarray)
    #plt.subplot(1,2,2)
    #plt.hist(image,bins=100)


    #display.display_by_pixel(image2,options)
    '''

    #. Perform some plots
    input('press button to quit')

    return
