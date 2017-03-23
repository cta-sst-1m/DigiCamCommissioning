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
from tqdm import tqdm

__all__ = ["create_histo", "perform_analysis", "display_results"]


def create_histo(options):
    """
    :return:
    """

    # Define the array
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

    # Load the histogram
    baseline = np.load(options.output_directory + options.histo_filename)['baseline']
    # Define Geometry
    geom = geometry.generate_geometry_0(pixel_list=options.pixel_list)
    fig,ax = plt.subplots(1,2)
    camera_visu = visualization.CameraDisplay(geom, ax=ax[0], title='', norm='lin', cmap='viridis',
                                              allow_pick=True)
    image = np.var(baseline - np.mean(baseline, axis=-1)[:, None], axis=-1)
    image2 = np.var(baseline - np.mean(baseline, axis=-1)[:, None], axis=-1)
    image2[image2>1]=1
    image[image > 0.2] = 0.2
    camera_visu.image = image
    camera_visu.add_colorbar()
    camera_visu.axes.set_xlabel('x [mm]')
    camera_visu.axes.set_ylabel('y [mm]')

    plt.subplot(1,2,2)
    plt.hist(image,bins=100)


    display.display_by_pixel(image2,options)

    #. Perform some plots
    input('press button to quit')

    return
