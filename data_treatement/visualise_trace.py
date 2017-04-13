import numpy as np
from ctapipe.io import zfits
from utils.peakdetect import spe_peaks_in_event_list
from utils.toy_reader import ToyReader
import logging
import sys
from utils.logger import TqdmToLogger
from ctapipe import visualization
from utils import geometry
from data_treatement import trigger
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from cts_core.camera import Camera



from tqdm import tqdm


def visualise(options):
    """
    Fill the adcs Histogram out of darkrun/baseline runs
    :param h_type: type of Histogram to produce: ADC for all samples adcs or SPE for only peaks
    :param hist: the Histogram to fill
    :param options: see analyse_spe.py
    :param prev_fit_result: fit result of a previous step needed for the calculations
    :return:
    """
    log = logging.getLogger(sys.modules['__main__'].__name__+'.'+__name__)
    event_number = options.event_min

    if not options.mc:
        log.info('Viewing on DigiCam data')
    else:
        log.info('Viewing on MC data')


    tqdm_out = TqdmToLogger(log, level=logging.INFO)

    geom = geometry.generate_geometry_0()
    camera = Camera(options.cts_directory + 'config/camera_config.cfg', options.cts_directory + 'config/cluster.p')
    fig_camera = plt.figure(figsize=(20, 10))
    axis_camera = fig_camera.add_subplot(121)
    #fig_readout = plt.figure(figsize=(15, 10))
    axis_readout = fig_camera.add_subplot(122)
    #axis_histogram = fig.add_subplot(121)
    #axis_camera = fig.add_subplot(122)
    camera_visu = visualization.CameraDisplay(geom, ax=axis_camera, title='', norm='lin', cmap='viridis',
                                              allow_pick=True)
    camera_visu.add_colorbar()
    camera_visu.colorbar.set_label('%s [ADC]'%options.camera_view)
    camera_visu.axes.set_xlabel('')
    camera_visu.axes.set_ylabel('')
    camera_visu.axes.set_xticks([])
    camera_visu.axes.set_yticks([])

    event_clicked_on = event_clicked()


    for file in options.file_list:
        # Open the file
        _url = options.directory + options.file_basename % file

        if not options.mc:

            inputfile_reader = zfits.zfits_event_source(url=_url, max_events=options.event_max)

        else:

            inputfile_reader = ToyReader(filename=_url, id_list=[0],
                                         max_events=options.event_max,
                                         )

        log.debug('--|> Moving to file %s' % _url)
        # Loop over event in this file
        #file_iterator = inputfile_reader.__iter__()
        for event in inputfile_reader:

            event_number += 1
            if event_number > options.event_max:

                break

            for telid in event.dl0.tels_with_data:

                data = np.array(list(event.dl0.tel[telid].adc_samples.values()))


                def draw_pulse_shape(pix_id):
                    event_clicked_on.ind[-1] = pix_id
                    axis_readout.cla()
                    axis_readout.step(4*np.arange(0, data.shape[-1], 1), data[pix_id], label='%s %d' %(options.view_type, pix_id))
                    axis_readout.set_xlabel('t [ns]')
                    axis_readout.set_ylabel('[ADC]')
                    axis_readout.legend(loc='upper right')
                    #axis_readout.yaxis.set_major_formatter(FormatStrFormatter('%d'))

                if not options.view_type=='pixel':

                    baseline = np.mean(data[..., 0:options.baseline_window_width], axis=1)
                    data = data - baseline[:, np.newaxis]


                    cluster_trace, patch_trace = trigger.compute_cluster_trace(data, camera, options, log)

                    for pixel_id in range(data.shape[0]):

                        if options.view_type == 'patch':

                            data[pixel_id] = patch_trace[camera.Pixels[pixel_id].patch]

                        elif options.view_type=='cluster_7':

                            data[pixel_id] = cluster_trace[camera.Pixels[pixel_id].patch]

                        elif options.view_type == 'cluster_9':

                            log.error('Cluster 19 not implemented')
                            break

                #else:

                   # log.error('Incorrect view typ %s' %options.view_type)

                if options.camera_view=='max':

                    camera_visu.image = np.max(data, axis=1)
                elif options.camera_view=='mean':

                    camera_visu.image = np.mean(data, axis=1)

                camera_visu.on_pixel_clicked = draw_pulse_shape
                camera_visu._on_pick(event_clicked_on)
                axis_camera.set_title('display mode : %s, #%d' %(options.view_type, event_number))
                #fig.canvas.mpl_connect('key_press_event', on_key)
                input("Press enter to go to next event")
    return

class event_clicked():

    def __init__(self):

        self.ind = [0, 700]




