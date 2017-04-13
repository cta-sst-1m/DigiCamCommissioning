import numpy as np
from ctapipe.io import zfits
from utils.toy_reader import ToyReader
from ctapipe import visualization
from utils import geometry
from data_treatement import trigger
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from cts_core.camera import Camera
from matplotlib.widgets import CheckButtons, Button
from matplotlib.offsetbox import AnchoredText
from itertools import cycle


class EventViewer():

    def __init__(self, options):

        plt.ioff()
        self.options = options
        self.filename = options.directory + options.file
        self.mc = options.mc
        self.baseline_window_width = options.baseline_window_width
        self.scale = options.scale

        if self.mc:

            self.event_iterator = ToyReader(filename=self.filename, id_list=[0], max_events=options.event_max)

        else:

            self.event_iterator = zfits.zfits_event_source(url=self.filename, max_events=options.event_max)


        self.telescope_id = options.telescope_id
        self.event_id = 1
        self.pixel_id = options.pixel_start
        self.data = np.array(
            list(self.event_iterator.__next__().dl0.tel[self.telescope_id].adc_samples.values()))

        self.event_clicked_on = Event_Clicked(options)
        self.geometry = geometry.generate_geometry_0()
        self.camera = Camera(options.cts_directory + 'config/camera_config.cfg',
                             options.cts_directory + 'config/cluster.p')

        self.view_type = options.view_type
        self.view_types = ['pixel', 'patch', 'cluster_7']#, 'cluster_9']
        self.iterator_view_type = cycle(self.view_types)
        self.camera_view = options.camera_view
        self.camera_views = ['sum', 'mean', 'max', 'std']
        self.iterator_camera_view = cycle(self.camera_views)
        self.figure = plt.figure(figsize=(20, 10))
        self.axis_next_event_button = self.figure.add_axes([0.03, 0.9, 0.1, 0.05], zorder=np.inf)
        self.axis_next_camera_view_button = self.figure.add_axes([0.03, 0.03, 0.1, 0.05], zorder=np.inf)
        self.axis_next_view_type_button = self.figure.add_axes([0.35, 0.03, 0.1, 0.05], zorder=np.inf)
        self.axis_readout = self.figure.add_subplot(122)
        self.axis_camera = self.figure.add_subplot(121)
        self.axis_camera.axis('off')

        self.camera_visu = visualization.CameraDisplay(self.geometry, ax=self.axis_camera, title='', norm=self.scale,
                                                       cmap='viridis',
                                                       allow_pick=True)
        self.camera_visu.add_colorbar(orientation='horizontal', pad=0.03, fraction=0.05)
        self.camera_visu.colorbar.set_label('%s [ADC]' % self.camera_view)
        self.camera_visu.axes.set_xlabel('')
        self.camera_visu.axes.set_ylabel('')
        self.camera_visu.axes.set_xticks([])
        self.camera_visu.axes.set_yticks([])

    def next(self, event=None):

        self.data = np.array(list(self.event_iterator.__next__().dl0.tel[self.telescope_id].adc_samples.values()))
        self.update()
        self.event_id += 1

    def next_camera_view(self, event=None):

        self.camera_view = next(self.iterator_camera_view)
        self.update()

    def next_view_type(self, event=None):

        self.view_type = next(self.iterator_view_type)
        self.update()

    def update(self):

        self.draw_readout(self.pixel_id)
        self.draw_camera()

    def draw(self):

        self.next()
        button_next_event = Button(self.axis_next_event_button, 'next')
        button_next_camera_view = Button(self.axis_next_camera_view_button, 'view')
        button_next_view_type = Button(self.axis_next_view_type_button, 'type')
        button_next_event.on_clicked(self.next)
        button_next_camera_view.on_clicked(self.next_camera_view)
        button_next_view_type.on_clicked(self.next_view_type)
        plt.show()

    def draw_readout(self, pix):

        x = 4 * np.arange(0, self.data.shape[-1], 1)
        y = self.compute_trace()[pix]
        self.pixel_id = pix
        self.event_clicked_on.ind[-1] = self.pixel_id
        self.axis_readout.cla()
        self.axis_readout.step(x, y,
                          label='%s %d' % (self.view_type, self.pixel_id))
        self.axis_readout.set_xlabel('t [ns]')
        self.axis_readout.set_ylabel('[ADC]')
        self.axis_readout.legend(loc='upper right')
        self.axis_readout.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        self.axis_readout.yaxis.set_major_locator(MultipleLocator(1))


    def draw_camera(self):

        self.camera_visu.image = self.compute_image()
        self.camera_visu.on_pixel_clicked = self.draw_readout
        self.camera_visu.colorbar.set_label('%s [ADC]' % self.camera_view)
        self.camera_visu._on_pick(self.event_clicked_on)
        anchored_text = AnchoredText('%s #%d' % ('MC' if self.mc else 'data',self.event_id), loc=1, prop=dict(size=14))
        self.axis_camera.add_artist(anchored_text)

    def compute_trace(self):

        image = self.data

        if self.view_type in self.view_types:

            if not self.view_type=='pixel':

                baseline = np.mean(image[..., 0:self.baseline_window_width], axis=1)
                image = image - baseline[:, np.newaxis]

                cluster_trace, patch_trace = trigger.compute_cluster_trace(image, self.camera, self.options)

                for pixel_id in range(self.data.shape[0]):

                    if self.view_type == 'patch':

                        image[pixel_id] = patch_trace[self.camera.Pixels[pixel_id].patch]

                    elif self.view_type == 'cluster_7':

                        image[pixel_id] = cluster_trace[self.camera.Pixels[pixel_id].patch]

                    elif self.view_type == 'cluster_9':

                        print('Cluster 19 not implemented')

                        image = np.zeros(self.data.shape)

        return image

    def compute_image(self):

        image = self.compute_trace()


        if self.camera_view in self.camera_views:

            if self.camera_view == 'mean':

                image = np.mean(image, axis=1)

            elif self.camera_view == 'std':

                image = np.std(image, axis=1)

            elif self.camera_view == 'max':

                image = np.max(image, axis=1)

            elif self.camera_view == 'sum':

                image = np.sum(image, axis=1)

        else:

            print('Cannot compute for camera type : %s' % self.camera_view)

        return image


class Event_Clicked():

    def __init__(self, options):
        self.ind = [0, options.pixel_start]




