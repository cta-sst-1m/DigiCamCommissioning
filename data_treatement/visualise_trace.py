import numpy as np
import sys
from ctapipe.io import zfits
from utils.toy_reader import ToyReader
from ctapipe import visualization
from utils import geometry
from data_treatement import trigger
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, MaxNLocator
from cts_core.camera import Camera
from matplotlib.widgets import CheckButtons, Button, RadioButtons, Slider
from matplotlib.offsetbox import AnchoredText
from itertools import cycle
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import matplotlib as mpl


class EventViewer():

    def __init__(self, options):

        plt.ioff()
        #mpl.figure.autolayout = False
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
        self.event_id = 0
        self.first_call = True
        self.time = options.bin_start
        self.n_bins = options.n_bins
        self.pixel_id = options.pixel_start
        self.data = np.array(list(self.event_iterator.__next__().dl0.tel[self.telescope_id].adc_samples.values()))

        self.event_clicked_on = Event_Clicked(options)
        self.geometry = geometry.generate_geometry_0()
        self.camera = Camera(options.cts_directory + 'config/camera_config.cfg',
                             options.cts_directory + 'config/cluster.p')

        self.view_type = options.view_type
        self.view_types = ['pixel', 'patch', 'cluster_7']#, 'cluster_9']
        self.iterator_view_type = cycle(self.view_types)
        self.camera_view = options.camera_view
        self.camera_views = ['sum', 'mean', 'max', 'std', 'time']
        self.iterator_camera_view = cycle(self.camera_views)
        self.figure = plt.figure(figsize=(20, 10))

        ## Readout

        self.readout_x = 4 * np.arange(0, self.data.shape[-1], 1)
        self.axis_readout = self.figure.add_subplot(122)
        self.axis_readout.set_xlabel('t [ns]')
        self.axis_readout.set_ylabel('[ADC]')
        self.axis_readout.legend(loc='upper right')
        self.axis_readout.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        self.axis_readout.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        self.trace_time_plot, = self.axis_readout.plot(np.array([self.time, self.time]) * 4, np.array([np.min(self.data[self.pixel_id]), np.max(self.data[self.pixel_id])]), color='r',
                               linestyle='--')
        self.trace_readout, = self.axis_readout.step(self.readout_x, self.data[self.pixel_id],
                               label='%s %d' % (self.view_type, self.pixel_id), where='mid')
        self.axis_readout.legend(loc='upper right')

        ## Camera

        self.axis_camera = self.figure.add_subplot(121)
        self.axis_camera.axis('off')
        self.camera_visu = visualization.CameraDisplay(self.geometry, ax=self.axis_camera, title='', norm=self.scale,
                                                       cmap='viridis',
                                                       allow_pick=True)
        self.camera.image = self.compute_image()
        self.camera_visu.cmap.set_bad(color='k')
        self.camera_visu.add_colorbar(orientation='horizontal', pad=0.03, fraction=0.05)
        #self.camera_visu.colorbar.set_norm(LogNorm(vmin=1, vmax=None, clip=False))
        self.camera_visu.colorbar.set_label('[ADC]')
        self.camera_visu.axes.get_xaxis().set_visible(False)
        self.camera_visu.axes.get_yaxis().set_visible(False)
        self.camera_visu.on_pixel_clicked = self.draw_readout

        ## Buttons

        readout_position = self.axis_readout.get_position()
        self.axis_next_event_button = self.figure.add_axes([0.35, 0.9, 0.13, 0.07], zorder=np.inf)
        self.axis_next_camera_view_button = self.figure.add_axes([0., 0.85, 0.1, 0.15], zorder=np.inf)
        self.axis_next_view_type_button = self.figure.add_axes([0., 0.18, 0.1, 0.15], zorder=np.inf)
        #self.axis_slider_time = self.figure.add_axes([readout_position.x0 - 0.018, readout_position.y1 + 0.005, readout_position.x1 - readout_position.x0 + 0.005, 0.02], facecolor='lightgoldenrodyellow', zorder=np.inf)
        self.axis_next_camera_view_button.axis('off')
        self.axis_next_view_type_button.axis('off')
        self.button_next_event = Button(self.axis_next_event_button, 'show event # %d' %(self.event_id + 1))
        self.radio_button_next_camera_view = RadioButtons(self.axis_next_camera_view_button, self.camera_views)
        self.radio_button_next_view_type = RadioButtons(self.axis_next_view_type_button, self.view_types)
        self.radio_button_next_view_type.set_active(self.view_types.index(self.view_type))
        self.radio_button_next_camera_view.set_active(self.camera_views.index(self.camera_view))
        #self.slider_time = Slider(self.axis_slider_time, '', 0, options.n_bins - 1, valinit=self.time, valfmt='%d')

    def next(self, event=None):

        if not self.first_call:
            self.data = np.array(list(self.event_iterator.__next__().dl0.tel[self.telescope_id].adc_samples.values()))

        self.update()
        self.first_call = False
        self.event_id += 1

    def next_camera_view(self, camera_view, event=None):

        self.camera_view = camera_view
        self.update()

    def next_view_type(self, view_type, event=None):

        self.view_type = view_type
        self.update()

    def set_time(self, time):

        self.time = time
        self.update()

    def update(self):

        self.draw_readout(self.pixel_id)
        self.draw_camera()
        self.button_next_event.label.set_text('show event #%d' %(self.event_id + 1))
        #self.slider_time.set_val(self.time)


    def draw(self):

        self.next()
        self.button_next_event.on_clicked(self.next)
        self.radio_button_next_camera_view.on_clicked(self.next_camera_view)
        self.radio_button_next_view_type.on_clicked(self.next_view_type)
        #self.slider_time.on_changed(self.set_time)
        self.figure.canvas.mpl_connect('key_press_event', self.press)
        self.camera_visu._on_pick(self.event_clicked_on)
        plt.show()

    def draw_readout(self, pix):

        y = self.compute_trace()[pix]
        limits_y = [np.min(y), np.max(y) + 1]
        self.pixel_id = pix
        self.event_clicked_on.ind[-1] = self.pixel_id
        self.trace_readout.set_ydata(y)
        self.trace_readout.set_label('%s : %d, time : %d' % (self.view_type, self.pixel_id, self.time))
        self.trace_time_plot.set_ydata(limits_y)
        self.trace_time_plot.set_xdata(self.time*4)
        #y_ticks = np.linspace(np.min(y), np.max(y) + (np.max(y)-np.min(y)//10), np.max(y)-np.min(y)//10)
        #self.axis_readout.set_yticks(np.linspace(np.min(y), np.max(y), 8).astype(int))
        self.axis_readout.set_ylim(limits_y)
        self.axis_readout.legend(loc='upper right')


        #self.axis_readout.cla()
        #self.axis_readout.plot(np.array([self.time, self.time])*4, np.array([np.min(y), np.max(y)]), color='r', linestyle='--')
        #self.axis_readout.step(x, y,
         #                 label='%s %d' % (self.view_type, self.pixel_id), where='mid')



    def draw_camera(self):

        self.camera_visu.image = self.compute_image()

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

            elif self.camera_view == 'time':

                image = image[:, self.time]

        else:

            print('Cannot compute for camera type : %s' % self.camera_view)

        return np.ma.masked_where(image<=0, image)

    def press(self, event):

        sys.stdout.flush()

        if event.key=='enter':

            self.next()

        if event.key=='right':

            self.set_time(self.time + 1)

        if event.key=='left':

            self.set_time(self.time - 1)

    def save(self, filename='test.png'):

        self.set_buttons_visible(False)
        self.figure.savefig(filename)
        self.set_buttons_visible(True)

        return self.figure

    def set_buttons_visible(self, visible=True):

        self.axis_next_camera_view_button.set_visible(visible)
        self.axis_next_view_type_button.set_visible(visible)
        #self.axis_slider_time.set_visible(visible)
        self.axis_next_event_button.set_visible(visible)









class Event_Clicked():

    def __init__(self, options):
        self.ind = [0, options.pixel_start]




