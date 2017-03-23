#!/usr/bin/env python3


from ctapipe import visualization
import numpy as np
from matplotlib import pyplot as plt
from utils.histogram import Histogram
from matplotlib.widgets import Button
import scipy.stats
from spectra_fit import fit_low_light
from astropy import units as u
from matplotlib.widgets import Button
import sys
from matplotlib.offsetbox import AnchoredText
from utils import geometry


class Histogram_Viewer():

    def __init__(self, histogram, options, pixel_start=0, level_start=0):

        self.histogram = histogram
        self.options = options
        self.geometry = geometry.generate_geometry_0(pixel_list=options.pixel_list)
        self.shape = self.histogram.data.shape
        self.pixel_start = pixel_start
        self.level_start = level_start
        self.fitted = False if np.isnan(self.histogram.fit_result) else True
        self.figure = plt.figure(figsize=(48, 27))
        self.axis_histogram = self.figure.add_subplot(121)
        self.axis_camera = self.figure.add_subplot(122)
        self.camera = visualization.CameraDisplay(self.geometry, ax=self.axis_camera, title='', norm='lin', cmap='viridis', allow_pick=True)

        if len(self.shape) >= 3:

            self.n_level = self.shape[0]
            self.n_pixel = self.shape[1]
            self.count_level = self.level_start
            self.count_pixel = self.pixel_start

        else:

            self.n_pixel = self.shape[0]
            self.count_pixel = self.pixel_start

        self.image = np.zeros(self.n_pixel)

        if self.fitted:

            self.parameter_start = 0
            self.n_parameter = self.histogram.fit_result.shape[-2]
            self.axis_parameter = self.figure.add_subplot(221)
            self.axis_histogram = self.figure.add_subplot(212)
            self.axis_camera = self.figure.add_subplot(222)


        self._update_count()


    def _press(self, event):

        sys.stdout.flush()
        if event.key == '+':

            self._next_pixel()

        elif event.key == '-':

            self._previous_pixel()

        elif event.key == '/':

            self._previous_level()

        elif event.key == '*':

            self._next_level()

        elif event.key == '.':

            self._next_param()

        else:
            print('Invalid key : %s' % event.key)

        self._update_count()
        self._update()



    def _next_pixel(self):

        if self.count_pixel<self.n_pixel:
            self.count_pixel +=1
        else:
            self.count_pixel = 0

    def _previous_pixel(self):

        if self.count_pixel>0:
            self.count_pixel -=1
        else:
            self.count_pixel = self.n_pixel

    def _next_level(self):
        if self.count_level<self.n_level:
            self.count_level +=1
        else:
            self.count_level = 0

    def _previous_level(self):
        if self.count_level>0:
            self.count_level -=1
        else:
            self.count_level = self.n_level


    def _next_param(self):

        if self.count_param<self.n_parameter:
            self.count_param += 1

        else:
            self.count_param = 0

    def _update_count(self):

        if len(self.shape) >= 3:
            self.count = (self.count_level, self.count_pixel, )

        else:
            self.count = (self.count_pixel, )



    def _update(self):

        self.camera.visu = d

