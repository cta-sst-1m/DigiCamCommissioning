import numpy as np
from scipy.optimize import splev, splrep, splint, splder


class PulseTemplate:

    def __init__(self, n_pixels=1296):

        self.n_pixels = n_pixels
        self.splines = [None]*self.n_pixels
        self.splines_derivative = [None]*self.n_pixels
        self.t_min = 0
        self.t_max = None

    def interpolate(self, pulse_data, pixels_id, **kwargs):

        time_data = np.arange(0, pulse_data.shape[-1], 1) * 4
        index_max = np.argmax(time_data)
        t_temp = np.linspace(time_data[index_max-1], time_data[index_max+1], 100)

        for i, pixel in enumerate(pixels_id):

            self.splines[pixel] = splrep(time_data, pulse_data[i], **kwargs)
            max = np.max(splev(t_temp, self.splines[pixel]))
            self.splines[pixel] = splrep(time_data, pulse_data[i]/max, **kwargs)

            self.spines_derivative[pixel] = splder(self.splines[pixel])

    def evaluate(self, time, pixels_id):

        return [splev(time, self.splines[pixel]) for pixel in pixels_id]

    def compute_integral(self, pixels_id):

        return [splint(self.splines[pixel]) for pixel in pixels_id]

    def compute_rise_time(self, pixels_id):

        pass













