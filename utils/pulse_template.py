import numpy as np
from scipy.interpolate import splev, splrep, splint
import pickle
import matplotlib.pyplot as plt


class PulseTemplate:

    def __init__(self, filename=None):

        if filename is not None:

            self._load(filename)

        else:

            n_pixels = 1296
            self.splines = [None]*n_pixels
            self.splines_square = [None]*n_pixels
            self.splines_derivative = [None]*n_pixels
            self.integral = np.zeros(n_pixels)
            self.integral_square = np.zeros(n_pixels)
            self.t_min = 0
            self.t_max = None

    def interpolate(self, pulse_data, pixels_id, **kwargs):

        time_data = np.arange(0, pulse_data.shape[-1], 1) * 4
        self.t_min = time_data[0]
        self.t_max = time_data[-1]
        index_max = np.argmax(pulse_data, axis=-1)

        for i, pixel in enumerate(pixels_id):

            start_index = max(index_max[i] - 1, 0)
            end_index = min(index_max[i] + 1, pulse_data.shape[-1] - 1)
            baseline = np.mean(pulse_data[i][min(start_index + 10, pulse_data.shape[-1] - 1):pulse_data.shape[-1]])
            pulse_data[i] = pulse_data[i] - baseline
            t_temp = np.linspace(time_data[start_index], time_data[end_index], 100)
            self.splines[pixel] = splrep(time_data, pulse_data[i], **kwargs)
            maximum = np.max(splev(t_temp, self.splines[pixel]))

            self.splines[pixel] = splrep(time_data, pulse_data[i]/maximum, **kwargs)
            self.splines_square[pixel] = splrep(time_data, (pulse_data[i]/maximum)**2, **kwargs)
            self.integral[pixel] = self.compute_integral([pixel])
            self.integral_square[pixel] = self.compute_integral([pixel], moment=2)

    def evaluate(self, time, pixels_id, derivative=0, moment=1):

        return [splev(time, self._get_spine(pixel, moment), der=derivative) for pixel in pixels_id]

    def compute_integral(self, pixels_id, moment=1, a=None, b=None):

        if a is None:
            a = self.t_min
        if b is None:
            b = self.t_max

        return np.array([splint(a, b, self._get_spine(pixel, moment)) for pixel in pixels_id])

    def compute_rise_time(self, pixels_id):

        pass

    def _get_spine(self, pixel, moment=1):

        if moment == 1:

            return self.splines[pixel]

        elif moment == 2:

            return self.splines_square[pixel]

        else:
            raise ValueError('Invalid value for moment = %d' % moment)

    def display(self, pixels_id, derivative=0, moment=1, axis=None, **kwargs):

        time = np.linspace(self.t_min, self.t_max, num=1000)
        lsb = self.evaluate(time=time, pixels_id=pixels_id, derivative=derivative, moment=moment)

        if axis is None:

            fig = plt.figure(figsize=(10, 10))
            axis = fig.add_subplot(111)

        for i, pixel in enumerate(pixels_id):
            axis.plot(time, lsb[i], label='pixel : %d' % pixel, **kwargs)
        axis.set_xlabel('t [ns]')
        axis.set_ylabel('[LSB]')
        axis.legend()

        return axis

    def save(self, filename):

        with open(filename, 'wb') as output:

            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

    def _load(self, filename):

        with open(filename, 'rb') as output:

            tmp_dict = pickle.load(output)
            self.__dict__.update(tmp_dict)


class NPulseTemplate:

    def __init__(self, shape=None, filename=None):

        if filename is not None:

            self._load(filename)

        else:

            self.shape = shape

            if len(shape) != 2:
                raise ValueError('Can treat only 2D pulse template')

            self.pulse_template = [[None]*shape[1]]*shape[0]

    def interpolate(self, pulse_data, pixels_id, **kwargs):

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):

                self.pulse_template[i][j] = PulseTemplate()
                self.pulse_template[i][j].interpolate(pulse_data[i, j], pixels_id, **kwargs)

    def display(self, pixel_list, indices=(0, 0, ), derivative=0, moment=1, axis=None):

        axis = self.pulse_template[indices[0]][indices[1]].display(pixel_list, derivative=derivative, moment=moment, axis=axis)
        return axis

    def save(self, filename):

        with open(filename, 'wb') as output:

            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

    def _load(self, filename):

        with open(filename, 'rb') as output:

            tmp_dict = pickle.load(output)
            self.__dict__.update(tmp_dict)

