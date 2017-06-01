import numpy as np
from scipy.optimize import splev, splrep, splint
import pickle
import matplotlib.pyplot as plt


class PulseTemplate:

    def __init__(self, filename=None):

        if filename is not None:

            self._load(filename)

        else:

            self.n_pixels = 1296
            self.splines = [None]*self.n_pixels
            self.splines_square = [None]*self.n_pixels
            self.splines_derivative = [None]*self.n_pixels
            self.integral = np.zeros(self.n_pixels)
            self.integral_square = np.zeros(self.n_pixels)
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
            self.splines_square[pixel] = splrep(time_data, (pulse_data[i]/max)**2, **kwargs)
            self.integral[pixel] = self.compute_integral([pixel])
            self.integral_square[pixel] = self.compute_integral([pixel], moment=2)

    def evaluate(self, time, pixels_id, derivative=0, moment=1):

        return [splev(time, self._get_spine(pixel, moment), der=derivative) for pixel in pixels_id]

    def compute_integral(self, pixels_id, moment=1):

        return [splint(self._get_spine(pixel, moment)) for pixel in pixels_id]

    def compute_rise_time(self, pixels_id):

        pass

    def _get_spine(self, pixel, moment=1):

        if moment == 1:

            return self.splines[pixel]

        elif moment == 2:

            return self.splines_square[pixel]

        else:
            raise ValueError('Invalid value for moment = %d' % moment)

    def display(self, pixels_id, derivative=0, moment=1, **kwargs):

        time = np.linspace(0, 92 * 4, num=1000)
        lsb = self.evaluate(time=time, pixels_id=pixels_id, derivative=derivative, moment=moment)

        plt.figure()
        for i, pixel in enumerate(pixels_id):
            plt.plot(time, lsb[i], label='pixel : %d' % pixel, **kwargs)
        plt.xlabel('t [ns]')
        plt.ylabel('[LSB]')
        plt.legend()
        plt.show()

    def save(self, filename):

        with open(filename, 'wb') as output:

            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

    def _load(self, filename):

        with open(filename, 'rb') as output:

            tmp_dict = pickle.load(output)
            self.__dict__.update(tmp_dict)














