import numpy as np
import datetime
import pickle

class Calibration_Container():

    """
    a container with all calibration parameters for SST-1M camera
    each field contains a dict() of np.array
    """

    def __init__(self, filename=None):

        if filename is not None:

            self._load(filename)

        else:

            self.pixel_id = [i for i in range(1296)]
            self.n_pixels = len(self.pixel_id)

            ### SiPMs

            self.gain = {'value' : [None]*self.n_pixels, 'error': [None]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}
            self.electronic_noise = {'value' : [None]*self.n_pixels, 'error': [None]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}
            self.gain_smearing = {'value' : [None]*self.n_pixels, 'error': [None]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}
            self.crosstalk = {'value' : [None]*self.n_pixels, 'error': [None]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}
            self.baseline = {'value' : [None]*self.n_pixels, 'error': [None]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}
            self.mean_temperature = {'value' : [None]*self.n_pixels, 'error': [None]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}

            ### LEDs

            self.ac_led = {'value' : [[None]*4]*self.n_pixels, 'error': [[None]*4]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}
            self.dc_led = {'value' : [[None]*2]*self.n_pixels, 'error': [[None]*2]*self.n_pixels, 'time_stamp': [None]*self.n_pixels}



    def update(self, field, indices, value, error=None):

        class_attribute = getattr(self, field)

        for i, index in enumerate(indices):

            class_attribute['value'][index] = value[i]

            if error[i] is not None:

                class_attribute['error'][index] = error[i]

            class_attribute['time_stamp'][index] = datetime.datetime.now()



    def save(self, filename):

        with open(filename, 'wb') as output:

            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)

    def _load(self, filename):

        with open(filename, 'rb') as output:

            tmp_dict = pickle.load(output)
            self.__dict__.update(tmp_dict)


if __name__ == '__main__':

    print(datetime.datetime.now())

    test = Calibration_Container()

    test.update('gain', [4], [1000], [0.5])


    print(test.gain['time_stamp'][4])

    test.update('gain', [4, 5, 6, 7, 8], [1000]*5, [0.5]*5)

    print(test.gain['time_stamp'][4:8])

    filename = 'camera_calibration_container.pk'
    test.save(filename)

    a = Calibration_Container(filename)

    print(a.gain['time_stamp'][4:8])
    print (a.pixel_id)
    print (a.dc_led['value'][1])




