from ctapipe.calib.camera.charge_extractors import Integrator
import numpy as np

class MovingWindowIntegrator(Integrator):
    name = 'MovingWindowIntegrator'

    def __init__(self, config, tool, **kwargs):
        super().__init__(config=config, tool=tool, **kwargs)

    def _get_window_width(self):
        '''
        Find the max within the width, then from this point on:
          - If it is below preamp saturation, reduce the integration width to the max or close to the max
          - It is is above, find the point where it goes below 1.p.e on each side

        Returns
        -------
        width : ndarray
            Numpy array containing the window width for each pixel. Shape =
            (n_chan, n_pix)
        '''
        return np.full((self._nchan, self._npix), self._nsamples,
                       dtype=np.intp)

    def _get_window_start(self, waveforms):
        '''
        Find the max within the width, then from this point on:
          - If it is below preamp saturation, reduce the integration width to the max or close to the max
          - It is is above, find the point where it goes below 1.p.e on each side

        Returns
        -------
        start : ndarray
            Numpy array containing the window start for each pixel. Shape =
            (n_chan, n_pix)

        '''
        return np.zeros((self._nchan, self._npix), dtype=np.intp)


    def extract_charge(self, waveforms):
        self.check_neighbour_set()
        self._nchan, self._npix, self._nsamples = waveforms.shape
        w_width = self._get_window_width()
        w_start = self._get_window_start(waveforms)
        self._check_window_width_and_start(w_width, w_start)
        window = self._define_window(w_start, w_width)
        windowed_waveforms = self._window_waveforms(waveforms, window)
        charge = self._integrate(windowed_waveforms)

        self.extracted_samples = window
        return charge

    @abstractmethod
    def _get_window_width(self):
        """
        Get the width of the integration window

        Returns
        -------
        width : ndarray
            Numpy array containing the window width for each pixel. Shape =
            (n_chan, n_pix)

        """

    @abstractmethod
    def _get_window_start(self, waveforms):
        """
        Get the starting point for the integration window

        Returns
        -------
        start : ndarray
            Numpy array containing the window start for each pixel. Shape =
            (n_chan, n_pix)

        """
