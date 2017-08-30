import numpy as np
import h5py
import logging
import sys
from ctapipe.io.containers import DataContainer, DigiCamCameraContainer


def hdf5_event_source(url):

    try:
        hdf5 = h5py.File(url, 'r')
    except:
        raise NameError('failed to open %s' % url)

    max_events = np.inf
    event_id = 0

    while event_id < max_events:

        data = DataContainer()
        data.meta['hdf5_input'] = url
        data.meta['hdf5_max_events'] = max_events
        data.r0.run_id = event_id
        data.r0.event_id = event_id
        data.r0.tels_with_data = [0, ]
        data.count = event_id

        for telescope_id in data.r0.tels_with_data:

            input_data = hdf5['data']
            max_events = input_data.shape[-1]

            data.inst.num_channels[telescope_id] = 1
            data.inst.num_pixels[telescope_id] = input_data.shape[0]
            data.r0.tel[telescope_id] = DigiCamCameraContainer()
            data.r0.tel[telescope_id].camera_event_number = event_id
            data.r0.tel[telescope_id].pixel_flags = np.ones(data.inst.num_pixels[telescope_id])
            data.r0.tel[telescope_id].local_camera_clock = 0
            data.r0.tel[telescope_id].num_samples = input_data.shape[1]
            data.r0.tel[telescope_id].adc_samples = dict(zip(np.arange(data.inst.num_pixels[telescope_id]), input_data[..., event_id]))

        event_id += 1

        yield data
