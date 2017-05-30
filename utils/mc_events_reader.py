import numpy as np
import h5py
import logging
import sys
from ctapipe.io.containers import DataContainer, DigiCamCameraContainer


def hdf5_mc_event_source(url, events_per_dc_level, events_per_ac_level, dc_start=0, ac_start=0, max_events=None, bootstrap=False):

    try:
        hdf5 = h5py.File(url, 'r')
    except:
        raise NameError('hdf5_mc_event_source failed to open %s' % url)

    dc_level = dc_start
    ac_level = ac_start
    count = 0

    for event_id in range(max_events):

        data = DataContainer()
        data.meta['hdf5_input'] = url
        data.meta['hdf5_max_events'] = max_events
        data.r0.run_id = event_id
        data.r0.event_id = event_id
        data.r0.tels_with_data = [0, ]
        data.count = event_id

        for telescope_id in data.r0.tels_with_data:

            data.inst.num_channels[telescope_id] = 1
            data.inst.num_pixels[telescope_id] = len(hdf5['simulation_parameters']['gain'])
            data.r0.tel[telescope_id] = DigiCamCameraContainer()
            data.r0.tel[telescope_id].camera_event_number = event_id
            data.r0.tel[telescope_id].pixel_flags = np.ones(data.inst.num_pixels[telescope_id])
            data.r0.tel[telescope_id].local_camera_clock = 0
            # data.r0.tel[telescope_id].event_type = 'MC : DigiCamToy'
            end_time = hdf5['simulation_parameters']['end_time'][()]
            start_time = hdf5['simulation_parameters']['start_time'][()]
            sampling_time = hdf5['simulation_parameters']['sampling_time'][()]

            data.r0.tel[telescope_id].num_samples = (start_time - end_time) // sampling_time + 1

            mc_data = hdf5['dc_level_%d_ac_level_%d' % (dc_level, ac_level)]['data']

            dict_data = {}

            for pixel in range(mc_data.shape[0]):

                if bootstrap:
                    event_index_in_file = random_state.randint(0, mc_data.shape[-1])

                else:
                    event_index_in_file = count

                dict_data[pixel] = mc_data[pixel, :, event_index_in_file]

            data.r0.tel[telescope_id].adc_samples = dict_data

        count += 1
        if events_per_ac_level != 0 and (count % events_per_ac_level) == 0:
            count = 0
            ac_level += 1

        if events_per_dc_level != 0 and (count % events_per_dc_level) == 0:
            count = 0
            dc_level += 1



        yield data
