import numpy as np
from ctapipe.io import zfits
from ctapipe.io.serializer import Serializer
import h5py


def run(file_basename, file_list, timeout, ac_level, n_events):

    data_list = []
    time_previous = 0.
    ac_level_id = 0
    first_event = True

    for file_number in file_list:

        file_name = file_basename % file_number

        event_stream = zfits.zfits_event_source(url=file_name)

        for event in event_stream:

            for telid in event.r0.tels_with_data:

                time = event.r0.tel[telid].local_camera_clock
                data = np.array(list(event.r0.tel[telid].adc_samples.values()))
                data_list.append(data)

                if first_event:

                    n_pixels, n_bins = data.shape[0], data.shape[1]
                    first_event = False

                if (time - time_previous) > timeout:

                    data_list = np.array(data_list)
                    data_list = data_list[-n_events:]
                    data_list = np.rollaxis(data_list, 0, 3)

                    with h5py.File(directory + "ac_level_%d.hdf5" % ac_level[ac_level_id], "w") as h5py_file:

                        data_set = h5py_file.create_dataset('data', (n_pixels, n_bins, n_events), dtype='i', compression='gzip', compression_opts=5)
                        data_set[...] = data_list

                    data_list = []
                    ac_level_id += 1

                time_previous = time
    return


if __name__ == '__main__':

    n_files = 23
    file_list = np.arange(0, n_files)
    directory = '/data/datasets/CTA/DATA/FULLSEQ/ac_scan_0/'
    file_basename = directory + 'CameraDigicam@localhost.localdomain_0_000.%d.run_516.fits.fz'
    timeout = 1E9
    ac_level = [0, 50, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 640, 680, 720, 760, 800]
    n_events = 5000