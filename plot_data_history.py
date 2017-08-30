import numpy as np
from ctapipe.io import zfits
from utils.hdf5_reader import hdf5_event_source
import matplotlib.pyplot as plt


def run(filename, file_list, pixels):
    data_max_min = []
    data_mean = []
    data_std = []
    time = []
    for file_number in file_list:

        file_name = filename % file_number

        print('reading file : ', file_number)

        if '.fits.fz' in filename:

            event_stream = zfits.zfits_event_source(url=file_name)

        elif '.hdf5' in filename:

            event_stream = hdf5_event_source(url=file_name)

        for event in event_stream:

            for telid in event.r0.tels_with_data:
                data = np.array(list(event.r0.tel[telid].adc_samples.values()))
                data = data[pixel]
                time.append(event.r0.tel[telid].local_camera_clock)
                data_max_min.append(np.max(data, axis=-1) - np.min(data, axis=-1))
                data_mean.append(np.mean(data, axis=-1))
                data_std.append(np.std(data, axis=-1))

    return np.array(data_max_min), np.array(data_mean), np.array(data_std), np.array(time)


if __name__ == '__main__':

    n_files = 64
    file_list = np.arange(0, n_files)
    file_basename = 'CameraDigicam@localhost.localdomain_0_000.%d.run_452.fits.fz'
    # file_basename = 'ac_level_%d.hdf5'
    directory = '/data/datasets/CTA/DATA/FULLSEQ/ac_scan_120/'
    temp_filename = 'temp_120_fullseq.npz'
    filename = directory + file_basename
    pixel = [60, 115]

#    data_max_min, data_mean, data_std, time = run(filename=filename, file_list=file_list, pixels=pixel)

#   np.savez(temp_filename, max_min=data_max_min, mean=data_mean, std=data_std, time=time)

    data = np.load(temp_filename)

    data_max_min = data['max_min']
    data_mean = data['mean']
    data_std = data['std']
    time = data['time']

    plt.figure()
    colors = ['g', 'r']
    # plt.semilogy(np.diff(time), label='$\Delta t$')
    for pixel in range(data_max_min.shape[1]):
        pass
        # plt.semilogy(data_max_min[:, pixel], label='max - min', marker='o', linestyle='None', color=colors[pixel])
        # plt.semilogy(data_mean[:, pixel], label='mean', marker='x', linestyle='None', color=colors[pixel])
        plt.semilogy(time, data_std[:, pixel], label='$\Delta std$', marker='v', linestyle='None', color=colors[pixel])

    plt.xlabel('event number')
    plt.legend()
    plt.show()
