import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    directory = '/data/datasets/CTA/DATA/data_recovery/'
    filename = 'temp_120.npz'

    data = np.load(directory + filename)
    print(data['time'].shape)
    time = data['time']
    std = data['std']
    time_diff = np.diff(time)
    # bins = np.logspace(np.log(np.min(time_diff)), np.log(np.max(time_diff)), num=100)
    bins = np.linspace(np.min(time_diff), np.max(time_diff), num=100)

    plt.figure()
    plt.hist(time_diff, bins=bins, log=True)
    plt.xlabel('[s]')


    plt.figure()
    plt.plot(std, marker='.', linestyle='None')
    plt.xlabel('event id')

    plt.figure()
    plt.plot(time, std, marker='.', linestyle='None')
    plt.xlabel('time [ns]')

    plt.figure()
    plt.plot(time, marker='.', linestyle='None')
    plt.xlabel('event id')
    plt.ylabel('time [ns]')

    plt.show()