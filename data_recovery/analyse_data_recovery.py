import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    directory = './'
    filename = 'temp_0_dc.npz'

    data = np.load(directory + filename)
    print(data['time'].shape)
    time = data['time']
    time = time - np.min(time)
    std = data['std']
    time_diff = np.diff(time)
    # bins = np.logspace(np.log(np.min(time_diff)), np.log(np.max(time_diff)), num=100)
    bins = np.linspace(np.min(time_diff), np.max(time_diff), num=100)

    # print(np.argwhere(time_diff > 5 * 1E9))
    # print(time[np.argwhere(time_diff > 5 * 1E9)[0][0]] * 1E-9)



    plt.figure()
    plt.hist(time_diff * 1E-9, bins=bins * 1E-9, log=True)
    plt.xlabel('[s]')


    plt.figure()
    plt.plot(std, marker='.', linestyle='None')
    plt.xlabel('event id')

    plt.figure()
    plt.plot(time * 1E-9, std, marker='.', linestyle='None')
    plt.xlabel('time [s]')

    plt.figure()
    plt.plot(time * 1E-9, marker='.', linestyle='None')
    plt.xlabel('event id')
    plt.ylabel('time [s]')

    plt.show()