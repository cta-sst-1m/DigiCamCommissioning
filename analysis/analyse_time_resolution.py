#!/usr/bin/env python3

# external modules

# internal modules
import logging
import sys
import numpy as np
import logging
from tqdm import tqdm
from utils.logger import TqdmToLogger
from data_treatement import timing
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py

__all__ = ["create_histo", "perform_analysis", "display_results"]


def create_histo(options):

    arrival_time = timing.run(options)
    np.savez(file=options.output_directory + options.timing_filename, arrival_time=arrival_time)

    return


def perform_analysis(options):
    """
    Perform a simple gaussian fit of the ADC histograms

    :param options: a dictionary containing at least the following keys:
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram
                                                 whose fit contains the gain,sigmas etc...(str)

    :return:
    """

    data = np.load(file=options.output_directory + options.timing_filename)
    arrival_time = data['arrival_time']
    mean_arrival_time = np.mean(arrival_time, axis=2)

    #print(arrival_time[1,0])

    timing_matrix = np.zeros((arrival_time.shape[0], arrival_time.shape[1], arrival_time.shape[1]))
    std_matrix = np.zeros((arrival_time.shape[0], arrival_time.shape[1], arrival_time.shape[1]))
    diff = np.zeros((arrival_time.shape[0], arrival_time.shape[1], arrival_time.shape[2], arrival_time.shape[1]))

    print(diff.shape)

    #print(diff.shape)

    for level in range(timing_matrix.shape[0]):
        x = np.reshape(arrival_time[level], arrival_time[level].shape + (1, ))
        diff[level] = x - x.transpose()

        timing_matrix[level, :, :] = np.mean(diff[level], axis=1)
        std_matrix[level, :, :] = np.std(diff[level], axis=1)


    np.savez(file=options.output_directory + options.timing_filename, arrival_time=arrival_time, mean_arrival_time=mean_arrival_time, timing_matrix=timing_matrix, std_matrix=std_matrix, diff=diff)

    return


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    data = np.load(file=options.output_directory + options.timing_filename)
    arrival_time = data['arrival_time']
    timing_matrix = data['timing_matrix']
    std_matrix = data['std_matrix']
    mean_arrival_time = data['mean_arrival_time']
    diff = data['diff']

    for level in options.scan_level:


        ticks = options.pixel_list
        ticks_position = np.arange(0, len(options.pixel_list), 1)
        plt.figure(figsize=(10, 10))
        plt.title('level %d' % level)
        plt.imshow(timing_matrix[level])
        plt.colorbar(label='$\Delta t$ [ns]', fraction=0.046, pad=0.04)
        plt.xticks(ticks_position, ticks, rotation='vertical')
        plt.yticks(ticks_position, ticks)
        plt.xlabel('pixel id')
        plt.ylabel('pixel id')
        plt.grid('off')

        plt.figure(figsize=(10, 10))
        plt.title('level %d' % level)
        plt.imshow(std_matrix[level])
        plt.xticks(ticks_position, ticks, rotation='vertical')
        plt.yticks(ticks_position, ticks)
        plt.xlabel('pixel id')
        plt.ylabel('pixel id')
        plt.colorbar(label='$\sigma_t$ [ns]', fraction=0.046, pad=0.04)
        plt.grid('off')

        temp = np.tril(timing_matrix[level], -1).ravel()
        plt.figure(figsize=(10, 10))
        #plt.title('level %d' % level)

        y, x = np.histogram(temp*1E3, bins=np.linspace(np.min(temp*1E3), np.max(temp*1E3), 10))

        plt.errorbar(x[0:-1], y, yerr=np.sqrt(y), label='mean : %0.2f [ps] \n std : %0.2f [ps]' %(np.mean(temp)*1E3, np.std(temp)*1E3), marker='o', color='k', linestyle='None')
        plt.step(x[0:-1], y, where='mid')
        plt.xlabel('$\Delta t$ [ps]')
        plt.legend()

        temp = np.tril(std_matrix[level], -1).ravel()
        temp = temp[temp>0]
        plt.figure(figsize=(10, 10))
        #plt.title('level %d' % level)

        y, x = np.histogram(temp*1E3, bins=np.linspace(np.min(temp*1E3), np.max(temp*1E3), 10))

        plt.errorbar(x[0:-1], y, yerr=np.sqrt(y), label='mean : %0.2f [ps] \n std : %0.2f [ps]' %(np.mean(temp)*1E3, np.std(temp)*1E3), marker='o', color='k', linestyle='None')
        plt.step(x[0:-1], y, where='mid')
        #plt.hist(temp*1E3, label='mean : %0.2f [ps] \n std : %0.2f [ps]' %(np.mean(temp)*1E3, np.std(temp)*1E3))
        plt.xlabel('$\sigma_t$ [ps]')
        plt.legend()




        plt.figure(figsize=(5,7))
        cmap = cm.get_cmap('viridis')
        count = 0
        for j in range(0,diff.shape[-3]):
            for i in range(1,diff.shape[-1]):
                if i>=j:continue

                col_curve = cmap(float(count) / float(diff.shape[1]*diff.shape[-1]/2 + 1))
                count += 1

                y,x = np.histogram(diff[level, j, :,i],bins=np.arange(-1.2,1.6,0.2))
                plt.plot((x[0:-1] + 0.1)*1000.,y, color=col_curve, linestyle='-', linewidth=2., alpha=0.2)

        plt.xlabel('$\Delta t$ [ps]')
        plt.ylabel('Counts')
        plt.show()
        input()




    for level in range(len(options.scan_level)):
        for i in range(arrival_time.shape[1]):
            for j in range(arrival_time.shape[1]):
                if i >= j: continue

                plt.figure(figsize=(10,10))
                plt.title('level index : %d' % options.scan_level[level])
                plt.hist(diff[level, i, :, j], bins='auto', label='pixels : (%d, %d)' % (options.pixel_list[i], options.pixel_list[j]))
                plt.legend()
                plt.xlabel('$\Delta t$')

                plt.figure(figsize=(10,10))
                plt.title('level index : %d' % options.scan_level[level])
                plt.hist(arrival_time[level, j], bins='auto', label='pixel : %d' % options.pixel_list[j])
                plt.legend()
                plt.xlabel('$t_0$')
                input()



    plt.show()
    0/0

    for level in range(arrival_time.shape[0]):

        mean = np.mean(arrival_time[level].ravel())
        std = np.std(arrival_time[level].ravel())
        bins = np.arange(np.min(arrival_time[level].ravel()), np.max(arrival_time[level].ravel()), 0.5)
        plt.figure(figsize=(10, 10))
        if options.mc:
            mc_parameters = h5py.File(options.directory + options.file_basename % options.file_list[0])[
               'simulation_parameters']
            true_time = mc_parameters['time_signal'].value
            jitter = mc_parameters['jitter_signal'].value
            nsb_rate = np.mean(mc_parameters['nsb_rate'][:, level])
            n_pe = mc_parameters['n_signal_photon'][0]
            plt.title('$f_{nsb} = $ %0.1f [MHz], $N_{p.e.} = $ %0.1f [p.e.]' % (nsb_rate, n_pe), fontsize=14)
            plt.axvline(x=true_time, label='true : \n $t_0 =$ %0.2f [ns] \n $\sigma_{t_0} = $ %0.2f [ns]' % (true_time, jitter),linestyle='--')

        plt.hist(arrival_time[level].ravel(), bins=bins, label='measured : \n $<t_0> =$ %0.2f [ns] \n $\sigma_{t_0} = $ %0.2f [ns]' % (mean, std))
        # plt.axvline(x=mean, label='Measured', linestyle=':', color='b')
        plt.xlabel('$t_0$ [ns]')
        plt.ylabel('count')
        plt.legend(loc='best', fontsize=12)

    plt.show()

    return
