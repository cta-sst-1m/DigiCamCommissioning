#!/usr/bin/env python3

# external modules

# internal modules
import logging
import sys
import numpy as np
import logging
from tqdm import tqdm
from utils.logger import TqdmToLogger
from data_treatement import cosmic
import matplotlib.pyplot as plt
import h5py
from matplotlib.collections import PolyCollection
import copy

__all__ = ["create_histo", "perform_analysis", "display_results"]


def create_histo(options):

    cosmic_info = cosmic.run(options)

    np.savez(file=options.output_directory + options.cosmic_filename, data=cosmic_info)

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

    data = np.load(options.output_directory + options.cosmic_filename)['data']
    selected_data = []

    for event_id in range(data.shape[0]):

        if data[event_id]['pixel'].shape[0] >= 3:

            selected_data.append(data[event_id])

    np.savez(file=options.output_directory + options.cosmic_filename_cut, data=selected_data)

    return


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    data = np.load(options.output_directory + options.cosmic_filename)['data']
    selected_data = np.load(options.output_directory + options.cosmic_filename_cut)['data']
    camera = options.cts.camera

    vertices = np.array([camera.Pixels[j].Vertices for j in range(len(camera.Pixels))])
    vertices = np.swapaxes(vertices, 1, 2)
    coll = PolyCollection(vertices, facecolors='w', linewidths=1, edgecolors='k')

    plt.ioff()

    for i in range(selected_data.shape[0]):

        sorted_order = np.argsort(selected_data[i]['time'])

        for key, val in selected_data[i].items():

            if key != 'event_id':

                selected_data[i][key] = val[sorted_order]

        x_pos = np.array([camera.Pixels[selected_data[i]['pixel'][j]].center[0] for j in range(len(selected_data[i]['pixel']))])
        y_pos = np.array([camera.Pixels[selected_data[i]['pixel'][j]].center[1] for j in range(len(selected_data[i]['pixel']))])
        fig, ax = plt.subplots(figsize=(12, 10))

        ax.add_collection(copy.copy(coll))
        cax = ax.scatter(x_pos, y_pos, s=selected_data[i]['charge'], c=selected_data[i]['time'], label='event # %d' % selected_data[i]['event_id'])
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_xlim([-550, 550])
        ax.set_ylim([-550, 550])
        fig.colorbar(cax, label=' t[ns]')
        ax.legend(loc='upper right')

        fig.savefig('/home/alispach/figures/cosmic/horizontal/' + 'event_%d.png' %i)

        r = np.sqrt(np.diff(x_pos)**2 + np.diff(y_pos)**2) * 1E-3

        if selected_data[i]['event_id'] in [344, 455]:


            print(selected_data[i]['time'], r)

            dt = np.diff(selected_data[i]['time']) * 1E-9
            v = np.sum(r) / np.sum(dt)



            print(v)

        plt.axis('equal')


    return
