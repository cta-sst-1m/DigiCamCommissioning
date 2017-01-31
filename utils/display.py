from ctapipe import visualization
import numpy as np
from matplotlib import pyplot as plt
from utils.histogram import Histogram
from matplotlib.widgets import Button

def display_fit_result(hist, geom = None , index_var=1, limits=[0.,10.], bin_width=0.2):
    """
    A function to display a vaiable both as an histogram and as a camera view

    :param hist: the histogram holding the fit result                      (utils.histogram.Histogram)
    :param geom: the geometry. if None only display histogram              (ctapipe.io.camera.CameraGeometry)
    :param title: the z-label in camera view, x-label in histogram view    (str)
    :param index_var: the index of the variable in the fit_result array    (int)
    :param limits:    the minimal and maximal values for the variable      (list(int))
    :param bin_width: the bin width for the variable                       (float)
    :return:
    """

    # Set the limits
    h = np.copy(hist.fit_result[:, index_var, 0])
    h_err = np.copy(hist.fit_result[:, index_var, 1])
    h[np.isnan(h_err)] = limits[0]
    h[h < limits[0]] = limits[0]
    h[h > limits[1]] = limits[1]
    f, ax = None,None
    if geom:
        f, ax = plt.subplots(1, 2, figsize=(20, 7))
        plt.subplot(1, 2, 1)
        vis_gain = visualization.CameraDisplay(geom, title='', norm='lin', cmap='viridis')
        vis_gain.add_colorbar()
        vis_gain.colorbar.set_label(hist.fit_result_label[index_var])
        vis_gain.image = h
        plt.subplot(1, 2, 2)
    else:
        f, ax = plt.subplots(1, 1, figsize=(10, 7))
        plt.subplot(1, 1, 1)
    # Create the variable histogram
    hh, bin_tmp = np.histogram(h, bins=np.arange(limits[0] - bin_width / 2, limits[1] + 1.5 * bin_width, bin_width),)
    hh_err = np.sqrt(hh)
    hh_err[hh==0]=1
    plt.step(np.arange(limits[0] + bin_width / 2 , limits[1] + 1.5 * bin_width, bin_width),hh,label='All pixels',color='k',lw='1')
    plt.errorbar(np.arange(limits[0], limits[1] + bin_width,  bin_width),hh,yerr = hh_err ,fmt='ok')
    plt.xlabel(hist.fit_result_label[index_var])
    plt.ylabel('$\mathrm{N_{pixel}/%.2f}$' % bin_width)
    plt.xlim(limits[0]+bin_width / 2,limits[1]+bin_width / 2)
    plt.ylim(np.min(hh),np.max(hh[1:-1])*1.25)
    plt.show()
