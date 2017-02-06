from ctapipe import visualization
import numpy as np
from matplotlib import pyplot as plt
from utils.histogram import Histogram
from matplotlib.widgets import Button



def draw_fit_result(axis, hist, index=1, limits=None, bin_width=0.2,**kwargs):
    """
    A function to display the histogram of a variable from fit_results

    :param axis: a matplotlib axis                                         (AxesSubplot)
    :param hist: the histogram holding the fit result                      (utils.histogram.Histogram)
    :param index: the index of the variable in the fit_result array    (int)
    :param limits:    the minimal and maximal values for the variable      (list(int))
    :param bin_width: the bin width for the variable                       (float)
    :return:
    """

    # Get the data and assign limits
    h = np.copy(hist.fit_result[:, index, 0])
    h_err = np.copy(hist.fit_result[:, index, 1])
    if not limits:
        limits = [np.min(h), np.max(h)]
    h[np.isnan(h_err)] = limits[0]
    h[h < limits[0]] = limits[0]
    h[h > limits[1]] = limits[1]
    hh, bin_tmp = np.histogram(h, bins=np.arange(limits[0] - bin_width / 2, limits[1] + 1.5 * bin_width, bin_width), )
    hh_err = np.sqrt(hh)
    hh_err[hh == 0] = 1
    # Plot it on the axis
    axis.step(np.arange(limits[0] + bin_width / 2, limits[1] + 1.5 * bin_width, bin_width), hh, label='All pixels',
              color='k', lw='1')
    axis.errorbar(np.arange(limits[0], limits[1] + bin_width, bin_width), hh, yerr=hh_err, fmt='ok')
    # Beautify
    axis.set_xlabel(hist.fit_result_label[index])
    axis.set_ylabel('$\mathrm{N_{pixel}/%.2f}$' % bin_width)
    axis.set_xlim(limits[0] + bin_width / 2, limits[1] + bin_width / 2)
    axis.set_ylim(np.min(hh), np.max(hh[1:-1]) * 1.25)
    axis.xaxis.get_label().set_ha('right')
    axis.xaxis.get_label().set_position((1, 0))
    axis.yaxis.get_label().set_ha('right')
    axis.yaxis.get_label().set_position((0, 1))
    # Legend TODO
    return h


def draw_hist(axis, hist, index=(0,), limits=None, draw_fit = False, label = 'Pixel %s',**kwargs):
    """
    A function to display the histogram of a variable from fit_results

    :param axis: a matplotlib axis                                         (AxesSubplot)
    :param hist: the histogram holding the fit result                      (utils.histogram.Histogram)
    :param index: the index of the histogram to display                    (tuple)
    :param limits:    the minimal and maximal values for the variable      (list(int))
    :param draw_fit:  should the fit be displayed?                         (bool)
    :return:
    """
    # Get the data and assign limits
    h = np.copy(hist.data[index])
    h_err = np.copy(hist.errors[index])
    if not limits:
        limits = hist.fit_slices[index].astype(dtype = int)
    h = h[limits[0]:limits[1]:1]
    h_err = h_err[limits[0]:limits[1]:1]
    axis.errorbar(hist.bin_centers[limits[0]:limits[1]:1], h, yerr=h_err, fmt='ok',label=label%index[-1])
    if not draw_fit:
        axis.step(hist.bin_centers[limits[0]:limits[1]:1]+0.5*hist.bin_width, h, color='k', lw='1')
    else :
        reduced_axis = hist.bin_centers[limits[0]:limits[1]:1]
        fit_axis = np.arange(reduced_axis[0], reduced_axis[-1], float(reduced_axis[1] - reduced_axis[0]) / 10)
        reduced_func = hist.fit_function
        axis.plot(fit_axis, reduced_func(hist.fit_result[index][:, 0], fit_axis), label='fit', color='r')
        text_fit_result = '$\chi^{2}/ndf = %f$\n'%(hist.fit_chi2_ndof[index][0]/hist.fit_chi2_ndof[index][1])
        for i in range(hist.fit_result.shape[-2]):
            if (i > hist.fit_result_label.shape[0]-1): continue #TODO log it in debug
            label = hist.fit_result_label[i]
            if label.count('Amplitude')>0: continue
            text_fit_result += str(label) + ' : ' + str(
                np.round(hist.fit_result[index + (i, 0,)],  int(2)))
            text_fit_result += ' $\pm$ ' + str(np.round(hist.fit_result[index + (i, 1,)], int(3)))
            text_fit_result += '\n'
        axis.text((limits[1]-limits[0])/2 +limits[0] ,  (np.max(h[1:-1]) * 1.25 - np.min(h))/2+np.min(h), text_fit_result)
    # Beautify
    axis.set_xlabel(hist.xlabel)
    axis.set_ylabel(hist.ylabel)
    axis.set_xlim(limits[0] + hist.bin_width / 2, limits[1] + hist.bin_width / 2)
    axis.set_ylim(np.min(h), np.max(h[1:-1]) * 1.25)
    axis.xaxis.get_label().set_ha('right')
    axis.xaxis.get_label().set_position((1, 0))
    axis.yaxis.get_label().set_ha('right')
    axis.yaxis.get_label().set_position((0, 1))
    # Legend TODO

    return


class pickable_visu(visualization.CameraDisplay):
    """
    A class to allow displaying an figure when clicking on a pixel
    """
    def __init__(self, hist, figure , visu_axis , pickable_figures, limits = None, draw_fit = False,  *args, **kwargs):# extra_plot, figure, slice_func, show_fit, axis_scale,config, *args, **kwargs):
        super(pickable_visu, self).__init__(*args, **kwargs)
        self.figure = figure
        self.hist = hist
        self.axis = visu_axis
        self.pickable_figures = pickable_figures
        self.draw_fit = draw_fit
        self.limits = limits

    def on_pixel_clicked(self, pix_id):
        self.axis.cla()
        for i, pickable_figure in enumerate(self.pickable_figures):
            pickable_figure(self.axis,self.hist,index = (pix_id,),limits = self.limits, draw_fit = self.draw_fit)
        try:
            self.figure.canvas.draw()
        except ValueError:
            print('some issue to plot')


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
    f, ax = None,None
    if geom:
        f, ax = plt.subplots(1, 2, figsize=(20, 7))
        plt.subplot(1, 2, 1)
        vis_gain = visualization.CameraDisplay(geom, title='', norm='lin', cmap='viridis')
        vis_gain.add_colorbar()
        vis_gain.colorbar.set_label(hist.fit_result_label[index_var])
        plt.subplot(1, 2, 2)
    else:
        f, ax = plt.subplots(1, 1, figsize=(10, 7))
        plt.subplot(1, 1, 1)
    # Create the variable histogram
    print(type(ax))
    axis = ax
    #TODO proper subplotting
    if isinstance(ax,np.ndarray):
        axis = ax[1]
    h = draw_fit_result(axis, hist, index=index_var, limits=limits, bin_width=bin_width)
    vis_gain.image = h
    f.canvas.draw()


def display_hist(hist, geom,index_default=(700,), param_to_display = -1, limits=None,limitsCam=None, draw_fit = False):
    """

    :return:
    """
    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    plt.subplot(1, 2, 1)
    pickable_figures = [draw_hist]
    vis_baseline = pickable_visu(hist, fig, ax[1], pickable_figures, limits,draw_fit , geom, title='', norm='lin')
    vis_baseline.add_colorbar()
    vis_baseline.colorbar.set_label(hist.label)
    plt.subplot(1, 2, 1)

    if param_to_display<0:
        _bin_tmp = np.repeat(np.reshape(hist.bin_centers,(1,hist.bin_centers.shape[0])),hist.data.shape[-2],axis=0)
        hist.data[hist.data==0]=1e-10
        peak = np.average(_bin_tmp,axis=-1,weights=hist.data)
        peak[np.isnan(peak)] = limitsCam[0]
        peak[peak < limitsCam[0]] = limitsCam[0]
        peak[peak > limitsCam[1]] = limitsCam[1]
    else:
        peak = np.copy(hist.fit_result[...,param_to_display,0])
        peak[np.isnan(peak)] = limitsCam[0]
        peak[peak < limitsCam[0]] = limitsCam[0]
        peak[peak > limitsCam[1]] = limitsCam[1]

    vis_baseline.axes.xaxis.get_label().set_ha('right')
    vis_baseline.axes.xaxis.get_label().set_position((1, 0))
    vis_baseline.axes.yaxis.get_label().set_ha('right')
    vis_baseline.axes.yaxis.get_label().set_position((0, 1))
    vis_baseline.image = peak

    # noinspection PyProtectedMember
    fig.canvas.mpl_connect('pick_event', vis_baseline._on_pick)
    vis_baseline.on_pixel_clicked(index_default[0])

    fig.canvas.draw()