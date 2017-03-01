from ctapipe import visualization
import numpy as np
from matplotlib import pyplot as plt
from utils.histogram import Histogram
from matplotlib.widgets import Button
import scipy.stats
from spectra_fit import fit_low_light


def draw_fit_result(axis, hist, index=1, display_fit=False):
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

    h_to_return = h

    ### Avoid NANs
    mask = (~np.isnan(h)) * (~np.isnan(h_err) * (h>0))
    h = h[mask]

<<<<<<< HEAD
    if limits:
        h[h<limits[0]]=limits[0]
        h[h>limits[1]]=limits[1]


    histo = axis.hist(h, bins='auto', histtype='step', align='left', label='All pixels', color='k', linewidth=1)
=======
    histo = axis.hist(h, bins='auto', histtype='step', align='left', label='all pixels', color='k', linewidth=1)
>>>>>>> d9bd0605ed92cd07f6a0c491af92ddb13722aaac

    bin_edges = histo[1][0:-1]
    bin_width = bin_edges[1] - bin_edges[0]
    histo = histo[0]

    if display_fit:

        gaussian = scipy.stats.norm
        fit_param = gaussian.fit(h)
        gaussian_fit = gaussian(fit_param[0], fit_param[1])
        x = np.linspace(min(bin_edges), max(bin_edges), 100)
        axis.plot(x, gaussian_fit.pdf(x)*np.sum(histo)*(bin_width), label='fit', color='r')
        text_fit_result = '$\mu$ = %0.2f \n $\sigma$ = %0.2f \n entries = %d' % (fit_param[0], fit_param[1], h_to_return.shape[0])
        axis.text(bin_edges[-2], max(histo), text_fit_result)


    axis.errorbar(bin_edges, histo, yerr=np.sqrt(histo), fmt='ok')
    # Beautify
    axis.set_xlabel(hist.fit_result_label[index])
    axis.set_ylabel('$\mathrm{N_{pixel}/%.2f}$' % bin_width)
    axis.xaxis.get_label().set_ha('right')
    axis.xaxis.get_label().set_position((1, 0))
    axis.yaxis.get_label().set_ha('right')
    axis.yaxis.get_label().set_position((0, 1))
    axis.legend()

    return h_to_return

def draw_fit_pull(axis, hist, index=1, true_value=5.6, limits=None, bin_width=None, display_fit=False, **kwargs):
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

    mask = (~np.isnan(h)) * (~np.isnan(h_err))
    h_err = h_err[mask]
    h = (true_value - h[mask])/h_err

    histo = axis.hist(h, bins='auto', histtype='step', align='left', label='All pixels', color='k', linewidth=1)

    bin_edges = histo[1][0:-1]
    bin_width = bin_edges[1] - bin_edges[0]
    histo = histo[0]

    if display_fit:

        gaussian = scipy.stats.norm
        fit_param = gaussian.fit(h)
        gaussian_fit = gaussian(fit_param[0], fit_param[1])
        x = np.linspace(min(bin_edges), max(bin_edges), 100)
        axis.plot(x, gaussian_fit.pdf(x)*np.sum(histo)*bin_width, label='fit', color='r')
        text_fit_result = '$\mu$ = %0.2f \n $\sigma$ = %0.2f' % (fit_param[0], fit_param[1])
        axis.text(bin_edges[-2], max(histo), text_fit_result)



    #axis.step(np.arange(limits[0] + bin_width / 2, limits[1] + 1.5 * bin_width, bin_width), hh, label='All pixels',
    #          color='k', lw='1')
    axis.errorbar(bin_edges, histo, yerr=np.sqrt(histo), fmt='ok')
    # Beautify
    axis.set_xlabel(hist.fit_result_label[index] + ' pull')
    axis.set_ylabel('$\mathrm{N_{pixel}/%.2f}$' % bin_width)
    axis.xaxis.get_label().set_ha('right')
    axis.xaxis.get_label().set_position((1, 0))
    axis.yaxis.get_label().set_ha('right')
    axis.yaxis.get_label().set_position((0, 1))
    axis.legend()

    return h

def draw_chi2(axis, hist, display_fit):
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
    h = np.copy(hist.fit_chi2_ndof[:,0]/ hist.fit_chi2_ndof[:,1])
    mask = (~np.isnan(h))
    h = h[mask]

    histo = axis.hist(h, bins='auto', histtype='step', align='left', label='All pixels', color='k', linewidth=1)

    bin_edges = histo[1][0:-1]
    bin_width = bin_edges[1] - bin_edges[0]
    histo = histo[0]

    if display_fit:

        gaussian = scipy.stats.norm
        fit_param = gaussian.fit(h)
        gaussian_fit = gaussian(fit_param[0], fit_param[1])
        x = np.linspace(min(bin_edges), max(bin_edges), 100)
        axis.plot(x, gaussian_fit.pdf(x)*np.sum(histo)*bin_width, label='fit', color='r')
        text_fit_result = '$\mu$ = %0.2f \n $\sigma$ = %0.2f' % (fit_param[0], fit_param[1])
        axis.text(bin_edges[-2], max(histo), text_fit_result)




    #axis.step(np.arange(limits[0] + bin_width / 2, limits[1] + 1.5 * bin_width, bin_width), hh, label='All pixels',
    #          color='k', lw='1')
    axis.errorbar(bin_edges, histo, yerr=np.sqrt(histo), fmt='ok')
    # Beautify
    axis.set_xlabel('$\chi^2 / ndf$')
    axis.set_ylabel('$\mathrm{N_{pixel}/%.2f}$' % bin_width)
    axis.xaxis.get_label().set_ha('right')
    axis.xaxis.get_label().set_position((1, 0))
    axis.yaxis.get_label().set_ha('right')
    axis.yaxis.get_label().set_position((0, 1))
    axis.legend()

def draw_hist(axis, hist, index, draw_fit=False):
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
    h_to_return = h

    ### Avoid NANs
    mask = (~np.isnan(h)) * (~np.isnan(h_err) * (h > 0))
    h = h[mask]
    h_err = h_err[mask]
    x = hist.bin_centers[mask]

    axis.step(x, h, color='k', where='mid')
    axis.errorbar(x, h, yerr=h_err, fmt='ok', label='pixel %s' % (index, ))
    text_fit_result = ''

    if draw_fit:
        reduced_axis = x
        fit_axis = np.linspace(reduced_axis[0], reduced_axis[-1], 10*reduced_axis.shape[0])
        reduced_func = hist.fit_function
        axis.plot(fit_axis, reduced_func(hist.fit_result[index][:, 0], fit_axis), label='fit', color='r')
        text_fit_result += '$\chi^{2}/ndf = %f$\n'%(hist.fit_chi2_ndof[index][0]/hist.fit_chi2_ndof[index][1])
        for i in range(hist.fit_result.shape[-2]):
            if (i > hist.fit_result_label.shape[0]-1): continue #TODO log it in debug
            if 'Amplitude' in hist.fit_result_label[i]: continue
            text_fit_result += hist.fit_result_label[i] + ' : %0.2f $\pm$ %0.2f' % (hist.fit_result[index][i, 0], hist.fit_result[index][i, 1])
            text_fit_result += '\n'
        axis.text(0.7, 0.7, text_fit_result, horizontalalignment='left', verticalalignment='center',
                      transform=axis.transAxes, fontsize=10)

    axis.set_xlabel(hist.xlabel)
    axis.set_ylabel(hist.ylabel)
    axis.xaxis.get_label().set_ha('right')
    axis.xaxis.get_label().set_position((1, 0))
    axis.yaxis.get_label().set_ha('right')
    axis.yaxis.get_label().set_position((0, 1))
    axis.set_yscale('log', nonposy='clip')
    axis.legend(loc='best')


    return h_to_return


def display_fit_result(hist, geom = None , index_var=1, limits=[0,4095], bin_width=None, display_fit=False):
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
    if geom: #TODO correct colobar on right instead of left
        fig = plt.figure(figsize=(20, 7))
        ax_left = fig.add_subplot(1,2,1)
        ax_right = fig.add_subplot(1,2,2)
        vis_gain = visualization.CameraDisplay(geom, ax=ax_left, title='', norm='lin', cmap='viridis')
        h = draw_fit_result(ax_right, hist, index=index_var, limits=limits, bin_width=bin_width, display_fit=display_fit)
<<<<<<< HEAD
        #vis_gain.image = h
=======

        h[np.isnan(h)*~np.isfinite(h)] = limits[1]
        h[h<limits[0]] = limits[0]
        h[h>limits[1]] = limits[1]

        vis_gain.image = h
        vis_gain.add_colorbar()
        vis_gain.colorbar.set_label(hist.fit_result_label[index_var])

>>>>>>> d9bd0605ed92cd07f6a0c491af92ddb13722aaac
    else: # TODO check this case
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(1, 1, 1)
        draw_fit_result(ax, hist, index=index_var, limits=limits, bin_width=bin_width, display_fit=display_fit)

    fig.canvas.draw()

    return fig

def display_fit_pull(hist, geom = None , index_var=1, limits=None, true_value=None, bin_width=None, display_fit=False):
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
    if geom: #TODO correct colobar on right instead of left
        fig = plt.figure(figsize=(20, 7))
        ax_left = fig.add_subplot(1,2,1)
        ax_right = fig.add_subplot(1,2,2)
        vis_gain = visualization.CameraDisplay(geom, ax=ax_left, title='', norm='lin', cmap='viridis')
        vis_gain.add_colorbar()
        vis_gain.colorbar.set_label(hist.fit_result_label[index_var])
        h = draw_fit_pull(ax_right, hist, index=index_var, limits=limits, bin_width=bin_width, true_value=true_value, display_fit=display_fit)
        vis_gain.image = h
    else:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(1, 1, 1)
        h = draw_fit_pull(ax, hist, index=index_var, limits=limits, bin_width=bin_width, true_value=true_value, display_fit=display_fit)

    # Create the variable histogram
    #h = draw_fit_result(ax[1], hist, index=index_var, limits=limits, bin_width=bin_width)
    fig.canvas.draw()

    return fig

def display_chi2(hist, geom = None, display_fit=False):
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
    if geom: #TODO correct colobar on right instead of left
        fig = plt.figure(figsize=(20, 7))
        ax_left = fig.add_subplot(1,2,1)
        ax_right = fig.add_subplot(1,2,2)
        vis_gain = visualization.CameraDisplay(geom, ax=ax_left, title='', norm='lin', cmap='viridis')
        vis_gain.add_colorbar()
        vis_gain.colorbar.set_label(hist.fit_chi2_ndof[0]/hist.fit_chi2_ndof[1])
        h = draw_chi2(ax_right, hist, display_fit=display_fit)
        vis_gain.image = h
    else:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(1, 1, 1)
        h = draw_chi2(ax, hist, display_fit=display_fit)

    # Create the variable histogram
    #h = draw_fit_result(ax[1], hist, index=index_var, limits=limits, bin_width=bin_width)
    fig.canvas.draw()


    return fig

def display_hist(hist, index, geom=None, param_to_display=None, draw_fit = False): #TODO check default pixel=700 better zero to avoid conflict with mc
    """

    :return:
    """

    fig = plt.figure(figsize=(20, 20))

    if geom is None and param_to_display is None:

        axis_histogram = fig.add_subplot(111)
        draw_hist(axis_histogram, hist, index=index, draw_fit=draw_fit)

    elif geom is not None and param_to_display is None:

        axis_histogram = fig.add_subplot(121)
        axis_camera = fig.add_subplot(122)
        camera_visu = visualization.CameraDisplay(geom, ax=axis_camera, title='', norm='lin', cmap='viridis', allow_pick=True)
        camera_visu.axes.set_xlabel('x [mm]')
        camera_visu.axes.set_ylabel('y [mm]')
        draw_hist(axis_histogram, hist, index=index, draw_fit=draw_fit)

        pixels_value = np.zeros(hist.data.shape[0])
        for i in range(hist.data.shape[0]):
            pixels_value[i] = np.average(hist.bin_centers, weights=hist.data[i])

        camera_visu.image = pixels_value

        camera_visu.add_colorbar(ticks=np.linspace(np.min(pixels_value), np.max(pixels_value), 10))
        camera_visu.colorbar.set_label('mean ADC')

    elif geom is not None and param_to_display is not None:

        axis_param = fig.add_subplot(221)
        axis_histogram = fig.add_subplot(212)
        axis_camera = fig.add_subplot(222)

        camera_visu = visualization.CameraDisplay(geom, ax=axis_camera, title='', norm='lin', cmap='viridis', allow_pick=True)
        draw_hist(axis_histogram, hist, index=index, draw_fit=draw_fit)
        camera_visu.axes.set_xlabel('x [mm]')
        camera_visu.axes.set_ylabel('y [mm]')
        fit_result =  draw_fit_result(axis_param, hist, index=param_to_display, display_fit=True)
        camera_visu.image = fit_result
        camera_visu.add_colorbar(ticks=np.linspace(0., np.max(fit_result[fit_result>0]), 10))
        camera_visu.colorbar.set_label(hist.fit_result_label[param_to_display])

    return fig