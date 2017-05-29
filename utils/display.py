import sys
import numpy as np
import scipy.stats
from astropy import units as u
from ctapipe import visualization
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.widgets import Button
from matplotlib.widgets import Button
from spectra_fit import fit_low_light
from spectra_fit import fit_low_light,fit_high_light
from utils.histogram import Histogram




def draw_fit_result(axis, hist, options, level=0, index=0, limits = None, display_fit=False):
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
    #TODO deal with various shape
    if len(hist.fit_result.shape)>3:
        h = np.copy(hist.fit_result[level, :, index, 0])
    else:
        h = np.copy(hist.fit_result[:, index, 0])

    h_to_return = h

    ### Avoid NANs
    mask = (~np.isnan(h) * (h>0))
    h_to_return[~mask] = np.nanmax(h_to_return)
    h = h[mask]

    if limits:
        h[(h>=limits[0]) * (h<=limits[1])]

    else:
        limits = [np.median(h) - np.std(h), np.median(h) + np.std(h)]
        h[(h>=limits[0]) * (h<=limits[1])]

    histo = axis.hist(h, bins='auto', histtype='step', align='left', color='k', linewidth=1)

    bin_edges = histo[1][0:-1]
    if len(bin_edges)==1:
        bin_width = 0.
    else:
        bin_width = bin_edges[1] - bin_edges[0]

    histo = histo[0]

    if display_fit:

        gaussian = scipy.stats.norm
        fit_param = gaussian.fit(h)
        gaussian_fit = gaussian(fit_param[0], fit_param[1])
        x = np.linspace(min(bin_edges), max(bin_edges), 100)
        axis.plot(x, gaussian_fit.pdf(x)*np.sum(histo)*(bin_width), label='fit', color='r')
        text_fit_result = '$\mu$ : %0.2f \n $\sigma$ : %0.2f \n entries : %d' % (fit_param[0], fit_param[1], h.shape[0])
        anchored_text = AnchoredText(text_fit_result, loc=2, prop=dict(size=18))
        axis.add_artist(anchored_text)


    axis.errorbar(bin_edges, histo, yerr=np.sqrt(histo), fmt='ok', label='level : %d' %options.scan_level[level] if hasattr(options, 'scan_level') else 0)
    # Beautify
    axis.set_xlabel(hist.fit_result_label[index])
    axis.set_ylabel('$\mathrm{N_{pixel}/%.2f}$' % bin_width)
    axis.xaxis.get_label().set_ha('right')
    axis.xaxis.get_label().set_position((1, 0))
    axis.yaxis.get_label().set_ha('right')
    axis.yaxis.get_label().set_position((0, 1))
    axis.legend(loc='upper right')

    return h_to_return

def draw_fit_result_level(axis, hist, options, pixel=0, index=0, scale='linear', dark_x=False):

    y = hist.fit_result[:, pixel, index, 0]
    yerr = hist.fit_result[:, pixel, index, 1]
    if dark_x:
        x = hist.fit_result[:, pixel, -1, 0]
        xerr = hist.fit_result[:, pixel, -1, 1]
        axis.set_xlabel(hist.fit_result_label[-1])
        if hist.fit_result_label[index]=='G/G$_{dark}$ []':

            x_model = np.logspace(np.log10(np.nanmin(x[x>0])), np.log10(np.nanmax(x[x>0])), 1000)
            r_bias = 1E4
            c_cell = 85 * 1E-15
            model = 1. / (1. + r_bias*x_model*1E9*c_cell)
            axis.plot(x_model, model, label='model : $R_{bias} = $ %d k$\Omega$, $C_{cell} =$ %d fF' % (r_bias/1E3, c_cell*1E15))

        axis.set_xscale('log')

    else:
        x = np.array(options.scan_level[0:len(y)])
        xerr = np.zeros(len(x))
        axis.set_xlabel('level [DAC]')

    axis.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='ok', label='pixel : %d' %(options.pixel_list[pixel]))
    axis.set_ylabel(hist.fit_result_label[index])
    axis.set_yscale(scale)
    axis.legend(loc='best')


    return


def draw_fit_pull(axis, hist, index=0, true_value=5.6, limits=None, bin_width=None, display_fit=False, **kwargs):
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

    if len(hist.fit_chi2_ndof.shape)>2:
        h = hist.fit_chi2_ndof[0, :, 0] / hist.fit_chi2_ndof[0, :, 1]
    else:
        h = hist.fit_chi2_ndof[:,0]/ hist.fit_chi2_ndof[:,1]

    limit = 100

    h_to_return = h
    mask = (~np.isnan(h) * np.isfinite(h) * (h<=limit))
    h_to_return[~mask] = limit
    h = h[mask]


    if np.sum(mask)>0:

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

    return h_to_return


def draw_pulse_shape(axis, pulse_shape, options, index, color='k'):



    if len(index)>=2:

        pixel_label = ' pixel : %d, level : %d' %(options.pixel_list[index[1]], options.scan_level[index[0]])

    else:

        pixel_label = ' pixel : %d' %(options.pixel_list[index[0]])

    h = np.max(pulse_shape[index[0], :, :, 0], axis=1)
    y = pulse_shape[index][:, 0]
    x = np.arange(0, y.shape[0], 1) * options.sampling_time
    yerr = pulse_shape[index][:, 1]

    axis.step(x, y, color=color, where='mid')
    axis.errorbar(x, y, yerr=yerr, fmt='ok', label=pixel_label)
    axis.legend(loc='upper right')
    axis.set_xlabel('t [ns]')
    axis.set_ylabel('LSB')

    return h

def draw_hist(axis, hist, options, index, draw_fit=False, color='k', scale = 'log'):

    """
    A function to display the histogram of a variable from fit_results

    :param axis: a matplotlib axis                                         (AxesSubplot)
    :param hist: the histogram holding the fit result                      (utils.histogram.Histogram)
    :param index: the index of the histogram to display                    (tuple)
    :param limits:    the minimal and maximal values for the variable      (list(int))
    :param draw_fit:  should the fit be displayed?                         (bool)
    :return:
    """

    if len(index)>=2:

        pixel_label = ' pixel : %d, level : %d' %(options.pixel_list[index[1]], options.scan_level[index[0]])

    else:

        pixel_label = ' pixel : %d' %(options.pixel_list[index[0]])



    #pixel_label[-1]= options.pixel_list[pixel_label[-1]] if not hasattr(options,'display_pixel' ) else options.display_pixel
    #pixel_label = tuple(pixel_label)
    #_tmp = list(hist.data.shape)
    #_tmp2 = _tmp.pop()
    #slice=[np.zeros(tuple(_tmp),dtype=int),np.ones(tuple(_tmp),dtype=int)*-1]
    #if draw_fit:
    #    slice = [hist.fit_slices[..., 0], hist.fit_slices[..., 1]]
    # Get the data and assign limits
    #h = np.copy(hist.data[index][slice[0][index]:slice[1][index]:1])
    #h_err = np.copy(hist.errors[index][slice[0][index]:slice[1][index]:1])
    #x = np.copy(hist.bin_centers[slice[0][index]:slice[1][index]:1])


    h = hist.data[index]
    h_err = hist.errors[index]
    x = hist.bin_centers


    #print(slice)
    h_to_return = h

    ### Avoid NANs
    mask = (~np.isnan(h) * (h>=1) )#* (x<500))



    if np.sum(mask)==0:
        mask = [True**len(h)]
    #print(mask)
    #print(index)
    h = h[mask]
    h_err = h_err[mask]
    x = x[mask]
    axis.step(x, h, color=color, where='mid')
    axis.errorbar(x, h, yerr=h_err, fmt='ok', label=pixel_label)
    text_fit_result = ''

    if draw_fit:
        reduced_axis = x
        fit_axis = np.linspace(reduced_axis[0], reduced_axis[-1]+1E-8, 10*reduced_axis.shape[0])
        reduced_func = hist.fit_function
        #if hist.fit_result[index][0, 0]<30.:
        #    import spectra_fit.fit_low_light as tmpfct
        #    reduced_func = tmpfct.fit_func
        #print(index)
        #print(hist.fit_result)
        axis.plot(fit_axis, reduced_func(hist.fit_result[index][:, 0], fit_axis), label='fit', color='r')

        text_fit_result += '$\chi^{2}/ndf : %f$\n'%(hist.fit_chi2_ndof[index][0]/hist.fit_chi2_ndof[index][1])
        for i in range(hist.fit_result.shape[-2]):
            if (i > hist.fit_result_label.shape[0]-1): continue #TODO log it in debug
            if 'Amplitude' in hist.fit_result_label[i]: continue
            text_fit_result += hist.fit_result_label[i] + ' : %0.2f $\pm$ %0.2f' % (hist.fit_result[index][i, 0], hist.fit_result[index][i, 1])
            text_fit_result += '\n'
        anchored_text = AnchoredText(text_fit_result, loc=4, prop=dict(size=18))
        axis.add_artist(anchored_text)
        #axis.text(0.7, 0.7, text_fit_result, horizontalalignment='left', verticalalignment='center',
        #              transform=axis.transAxes, fontsize=10)

    axis.set_xlabel(hist.xlabel)
    axis.set_ylabel(hist.ylabel)
    #axis.set_ylim(bottom=1)
    axis.xaxis.get_label().set_ha('right')
    axis.xaxis.get_label().set_position((1, 0))
    axis.yaxis.get_label().set_ha('right')
    axis.yaxis.get_label().set_position((0, 1))
    axis.set_yscale(scale, nonposy='clip')
    axis.legend(loc='upper right')


    return h_to_return


def display_fit_result(hist, options, geom = None, limits=[0,4095], display_fit=False):
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

    fig = plt.figure(figsize=(20, 7))
    counter = Counter(hist.data.shape, param_len=hist.fit_result.shape[-2])

    def press(event):

        sys.stdout.flush()
        if event.key == '.':

            counter.next_param()
            axis_param.cla()
            image = draw_fit_result(axis_param, hist, options=options, index=counter.count_param, display_fit=display_fit)

            if geom is not None:
                camera_visu.image = image
                camera_visu.colorbar.set_label(hist.fit_result_label[counter.count_param])
        else:

            print('Invalid key : %s' %event.key)

    fig.canvas.mpl_connect('key_press_event', press)


    # Set the limits
    if geom:

        axis_param = fig.add_subplot(1,2,1)
        axis_camera = fig.add_subplot(1,2,2)
        camera_visu = visualization.CameraDisplay(geom, ax=axis_camera, title='', norm='lin', cmap='viridis')
        h = draw_fit_result(axis_param, hist, options=options, index=counter.count_param, display_fit=display_fit)

        h[np.isnan(h)*~np.isfinite(h)] = limits[1]
        h[h<limits[0]] = limits[0]
        h[h>limits[1]] = limits[1]

        camera_visu.image = h
        camera_visu.add_colorbar()
        camera_visu.colorbar.set_label(hist.fit_result_label[counter.count_param])
        camera_visu.axes.get_xaxis().set_visible(False)
        camera_visu.axes.get_yaxis().set_visible(False)

    else: # TODO check this case
        axis_param = fig.add_subplot(1, 1, 1)
        draw_fit_result(axis_param, hist, options=options, index=counter.count_param, display_fit=display_fit)

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
        vis_gain.colorbar.set_label('$\chi^2 / ndf$')
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

def display_hist(hist, options, geom=None, display_parameter=False, draw_fit = False, scale = 'log'): #TODO check default pixel=700 better zero to avoid conflict with mc
    """

    :return:
    """

    fig = plt.figure(figsize=(48, 27))

    min_hist_x = np.min(hist.bin_centers)
    max_hist_x = np.min(hist.bin_centers)
    min_hist_y = np.min(hist.data)
    max_hist_y = np.min(hist.data)



    if display_parameter:
        counter = Counter(hist.data.shape, param_len=hist.fit_result.shape[-2])

    else:
        counter = Counter(hist.data.shape)

    def press(event):

        sys.stdout.flush()
        if event.key == '+':

            counter.next_pixel()

        elif event.key == '-':

            counter.previous_pixel()

        elif event.key == '/':

            counter.previous_level()

        elif event.key == '*':

            counter.next_level()

        elif event.key == '.':

            counter.next_param()

        else:
            print('Invalid key : %s' %event.key)

        if display_parameter:
            axis_param.cla()
            image = draw_fit_result(axis_param, hist, options=options, level=counter.count_level, index=counter.count_param,
                                display_fit=True)
            image = np.ma.masked_where(image <= 0, image)
        axis_histogram.cla()
        draw_hist(axis_histogram, hist, options=options, index=counter.count, draw_fit=draw_fit, scale=scale)
        #axis_histogram.set_xlim([min_hist_x, max_hist_x])
        #axis_histogram.set_ylim([min_hist_y, max_hist_y])

        if geom is not None and display_parameter:
            camera_visu.cmap.set_bad(color='k')
            camera_visu.image = image
            camera_visu.colorbar.set_label(hist.fit_result_label[counter.count_param])

    fig.canvas.mpl_connect('key_press_event', press)

    if geom is None and not display_parameter:

        axis_histogram = fig.add_subplot(111)
        draw_hist(axis_histogram, hist, options=options, index=counter.count, draw_fit=draw_fit, scale = scale)


    elif geom is not None:

        if not display_parameter:
            axis_histogram = fig.add_subplot(121)
            axis_camera = fig.add_subplot(122)

            if len(hist.data.shape) > 2:
                image = np.zeros(hist.data.shape[1])
                for i in range(hist.data.shape[1]):
                    if np.sum(hist.data[0,i])==0:
                        image[i] = 0.
                    else:
                        image[i] = np.average(hist.bin_centers, weights=hist.data[0, i])

            else:
                hist.data = hist.data[...,0:10000]
                image = np.zeros(hist.data.shape[0])

                for i in range(hist.data.shape[0]):
                    if np.sum(hist.data[i]) == 0:
                        image[i] = 0.
                    else:
                        image[i] = np.average(hist.bin_centers, weights=hist.data[i])

        elif display_parameter:

            axis_param = fig.add_subplot(221)
            axis_histogram = fig.add_subplot(212)
            axis_camera = fig.add_subplot(222)
            image = draw_fit_result(axis_param, hist, options=options, index=counter.count_param, display_fit=True)


        draw_hist(axis_histogram, hist, options=options, index=counter.count, draw_fit=draw_fit, scale = scale)
        print(geom)
        camera_visu = visualization.CameraDisplay(geom, ax=axis_camera, title='', norm='lin', cmap='viridis', allow_pick=True)
        image = np.ma.masked_where(image <= 0, image)
        camera_visu.cmap.set_bad(color='k')
        camera_visu.image = image
        camera_visu.add_colorbar()
        #camera_visu.colorbar.set_clim(np.nanmin(image), np.nanmax(image))

        if display_parameter:

            camera_visu.colorbar.set_label(hist.fit_result_label[counter.count_param])

        camera_visu.axes.set_xlabel('')
        camera_visu.axes.set_ylabel('')
        camera_visu.axes.set_xticks([])
        camera_visu.axes.set_yticks([])

    return fig


def display_pulse_shape(hist, options, geom=None, display_parameter=False, draw_fit = False): #TODO rename hist to pulse_shapes
    """

    :return:
    """

    fig = plt.figure(figsize=(48, 27))

    if display_parameter:
        counter = Counter(hist.data.shape, param_len=hist.fit_result.shape[-2])

    else:
        counter = Counter(hist.data.shape)

    def press(event):

        sys.stdout.flush()
        if event.key == '+':

            counter.next_pixel()

        elif event.key == '-':

            counter.previous_pixel()

        elif event.key == '/':

            counter.previous_level()

        elif event.key == '*':

            counter.next_level()

        else:

            print('Invalid key : %s' %event.key)

        axis_histogram.cla()
        image = draw_pulse_shape(axis_histogram, hist, options=options, index=counter.count)
        if geom is not None:
            camera_visu.image = image

    fig.canvas.mpl_connect('key_press_event', press)

    if geom is None and not display_parameter:

        axis_histogram = fig.add_subplot(111)
        draw_pulse_shape(axis_histogram, hist, options=options, index=counter.count)


    elif geom is not None:

        if not display_parameter:
            axis_histogram = fig.add_subplot(121)
            axis_camera = fig.add_subplot(122)

            if len(hist.data.shape) > 3:
                image = np.zeros(hist.shape[1])
                for i in range(hist.shape[1]):
                    temp = np.sum(hist[0,i,:,0])
                    if temp==0:
                        image[i] = 0.
                    else:
                        image[i] = np.mean(hist[0, i, :, 0])

            else:
                exit()
                image = np.zeros(hist.data.shape[0])
                for i in range(hist.data.shape[0]):
                    if np.sum(hist.data[i]) == 0:
                        image[i] = 0.
                    else:
                        image[i] = np.mean(hist.data[i])

        elif display_parameter:

            axis_param = fig.add_subplot(221)
            axis_histogram = fig.add_subplot(212)
            axis_camera = fig.add_subplot(222)
            image = draw_fit_result(axis_param, hist, options=options, index=counter.count_param, display_fit=True)


        draw_pulse_shape(axis_histogram, hist, options=options, index=counter.count)
        camera_visu = visualization.CameraDisplay(geom, ax=axis_camera, title='', norm='lin', cmap='viridis', allow_pick=True)
        camera_visu.image = image
        camera_visu.add_colorbar()
        #camera_visu.colorbar.set_clim(np.nanmin(image), np.nanmax(image))

        if display_parameter:

            camera_visu.colorbar.set_label(hist.fit_result_label[counter.count_param])

        camera_visu.axes.set_xlabel('')
        camera_visu.axes.set_ylabel('')
        camera_visu.axes.set_xticks([])
        camera_visu.axes.set_yticks([])

    return fig

def display_fit_result_level(hist, options, scale='linear', dark_x=False):
    """
    """

    fig = plt.figure(figsize=(20, 20))
    counter = Counter(hist.data.shape, param_len=hist.fit_result.shape[-2])

    def press(event):

        sys.stdout.flush()

        if event.key == '+':

            counter.next_pixel()

        elif event.key == '-':

            counter.previous_pixel()

        elif event.key == '.':

            counter.next_param()

        else:

            print('Invalid key : %s' %event.key)

        axis_param.cla()
        draw_fit_result_level(axis_param, hist, options=options, pixel=counter.count_pixel, index=counter.count_param,
                              scale=scale, dark_x=dark_x)

    fig.canvas.mpl_connect('key_press_event', press)

    axis_param = fig.add_subplot(1, 1, 1)
    draw_fit_result_level(axis_param, hist, options=options, pixel=counter.count_pixel, index=counter.count_param, dark_x=dark_x)

    fig.canvas.draw()

    return fig

def display_xy_pixel(x, y):

    fig = plt.figure(figsize=(20, 20))
    counter = Counter(x.shape)

    def press(event):

        sys.stdout.flush()

        if event.key == '+':

            counter.next_pixel()
            axis_param.cla()
            draw_fit_result_level(axis_param, hist, options=options, pixel=counter.count_pixel, index=counter.count_param)

        elif event.key == '-':

            counter.previous_pixel()
            axis_param.cla()
            draw_fit_result_level(axis_param, hist, options=options, pixel=counter.count_pixel, index=counter.count_param)

        else:

            print('Invalid key : %s' %event.key)

    fig.canvas.mpl_connect('key_press_event', press)

def draw_xy():
    return

class Counter():

    def __init__(self, index, param_len=None):

        self.shape = index

        if param_len is not None:

            self.count_param = 0
            self.min_param = 0
            self.max_param = param_len

        if len(self.shape) >= 3:

            self.count_pixel = 0
            self.min_pixel = 0
            self.max_pixel = self.shape[1] - 1

            self.count_level = 0
            self.min_level = 0
            self.max_level = self.shape[0] - 1
            self.count = (self.count_level, self.count_pixel,)

        elif len(self.shape) == 2:

            self.count_pixel = 0
            self.min_pixel = 0
            self.max_pixel = self.shape[0] - 1

            self.count_level = 0
            self.min_level = 0
            self.max_level = 0
            self.count = (self.count_pixel,)

    def next_pixel(self):
        if self.count_pixel + 1<self.max_pixel:
            self.count_pixel +=1
        else:
            self.count_pixel = self.min_pixel

        self._update()

    def previous_pixel(self):
        if self.count_pixel>self.min_pixel:
            self.count_pixel -=1
        else:
            self.count_pixel = self.max_pixel

        self._update()

    def next_level(self):
        if self.count_level<self.max_level:
            self.count_level +=1
        else:
            self.count_level = self.min_level

        self._update()

    def previous_level(self):
        if self.count_level>self.min_level:
            self.count_level -=1
        else:
            self.count_level = self.max_level

        self._update()

    def next_param(self):

        if self.count_param + 1<self.max_param:
            self.count_param += 1

        else:
            self.count_param = self.min_param

    def _update(self):

        if len(self.shape) >= 3:
            self.count = (self.count_level, self.count_pixel, )

        elif len(self.shape) == 2:
            self.count = (self.count_pixel, )

    #TODO (in# itialise at value in options.pixel_display)


def display_by_pixel(hist,options):
    print(np.max(hist),options.pixel_list)
    h = np.zeros((1296,),dtype=float)
    h1 = np.zeros((12,),dtype=float)
    for module in options.cts.camera.Modules:
        for pix in module.pixelsID_inModule:
            if module.pixels[pix-1].ID in options.pixel_list:
                if hist[options.pixel_list.index(module.pixels[pix-1].ID)]>0.05:
                    h1[(pix - 1)]=h1[(pix - 1)]+hist[options.pixel_list.index(module.pixels[pix-1].ID)]
                h[(module.ID - 1) * 12 + (pix - 1)]=hist[options.pixel_list.index(module.pixels[pix-1].ID)]


    plt.figure()
    plt.step(np.arange(0,1296),h, where='mid')
    plt.show()
    plt.figure()
    plt.step(np.arange(0,12),h1, where='mid')
    plt.show()
