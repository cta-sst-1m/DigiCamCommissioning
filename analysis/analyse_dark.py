#!/usr/bin/env python3

# external modules

# internal modules
from utils import display, histogram, geometry
import matplotlib.pyplot as plt
from data_treatement import adc_hist
import logging,sys
import numpy as np
import peakutils
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep
from spectra_fit import fit_dark_adc, fit_multiple_gaussians_spe, fit_gaussian_sum

import matplotlib.cm as cm



__all__ = ["create_histo", "perform_analysis", "display_results"]


def create_histo(options):
    """
    Create a list histograms and fill it with data
    :param options: a dictionary containing at least the following keys:
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram      (str)
        - 'file_basename'    : the base name of the input files                   (str)
        - 'directory'        : the path of the directory containing input files   (str)
        - 'file_list'        : the list of base filename modifiers for the input files
                                                                                  (list(str))
        - 'evt_max'          : the maximal number of events to process            (int)
        - 'n_evt_per_batch'  : the number of event per fill batch. It can be
                               optimised to improve the speed vs. memory          (int)
        - 'n_pixels'         : the number of pixels to consider                   (int)
        - 'adcs_min'         : the minimum adc value in histo                     (int)
        - 'adcs_max'         : the maximum adc value in histo                     (int)
        - 'adcs_binwidth'    : the bin width for the adcs histo                   (int)
    :return:
    """
    dark = None

    if options.analysis_type == 'step_function':
        dark = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                                bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list),),
                                label='Dark step function', xlabel='threshold LSB', ylabel='entries')

        # Get the adcs
        adc_hist.run(dark, options, h_type='STEPFUNCTION')

    elif options.analysis_type == 'single_photo_electron':
        print('#################################3 here')
        print(dir(options))
        dark = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                                   bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list),),
                                   label='Pixel SPE', xlabel='Pixel ADC', ylabel='Count / ADC')
        print('#################################3 there')
        baseline_fit = histogram.Histogram(filename=options.output_directory + options.histo_dark,fit_only=True).fit_result
        # Get the adcs
        adc_hist.run(dark, options, 'SPE', prev_fit_result=baseline_fit)

    elif options.analysis_type == 'adc_template' or options.analysis_type == 'fit_baseline' :
        dark = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                                   bin_width=options.adcs_binwidth, data_shape=(len(options.pixel_list),),
                                   label='Dark LSB', xlabel='LSB', ylabel='entries')
        # Get the adcs
        adc_hist.run(dark, options, 'ADC')


    # Save the histogram
    dark.save(options.output_directory + options.histo_filename)

    del dark

    return


def perform_analysis(options):
    """
    Extract the dark rate and cross talk, knowing baseline gain sigmas from previous
    runs
    :param options: a dictionary containing at least the following keys:
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram      (str)
    :return:
    """
    if options.analysis_type == 'step_function':
        step_function(options)
    elif options.analysis_type == 'adc_template':
        adc_template(options)
    elif options.analysis_type == 'fit_baseline':
        fit_baseline(options)
    elif options.analysis_type == 'single_photo_electron':
        single_photo_electron(options)


def display_results(options):
    """
    Display the analysis results
    :param options:
    :return:
    """
    if options.analysis_type == 'step_function':
        # Load the data
        dark_step_function = histogram.Histogram(filename=options.output_directory + options.histo_filename)
        dark_count_rate = dark_step_function.fit_result[:, 0, 0]
        cross_talk = dark_step_function.fit_result[:, 1, 0]

        mask = ~np.isnan(dark_count_rate)
        dark_count_rate = dark_count_rate[mask]
        cross_talk = cross_talk[mask]
        # Display step function
        display.display_hist(dark_step_function, options=options)

        # Display histograms of dark count rate and cross talk
        plt.figure()
        plt.hist(dark_count_rate * 1E3, bins='auto')
        plt.xlabel('$f_{dark}$ [MHz]')

        plt.figure()
        plt.hist(cross_talk, bins='auto')
        plt.xlabel('XT')

        plt.show()

    elif options.analysis_type == 'single_photo_electron':

        dark_spe = histogram.Histogram(filename=options.output_directory + options.histo_filename)
        from utils.peakdetect import peakdetect
        plt.subplots(1, 1, figsize=(10, 16))
        fadc_list = []
        for i in range(dark_spe.data.shape[0]):
            pixel_idx = i
            cmap = cm.get_cmap('viridis')
            fadc_mult = options.cts.camera.Pixels[options.pixel_list[i]].fadc
            sector = options.cts.camera.Pixels[options.pixel_list[i]].sector
            fadc = fadc_mult + 9 * (sector - 1)
            # col_curve = cmap(float(i)/float(dark_adc.data.shape[0] + 1))
            col_curve = cmap(float(fadc) / float(9 * 3))
            if sector == 1:
                col_curve =(1.,0.,0.,0.4)#, (1. - (float(fadc_mult - 1) / 9./2.+0.5), 0., 0., 0.4)
            elif sector == 2:
                col_curve = (0.,1.,0.,0.4)#(0., 1. - (float(fadc_mult - 1) / 9./2.+0.5), 0., 0.4)
            elif sector == 3:
                col_curve = (0.,0.,1.,0.4)#(0., 0., 1. - (float(fadc_mult - 1) /9./2.+0.5), 0.4)
            col_curve=(0.,0.,0.,0.05)

            if dark_spe.fit_result[pixel_idx, 1, 0]<15: continue
            if fadc in fadc_list:
                plt.subplot(1, 1,1)
                x_1 = (dark_spe.bin_centers-dark_spe.fit_result[pixel_idx, 0, 0])/dark_spe.fit_result[pixel_idx, 1, 0]
                y_1 =  dark_spe.data[pixel_idx]
                plt.plot(x_1,y_1, color=col_curve, linestyle='-', linewidth=2)
            else:
                plt.subplot( 1, 1,1)
                plt.plot(dark_spe.bin_centers,
                         dark_spe.data[ pixel_idx], color=col_curve, linestyle='-', linewidth=2)#,
                         #label="FADC %d Sector %d" % (fadc_mult, sector))
            fadc_list.append(fadc)

        plt.subplot(1, 1, 1)

        plt.yscale('log')
        plt.xlim(0,10)

        n_pes = (np.repeat(dark_spe.bin_centers.reshape(1,-1),dark_spe.data.shape[0], axis=0)-dark_spe.fit_result[:, 0, 0].reshape(-1,1))/dark_spe.fit_result[:, 1, 0].reshape(-1,1)
        entries = dark_spe.data
        n_pes=n_pes[dark_spe.fit_result[:, 1, 0]>15]
        entries = entries[dark_spe.fit_result[:, 1, 0]>15]

        x = np.arange(-1.,9.1,0.1)
        distrib = np.ones((x.shape[0],entries.shape[0]*entries.shape[1]))*np.nan
        r_npes = n_pes.reshape(-1)
        r_entries = entries.reshape(-1)
        for i,j in enumerate(x):
            print(i,j)
            mask = ((r_npes>(j-0.05))* (r_npes<(j+0.05)))
            entries_reduced = r_entries[mask]
            distrib[i,0:entries_reduced.shape[0]]=entries_reduced

        #plt.subplot(1, 1, 1)
        print(x)
        n_sigma = 1
        plt.fill_between(x,np.nanmean(distrib,axis=-1)+n_sigma*np.nanstd(distrib,axis=-1),
                         np.nanmean(distrib, axis=-1) - n_sigma*np.nanstd(distrib, axis=-1),
                         alpha=0.5, edgecolor='r' , facecolor='r',
                        zorder = 10, label='%d $\sigma$' %n_sigma)


        plt.plot(x,np.nanmean(distrib,axis=-1),color='r')
        plt.xlabel('N(p.e.)')
        plt.ylabel('$N_{peaks}$')
        plt.legend()
        plt.show()

        #dark_spe.data = (np.cumsum(dark_spe.data,axis=-1)-np.sum(dark_spe.data,axis=-1).reshape(-1,1))*-1
        #dark_spe.data = np.diff(dark_spe.data,n=1,axis=-1)*-1.
        #dark_spe.data = np.diff(dark_spe.data,n=1,axis=-1)*-1.
        #dark_spe.data = np.diff(dark_spe.data,n=1,axis=-1)*-1.

        display.display_hist(dark_spe, options=options, draw_fit=True, geom=geometry.generate_geometry(options.cts, all_camera=True)[0])

        for i in [0,1,2,3]:#range(dark_spe.fit_result.shape[1]):
            fig = plt.figure(figsize=(10, 10))
            axis = fig.add_subplot(111)
            display.draw_fit_result(axis, dark_spe, options, level=0, index=i, limits=None, display_fit=True)


        plt.show()
        plt.subplots(1,2)
        plt.subplot(1,2,1)
        one = dark_spe.fit_result[:,4,0]
        xt = dark_spe.fit_result[:,5,0][one>0]/ (one[one>0] + dark_spe.fit_result[:,5,0][one>0])
        xt=xt[xt<0.5]
        xt=xt[xt>0.]
        print(xt)
        print(np.min(xt),np.max(xt))
        dc = (dark_spe.fit_result[:,5,0]+dark_spe.fit_result[:,6,0]+dark_spe.fit_result[:,7,0]+
              dark_spe.fit_result[:,8,0] +dark_spe.fit_result[:,9,0]+dark_spe.fit_result[:,4,0])/(4. * 42. * 1000000.) * 1E3
        plt.hist(xt,bins=100)

        plt.subplot(1,2,2)
        plt.hist(dc[~np.isnan(dc)],bins=50)
        plt.show()

        display.display_chi2(dark_spe, geom=geometry.generate_geometry(options.cts, all_camera=True)[0])

        input('press a key')

    elif options.analysis_type == 'fit_baseline':

        dark_adc = histogram.Histogram(filename=options.output_directory + options.histo_filename)
        #print(col_curve)
        #print(col_curve.shape)
        plt.subplots(1,1,figsize=(10,8))
        fadc_list = []
        for i in range(dark_adc.data.shape[0]):
            cmap = cm.get_cmap('nipy_spectral')
            fadc_mult = options.cts.camera.Pixels[options.pixel_list[i]].fadc
            sector = options.cts.camera.Pixels[options.pixel_list[i]].sector
            fadc = fadc_mult + 9 * (sector-1)
            #col_curve = cmap(float(i)/float(dark_adc.data.shape[0] + 1))
            col_curve = (0.,0.,0.,0.05)#cmap(float(fadc)/float(9*3))
            if dark_adc.fit_chi2_ndof[i,0]/dark_adc.fit_chi2_ndof[i,1]>5000 or dark_adc.fit_result[i,1,0]>50: continue

            if fadc in fadc_list:
                plt.plot(dark_adc.bin_centers,dark_adc.data[i], color=col_curve, linestyle='-', linewidth=2)
            else:
                plt.plot(dark_adc.bin_centers,dark_adc.data[i], color=col_curve, linestyle='-', linewidth=2,label= "FADC %d Sector %d"%(fadc_mult,sector))
                print(fadc,float(fadc)/float(9*3))
            fadc_list.append(fadc)
        plt.xlabel('LSB')
        plt.ylabel('Counts')
        plt.legend()
        #plt.yscale('log')
        plt.ylim(1.,np.max(dark_adc.data*1.5))

        #dark_adc.data = (np.cumsum(dark_adc.data,axis=-1)-np.sum(dark_adc.data,axis=-1).reshape(-1,1))*-1
        # dark_spe.data = np.diff(dark_spe.data,n=1,axis=-1)*-1.
        # dark_spe.data = np.diff(dark_spe.data,n=1,axis=-1)*-1.
        # dark_spe.data = np.diff(dark_spe.data,n=1,axis=-1)*-1.
        display.display_hist(dark_adc, options=options, draw_fit=True,scale='linear')

        for i in [0,1,2]:#range(dark_spe.fit_result.shape[1]):
            fig = plt.figure(figsize=(10, 10))
            axis = fig.add_subplot(111)
            display.draw_fit_result(axis, dark_adc, options, level=0, index=i, limits=None, display_fit=True)
        plt.show()
        input('press a key')

    return


def step_function(options):

    dark_step_function = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    log = logging.getLogger(sys.modules['__main__'].__name__ + __name__)

    y = dark_step_function.data
    x = np.tile(dark_step_function.bin_centers, (y.shape[0], 1))

    y = - np.diff(np.log(y)) / np.diff(x)
    x = x[..., :-1] + np.diff(x) / 2.

    y[np.isnan(y)] = 0
    y[np.isinf(y)] = 0
    y[y < 0] = 0

    threshold = 0.15
    width = 3

    dark_count = np.zeros(y.shape[0])
    cross_talk = np.zeros(y.shape[0])

    n_samples = options.n_samples - options.baseline_per_event_limit - options.window_width + 1
    time = (options.max_event - options.min_event) * 4 * n_samples

    for pixel in range(y.shape[0]):
        max_indices = peakutils.indexes(y[pixel], thres=threshold, min_dist=options.min_distance)

        try:
            max_x = peakutils.interpolate(x[pixel], y[pixel], ind=max_indices, width=width)

        except RuntimeError:

            log.warning('Could not interpolate for pixel %d taking max as indices' % options.pixel_list[pixel])
            max_x = x[pixel][max_indices]

        gain = np.mean(np.diff(max_x))
        min_x = max_x + 0.5 * gain
        f = interp1d(dark_step_function.bin_centers, dark_step_function.data[pixel], kind='cubic')

        try:

            counts = [f(min_x[0]), f(min_x[1])]

        except IndexError:

            log.warning('Could not find 0.5 p.e. and 1.5 p.e. for pixel %d' % options.pixel_list[pixel])
            counts = [np.nan, np.nan]

        # spline step function method

        spline_step_function = splrep(dark_step_function.bin_centers, dark_step_function.data[pixel], k=3, w=1./np.sqrt(dark_step_function.data[pixel]), s=10)

        # x_around_minima = np.linspace(0, max_x[1] + 0.5 * gain, 100)
        # spline_step_function_second_derivative = splev(x_around_minima, tck=spline_step_function, der=2)

        dark_count[pixel] = counts[0] / time
        cross_talk[pixel] = counts[1] / counts[0]

        if options.verbose:
            x_spline = np.linspace(dark_step_function.bin_centers[0], dark_step_function.bin_centers[-1],
                                   num=len(dark_step_function.bin_centers) * 20)
            plt.figure()
            plt.semilogy(x_spline, splev(x_spline, spline_step_function), label='spline')
            plt.semilogy(dark_step_function.bin_centers, dark_step_function.data[pixel], label='data', linestyle='None', marker='o')
            plt.axvline(min_x[0])
            plt.axvline(min_x[1])
            plt.legend()

            plt.figure()
            plt.plot(x_spline, ((splev(x_spline, spline_step_function, der=2))), label='spline second der')
            plt.legend()
            plt.show()

    dark_step_function.fit_result = np.zeros((dark_step_function.data.shape[0], 4, 2))
    dark_step_function.fit_result[:, 0, 0] = dark_count
    dark_step_function.fit_result[:, 1, 0] = cross_talk

    dark_step_function.save(options.output_directory + options.histo_filename)
    return


def adc_template(options):
    log = logging.getLogger(sys.modules['__main__'].__name__ + __name__)
    log.info('Perform an analytic extraction of mu_XT ( ==> baseline and sigmae for full mpe )')

    mpes_full = histogram.Histogram(filename=options.output_directory + options.full_histo_filename, fit_only=True)
    mpes_full_fit_result = np.copy(mpes_full.fit_result)
    del mpes_full

    # Load the histogram
    dark_hist = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    dark_hist_for_baseline = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    dark_hist_for_baseline.fit(fit_dark_adc.fit_func, fit_dark_adc.p0_func, fit_dark_adc.slice_func,
                               fit_dark_adc.bounds_func, \
                               labels_func=fit_dark_adc.labels_func)

    dark_hist.fit_result = np.zeros((len(options.pixel_list), 3, 2))
    dark_hist.fit_result_label = np.array(['baseline [LSB]', '$f_{dark}$ [MHz]', '$\mu_{XT}$'])

    x = dark_hist.bin_centers

    # print(mpes_full_fit_result.shape)
    # print(mpes_full_fit_result[1 , 0, 0])
    # print(mpes_full_fit_result[1 , 1, 0])
    # print(mpes_full_fit_result[1 , 2, 0])
    # print(mpes_full_fit_result[1 , 3, 0])
    baseline = dark_hist.fit_result[..., 1, 0]
    baseline_error = dark_hist.fit_result[..., 1, 1]
    gain = mpes_full_fit_result[..., 1, 0]
    gain_error = mpes_full_fit_result[..., 1, 1]
    sigma_e = mpes_full_fit_result[..., 2, 0]
    sigma_e_error = mpes_full_fit_result[..., 2, 1]
    sigma_1 = mpes_full_fit_result[..., 3, 0]
    sigma_1_error = mpes_full_fit_result[..., 3, 1]

    print(baseline)

    integ = np.load(options.output_directory + options.pulse_shape_filename)
    integral = integ['integrals']
    integral_square = integ['integrals_square']

    for pixel in range(len(options.pixel_list)):

        y = dark_hist.data[pixel]

        if options.mc:
            baseline = 2010
            gain = 5.6
            sigma_1 = 0.48
            sigma_e = np.sqrt(0.86 ** 2.)

        dark_parameters = compute_dark_parameters(x, y, baseline[pixel], gain[pixel], sigma_1[pixel], sigma_e[pixel],
                                                  integral[pixel], integral_square[pixel])

        # print(pixel)
        # print(baseline)
        # print(dark_hist.fit_result.shape)

        dark_hist.fit_result[pixel, 1, 0] = dark_parameters[0, 0]
        dark_hist.fit_result[pixel, 1, 1] = dark_parameters[0, 1]
        dark_hist.fit_result[pixel, 2, 0] = dark_parameters[1, 0]
        dark_hist.fit_result[pixel, 2, 1] = dark_parameters[1, 1]

    dark_hist.fit_result[:, 0, 0] = baseline
    dark_hist.fit_result[:, 0, 1] = baseline_error

    # dark_hist.save(options.output_directory + options.histo_filename.split('.npz')[0]+'_xt.npz')
    dark_hist.save(options.output_directory + options.histo_filename)
    del dark_hist


def compute_dark_parameters(x, y, baseline, gain, sigma_1, sigma_e, integral,integral_square):
    '''
    In developement
    :param x:
    :param y:
    :param baseline:
    :param gain:
    :param sigma_1:
    :param sigma_e:
    :return:
    '''

    x = x - baseline
    sigma_1 = sigma_1/gain
    mean_adc = np.average(x, weights=y)
    sigma_2_adc = np.average((x - mean_adc) ** 2, weights=y) - 1./12.
    pulse_shape_area = integral * gain
    pulse_shape_2_area = integral_square * gain**2
    alpha = (mean_adc * pulse_shape_2_area)/((sigma_2_adc - sigma_e**2)*pulse_shape_area)

    if (1./alpha - sigma_1**2)<0 or np.isnan(1./alpha - sigma_1**2):
        mu_borel = np.nan
        mu_xt_dark = np.nan
        f_dark = np.nan

    elif np.sqrt(1./alpha - sigma_1**2)<1:

        mu_xt_dark = 0.
        f_dark = mean_adc / pulse_shape_area



    else:

        mu_borel = np.sqrt(1./alpha - sigma_1**2)
#        mu_borel = 1./(1.-0.06)
        mu_xt_dark = 1. - 1./mu_borel
        f_dark = mean_adc / mu_borel / pulse_shape_area

    f_dark_error = np.nan
    mu_xt_dark_error = np.nan

    """
    print('gain [ADC/p.e.]: %0.4f'%gain)
    print('baseline [LSB]: %0.4f'%baseline)
    print('sigma_e [LSB]: %0.4f'%sigma_e)
    print('sigma_1 [LSB]: %0.4f'%(sigma_1*gain))
    print('mean adc [LSB]: %0.4f' % mean_adc)
    print('sigma_2 adc [LSB]: %0.4f' % sigma_2_adc)
    print ('mu_borel : %0.4f [p.e.]'%mu_borel)
    print('f_dark %0.4f [MHz]' %(f_dark*1E3))
    print('dark XT : %0.4f [p.e.]' %mu_xt_dark)
    """

    return np.array([[f_dark*1E3, f_dark_error*1E3], [mu_xt_dark, mu_xt_dark_error]])


def single_photo_electron(options):

    from utils.peakdetect import peakdetect

    dark_spe = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    # dark_spe.data = (np.cumsum(dark_spe.data,axis=-1)-np.sum(dark_spe.data,axis=-1).reshape(-1,1))*-1
    # dark_spe.data = np.diff(dark_spe.data,n=1,axis=-1)*-1.
    # dark_spe.data = np.diff(dark_spe.data,n=1,axis=-1)*-1.
    # dark_spe.data = np.diff(dark_spe.data,n=1,axis=-1)*-1.

    hist_dark_fit_result = histogram.Histogram(filename=options.output_directory + options.histo_dark, fit_only=True).fit_result
    display.display_hist(dark_spe, options=options)
    #fixed_param = [ [0, (1, 0)],[2, (2, 0)] ]
    fixed_param = [[2, (2, 0)]]


    """
    threshold = 0.1
    min_dist = 10

    n_removed_points = np.zeros(dark_spe.data.shape[0])

    for pixel in range(dark_spe.data.shape[0]):

        y = dark_spe.data[pixel]
        if np.sum(y) != 0:
            print(y)
            peak_index = peakutils.indexes(y, threshold, min_dist)
            #peak_index = dark_spe.bin_centers[peak_index]
            print(peak_index)
            indices_around = min(6, min_dist)
            list_indices = [peak_index[j] + i for j in range(len(peak_index)) for i in range(-indices_around,
                                                                                             indices_around + 1, 1)]
            print(list_indices)
            list_indices = list(set(range(len(dark_spe.bin_centers))) - set(list_indices))
            print(dark_spe.data.shape)
            print(list_indices)
            print(pixel)

            n_removed_points[pixel] = len(list_indices)
            dark_spe.errors[pixel][list_indices] = 1E6

    """

    dark_spe.fit(fit_multiple_gaussians_spe.fit_func, fit_multiple_gaussians_spe.p0_func, fit_multiple_gaussians_spe.slice_func, fit_multiple_gaussians_spe.bounds_func, \
             labels_func=fit_multiple_gaussians_spe.labels_func, config=hist_dark_fit_result)  #, limited_indices=(7,))#, fixed_param=fixed_param)

    """
    dark_spe.fit(fit_multiple_gaussians_full_mpe.fit_func, fit_multiple_gaussians_full_mpe.p0_func,
                 fit_multiple_gaussians_full_mpe.slice_func, fit_multiple_gaussians_full_mpe.bounds_func, \
                 labels_func=fit_multiple_gaussians_full_mpe.labels_func, config=None,
                 limited_indices=(7,))  # , fixed_param=fixed_param)
  

    dark_spe.fit_chi2_ndof[..., 1] -= n_removed_points
    dark_spe.errors = np.sqrt(dark_spe.data)
    dark_spe.errors[dark_spe.errors == 0] = 1
    """

    display.display_hist(dark_spe, options=options, draw_fit=True)
    dark_spe.save(options.output_directory + options.histo_filename)


    """
    dark_spe.fit_result = np.zeros((dark_spe.data.shape[0], 3, 2))
    dark_spe.fit_result_label = np.array(['$f_{dark}$ [MHz]', 'XT', 'Gain [LSB/p.e.]'])

    n_samples = options.n_samples - options.baseline_per_event_limit - options.window_width + 1



    for pixel in range(dark_spe.data.shape[0]):

        index_fifteen = (dark_spe.bin_centers <= 15)
        dark_spe.data[pixel][index_fifteen] = 0
        peaks = np.array(peakdetect(dark_spe.data[pixel], lookahead=2)[0])
        n_events = options.max_event - options.min_event



        try:
            gain = np.mean(np.diff(peaks[:, 0][0:min(3, peaks.shape[0])]))
            one_pe_count = dark_spe.data[pixel][
                           max(0, peaks[0, 0] - gain // 2):min(dark_spe.data.shape[1], peaks[0, 0] + gain // 2):1]
            one_pe_count = np.sum(one_pe_count)

            #dark_count_rate = one_pe_count / (4. * n_samples * n_events) * 1E3
            dark_count_rate = np.sum(dark_spe.data[pixel]) / (4. * n_samples * n_events) * 1E3

            gain = dark_spe.bin_centers[peaks[0, 0]]
        except:

            dark_count_rate = np.nan

        try :

            #cross_talk = peaks[1][1]/peaks[0][1]
            cross_talk = np.sum(dark_spe.data[pixel][
                           max(0, peaks[1, 0] - gain // 2):min(dark_spe.data.shape[1], peaks[1, 0] + gain // 2):1])/ np.sum(dark_spe.data[pixel])

        except:

            cross_talk = np.nan
        dark_spe.fit_result[pixel, 0, 0] = dark_count_rate
        dark_spe.fit_result[pixel, 1, 0] = cross_talk
        dark_spe.fit_result[pixel, 2, 0] = gain

    #limits = [[1.5, 2.75],[0.05, 0.2], [18, 26]]
    for i in range(dark_spe.fit_result.shape[1]):
        fig = plt.figure(figsize=(10, 10))
        axis = fig.add_subplot(111)
        display.draw_fit_result(axis, dark_spe, options, level=0, index=i, limits=None, display_fit=True)
    #display.draw_fit_result(axis, dark_spe, options, level=0, index=1, limits=None)
    #display.draw_fit_result(axis, dark_spe, options, level=0, index=2, limits=None)


    #display.display_fit_result(dark_spe, options, limits=[1.8, 2.25], display_fit=True)


    cmap = cm.get_cmap('viridis')


    fig = plt.figure()
    axis_good = fig.add_subplot(121)
    axis_bad = fig.add_subplot(122)

    fig_2 = plt.figure(figsize=(10, 10))
    axis_all = fig_2.add_subplot(111)

    for i in range(dark_spe.data.shape[0]):
        col_curve = cmap(float(i) / float(len(options.pixel_list) + 1))

        if dark_spe.fit_result[i, 0, 0] > 1.:

            axis_good.semilogy(dark_spe.bin_centers/dark_spe.fit_result[pixel, 2, 0], dark_spe.data[i], color=col_curve, linestyle='-', linewidth=2, alpha=0.2)

        else:

            axis_bad.semilogy(dark_spe.bin_centers/dark_spe.fit_result[pixel, 2, 0], dark_spe.data[i], color=col_curve, linestyle='-', linewidth=2, alpha=0.2)

        axis_all.semilogy(dark_spe.bin_centers / dark_spe.fit_result[pixel, 2, 0], dark_spe.data[i], color=col_curve,
                          linestyle='-', linewidth=2, alpha=0.2)

        axis_all.set_xlabel('[LSB]')
        axis_all.set_ylabel('Counts')


    plt.show()
    input('press a key')

    """
    return


def fit_baseline(options):
    log = logging.getLogger(sys.modules['__main__'].__name__ + __name__)
    log.info('Perform a gaussian fit of the left side of the dark (==> baseline and sigmae for full mpe)')

    # Load the histogram
    adcs = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    # Fit the baseline and sigma_e of all pixels
    adcs.fit(fit_dark_adc.fit_func, fit_dark_adc.p0_func, fit_dark_adc.slice_func, fit_dark_adc.bounds_func, \
             labels_func=fit_dark_adc.labels_func)  # , limited_indices=tuple(options.pixel_list))
    # adcs.fit(fit_2_gaussian.fit_func, fit_2_gaussian.p0_func, fit_2_gaussian.slice_func, fit_2_gaussian.bounds_func, \
    #         labels_func=fit_2_gaussian.label_func)#, limited_indices=tuple(options.pixel_list))

    # Save the fit
    adcs.save(options.output_directory + options.histo_filename)

    # Delete the histograms
    del adcs
