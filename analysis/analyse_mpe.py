#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import mpe_hist
from spectra_fit import fit_low_light,fit_high_light
from utils import display, histogram, geometry
import logging,sys
import numpy as np
import logging
from tqdm import tqdm
from utils.logger import TqdmToLogger

__all__ = ["create_histo", "perform_analysis", "display_results"]


def create_histo(options):
    """
    Create a list of ADC histograms and fill it with data

    :param options: a dictionary containing at least the following keys:
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram      (str)
        - 'synch_histo_filename'   : the name of the file containing the histogram      (str)
        - 'file_basename'    : the base name of the input files                   (str)
        - 'directory'        : the path of the directory containing input files   (str)
        - 'file_list'        : the list of base filename modifiers for the input files
                                                                                  (list(str))
        - 'evt_max'          : the maximal number of events to process            (int)
        - 'n_evt_per_batch'  : the number of event per fill batch. It can be
                               optimised to improve the speed vs. memory          (int)
        - 'n_pixels'         : the number of pixels to consider                   (int)
        - 'scan_level'       : number of unique poisson dataset                   (int)
        - 'adcs_min'         : the minimum adc value in histo                     (int)
        - 'adcs_max'         : the maximum adc value in histo                     (int)
        - 'adcs_binwidth'    : the bin width for the adcs histo                   (int)

    :return:
    """

    # Define the histograms
    mpes = histogram.Histogram(bin_center_min=options.adcs_min, bin_center_max=options.adcs_max,
                               bin_width=options.adcs_binwidth, data_shape=(len(options.scan_level),len(options.pixel_list),),
                               label='MPE',xlabel='Peak ADC',ylabel = '$\mathrm{N_{entries}}$')

    # Get the reference sampling time
    peaks = histogram.Histogram(filename = options.output_directory + options.synch_histo_filename)

    # Construct the histogram
    mpe_hist.run(mpes, options, peak_positions=peaks.data)

    # Save the histogram
    mpes.save(options.output_directory + options.histo_filename)

    # Delete the histograms
    del mpes,peaks

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
    # Fit the baseline and sigma_e of all pixels
    mpes = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    nlevel = mpes.data.shape[0]
    mpes_full = histogram.Histogram(filename=options.output_directory + options.full_histo_filename, fit_only= True)
    mpes_full_fit_result = np.copy(mpes_full.fit_result)
    dark = histogram.Histogram(filename=options.output_directory + options.dark_histo_filename, fit_only= True)
    dark_fit_result = np.copy(dark.fit_result)
    del dark
    del mpes_full
    #mpes_full_fit_result[...,0,0]=dark_fit_result[...,1,0]
    #mpes_full_fit_result[...,0,1]=dark_fit_result[...,1,1]
    mpes_full_fit_result = mpes_full_fit_result.reshape((1,) + mpes_full_fit_result.shape)
    mpes_full_fit_result = np.repeat(mpes_full_fit_result, nlevel, axis=0)
    log = logging.getLogger(sys.modules['__main__'].__name__+__name__)
    pbar = tqdm(total=mpes.data.shape[0]*mpes.data.shape[1])
    tqdm_out = TqdmToLogger(log, level=logging.INFO)
    def std_dev(x, y):
        if np.sum(y)<=0: return 0.
        avg = np.average(x, weights=y)
        return np.sqrt(np.average((x - avg) ** 2, weights=y))

    ## Now perform the mu and mu_XT fits
    for pixel,real_pix in enumerate(options.pixel_list):

        _tmp_level = -1
        #if hasattr(options,'pixel_list') and pixel not in options.pixel_list:
        #    continue

        force_xt = False
        if pixel > 0: log.debug('Pixel #' + str(pixel - 1)+' treated')
        for level in range(mpes.data.shape[0]):
            pbar.update(1)
            if np.isnan(mpes_full_fit_result[0,pixel,0,0]): continue
            if np.nonzero(mpes.data[level, pixel])[0].shape[0] == 1: continue
            if std_dev(mpes.bin_centers, mpes.data[level, pixel]) > 400: continue
            if mpes.data[level, pixel, -1] > 0.02 * np.sum(mpes.data[level, pixel]): continue
            if np.sum(mpes.data[level, pixel])<1e-8 or np.sum(mpes.data[level, pixel])<options.events_per_level/10:continue
            if _tmp_level!= level-1: continue
            #print(np.sum(mpes.data[level, pixel]),std_dev(mpes.bin_centers, mpes.data[level, pixel]))
            #print(mpes.data[level, pixel])
            # check if the mu of the previous level is above 5
            _tmp_level = level
            fixed_param = []
            _fit_spectra = fit_low_light
            #if level > 0:
                #print('####################### MU :',mpes.fit_result[level - 1, pixel, 0, 0] )
            #print('pixel %d, level %d'%(pixel,level))
            successful = False
            while not successful:

                #if level > 0 :
                #    print('level:',level-1,mpes.fit_result[level - 1, pixel, 0, 0],mpes.fit_result[level - 1, pixel, 1, 0],mpes.fit_result[level - 1, pixel, 0, 1],mpes.fit_result[level - 1, pixel, 1, 1])

                if level > 0 and (mpes.fit_result[level - 1, pixel, 0, 0] > 25. or _fit_spectra is fit_high_light):
                   #mpes.save(options.output_directory + options.histo_filename)
                    #sys.exit()
                    fixed_param = [
                        # in this case assign the cross talk estimation with smallest error
                        [1, mpes.fit_result[np.argmin(mpes.fit_result[5:level:1, pixel, 1, 1]), pixel, 1, 0]],
                        # start from level 5 to avoid taking dark or hv off
                        [2, (1, 0)],  # gain
                        [3, (0, 0)],  # baseline
                        # [4,(2,0)], # sigma_e
                        [5, (3, 0)],  # sigma_1
                        [7, 0.]  # offset
                    ]
                    _fit_spectra = fit_high_light
                elif (level > 0 and mpes.fit_result[
                        level - 1, pixel, 0, 0] > 10.) or force_xt:  # TODO Sometimes mu_xt error min is for mu_xt =0. (dark/hv off)
                    fixed_xt = mpes.fit_result[np.argmin(mpes.fit_result[0:level:1, pixel, 1,
                                                         1]), pixel, 1, 0]  # start from level 5 to avoid taking dark or hv off
                    fixed_param = [
                        # in this case assign the cross talk estimation with smallest error
                        [1, fixed_xt],
                        [2, (1, 0)],  # gain
                        #[3, (0, 0)],  # baseline
                        [4, (2, 0)],  # sigma_e
                        [5, (3, 0)],  # sigma_1
                        [7, 0.]  # offset
                    ]
                else:
                    fixed_param = [
                        [2, (1, 0)],  # gain
                        #[3, (0, 0)],  # baseline
                        #[4, (2, 0)],  # sigma_e
                        #[5, (3, 0)],  # sigma_1
                        [7, 0.]  # offset
                    ]
                mpes.fit(_fit_spectra.fit_func, _fit_spectra.p0_func, _fit_spectra.slice_func,
                         _fit_spectra.bounds_func, config=mpes_full_fit_result, fixed_param=fixed_param
                         , limited_indices=[(level, pixel,)], force_quiet=True, labels_func=_fit_spectra.label_func)
                successful=True
    mpes.save(options.output_directory + options.histo_filename)


def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    # Load the histogram
    adcs = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    # Define Geometry
    geom = geometry.generate_geometry_0(pixel_list=options.pixel_list)

    import matplotlib.pyplot as plt

    pixel_start = options.pixel_list[2]

    # Perform some plots
    if options.mc:

        for level in options.scan_level:

            fig = plt.figure()
            axis = fig.add_subplot(111)
            display.draw_hist(axis, adcs, index=(level, pixel_start,), limits=[2005, 2150], draw_fit=True, label='Pixel %s')

        x = np.array(options.scan_level)*5.
        y = adcs.fit_result[:,int(options.n_pixels-1),0,0]
        yerr = adcs.fit_result[:,int(options.n_pixels-1),0,1]
        mask = np.isfinite(x)*np.isfinite(y)*np.isfinite(yerr)

        param = np.polyfit(x[mask], y[mask], 4, w=1./yerr[mask])
        text_param = ''
        for i in range(len(param)):
            text_param += 'p_%d = %0.9f  \n' %(i, param[i])

        true_param = np.array([11 * 1E-8, 0., 0., 0., 0.])

        fig = plt.figure()
        ax_up = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
        ax_down = plt.subplot2grid((4,4), (3,0), colspan=4, sharex=ax_up)
        #ax_down_2 = plt.subplot2grid((4,4), (3,0), colspan=4, sharex=ax_up)
        ax_up.plot(x, np.polyval(param, x), label='MC observed best fit p_0 = %0.4f [p.e.]' %param[-1])
        ax_up.plot(x, np.polyval(true_param, x), label='MC generated')
        ax_up.errorbar(x[mask], y[mask], yerr=yerr[mask], label='MC observed', linestyle='None', barsabove=True, markersize=12, marker='o')
        ax_down.plot(x[mask], np.abs(np.polyval(param, x[mask])-np.polyval(true_param, x[mask]))/np.polyval(param, x[mask]), label='bias polynomial')
        ax_down.plot(x[mask], np.abs(y[mask]-np.polyval(true_param, x[mask]))/y[mask], label='bias measurements')
        #ax_down_2.plot(x[mask], np.abs(y[mask]-np.polyval(true_param, x[mask]))/yerr[mask], label='pull')
        #ax_up.text(x[-3], y[-3], text_param)
        ax_down.set_xlabel('DAC')
        ax_up.set_ylabel('$\mu$ [p.e.]')
        #ax_down.set_ylabel('$\\frac{\mu_{t}- \mu_{m}}{\sigma_{m}}$')
        fig.subplots_adjust(hspace=0.1)
        plt.setp(ax_up.get_xticklabels(), visible=False)
        #plt.setp(ax_down.get_xticklabels(), visible=False)
        ax_up.set_yscale('log')
        ax_down.set_yscale('log')
        #ax_down_2.set_yscale('log')
        ax_up.legend()
        ax_down.legend()
        #ax_down_2.legend()


    else:

        display.display_hist(adcs, options=options, geom=geom,draw_fit=True)
    input('press button to quit')

    return
