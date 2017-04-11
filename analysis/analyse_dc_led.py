
#!/usr/bin/env python3

# external modules
import logging,sys
import numpy as np
from tqdm import tqdm
from utils.logger import TqdmToLogger
import scipy
from numpy.linalg import inv


# internal modules
from data_treatement import mpe_hist
from spectra_fit import fit_dc_led
from utils import display, histogram, geometry
from ctapipe import visualization


__all__ = ["create_histo", "perform_analysis", "display_results", 'save']


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

    dc_led = histogram.Histogram(data=np.zeros((len(options.pixel_list), len(options.scan_level, )) ), \
                                 bin_centers=np.array(options.scan_level), label='DC LED', \
                                 xlabel='DC level', ylabel='$f_{NSB} [MHz]$')

    dc_led.errors = np.zeros((len(options.pixel_list), len(options.scan_level)))

    pulse_shape = histogram.Histogram(filename=options.directory + options.pulse_shape_filename)
    nsb = histogram.Histogram(filename=options.output_directory + options.nsb_histo_filename)
    mpes = histogram.Histogram(filename=options.output_directory + options.mpe_histo_filename)

    if options.full_mpe_histo_filename is not None:

        full_mpe = histogram.Histogram(filename=options.directory + options.full_mpe_histo_filename)

    else:

        full_mpe = histogram.Histogram(data=np.zeros((len(options.pixel_list))), \
                                 bin_centers=np.array(options.scan_level), label='Full MPE', \
                                 xlabel='ADC', ylabel='$N_{entries}$')

        full_mpe.fit_result = np.zeros((len(options.pixel_list), 2, 2))
        full_mpe.fit_result[:, 1, 0] = np.ones(full_mpe.data.shape[0])*22.7
        full_mpe.fit_result[:, 1, 1] = np.zeros(full_mpe.data.shape[0])

    log = logging.getLogger(sys.modules['__main__'].__name__ + '.' + __name__)
    pbar = tqdm(total=dc_led.data.shape[0])
    tqdm_out = TqdmToLogger(log, level=logging.INFO)

    xt = 0.08
    xt_error = 0.



    for pixel in range(dc_led.data.shape[0]):

        b = nsb.fit_result[:, pixel, 0, 0] - nsb.fit_result[0, pixel, 0, 0]
        delta_b = np.sqrt((nsb.fit_result[:, pixel, 0, 1] /nsb.fit_result[:, pixel, 0, 0])**2 + (nsb.fit_result[0, pixel, 0, 1]/nsb.fit_result[0, pixel, 0, 0])**2)
        d = (1. - xt)
        delta_d = xt_error
        a = b * d
        delta_a = np.abs(a) * np.sqrt((delta_b/b)**2 + (delta_d/d)**2)
        c = pulse_shape.fit_result[options.ac_level_for_pulse_shape_integral, pixel, 3, 0] * mpes.fit_result[:, pixel, 1, 0] * full_mpe.fit_result[pixel, 1, 0]
        delta_c = np.abs(c) * np.sqrt((pulse_shape.fit_result[options.ac_level_for_pulse_shape_integral, pixel, 3, 1]/pulse_shape.fit_result[options.ac_level_for_pulse_shape_integral, pixel, 3, 0])**2  + (mpes.fit_result[:, pixel, 1, 1]/mpes.fit_result[:, pixel, 1, 0])**2 + (full_mpe.fit_result[pixel, 1, 1]/full_mpe.fit_result[pixel, 1, 0])**2)
        f_nsb = a/c
        delta_f_nsb = np.abs(f_nsb) * np.sqrt((delta_a/a)**2 + (delta_c/c)**2)

        dc_led.data[pixel] = f_nsb * 1E3
        #dc_led.errors[pixel] = delta_f_nsb * 1E3
        dc_led.errors[pixel] = np.ones(dc_led.data[pixel].shape)

        '''
        dc_led.data[pixel] = nsb.fit_result[:, pixel, 0, 0] - nsb.fit_result[0, pixel, 0, 0]
        dc_led.data[pixel] /= mpes.fit_result[:, pixel, 1, 0] * full_mpe.fit_result[pixel, 1, 0]
        dc_led.data[pixel] /= pulse_shape.fit_result[options.ac_level_for_pulse_shape_integral, pixel, 3, 0]
        dc_led.data[pixel] *= 1E3 * (1. - xt)
        #dc_led.errors[pixel] = np.sqrt(nsb.fit_result[:, pixel, 0, 1]**2 + nsb.fit_result[0, pixel, 0, 1]**2)
        dc_led.errors[pixel] = np.ones(dc_led.data[pixel].shape)
        '''
        """
        if options.pixel_list[pixel]==627:
            print(nsb.fit_result[:, pixel, 0, 0])
            print(nsb.fit_result[0, pixel, 0, 0])
            print(mpes.fit_result[:, pixel, 1, 0])
            print(full_mpe.fit_result[pixel, 1, 0])
            print(pulse_shape.fit_result[options.ac_level_for_pulse_shape_integral, pixel, 3, 0])
        """

        pbar.update(1)

    dc_led.save(options.output_directory + options.histo_filename)

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

    dc_led = histogram.Histogram(filename=options.output_directory + options.histo_filename)


    #dc_led.fit_result_label = ['$p_0$', '$p_1$', '$p_2$', '$p_3$', '$p_4$', '$p_5$', '$p_6$']
    #dc_led.fit_result = np.zeros((dc_led.data.shape[:-1])+(len(dc_led.fit_result_label),2,))

    deg = 6

    log = logging.getLogger(sys.modules['__main__'].__name__ + '.' + __name__)
    pbar = tqdm(total=dc_led.data.shape[0])
    tqdm_out = TqdmToLogger(log, level=logging.INFO)

    dc_led.fit_function = np.polyval

    for pixel in range(dc_led.data.shape[0]):

        x = dc_led.bin_centers
        y = dc_led.data[pixel]
        yerr = dc_led.errors[pixel]

        #mask = ~np.isnan(y) * ~np.isnan(yerr) * (y>1E2) * (y<1E6)
        #x = x[mask]
        #y = y[mask]
        #yerr = yerr[mask]

        #if np.sum(mask)!=0:

        #try:

        dc_led.fit(fit_dc_led.fit_func, fit_dc_led.p0_func, fit_dc_led.slice_func,
                          fit_dc_led.bounds_func, config=None, limited_indices=(pixel,), force_quiet=True,
                          labels_func=fit_dc_led.label_func)

         #   print('hello')

            #fit_result = np.polyfit(x, y, deg=deg, w=1./yerr, cov=True)
            #print(fit_result)
            #dc_led.fit_result[pixel, 0:deg + 2, 0] = fit_result[0]
            #dc_led.fit_result[pixel, 0:deg + 2, 1] = np.sqrt(np.diag(fit_result[1]))

        #except:

        #    log.info('Could not fit DC LED for pixel %d' %options.pixel_list[pixel])

        pbar.update(1)

    dc_led.save(options.output_directory + options.histo_filename)



def display_results(options):
    """
    Display the analysis results
    :param options:
    :return:
    """

    # Load the histogram
    dc_led = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    # Define Geometry
    geom = geometry.generate_geometry_0(pixel_list=options.pixel_list)

    #display.display_hist(dc_led, options, geom=geom, scale='linear')
    #display.display_hist(dc_led, options, geom=geom, display_parameter=True)
    display.display_hist(dc_led, options, geom=geom, draw_fit=True, scale='log')
    #display.display_hist(dc_led, options, geom=geom, draw_fit=True, scale='linear')
    display.display_chi2(dc_led, geom=geom)


    return


def save(options): # convert f to dac

    dc_led = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    dac_limits = [0, 1000]

    def inverse_func(param, f):

        return (1./param[1] * np.log(f/param[0])).astype(int)

    dac = np.zeros((dc_led.fit_result.shape[0], len(options.frequency_list)))

    if np.any(np.array(options.frequency_list)>1E4) or np.any(np.array(options.frequency_list)<1E1):

        pass

    else:

        for pixel in range(dac.shape[0]):

            dac[pixel, :] = inverse_func(dc_led.fit_result[pixel, :, 0], np.array(options.frequency_list))

            if np.any(dac[pixel, :]<dac_limits[0]) or np.any(dac[pixel, :]>dac_limits[1]) or np.any(np.isnan(dac[pixel, :])):

                dac[pixel, :] = np.zeros(dac.shape[1])

    coeff = np.zeros((dc_led.data.shape[0], 2))
    coeff[:, 0] = dc_led.fit_result[:, 0, 0]
    coeff[:, 1] = dc_led.fit_result[:, 1, 0]

    np.savetxt(fname=options.output_directory + 'coeff.txt', X=coeff)
    np.savetxt(fname=options.output_directory + options.frequency_filename, X=dac, fmt='%d')



