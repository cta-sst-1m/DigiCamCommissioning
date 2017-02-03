#!/usr/bin/env python3
from cts import cameratestsetup as cts
from utils.geometry import generate_geometry,generate_geometry_0, generate_geometry_MC
from utils.plots import pickable_visu_mpe,pickable_visu_led_mu
from utils.pdf import mpe_distribution_general,mpe_distribution_general_sh
from optparse import OptionParser
from utils.histogram import Histogram
import peakutils
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from data_treatement import mpe_hist,synch_hist
from utils.plots import pickable_visu
from spectra_fit import fit_low_light,fit_full_mpe
from utils.plots import display, display_var, display_var_biais, display_chi2
import logging, sys

parser = OptionParser()

logging.basicConfig()

# Job configuration
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

parser.add_option("-c", "--create_histo", dest="create_histo", action="store_true",
                  help="load the mpe histo from file", default=False)

parser.add_option("-t", "--create_time_histo", dest="create_time_histo", action="store_true",
                  help="load the mpe histo from file", default=False)

parser.add_option("-k", "--create_full_histo", dest="create_full_histo", action="store_true",
                  help="load the mpe full histo from file", default=False)

parser.add_option("-p", "--perform_fit_gain", dest="perform_fit_gain", action="store_true",
                  help="perform fit of all mpe to get gain, sigma_e, sigma1", default=False)

parser.add_option("-f", "--file_list", dest="file_list",
                  help="input filenames separated by ','", default=
                  '124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149')

parser.add_option("-w", "--weights", dest="weights",
                  help="weights for the construction of full mpe ','", default=
                  '1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1')


parser.add_option("-l", "--scan_level", dest="scan_level",
                  help="list of scans DC level, separated by ',', if only three argument, min,max,step",
                  default=
                  '''
0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120,
130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180,
185, 190, 195, 200, 210, 220, 230, 240, 250, 260, 270,
280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380,
390, 400, 410, 420, 430, 440, 450,460, 470, 480, 490,
500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600
''')

parser.add_option("-e", "--events_per_level", dest="events_per_level",
                  help="number of events per level", default=5000,type=int)

parser.add_option("--evt_max", dest="evt_max",
                  help="maximal number of events", default=5e10, type=int)

parser.add_option("-n", "--n_evt_per_batch", dest="n_evt_per_batch",
                  help="number of events per batch", default=1000, type=int)

parser.add_option("--n_pixels", dest="n_pixels",
                  help="number of of pixels ", default=1296, type=int)

# File management
parser.add_option("--file_basename", dest="file_basename",
                  help="file base name ", default="CameraDigicam@localhost.localdomain_0_000.%s.fits.fz")

parser.add_option("-d", "--directory", dest="directory",
                  help="input directory", default="/data/datasets/CTA/DATA/20161214/")

parser.add_option("--histo_filename", dest="histo_filename",
                  help="Histogram SPE file name", default="mpe_scan_0_195_5_200_600_10.npz")

parser.add_option( "--peak_histo_filename", dest="peak_histo_filename",
                  help="name of peak histo file", default='peaks.npz')

parser.add_option("--output_directory", dest="output_directory",
                  help="directory of histo file", default='/data/datasets/CTA/DarkRun/20161214/')

parser.add_option("--fit_filename", dest="fit_filename",
                  help="name of fit file with MPE", default='mpe_scan_0_195_5_200_600_10_fit.npz')

parser.add_option("--input_fit_hvoff_filename", dest="input_hvoff_filename",
                  help="Input fit file name", default="adc_hv_off_fit.npz")

parser.add_option("--input_fit_dark_filename", dest="input_dark_filename",
                  help="Input fit file name", default="spe_hv_on_fit.npz")

parser.add_option("--toy_test", dest="toy_test", action="store_true",
                  help="Perform ac dac scan on toy data", default=False)



# Arange the options
(options, args) = parser.parse_args()

options.histo_filename = options.file_basename[0:8] + '.npz'
options.peak_histo_filename = 'peaks_' + options.histo_filename
options.fit_filename = 'fit_' + options.histo_filename

options.file_list = options.file_list.split(',')
options.weights = np.array(options.weights.split(','), dtype=float)
#options.weights = options.weights * np.sqrt(1.12**2 + np.arange(len(options.weights))*(1+0.06)*0.56**2)#np.cumsum(options.weights)
#options.weights = np.cumsum(options.weights)
options.scan_level = [int(level) for level in options.scan_level.split(',')]
if len(options.scan_level)==3:
    options.scan_level= np.arange(options.scan_level[0],options.scan_level[1]+options.scan_level[2],options.scan_level[2])
else:
    options.scan_level = np.array(options.scan_level)

# Prepare the mpe histograms
mpes = Histogram(bin_center_min=0., bin_center_max=4000., bin_width=1.,
                 data_shape=(options.scan_level.shape+(options.n_pixels,)),
                 xlabel='Peak ADC', ylabel='$\mathrm{N_{entries}}$', label='MPE')

peaks = Histogram(bin_center_min=0., bin_center_max=50., bin_width=1.,
                  data_shape=((options.n_pixels,)),
                  xlabel='Peak maximum position [4ns]', ylabel='$\mathrm{N_{entries}}$', label='peak position')
# Where do we take the data from
if options.create_time_histo:
    # Loop over the files
    synch_hist.run(peaks, options, min_evt=0, max_evt=50000)
    del(peaks)



if options.verbose:
    print('--|> Recover data from %s' % (options.output_directory+options.peak_histo_filename))
file = np.load(options.output_directory+options.peak_histo_filename)
peaks = Histogram(data=file['peaks'], bin_centers=file['peaks_bin_centers'], xlabel ='sample [$\mathrm{4 ns^{1}}$]',
                  ylabel = '$\mathrm{N_{trigger}}$', label='synchrone peak position')


if options.create_histo:
    # Loop over the files
    if options.toy_test:
        if options.verbose:
            print('--|> Recover data from Toy %s' % (options.directory + options.file_basename))

        mpe_hist.run([mpes], options, peak_positions=peaks.data)
    else:

        mpe_hist.run([mpes], options, peak_positions = None )#peaks.data)

del(mpes)



prev_fit_result = np.ones((options.n_pixels, 4, 2))

if options.verbose: print('--|> Recover data from %s' % (options.output_directory+options.histo_filename))
file = np.load(options.output_directory+options.histo_filename)
mpes = Histogram(data=np.copy(file['mpes']), bin_centers=np.copy(file['mpes_bin_centers']), xlabel ='Peak ADC',
                 ylabel='$\mathrm{N_{trigger}}$', label='MPE from peak value')
file.close()

if options.create_full_histo:
    # Add an Histogram corresponding to the sum of all other only if the mu is above a certain threshold
    print('--|> Create summed MPE')
    mpe_tmp = np.copy(mpes.data)
    mpe_tmp[mpe_tmp == 0] = 1e-6
    mpe_mean = np.average(np.repeat(
        np.repeat(
            mpes.bin_centers[1:-1:1].reshape(1, 1, -1), mpe_tmp.shape[0], axis=0), mpe_tmp.shape[1], axis=1),
        weights=mpe_tmp[..., 1:-1:1], axis=2)
    del (mpe_tmp)
    # subtract the baseline
    mpe_mean = np.subtract(mpe_mean, np.repeat(prev_fit_result[:, 0, 0].reshape((1,) + prev_fit_result[:, 0, 0].shape),
                                               mpe_mean.shape[0], axis=0))

    mpe_tmp = np.copy(mpes.data)
    #for i in range(mpe_tmp.shape[0]):
    #    for j in range(mpe_tmp.shape[1]):
            #if mpe_mean[i, j] < 5 or np.where(mpe_tmp[i, j] != 0)[0].shape[0] < 2: mpe_tmp[i, j, :] = 0
            #if mpe_mean[i, j] < 10 or np.where(mpe_tmp[i, j] != 0)[0].shape[0] < 2: mpe_tmp[i, j, :] = 0

    mpe_tmp = np.sum(mpe_tmp, axis=0)
    mpes_full = Histogram(data=np.copy(mpe_tmp), bin_centers=mpes.bin_centers, xlabel='ADC',
                          ylabel='$\mathrm{N_{trigger}}$', label='Summed MPE')
    np.savez_compressed(options.output_directory + 'full_' + options.histo_filename,
                        full_mpe=mpes_full.data, full_mpe_bin_centers=mpes_full.bin_centers)
    del (mpe_tmp)
    # for i in range(1296):
    #    print(np.mean(mpes_full.data[i]))

if options.verbose: print('--|> Recover data from %s' % (options.output_directory+'full_' +options.histo_filename))
file = np.load(options.output_directory+'full_' +options.histo_filename)
mpes_full = Histogram(data=np.copy(file['full_mpe']), bin_centers=np.copy(file['full_mpe_bin_centers']), xlabel ='ADC',
                      ylabel='$\mathrm{N_{trigger}}$', label='Summed MPE')
file.close()


def find_n_peaks(bin_centers, data):

    if options.toy_test:

        return len(options.weights)

    else:

        threshold = 0.05
        min_dist = 3
        x = bin_centers
        y = data
        x = x[y>0]
        y = y[y>0]
        log_func = - np.diff(np.log(y)) / np.diff(x)
        peak_index = peakutils.indexes(log_func, threshold, min_dist) - 1
        n_peaks = len(peak_index)

        return n_peaks

if options.perform_fit_gain :

    print('--|> Perform full MPE fit')

    for pixel in range(mpes_full.data.shape[0]):

        n_peaks = find_n_peaks(mpes_full.bin_centers, mpes_full.data[pixel])

        logging.debug('Initial number of peaks : %d for pixel : %d ', n_peaks, pixel)

        n_peaks_max = n_peaks + 2
        n_peaks_min = n_peaks - 2

        i = n_peaks_min

        best_fit_result = np.inf

        while i>=n_peaks_min and i<=n_peaks_max:

            reduced_bounds = lambda *args, config=None, **kwargs: fit_full_mpe.bounds_func(*args, n_peaks=i,
                                                                                           config=config, **kwargs)
            reduced_p0 = lambda *args, config=None, **kwargs: fit_full_mpe.p0_func(*args, n_peaks=i,
                                                                                   config=config, **kwargs)
            mpes_full.fit(fit_full_mpe.fit_func, reduced_p0, fit_full_mpe.slice_func,
                          reduced_bounds, config=prev_fit_result, limited_indices=(pixel,),force_quiet=True)

            if mpes_full.fit_chi2_ndof[pixel, 0]<best_fit_result:

                best_fit_result = mpes_full.fit_chi2_ndof[pixel, 0]
                n_peaks = i

            i = i + 1

        logging.debug('n_peaks : %d for pixel : %d', n_peaks, pixel)

        reduced_bounds = lambda *args, config=None, **kwargs: fit_full_mpe.bounds_func(*args, n_peaks=n_peaks,
                                                                                       config=config, **kwargs)
        reduced_p0 = lambda *args, config=None, **kwargs: fit_full_mpe.p0_func(*args, n_peaks=n_peaks,
                                                                               config=config, **kwargs)
        mpes_full.fit(fit_full_mpe.fit_func, reduced_p0, fit_full_mpe.slice_func,
                      reduced_bounds, config=prev_fit_result, limited_indices=(pixel,), force_quiet=True)

        if np.any(np.isnan(mpes_full.fit_result[pixel,:,1])) and not np.all(np.isnan(pix_fit_result[pixel,:,0])):
            print('-----|> Pixel %d is still badly fitted'%pixel)

    print('--|> Save the full mpe fit result to %s' % (options.output_directory + options.fit_filename))
    np.savez_compressed(options.output_directory + 'full_'+ options.fit_filename,
                        full_mpe_fit_result=mpes_full.fit_result)

if options.verbose:
    print('--|> Recover fit results for G and sigmas from %s' % (options.output_directory + options.fit_filename))
file = np.load(options.output_directory+ 'full_' + options.fit_filename)
mpes_full.fit_result = np.copy(file['full_mpe_fit_result'])
mpes_full.fit_function = fit_full_mpe.fit_func

print

#del(mpes_full)
file.close()

# Leave the hand
plt.ion()

# Define Geometry

#geom = generate_geometry_MC(n_pixels=options.n_pixels)
geom= generate_geometry_0(n_pixels=options.n_pixels)
show_fit = True

display_var(mpes_full, geom, title='Baseline [ADC]', index_var=0, show_fit=show_fit)
display_var(mpes_full, geom, title='Gain [ADC/p.e.]', index_var=1, show_fit=show_fit)
display_var(mpes_full, geom, title='$\sigma_e$ [ADC]', index_var=2, show_fit=show_fit)
display_var(mpes_full, geom, title='$\sigma_1$ [ADC]', index_var=3, show_fit=show_fit)
display_var_biais(mpes_full, geom, title='Baseline [ADC]', index_var=0, true_param=2010, show_fit=show_fit)
display_var_biais(mpes_full, geom, title='Gain [ADC/p.e.]', index_var=1, true_param=5.6, show_fit=show_fit)
display_var_biais(mpes_full, geom, title='$\sigma_e$ [ADC]', index_var=2, true_param=0.86, show_fit=show_fit)
display_var_biais(mpes_full, geom, title='$\sigma_1$ [ADC]', index_var=3, true_param=0.48, show_fit=show_fit)
display_chi2(mpes_full, geom, show_fit=show_fit)

display([mpes_full], geom, fit_full_mpe.slice_func, norm='linear',pix_init=0, config=mpes_full.fit_result)