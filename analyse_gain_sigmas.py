#!/usr/bin/env python3
from cts import cameratestsetup as cts
from utils.geometry import generate_geometry,generate_geometry_0
from utils.plots import pickable_visu_mpe,pickable_visu_led_mu
from utils.pdf import mpe_distribution_general,mpe_distribution_general_sh
from optparse import OptionParser
from utils.histogram import Histogram
from matplotlib import pyplot as plt
import numpy as np
from data_treatement import mpe_hist,synch_hist
from spectra_fit import fit_full_mpe
from utils.plots import display, display_var

parser = OptionParser()

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

# File management
parser.add_option("--file_basename", dest="file_basename",
                  help="file base name ", default="CameraDigicam@localhost.localdomain_0_000.%s.fits.fz")

parser.add_option("-d", "--directory", dest="directory",
                  help="input directory", default="/data/datasets/CTA/DATA/20161214/")

parser.add_option("--histo_filename", dest="histo_filename",
                  help="Histogram SPE file name", default="mpe_0_195_5_200_600_10.npz")

parser.add_option( "--peak_histo_filename", dest="peak_histo_filename",
                  help="name of peak histo file", default='peaks.npz')

parser.add_option("--output_directory", dest="output_directory",
                  help="directory of histo file", default='/data/datasets/CTA/DarkRun/20161214/')

parser.add_option("--fit_filename", dest="fit_filename",
                  help="name of fit file with MPE", default='mpe_scan_200_600_10_fit.npz')

parser.add_option("--input_fit_hvoff_filename", dest="input_hvoff_filename",
                  help="Input fit file name", default="adc_hv_off_fit.npz")

parser.add_option("--input_fit_dark_filename", dest="input_dark_filename",
                  help="Input fit file name", default="spe_hv_on_fit.npz")



# Arange the options
(options, args) = parser.parse_args()
options.file_list = options.file_list.split(',')
options.scan_level = [int(level) for level in options.scan_level.split(',')]
if len(options.scan_level)==3:
    options.scan_level= np.arange(options.scan_level[0],options.scan_level[1]+options.scan_level[2],options.scan_level[2])
else:
    options.scan_level = np.array(options.scan_level)

# Prepare the mpe histograms
mpes = Histogram(bin_center_min=1950., bin_center_max=4095., bin_width=1.,
                 data_shape=(options.scan_level.shape+(1296,)),
                 xlabel='Peak ADC', ylabel='$\mathrm{N_{entries}}$', label='MPE')

peaks = Histogram(bin_center_min=0., bin_center_max=50., bin_width=1.,
                  data_shape=((1296,)),
                  xlabel='Peak maximum position [4ns]', ylabel='$\mathrm{N_{entries}}$', label='peak position')


# Where do we take the data from
if options.create_time_histo:
    # Loop over the files
    synch_hist.run(peaks, options)
    del(peaks)

if options.verbose:
    print('--|> Recover data from %s' % (options.output_directory+options.peak_histo_filename))
file = np.load(options.output_directory+options.peak_histo_filename)
peaks = Histogram(data=file['peaks'], bin_centers=file['peaks_bin_centers'], xlabel ='sample [$\mathrm{4 ns^{1}}$]',
                  ylabel = '$\mathrm{N_{trigger}}$', label='synchrone peak position')


if options.create_histo:
    # Loop over the files
    mpe_hist.run([mpes], options, peak_positions = None )#peaks.data)
    del(mpes)

# recover previous fit
if options.verbose: print(
    '--|> Recover fit results from %s' % (options.output_directory + options.input_dark_filename))
file = np.load(options.output_directory + options.input_dark_filename)
spes_fit_result = np.copy(file['adcs_fit_result'])
file.close()
if options.verbose: print(
    '--|> Recover fit results from %s' % (options.output_directory + options.input_hvoff_filename))
file = np.load(options.output_directory + options.input_hvoff_filename)
adcs_fit_result = np.copy(file['adcs_fit_result'])
file.close()
# Now build a fake fit result for stating point
# get the baseline
prev_fit_result = np.expand_dims(adcs_fit_result[:, 1] + spes_fit_result[:, 6], axis=1)
prev_fit_result = np.append(prev_fit_result, np.expand_dims(spes_fit_result[:, 2], axis=1), axis=1)
prev_fit_result = np.append(prev_fit_result, np.expand_dims(spes_fit_result[:, 0], axis=1), axis=1)
prev_fit_result = np.append(prev_fit_result, np.expand_dims(spes_fit_result[:, 1], axis=1), axis=1)

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
    for i in range(mpe_tmp.shape[0]):
        for j in range(mpe_tmp.shape[1]):
            if mpe_mean[i, j] < 5 or np.where(mpe_tmp[i, j] != 0)[0].shape[0] < 2: mpe_tmp[i, j, :] = 0

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

if options.perform_fit_gain :
    reduced_bounds = lambda *args,config=None, **kwargs: fit_full_mpe.bounds_func(*args,n_peaks = 22, config=config, **kwargs)
    reduced_p0 = lambda *args,config=None, **kwargs: fit_full_mpe.p0_func(*args,n_peaks = 22, config=config, **kwargs)
    mpes_full.fit(fit_full_mpe.fit_func, reduced_p0, fit_full_mpe.slice_func,
                  reduced_bounds, config=prev_fit_result)
    # get the bad fits
    print('Try to correct the pixels with wrong fit results')
    for pix,pix_fit_result in enumerate(mpes_full.fit_result):
        if np.isnan(pix_fit_result[0,1]) and not np.isnan(pix_fit_result[0,0]):
            print('Pixel %d refit',pix)
            i = 25
            while  np.isnan(mpes_full.fit_result[pix,0,1]) and i > 15:
                reduced_bounds = lambda *args, config=None, **kwargs: fit_full_mpe.bounds_func(*args, n_peaks = i ,
                                                                                             config=config, **kwargs)
                reduced_p0 = lambda *args, config=None, **kwargs: fit_full_mpe.p0_func(*args, n_peaks = i , config=config,
                                                                                       **kwargs)
                mpes_full.fit(fit_full_mpe.fit_func, reduced_p0, fit_full_mpe.slice_func,
                              reduced_bounds, config=prev_fit_result ,limited_indices=(pix,),force_quiet=True)
                i-=1

    for pix,pix_fit_result in enumerate(mpes_full.fit_result):
        if np.isnan(pix_fit_result[0,1]) and not np.isnan(pix_fit_result[0,0]): print('-----|> Pixel %d is still badly fitted'%pix)
    print('--|> Save the full mpe fit result to %s' % (options.output_directory + 'full_'+ options.fit_filename))
    np.savez_compressed(options.output_directory + 'full_'+ options.fit_filename,
                        full_mpe_fit_result=mpes_full.fit_result)

if options.verbose:
    print('--|> Recover fit results for G and sigmas from %s' % (options.output_directory + 'full_'+ options.fit_filename))
file = np.load(options.output_directory+ 'full_' + options.fit_filename)
mpes_full.fit_result = np.copy(file['full_mpe_fit_result'])
mpes_full.fit_function = fit_full_mpe.fit_func


#del(mpes_full)
file.close()

# Leave the hand
plt.ion()

# Define Geometry
geom= generate_geometry_0()

display_var(mpes_full, geom, title='$\sigma_e$ [ADC]', index_var=2, limit_min=0.8, limit_max=1.2, bin_width=0.05)
display_var(mpes_full, geom, title='$\sigma_i$ [ADC]', index_var=3, limit_min=0.4, limit_max=0.5, bin_width=0.002)
display_var(mpes_full, geom, title='Gain [ADC/p.e.]' , index_var=1, limit_min=5.1, limit_max=6., bin_width=0.04)

display([mpes_full], geom, fit_full_mpe.slice_func, norm='linear',pix_init=10, config=prev_fit_result)




#show_level(10,mpes)
'''
plt.figure()
plt.errorbar(peaks.bin_centers,peaks.data[485],yerr=peaks.errors[485],label='485_%d')
plt.errorbar(peaks.bin_centers,peaks.data[486],yerr=peaks.errors[486],label='486_%d')
plt.errorbar(peaks.bin_centers,peaks.data[700],yerr=peaks.errors[700],label='700_%d')
plt.legend()
plt.figure()
for v in [55,56,57,58,59,60]:
    plt.errorbar(mpes.bin_centers,mpes.data[v,485],yerr=mpes.errors[v,485],label='485_%d'%v)

plt.legend()

plt.figure()
for v in [25,26,27,28,29,30,31,32,33]:
    plt.errorbar(mpes.bin_centers,mpes.data[v,486],yerr=mpes.errors[v,486],label='486_%d'%v)

plt.legend()
plt.figure()

for v in [35,36,37,38,39,40]:
    plt.errorbar(mpes.bin_centers,mpes.data[v,700],yerr=mpes.errors[v,700],label='700_%d'%v)

plt.legend()
plt.show()

'''