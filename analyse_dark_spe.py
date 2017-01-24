#!/usr/bin/env python3
from optparse import OptionParser
import matplotlib.pyplot as plt
import numpy as np
from utils.plots import display, display_var
from data_treatement import adc_hist
from utils.geometry import generate_geometry_0
from utils.histogram import Histogram
from spectra_fit import fit_dark_spe

parser = OptionParser()
# Job configuration
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

parser.add_option("-c", "--create_histo", dest="create_histo", action="store_true",
                  help="load the SPE from dark run histograms from file", default=False)

parser.add_option("-p", "--perform_fit", dest="perform_fit", action="store_true",
                  help="perform fit of SPE from dark run", default=False)

parser.add_option("-f", "--file_list", dest="file_list",
                  help="input filenames separated by ','", default='117,118,119,120,121')

parser.add_option("--evt_max", dest="evt_max",
                  help="maximal number of events", default=50000, type=int)

parser.add_option("-n", "--n_evt_per_batch", dest="n_evt_per_batch",
                  help="number of events per batch", default=1000, type=int)

# File management
parser.add_option("--file_basename", dest="file_basename",
                  help="file base name ", default="CameraDigicam@localhost.localdomain_0_000.%s.fits.fz")
parser.add_option("-d", "--directory", dest="directory",
                  help="input directory", default="/data/datasets/CTA/DATA/20161214/")

parser.add_option("--histo_filename", dest="histo_filename",
                  help="Histogram SPE file name", default="spe_hv_on.npz")

parser.add_option("--output_directory", dest="output_directory",
                  help="directory of histo file", default='/data/datasets/CTA/DarkRun/20161214/')

parser.add_option("--fit_filename", dest="fit_filename",
                  help="name of fit file with SPE", default='spe_hv_on_fit.npz')

parser.add_option("--input_fit_filename", dest="input_fit_filename",
                  help="Input fit file name", default="adc_hv_off_fit.npz")

# Arrange the options
(options, args) = parser.parse_args()
options.file_list = options.file_list.split(',')

# Define the histograms
adcs = Histogram(bin_center_min=0., bin_center_max=4095., bin_width=1., data_shape=(1296,))

# Get the fit results from the HV OFF run
if options.verbose:
    print('--|> Recover data from %s' % (options.output_directory + options.input_fit_filename))
file = np.load(options.output_directory + options.input_fit_filename)
prev_fit_result = np.copy(file['adcs_fit_result'])

# Get the adcs
if options.create_histo:
    # Fill the adcs hist from data
    adc_hist.run(adcs, options, 'SPE',prev_fit_result=prev_fit_result)

if options.verbose:
    print('--|> Recover data from %s' % (options.output_directory + options.histo_filename))
file = np.load(options.output_directory + options.histo_filename)
adcs = Histogram(data=np.copy(file['adcs']), bin_centers=np.copy(file['adcs_bin_centers']))


# Recover fit from the HV off
if options.perform_fit:
    print('--|> Compute gain, cross talk sigma_i and sigma_e from ADC distributions with HV OFF')
    #TODO full fit including cross talk
    # Fit the baseline and sigma_e of all pixels
    adcs.fit(fit_dark_spe.fit_func, fit_dark_spe.p0_func, fit_dark_spe.slice_func,
             fit_dark_spe.bounds_func, config=prev_fit_result)
    if options.verbose:
        print('--|> Save the data in %s' % (options.output_directory + options.fit_filename))
    np.savez_compressed(options.output_directory + options.fit_filename,
                        adcs_fit_result=adcs.fit_result)

if options.verbose:
    print('--|> Recover data from %s' % (options.output_directory + options.fit_filename))
file = np.load(options.output_directory + options.fit_filename)
adcs.fit_result = np.copy(file['adcs_fit_result'])
adcs.fit_function = fit_dark_spe.fit_func


# Leave the hand
plt.ion()

# Define Geometry
geom = generate_geometry_0()

# Perform some plots
display_var(adcs, geom, title='$\sigma_e$ [ADC]', index_var=0, limit_min=0., limit_max=1., bin_width=0.05)
display_var(adcs, geom, title='$\sigma_i$ [ADC]', index_var=1, limit_min=0., limit_max=1., bin_width=0.05)
display_var(adcs, geom, title='Gain [ADC/p.e.]' , index_var=2, limit_min=4., limit_max=6., bin_width=0.05)
display_var(adcs, geom, title='Offset to baseline' , index_var=6, limit_min=-100., limit_max=100., bin_width=1)
display([adcs], geom, fit_dark_spe.slice_func, norm='log', config=prev_fit_result)
