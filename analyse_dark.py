#!/usr/bin/env python3
from optparse import OptionParser
import matplotlib.pyplot as plt
import numpy as np
from utils.plots import display, display_var
from data_treatement import adc_hist
from utils.geometry import generate_geometry_0
from utils.histogram import histogram
from spectra_fit import fit_dark_adc

parser = OptionParser()
# Job configuration
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

parser.add_option("-c", "--create_histo", dest="create_histo", action="store_true",
                  help="load the ADC with HV ON histograms from file", default=False)

parser.add_option("-p", "--perform_fit", dest="perform_fit", action="store_true",
                  help="perform fit of ADC from dark run", default=False)

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
                  help="Histogram SPE file name", default="adc_hv_on.npz")

parser.add_option("--output_directory", dest="output_directory",
                  help="directory of histo file", default='/data/datasets/CTA/DarkRun/20161214/')

parser.add_option("--fit_filename", dest="fit_filename",
                  help="name of fit file with SPE", default='adc_hv_on_fit.npz')

# Arrange the options
(options, args) = parser.parse_args()
options.file_list = options.file_list.split(',')

# Define the histograms
adcs = histogram(bin_center_min=0., bin_center_max=4095., bin_width=1., data_shape=(1296,))


# Get the adcs
if options.create_histo:
    # Fill the adcs hist from data
    adc_hist.run(adcs, options, 'ADC')

if options.verbose:
    print('--|> Recover data from %s' % (options.output_directory + options.histo_filename))
file = np.load(options.output_directory + options.histo_filename)
adcs = histogram(data=np.copy(file['adcs']), bin_centers=np.copy(file['adcs_bin_centers']))

# Recover fit from the HV off
'''
TODO:
1- evaluation from steps
2- evaluation from pdfs

if options.perform_fit:
    print('--|> Compute gain, cross talk sigma_i and sigma_e from ADC distributions with HV OFF')
    # Fit the baseline and sigma_e of all pixels
    adcs.fit(fit_dark.fit_func, fit_dark.p0_func, fit_dark.slice_func,
             fit_dark.bounds_func, config=prev_fit_result)
    if options.verbose:
        print('--|> Save the data in %s' % (options.output_directory + options.fit_filename))
    np.savez_compressed(options.output_directory + options.fit_filename,
                        adcs_fit_result=adcs.fit_result)
else:
    if options.verbose:
        print('--|> Recover data from %s' % (options.output_directory + options.fit_filename))
    file = np.load(options.output_directory + options.fit_filename)
    adcs.fit_result = np.copy(file['adcs_fit_result'])
    adcs.fit_function = fit_dark.fit_func
'''

# Leave the hand
plt.ion()

# Define Geometry
geom = generate_geometry_0()

# Perform some plots
display_var(adcs, geom, title='$\sigma_e$ [ADC]', index_var=2, limit_min=0., limit_max=2., bin_width=0.05)
display_var(adcs, geom, title='Baseline [ADC]', index_var=1, limit_min=1950., limit_max=2050., bin_width=10.)
display([adcs],geom,norm='log')
