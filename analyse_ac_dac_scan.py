#!/usr/bin/env python3

from kapteyn import kmpfit
from cts import cameratestsetup as cts
from utils.geometry import generate_geometry,generate_geometry_0
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
from spectra_fit import fit_low_light,fit_high_light,fit_full_mpe
from utils.plots import display, display_var

parser = OptionParser()

# Job configuration
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

parser.add_option("-p", "--perform_fit_mu", dest="perform_fit_mu", action="store_true",
                  help="perform fit of mpe", default=False)

parser.add_option("-a", "--perform_fit_acled", dest="perform_fit_acled", action="store_true",
                  help="perform fit of led parameter", default=False)

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

parser.add_option("-n", "--n_evt_per_batch", dest="n_evt_per_batch",
                  help="number of events per batch", default=1000, type=int)

# File management
parser.add_option("--file_basename", dest="file_basename",
                  help="file base name ", default="CameraDigicam@localhost.localdomain_0_000.%s.fits.fz")

parser.add_option("-d", "--directory", dest="directory",
                  help="input directory", default="/data/datasets/CTA/DATA/20161214/")

parser.add_option("--histo_filename", dest="histo_filename",
                  help="Histogram SPE file name", default="mpe_scan_0_195_5_200_600_10.npz")

parser.add_option("--output_directory", dest="output_directory",
                  help="directory of histo file", default='/data/datasets/CTA/DarkRun/20161214/')

parser.add_option("--fit_filename", dest="fit_filename",
                  help="name of fit file with MPE", default='mpe_scan_0_195_5_200_600_10_fit.npz')

parser.add_option("--led_fit_filename", dest="led_fit_filename",
                  help="name of fit file with LED parameters", default='led_0_195_5_200_600_10_fit.npz')


# Arange the options
(options, args) = parser.parse_args()
options.scan_level = [int(level) for level in options.scan_level.split(',')]
if len(options.scan_level)==3:
    options.scan_level= np.arange(options.scan_level[0],options.scan_level[1]+options.scan_level[2],options.scan_level[2])
else:
    options.scan_level = np.array(options.scan_level)

# Prepare the mpe histograms
mpes = Histogram(bin_center_min=1950., bin_center_max=4095., bin_width=1.,
                 data_shape=(options.scan_level.shape+(1296,)),
                 xlabel='Peak ADC', ylabel='$\mathrm{N_{entries}}$', label='MPE')


if options.verbose: print('--|> Recover data from %s' % (options.output_directory+options.histo_filename))
file = np.load(options.output_directory+options.histo_filename)
mpes = Histogram(data=np.copy(file['mpes']), bin_centers=np.copy(file['mpes_bin_centers']), xlabel ='Peak ADC',
                 ylabel='$\mathrm{N_{trigger}}$', label='MPE from peak value')
file.close()


if options.verbose:
    print('--|> Recover fit results for G and sigmas from %s' % (options.output_directory + 'full_'+ options.fit_filename))
file = np.load(options.output_directory+ 'full_' + options.fit_filename)

mpes_full_fit_result = np.copy(file['full_mpe_fit_result'])

mpes_full_fit_result=mpes_full_fit_result.reshape((1,)+mpes_full_fit_result.shape)
mpes_full_fit_result = np.repeat(mpes_full_fit_result,mpes.data.shape[0],axis=0)

## TODO perform the fit of concatenated with baseline only (need database)

#del(mpes_full)
#mpes_full.fit_function = fit_full_mpe.fit_func
file.close()

def std_dev(x,y):
    avg = np.average(x, weights=y)
    return np.sqrt(np.average((x - avg ) ** 2,weights=y))


## Now perform the mu and mu_XT fits
if options.perform_fit_mu:
    for pixel in range(mpes.data.shape[1]):
        force_xt = False
        if pixel>0: print('\nTreated Pixel #'+str(pixel-1))
        for level in range(mpes.data.shape[0]):
            print("Treating Pixel #"+str(pixel)+": Fit Progress {:2.1%}".format(float(level) / mpes.data.shape[0]), end="\r")

            if np.where(mpes.data[level,pixel] != 0)[0][0]==0 and np.where(mpes.data[level,pixel] != 0)[0].shape==(1,):continue
            if std_dev(mpes.bin_centers,mpes.data[level,pixel])>400: continue
            if mpes.data[level,pixel,-1]>0.02*np.sum(mpes.data[level,pixel]):continue

            # check if the mu of the previous level is above 5
            fixed_param = []
            _fit_spectra = fit_low_light
            if level> 0 and mpes.fit_result[level-1,pixel,0,0]>30.:
                fixed_param = [
                    # in this case assign the cross talk estimation with smallest error
                    [1,mpes.fit_result[np.argmin(mpes.fit_result[0:level - 1:1, pixel, 1, 1]),pixel,1,0]],
                    [2,(1,0)], # gain
                    [3,(0,0)], # baseline
                    #[4,(2,0)], # sigma_e
                    [5,(3,0)], # sigma_1
                    [7, 0.]  # offset
                ]
                _fit_spectra = fit_high_light
            elif (level> 0 and mpes.fit_result[level-1,pixel,0,0]>5.) or force_xt:
                force_xt = True
                fixed_param = [
                    # in this case assign the cross talk estimation with smallest error
                    [1,mpes.fit_result[np.argmin(mpes.fit_result[0:level - 1:1, pixel, 1, 1]),pixel,1,0]],
                    [2,(1,0)], # gain
                    [3,(0,0)], # baseline
                    [4,(2,0)], # sigma_e
                    [5,(3,0)], # sigma_1
                    [7, 0.]  # offset
                ]
            else:
                fixed_param = [
                    [2,(1,0)], # gain
                    [3,(0,0)], # baseline
                    [4,(2,0)], # sigma_e
                    [5,(3,0)], # sigma_1
                    [7, 0.]  # offset
                ]
            mpes.fit(_fit_spectra.fit_func, _fit_spectra.p0_func, _fit_spectra.slice_func,
                     _fit_spectra.bounds_func, config=mpes_full_fit_result,fixed_param=fixed_param
                     ,limited_indices=[(level,pixel,)],force_quiet = True)

    np.savez_compressed(options.output_directory + options.fit_filename,mpes_fit_result=mpes.fit_result)

if options.verbose:
    print('--|> Recover fit results for mu and mu_XT from %s' % (options.output_directory + options.fit_filename))
file = np.load(options.output_directory + options.fit_filename)
mpes.fit_result = np.copy(file['mpes_fit_result'])
mpes.fit_function = fit_low_light.fit_func
file.close()



## Now perform the AC LED
# Leave the hand
if options.perform_fit_acled:
    print('--|> Perform the fit for the AC LEDs')
    parameters = np.ones((mpes.data.shape[1],5,7),dtype = np.float)*np.nan

    for pixel in range(mpes.data.shape[1]):
        slicemax = 40
        y = mpes.fit_result[0:slicemax:1, pixel, 0, 0]
        yerr = mpes.fit_result[0:slicemax:1, pixel, 0, 1]
        x = np.array(options.scan_level,dtype = np.float)
        x1 = np.array(options.scan_level,dtype = np.float)
        x = x[0:slicemax:1]
        index_keep =~np.isnan(y) * ~np.isnan(yerr)
        y = y[index_keep]
        yerr = yerr[index_keep]
        x = x[index_keep]
        if x.shape[0]==0:continue
        deg = int(4)
        parameters[pixel, :, 0], parameters[pixel, :, 2:7:1] = np.polyfit(x, y, deg=deg, w=1. / yerr, cov=True)
        parameters[pixel, :, 1] = np.sqrt(np.diag(parameters[pixel, :, 2:7:1]))

    print('--|> Save fit results for LEDs in %s' % (options.output_directory + options.led_fit_filename))
    np.savez_compressed(options.output_directory + options.led_fit_filename,parameters=parameters)

# Leave the hand
plt.ion()

if options.verbose:
    print('--|> Recover fit results for LEDs from %s' % (options.output_directory + options.led_fit_filename))
file = np.load(options.output_directory + options.led_fit_filename)
led_parameters = np.copy(file['parameters'])
file.close()



# Define Geometry
geom= generate_geometry_0()

def show_level(level,hist):
    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    plt.subplot(1, 2, 1)
    vis_baseline = pickable_visu_mpe([hist], ax[1], fig, fit_low_light.slice_func, level,True, geom, title='', norm='lin',
                                     cmap='viridis', allow_pick=True)
    vis_baseline.add_colorbar()
    vis_baseline.colorbar.set_label('Peak position [4ns]')
    plt.subplot(1, 2, 1)
    val = np.mean(hist.data[0],axis=1)
    val[np.isnan(val)]=0
    val[val<1.]=1.
    val[val>10.]=10.
    vis_baseline.axes.xaxis.get_label().set_ha('right')
    vis_baseline.axes.xaxis.get_label().set_position((1, 0))
    vis_baseline.axes.yaxis.get_label().set_ha('right')
    vis_baseline.axes.yaxis.get_label().set_position((0, 1))
    vis_baseline.image = val
    fig.canvas.mpl_connect('pick_event', vis_baseline._on_pick)
    vis_baseline.on_pixel_clicked(700)
    plt.show()

def show_mu(level,hist):
    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    plt.subplot(1, 2, 1)
    vis_baseline = pickable_visu_led_mu([hist], ax[1], fig, slice_fun, level,True, geom, title='', norm='lin',
                                     cmap='viridis', allow_pick=True)
    vis_baseline.add_colorbar()
    vis_baseline.colorbar.set_label('Peak position [4ns]')
    plt.subplot(1, 2, 1)
    val = hist.fit_result[level,:,2,0]
    val[np.isnan(val)]=0
    val[val<1.]=1.
    val[val>10.]=10.
    vis_baseline.axes.xaxis.get_label().set_ha('right')
    vis_baseline.axes.xaxis.get_label().set_position((1, 0))
    vis_baseline.axes.yaxis.get_label().set_ha('right')
    vis_baseline.axes.yaxis.get_label().set_position((0, 1))
    vis_baseline.image = val
    fig.canvas.mpl_connect('pick_event', vis_baseline._on_pick)
    vis_baseline.on_pixel_clicked(10)
    plt.show()


from matplotlib.colors import LogNorm

def display_fitparam(hist,param_ind,pix,param_label,range=[0.9,1.1]):
    fig, ax = plt.subplots(1,2)
    param = hist.fit_result[:, :, param_ind, 0]
    param_err = hist.fit_result[:, :, param_ind, 1]
    param_ratio = np.divide(param, param_err[0])
    param_ratio[np.isnan(param_ratio)]=0.
    print(param_ratio.shape)
    plt.subplot(1,2,1)
    plt.errorbar(np.arange(50, 260, 10), param_ratio[:,pix], yerr=param_err[:,pix], fmt='ok')
    plt.ylim(range)
    plt.ylabel(param_label)
    plt.xlabel('AC LED DAC')
    plt.subplot(1,2,2)
    xaxis = np.repeat(np.arange(50, 260, 10).reshape((1,)+np.arange(50, 260, 10).shape),param.shape[1],axis=0).reshape(np.prod([param_ratio.shape]))
    y_axis = param_ratio.reshape(np.prod([param_ratio.shape]))
    plt.hist2d(xaxis,y_axis,bins=20,range=[[50, 250], range],norm=LogNorm())
    plt.colorbar()
    plt.show()

def display_fitparam_err(hist,param_ind,pix,param_label,range=[0.9,1.1]):
    fig, ax = plt.subplots(1,2)
    param = hist.fit_result[:, :, param_ind, 0]
    param_err = hist.fit_result[:, :, param_ind, 1]
    param_ratio = np.divide(param, param[np.nanargmin(param_err,axis=0)])
    param_ratio[np.isnan(param_ratio)]=0.
    print(param_ratio.shape)
    plt.subplot(1,2,1)
    plt.plot(np.arange(50, 260, 10), param_err[:,pix], color='k')
    plt.ylim(range)
    plt.ylabel(param_label)
    plt.xlabel('AC LED DAC')
    plt.subplot(1,2,2)
    xaxis = np.repeat(np.arange(50, 260, 10).reshape((1,)+np.arange(50, 260, 10).shape),param.shape[1],axis=0).reshape(np.prod([param_ratio.shape]))
    y_axis = param_ratio.reshape(np.prod([param_ratio.shape]))
    plt.hist2d(xaxis,y_axis,bins=20,range=[[50, 250], range],norm=LogNorm())
    plt.colorbar()
    plt.show()



def display_led_fit(pixel):
    slicemax = 80
    y = mpes.fit_result[0:slicemax:1, pixel, 0, 0]
    yerr = mpes.fit_result[0:slicemax:1, pixel, 0, 1]
    x = np.array(options.scan_level, dtype=np.float)
    x1 = np.array(options.scan_level, dtype=np.float)
    x = x[0:slicemax:1]
    index_keep = ~np.isnan(y) * ~np.isnan(yerr)
    y = y[index_keep]
    yerr = yerr[index_keep]
    x = x[index_keep]
    if x.shape[0] == 0: return
    deg = int(4)
    param , covariance = led_parameters[pixel, :, 0], led_parameters[pixel, :, 2:7:1]
    param_err = led_parameters[pixel, :, 1]
    xx = np.vstack([x1 ** (deg - i) for i in range(deg + 1)]).T
    yi = np.dot(xx, param)
    C_yi = np.dot(xx, np.dot(covariance, xx.T))
    sig_yi = np.sqrt(np.diag(C_yi))
    y_fit = np.polyval(param, x1)
    y_fit_max = np.polyval(param + param_err, x1)
    y_fit_min = np.polyval(param - param_err, x1)
    ax = plt.subplot(2, 1, 1)
    ax.cla()
    plt.errorbar(x, y, yerr=yerr, fmt='ok')
    ax.set_yscale('log')
    #plt.fill_between(x1, y_fit_max, y_fit_min, alpha=0.5, facecolor='blue', label='polyfit confidence level')
    plt.fill_between(x1, yi + sig_yi, yi - sig_yi, alpha=0.5, facecolor='red', label='polyfit confidence level')
    ax1 = plt.subplot(2, 1, 2)
    ax1.cla()
    ax1.set_yscale('log')
    #plt.plot(x1, (y_fit_max - y_fit_min) / yi, label='polyfit + 1 $\sigma$')
    # plt.plot(x1, y_fit_min, label='polyfit - 1 $\sigma$')
    # plt.fill_between(x1, yi + sig_yi, yi - sig_yi, alpha=0.5, facecolor='red', label='polyfit confidence level')
    plt.fill_between(x1, 2 * sig_yi / yi, alpha=0.5, facecolor='red', label='polyfit confidence level')
    print('param ', param, ' ± ', param_err)
    plt.show()
    input('bla')


show_level(44,mpes)


plt.subplots(2, 1)
for i in range(600,701):
    display_led_fit(i)

'''
        xx = np.vstack([x1 ** (deg - i) for i in range(deg + 1)]).T
        yi = np.dot(xx, param)
        C_yi = np.dot(xx, np.dot(covariance, xx.T))
        sig_yi = np.sqrt(np.diag(C_yi))
        y_fit = np.polyval(param, x1)
        y_fit_max = np.polyval(param + param_err, x1)
        y_fit_min = np.polyval(param - param_err, x1)

        ax = plt.subplot(2,1,1)
        ax.cla()
        plt.errorbar(x, y, yerr=yerr, fmt='ok')
        ax.set_yscale('log')
        plt.fill_between(x1, y_fit_max, y_fit_min, alpha=0.5, facecolor='blue', label='polyfit confidence level')
        plt.fill_between(x1, yi + sig_yi, yi - sig_yi, alpha=0.5, facecolor='red', label='polyfit confidence level')
        ax1 = plt.subplot(2,1,2)
        ax1.cla()
        plt.plot(x1, (y_fit_max-y_fit_min)/yi, label='polyfit + 1 $\sigma$')
        #plt.plot(x1, y_fit_min, label='polyfit - 1 $\sigma$')
        #plt.fill_between(x1, yi + sig_yi, yi - sig_yi, alpha=0.5, facecolor='red', label='polyfit confidence level')
        plt.fill_between(x1,2* sig_yi/yi, alpha=0.5, facecolor='red', label='polyfit confidence level')
        print('param ', param, ' ± ', param_err)
        plt.show()
        input('bla')
'''

v = input('preskey')