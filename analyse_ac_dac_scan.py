#!/usr/bin/env python3
from cts import cameratestsetup as cts
from utils.geometry import generate_geometry,generate_geometry_0
from utils.plots import pickable_visu_mpe,pickable_visu_led_mu
from utils.pdf import mpe_distribution_general,mpe_distribution_general_sh
from optparse import OptionParser
from utils.histogram import histogram
import peakutils
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from data_treatement import mpe_hist,synch_hist
from utils.plots import pickable_visu
from spectra_fit import fit_low_light,fit_full_mpe
from utils.plots import display, display_var

parser = OptionParser()

# Job configuration
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")

parser.add_option("-p", "--perform_fit_mu", dest="perform_fit_mu", action="store_true",
                  help="perform fit of mpe", default=False)

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
                  help="Histogram SPE file name", default="mpe_scan_0_195_5_200_600_10_max.npz")

parser.add_option("--output_directory", dest="output_directory",
                  help="directory of histo file", default='/data/datasets/CTA/DarkRun/20161214/')

parser.add_option("--fit_filename", dest="fit_filename",
                  help="name of fit file with MPE", default='mpe_scan_200_600_10_fit_max.npz')


# Arange the options
(options, args) = parser.parse_args()
options.file_list = options.file_list.split(',')
options.scan_level = [int(level) for level in options.scan_level.split(',')]
if len(options.scan_level)==3:
    options.scan_level= np.arange(options.scan_level[0],options.scan_level[1]+options.scan_level[2],options.scan_level[2])
else:
    options.scan_level = np.array(options.scan_level)

# Prepare the mpe histograms
mpes = histogram(bin_center_min=1950., bin_center_max=4095., bin_width=1.,
                       data_shape=(options.scan_level.shape+(1296,)),
                 xlabel='Peak ADC',ylabel='$\mathrm{N_{entries}}$',label='MPE')


if options.verbose: print('--|> Recover data from %s' % (options.output_directory+options.histo_filename))
file = np.load(options.output_directory+options.histo_filename)
mpes = histogram(data=np.copy(file['mpes']),bin_centers=np.copy(file['mpes_bin_centers']),xlabel = 'Peak ADC',
                 ylabel='$\mathrm{N_{trigger}}$',label='MPE from peak value')
file.close()


if options.verbose:
    print('--|> Recover fit results for G and sigmas from %s' % (options.output_directory + 'full_'+ options.fit_filename))
file = np.load(options.output_directory+ 'full_' + options.fit_filename)

mpes_full_fit_result = np.copy(file['full_mpe_fit_result'])

mpes_full_fit_result=mpes_full_fit_result.reshape((1,)+mpes_full_fit_result.shape)
mpes_full_fit_result = np.repeat(mpes_full_fit_result,mpes.data.shape[0],axis=0)

#del(mpes_full)
#mpes_full.fit_function = fit_full_mpe.fit_func
file.close()


## Now perform the mu and mu_XT fits
if options.perform_fit_mu:
    fixed_param = [
        [2,(1,0)], # gain
        [3,(0,0)], # baseline
        [4,(2,0)], # sigma_e
        [5,(3,0)], # sigma_1
        [7, 0.]  # offset
    ]
    mpes.fit(fit_low_light.fit_func, fit_low_light.p0_func, fit_low_light.slice_func,
             fit_low_light.bounds_func, config=mpes_full_fit_result,fixed_param=fixed_param)#,limited_indices=[(30, 387)])#(i,4,) for i in range(20)])
    np.savez_compressed(options.output_directory + options.fit_filename,mpes_fit_result=mpes.fit_result)

if options.verbose:
    print('--|> Recover fit results for mu and mu_XT from %s' % (options.output_directory + options.fit_filename))
file = np.load(options.output_directory + options.fit_filename)
mpes.fit_result = np.copy(file['mpes_fit_result'])
mpes.fit_function = fit_low_light.fit_func
file.close()


5./0
# Leave the hand
plt.ion()

# Define Geometry
geom= generate_geometry_0()

display_var(mpes_full, geom, title='$\sigma_e$ [ADC]', index_var=2, limit_min=0.8, limit_max=1.2, bin_width=0.05)
display_var(mpes_full, geom, title='$\sigma_i$ [ADC]', index_var=3, limit_min=0.4, limit_max=0.5, bin_width=0.002)
display_var(mpes_full, geom, title='Gain [ADC/p.e.]' , index_var=1, limit_min=5.1, limit_max=6., bin_width=0.04)

display([mpes_full], geom, fit_full_mpe.slice_func, norm='log',pix_init=485, config=prev_fit_result)


def show_level(level,hist):
    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    plt.subplot(1, 2, 1)
    vis_baseline = pickable_visu_mpe([hist], ax[1], fig, fit_low_light.slice_func, level,False, geom, title='', norm='lin',
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
    vis_baseline.on_pixel_clicked(486)
    plt.show()



#show_level(10,mpes)
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
#display([peaks])

# Fit them

if options.perform_fit:
    # recover previous fit
    if options.verbose: print(
        '--|> Recover fit results from %s' % (options.dark_calibration_directory + options.saved_spe_fit_filename))
    file = np.load(options.dark_calibration_directory + options.saved_spe_fit_filename)
    spes_fit_result = np.copy(file['spes_fit_results'])
    if options.verbose: print(
        '--|> Recover fit results from %s' % (options.dark_calibration_directory + options.saved_adc_fit_filename))
    file = np.load(options.dark_calibration_directory + options.saved_adc_fit_filename)
    adcs_fit_result = np.copy(file['adcs_fit_results'])
    if options.verbose:
        print('--|> Recover data from %s' % (options.dark_calibration_directory + options.saved_adc_histo_filename))
    file = np.load(options.dark_calibration_directory + options.saved_adc_histo_filename)
    adcs = histogram(data=np.copy(file['adcs']), bin_centers=np.copy(file['adcs_bin_centers']))


    prev_fit = np.append( adcs_fit_result.reshape((1,) + adcs_fit_result.shape),
                          spes_fit_result.reshape((1,) + spes_fit_result.shape),axis=2)
    # reodred (this will disapear once dark fit is implemented properly)
    # amp0,baseline,sigma_e, sigma_e,sigma_i,gain,amp1,amp2,amp3,baseline,offset,amp4,amp5
    prev_fit[...,[0,1,2,3,4,5,6,7,8,9,10,11,12],:] = prev_fit[...,[12,11,5,1,2,4,0,10,3,6,7,8,9],:]
    prev_fit = np.delete(prev_fit,[8,9,10,11,12],axis=2)
    # fix the cross talk for now...
    prev_fit[...,1,:]=[0.08,10.]
    print(prev_fit.shape)
    #print(options.scan_level)

    #intialise the fit result
    tmp_shape = prev_fit.shape
    tmp_shape=mpes_peaks.data.shape[:1]+tmp_shape[1:]
    print(tmp_shape)
    mpes_peaks.fit_result = np.ones(tmp_shape)*np.nan

    for level in range(len(options.scan_level)):
        if options.verbose: print("################################# Level",level)
        if level == 0:
            ## Take from spe fit
            for pix in range(mpes_peaks.data[level].shape[0]):
                if np.any(np.isnan(mpes_peaks.data[level, pix])):
                    if options.verbose: print('----> Pix',pix,'abort')
                    continue
                elif abs(np.nanmean(mpes_peaks.data[level, pix]) - np.nanmean(adcs.data[pix])) < 0.1:
                    if options.verbose: print('----> Pix',pix,'as dark')
                    ## to be replaced by fit_dark
                    mpes_peaks.fit_result[level,pix] = [[2., 1.e3], [prev_fit[0, pix, 1]], [prev_fit[0, pix, 2]], [prev_fit[0, pix, 3]],
                                  [prev_fit[0, pix, 4]], [0.7, 10.], [1000, 1.e8],[prev_fit[0, pix, 7]]]
                else:
                    if options.verbose: print('----> Pix',pix,'low light')
                    mpes_peaks.fit_result[level, pix] = \
                        mpes_peaks._axis_fit((level,pix,),
                                                    fit_low_light.fit_func,
                                                    fit_low_light.p0_func(mpes_peaks.data[level,pix],
                                                                          mpes_peaks.bin_centers,
                                                                          config=prev_fit[0,pix]),
                                                    slice=fit_low_light.slice_func(mpes_peaks.data[level,pix],
                                                                                   mpes_peaks.bin_centers,
                                                                                   config=prev_fit[0,pix]),
                                                    bounds=fit_low_light.bounds_func(mpes_peaks.data[level,pix],
                                                                                     mpes_peaks.bin_centers,
                                                                                     config=prev_fit[0,pix]),
                                                    # mu_xt,baseline,sigma_e,offset
                                                    fixed_param = np.array([[i for i in [1,3,4,7]],[prev_fit[0, pix, i , 0] for i in [1,3,4,7]]])
                                                    )
        else:
            ## Take from previous
            for pix in range(mpes_peaks.data[level].shape[0]):
                if np.any(np.isnan(mpes_peaks.data[level, pix])):
                    if options.verbose: print('----> Pix',pix,'abort')
                    continue

                elif abs(np.nanmean(mpes_peaks.data[level, pix]) - np.nanmean(adcs.data[pix])) < 0.1:
                    if options.verbose: print('----> Pix',pix,'as dark')
                    ## to be replaced by fit_dark
                    mpes_peaks.fit_result[level,pix] = [[2., 1.e3], [prev_fit[0, pix, 1]], [prev_fit[0, pix, 2]], [prev_fit[0, pix, 3]],
                                  [prev_fit[0, pix, 4]], [0.7, 10.], [1000, 1.e8],[prev_fit[0, pix, 7]]]

                elif mpes_peaks.fit_result[level-1,pix,0,0]<10.:
                    if options.verbose: print('----> Pix',pix,'low light')
                    mpes_peaks.fit_result[level, pix] = \
                        mpes_peaks._axis_fit((level,pix,),
                                                    fit_low_light.fit_func,
                                                    fit_low_light.p0_func(mpes_peaks.data[level, pix],
                                                                          mpes_peaks.bin_centers,
                                                                          config=mpes_peaks.fit_result[level-1, pix]),
                                                    slice=fit_low_light.slice_func(mpes_peaks.data[level, pix],
                                                                                   mpes_peaks.bin_centers,
                                                                                   config=mpes_peaks.fit_result[level-1, pix]),
                                                    bounds=fit_low_light.bounds_func(mpes_peaks.data[level, pix],
                                                                                     mpes_peaks.bin_centers,
                                                                                     config=mpes_peaks.fit_result[level-1, pix]),
                                                    # mu_xt,baseline,sigma_e,offset
                                                    fixed_param = np.array([[i for i in [1,3,4,7]],[mpes_peaks.fit_result[level-1, pix, i , 0] for i in [1,3,4,7]]])
                                                    )
                else:
                    if options.verbose: print('----> Pix',pix,'high light')
                    mpes_peaks.fit_result[level, pix] = \
                        mpes_peaks._axis_fit((level,pix,),
                                                    fit_high_light.fit_func,
                                                    fit_high_light.p0_func(mpes_peaks.data[level, pix],
                                                                          mpes_peaks.bin_centers,
                                                                          config=mpes_peaks.fit_result[level - 1, pix]),
                                                    slice=fit_high_light.slice_func(mpes_peaks.data[level, pix],
                                                                                   mpes_peaks.bin_centers,
                                                                                   config=mpes_peaks.fit_result[
                                                                                       level - 1, pix]),
                                                    bounds=fit_high_light.bounds_func(mpes_peaks.data[level, pix],
                                                                                     mpes_peaks.bin_centers,
                                                                                     config=mpes_peaks.fit_result[
                                                                                         level - 1, pix]),
                                                    # fix all but mu and amplitude
                                                    fixed_param=np.array([[i for i in [1,2,3,4,5,7]],[mpes_peaks.fit_result[level-1, pix, i , 0] for i in [1,2,3,4,5,7]]])
                                                    )


    # Save the parameters
    if options.verbose: print('--|> Save the fit result in %s' % (options.saved_histo_directory + options.saved_fit_filename))
    np.savez_compressed(options.saved_histo_directory + options.saved_fit_filename, mpes_fit_results=mpes_peaks.fit_result)
else :
    if options.verbose: print('--|> Load the fit result from %s' % (options.saved_histo_directory + options.saved_fit_filename))
    h = np.load(options.saved_histo_directory + options.saved_fit_filename)
    mpes_peaks.fit_result = h['mpes_fit_results']
    mpes_peaks.fit_function = mpe_distribution_general


# Plot them
def slice_fun(x, **kwargs):
    if np.where(x != 0)[0][0]== np.where(x != 0)[0][-1]:return [0,1,1]
    return [np.where(x != 0)[0][0], np.where(x != 0)[0][-1], 1]


def show_level(level,hist):
    fig, ax = plt.subplots(1, 2, figsize=(30, 10))
    plt.subplot(1, 2, 1)
    vis_baseline = pickable_visu_mpe([hist], ax[1], fig, slice_fun, level,True, geom, title='', norm='lin',
                                     cmap='viridis', allow_pick=True)
    vis_baseline.add_colorbar()
    vis_baseline.colorbar.set_label('Peak position [4ns]')
    plt.subplot(1, 2, 1)
    val = hist.fit_result[3,:,2,0]
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



show_level(0,mpes_peaks)



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
    vis_baseline.on_pixel_clicked(700)
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
    #plt.axes().xaxis.get_label().set_ha('right')
    #plt.axes().xaxis.get_label().set_position((1, 0))
    #plt.axes().yaxis.get_label().set_ha('right')
    #plt.axes().yaxis.get_label().set_position((0, 1))
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
    #plt.axes().xaxis.get_label().set_ha('right')
    #plt.axes().xaxis.get_label().set_position((1, 0))
    #plt.axes().yaxis.get_label().set_ha('right')
    #plt.axes().yaxis.get_label().set_position((0, 1))
    plt.subplot(1,2,2)
    xaxis = np.repeat(np.arange(50, 260, 10).reshape((1,)+np.arange(50, 260, 10).shape),param.shape[1],axis=0).reshape(np.prod([param_ratio.shape]))
    y_axis = param_ratio.reshape(np.prod([param_ratio.shape]))
    plt.hist2d(xaxis,y_axis,bins=20,range=[[50, 250], range],norm=LogNorm())
    plt.colorbar()
    plt.show()

display_fitparam(mpes_peaks,2,700,'Gain',[0.9,1.1]) #<N(p.e.)>@DAC=x
'''