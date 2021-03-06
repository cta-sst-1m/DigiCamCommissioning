#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import mpe_hist
from spectra_fit import fit_low_light,fit_high_light
from utils import display, histogram, geometry, pdf
from scipy.interpolate import LSQUnivariateSpline,UnivariateSpline
import logging,sys
import numpy as np
import logging
from tqdm import tqdm
from utils.logger import TqdmToLogger
import scipy
import pickle

from ctapipe import visualization

from numpy.linalg import inv

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

    return


def poly(p,x):
    # get the roots
    #p3 = np.poly1d([p[0], p[1], p[2], p[3]])
    #roots=np.roots(p3)
    #if np.max(roots[np.isreal(roots)])<1e-8: return 1e8
    return p[4]+p[3]*(x)+p[2]*(x)*(x)+p[1]*(x)*(x)*(x)+p[0]*(x)*(x)*(x)*(x)


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
    mpes = histogram.Histogram(filename=options.output_directory + options.mpe_histo_filename, fit_only= True)
    ac_led = histogram.Histogram(data= np.swapaxes(mpes.fit_result[...,0,0],0,1),bin_centers=np.array(options.scan_level))
    ac_led.errors = np.swapaxes(mpes.fit_result[...,0,1],0,1)
    ac_led.fit_result = np.ones((ac_led.data.shape[0],5,7),dtype = np.float)*np.nan
    chi2ndof = mpes.fit_chi2_ndof[..., 0] / mpes.fit_chi2_ndof[..., 1]
    chi2ndof = np.swapaxes(chi2ndof,0,1)
    spline_knot = []
    spline_coef = []
    dict_spline = {}
    for pixel in range(ac_led.data.shape[0]):
        y = ac_led.data[pixel,0:-1]
        y = y - y[0]
        yerr = ac_led.errors[ pixel,0:-1]
        chi2 = chi2ndof[pixel,0:-1]
        chi2[0]=0.
        ac_led.errors[pixel,0]=1e-6
        x = np.array(options.scan_level,dtype = np.float)[0:-1]
        index_keep =~np.isnan(y) * ~np.isnan(yerr)
        index_keep = index_keep * (chi2<50)
        index_keep_2 = index_keep
        index_keep_2 = index_keep_2 * (y>0.5)
        index_keep_2 = index_keep_2 * (np.append(np.diff(y),[1.])>0.)

        index_keep = index_keep * (y>1)

        if not np.all(index_keep==False):
            index_keep[0]=True
        if not np.all(index_keep_2==False):
            index_keep_2[0]=True
        y_2 = np.copy(y)
        yerr_2 = np.copy(yerr)
        x_2 = np.copy(x)
        y = y[index_keep]
        yerr = yerr[index_keep]
        x = x[index_keep]
        y_2 = y_2[index_keep_2]
        yerr_2 = yerr_2[index_keep_2]
        x_2 = x_2[index_keep_2]
        # skip points with chi2 too high
        if x_2.shape[0]>0:
            x_2 = np.delete(x_2,0)
            y_2 = np.delete(y_2,0)
            yerr_2 = np.delete(yerr_2,0)
        if x.shape[0]==0:continue
        deg = int(4)


        #ac_led.fit_result[pixel, :, 0]= np.polynomial.legendre.legfit(x, y, deg=deg, w=1. / yerr, full=True)[0]
        ac_led.fit_result[pixel, :, 0], ac_led.fit_result[pixel, :, 2:7:1] = np.polyfit(x, y, deg=deg, w=1. / yerr, cov=True)
        ac_led.fit_result[pixel, :, 1] = np.sqrt(np.diag(ac_led.fit_result[pixel, :, 2:7:1]))
        ## Now for pe below 10 do a spline fit

        x_line_high = np.linspace(x_2[-1],1000,1000-x_2[-1])
        y_line_high = np.polyval(ac_led.fit_result[pixel, :, 0], x_line_high)
        '''
        y_fit_max = np.polyval(ac_led.fit_result[pixel, :, 0]+ac_led.fit_result[pixel, :, 1], x_line_high)
        y_fit_min = np.polyval(ac_led.fit_result[pixel, :, 0]-ac_led.fit_result[pixel, :, 1], x_line_high)
        x_2 = np.append(x_2, x_line_high)
        y_2 = np.append(y_2, y_line_high)
        w_2 = np.append(yerr_2, (y_fit_max-y_fit_min)/2)
        '''
        spl = UnivariateSpline(x_2, y_2, w = 1./yerr_2, s=0, k=3)
        x_line = np.linspace(x_2[0],x_2[-1],x_2[-1]-x_2[0])
        x_start = np.linspace(0, x_line[0], x_line[0])

        dict_spline[options.pixel_list[pixel]] ={'DAC': np.append(x_start,np.append(x_line,x_line_high)) ,
                                                 'NPE': np.append(np.zeros(x_start.shape,dtype=float),np.append(spl(x_line),y_line_high)),
                                                 'LED': options.cts.pixel_to_led['AC'][options.pixel_list[pixel]]}

        knt = spl.get_knots()
        wei = spl.get_coeffs()
        spline_knot.append(knt)
        spline_coef.append(wei)

    pickle.dump(dict_spline, open(options.output_directory + 'ac_led_calib_spline_120.p', "wb"))
    ac_led.fit_result_label=['a0','a1','a2','a3','a4']
    ac_led.save(options.output_directory + options.histo_filename)
    np.savez_compressed(options.output_directory + options.histo_filename.split('.')[0]+'_spline.npz',
                        ac_spline_knot=spline_knot, ac_spline_coef=spline_coef)
    patch_ac_led = {}
    for patch in options.cts.camera.Patches:
        missing_pixel = False
        for p in patch.pixelsID:
            if not (p in dict_spline.keys()): missing_pixel = True
        if missing_pixel:
            continue
        patch_ac_led[patch.ID]={'DAC':dict_spline[patch.pixelsID[0]]['DAC'],
                                'NPE':dict_spline[patch.pixelsID[0]]['NPE']+dict_spline[patch.pixelsID[1]]['NPE']+dict_spline[patch.pixelsID[2]]['NPE'],
                                'LED': options.cts.patch_camera_to_patch_led[patch.ID]}

    pickle.dump(patch_ac_led , open(options.output_directory + 'ac_patch_calib_spline_120.p', "wb"))



def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    # Load the histogram
    ac_led = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    spline_knot = np.load(options.output_directory + options.histo_filename.split('.')[0]+'_spline.npz')['ac_spline_knot']
    spline_coef = np.load(options.output_directory + options.histo_filename.split('.')[0]+'_spline.npz')['ac_spline_coef']

    mpes = histogram.Histogram(filename=options.output_directory + options.mpe_histo_filename, fit_only=True)
    chi2ndof = mpes.fit_chi2_ndof[..., 0] / mpes.fit_chi2_ndof[..., 1]
    chi2ndof = np.swapaxes(chi2ndof,0,1)

    # Define Geometry
    geom, pixel_list = geometry.generate_geometry(options.cts)
    #geom = geometry.generate_geometry_0(pixel_list=options.pixel_list)
    pixelssss = []
    for i in [518,552,553,554,555,588,589,590,591,592,624,625,626,627,660,661,662,663,664,697,698]:
        pixelssss+=[options.pixel_list.index(i)]

    import matplotlib.pyplot as plt

    dict_spline = pickle.load( open( options.output_directory + 'ac_led_calib_spline_120.p' , "rb" ) )

    #h = input('see some pixels')
    for pixel in [250, 272, 273, 274, 275, 296, 297, 298, 299, 300, 320, 321, 322, 323, 344, 345, 346, 347, 348, 369, 370]:
        #if options.pixel_list[pixel]!=661:continue
        y = ac_led.data[pixel][0:-1]
        yerr = ac_led.errors[pixel][0:-1]
        yerr2 = ac_led.errors[pixel][0:-1]
        x = np.array(options.scan_level, dtype=np.float)[0:-1]
        x2 = np.array(options.scan_level, dtype=np.float)[0:-1]
        y2 = np.copy(y)
        y = y - y[0]
        index_keep = ~np.isnan(y) * ~np.isnan(yerr)

        chi2 = chi2ndof[pixel, 0:-1]
        #index_keep = index_keep * (chi2 < 10.)
        index_keep = index_keep
        y = y[index_keep]
        yerr = yerr[index_keep]
        x = x[index_keep]
        function = np.polyval#lambda p,xx : p[4]+p[3]*(xx)+p[2]*(xx)*(xx)+p[1]*(xx)*(xx)*(xx)+p[0]*(xx)*(xx)*(xx)*(xx)
        x_spline = dict_spline[pixel_list[pixel]]['DAC']
        y_spline = dict_spline[pixel_list[pixel]]['NPE']
        x_spline=x_spline[~np.isnan(y_spline)]
        y_spline=y_spline[~np.isnan(y_spline)]
        x1 = np.arange(x[0],options.scan_level[-1],1, dtype=np.float)
        if x.shape[0] == 0: return
        deg = int(4)
        param, covariance = ac_led.fit_result[pixel, :, 0], ac_led.fit_result[pixel, :, 2:7:1]

        param_err = ac_led.fit_result[pixel, :, 1]
        xx = np.vstack([x1 ** (deg - i) for i in range(deg + 1)]).T
        yi = np.dot(xx, param)
        C_yi = np.dot(xx, np.dot(covariance, xx.T))
        n_sigma = 5
        sig_yi = n_sigma * np.sqrt(np.diag(C_yi))

        y_fit = function(param,x1)

        y_fit_max = function(param + param_err, x1)
        y_fit_min = function(param - param_err, x1)

        ax = plt.subplot(1, 2, 1)
        ax.cla()
        plt.errorbar(x, y, yerr=yerr, fmt='ok',markersize='7')
        # ax.set_yscale('log')
        ax.set_ylabel('$N_{\gamma}$ [p.e.]')
        ax.set_xlabel('LED DAC')
        ax.set_yscale('log')
        ax.set_ylim(5e-1,1e5)
        ax.set_xlim(0.,1000.)
        #
        plt.plot(x1,y_fit,color='r')
        plt.fill_between(x1, y_fit_max, y_fit_min, alpha=0.5, facecolor='red', label='polyfit')
        #plt.fill_between(x1, yi + sig_yi, yi - sig_yi, alpha=0.5, facecolor='blue', label='polyfit confidence level')
        plt.plot(x_spline,y_spline, color = 'k', linestyle = '--')

        #plt.fill_between(x1, y_fit_max, y_fit_min, alpha=0.5, facecolor='blue', label='polyfit confidence level')
        plt.fill_between(x1, yi + sig_yi, yi - sig_yi, alpha=0.5, facecolor='red', label='%d $\sigma$ confidence level' %n_sigma)
        plt.legend(loc='best')
        plt.show()
"""
         # Use a poly1d to represent the polynomial.
        inv_p = lambda y : np.max( np.real((np.poly1d(param) - y).roots))
        inv_p_min = lambda y : np.max( np.real((np.poly1d(param-param_err) - y).roots))
        inv_p_max = lambda y : np.max( np.real((np.poly1d(param+param_err) - y).roots))
        ax0 = plt.subplot(1, 2, 2)

        ax0.cla()
        yy, yy_min,yy_max = [],[],[]
        for xx in np.arange(0,3000):
            yy.append(inv_p(xx))
            yy_min.append(inv_p_min(xx))
            yy_max.append(inv_p_max(xx))

        plt.fill_between(np.arange(0,3000), yy_min, yy_max, alpha=0.5, facecolor='red', label='polyfit confidence level')
        plt.plot(np.arange(0,3000),yy,color='red')

        ax0.set_xlabel('$N_{\gamma}$')
        ax0.set_ylabel('LED DAC %d'%options.pixel_list[pixel])
        ax0.set_xscale('log')
        ax0.set_xlim(10.,3000)
        '''
        ax1 = plt.subplot(1, 3, 3)
        ax1.cla()
        ax1.set_ylabel('$N_{\gamma}$, $68\%$ C.L. (relative)')
        ax1.set_xlabel('LED DAC')
        # plt.plot(x1, (y_fit_max - y_fit_min) / yi, label='polyfit + 1 $\sigma$')
        # plt.plot(x1, y_fit_min, label='polyfit - 1 $\sigma$')
        #plt.fill_between(x1, yi + sig_yi, yi - sig_yi, alpha=0.5, facecolor='red', label='polyfit confidence level')
        plt.fill_between(x1, 2 * np.max(np.append(y_fit_max / yi,y_fit_min / yi),axis=-1), alpha=0.5, facecolor='red', label='polyfit confidence level')
        '''
        print('param ', param, ' ± ', param_err)

        plt.show()
        input('bla')

    return
"""