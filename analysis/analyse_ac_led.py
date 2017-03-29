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
import scipy

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
    for pixel in range(ac_led.data.shape[0]):
        y = ac_led.data[pixel,1:-1]
        yerr = ac_led.errors[ pixel,1:-1]
        chi2 = chi2ndof[pixel,1:-1]
        x = np.array(options.scan_level,dtype = np.float)[1:-1]
        index_keep =~np.isnan(y) * ~np.isnan(yerr)
        index_keep = index_keep * (chi2<50)
        index_keep = index_keep * (y>1.)
        y = y[index_keep]
        yerr = yerr[index_keep]
        x = x[index_keep]
        # skip points with chi2 too high

        if x.shape[0]==0:continue
        deg = int(4)

        '''
        residual = lambda p, x, y, y_err: (y - (p[4]+p[3]*(x)+p[2]*(x)*(x)+p[1]*(x)*(x)*(x)+p[0]*(x)*(x)*(x)*(x))) / y_err
        out = scipy.optimize.least_squares(residual, [1.,1.,1.,1.,1.], args=(x,y,yerr),bounds=([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf,np.inf,np.inf]))
        weight_matrix = np.diag(1. / yerr)
        cov = inv(np.dot(np.dot(out.jac.T,weight_matrix), out.jac))
        '''
        ac_led.fit_result[pixel, :, 0], ac_led.fit_result[pixel, :, 2:7:1] = np.polyfit(x, y, deg=deg, w=1. / yerr, cov=True)
        ac_led.fit_result[pixel, :, 1] = np.sqrt(np.diag(ac_led.fit_result[pixel, :, 2:7:1]))
    ac_led.fit_result_label=['a0','a1','a2','a3','a4']
    ac_led.save(options.output_directory + options.histo_filename)



def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """

    # Load the histogram
    ac_led = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    mpes = histogram.Histogram(filename=options.output_directory + options.mpe_histo_filename, fit_only=True)
    chi2ndof = mpes.fit_chi2_ndof[..., 0] / mpes.fit_chi2_ndof[..., 1]
    chi2ndof = np.swapaxes(chi2ndof,0,1)

    # Define Geometry
    geom = geometry.generate_geometry_0(pixel_list=options.pixel_list)


    import matplotlib.pyplot as plt
    '''
    fig,ax=plt.subplots(2,3)
    camera_visu = visualization.CameraDisplay(geom, ax=ax[0,0], title='', norm='lin', cmap='viridis',
                                              allow_pick=False)

    #camera_visu.add_colorbar()
    image = np.log(np.abs(ac_led.fit_result[:, 0, 0]))*np.abs(ac_led.fit_result[:, 0, 0])/(ac_led.fit_result[:, 0, 0])
    print(image[0])
    image[np.isnan(image)]=0
    image[image>-15]=-15
    image[image<-17]=-17


    camera_visu.image = image

    camera_visu2 = visualization.CameraDisplay(geom, ax=ax[0 ,1], title='', norm='lin', cmap='viridis',
                                              allow_pick=False)
    image2 = np.log(np.abs(ac_led.fit_result[:, 1, 0]))*np.abs(ac_led.fit_result[:, 1, 0])/(ac_led.fit_result[:, 1, 0])
    print(image2[0])
    image2[np.isnan(image2)]=0
    image2[image2>12]=12
    image2[image2<7]=7

    camera_visu2.image = image2
    camera_visu3 = visualization.CameraDisplay(geom, ax=ax[0 ,2], title='', norm='lin', cmap='viridis',
                                              allow_pick=False)
    image3 = np.log(np.abs(ac_led.fit_result[:, 2, 0]))*np.abs(ac_led.fit_result[:, 2, 0])/(ac_led.fit_result[:, 2, 0])
    print(image3[0])
    image3[np.isnan(image3)]=0
    image3[image3>0]=0
    image3[image3<-10]=-10

    camera_visu3.image = image3
    plt.subplot(2,3,4)
    plt.errorbar(np.arange(528),ac_led.fit_result[:, 3, 0],yerr=ac_led.fit_result[:, 3, 1],fmt='ok')
    plt.subplot(2,3,5)
    plt.errorbar(np.arange(528),ac_led.fit_result[:, 4, 0],yerr=ac_led.fit_result[:, 4, 1],fmt='ok')

    plt.show()
    '''
    h = input('see some pixels')
    for pixel in range(528):
        y = ac_led.data[pixel][1:-1]
        yerr = ac_led.errors[pixel][1:-1]
        x = np.array(options.scan_level, dtype=np.float)[1:-1]
        index_keep = ~np.isnan(y) * ~np.isnan(yerr)

        chi2 = chi2ndof[pixel, 1:-1]
        index_keep = index_keep * (chi2 < 50.)
        index_keep = index_keep * (y>1.)
        y = y[index_keep]
        yerr = yerr[index_keep]
        x = x[index_keep]
        function = np.polyval #lambda p,xx : p[4]+p[3]*(xx)+p[2]*(xx)*(xx)+p[1]*(xx)*(xx)*(xx)+p[0]*(xx)*(xx)*(xx)*(xx)

        x1 = np.arange(x[0],options.scan_level[-1],1, dtype=np.float)
        if x.shape[0] == 0: return
        deg = int(4)
        param, covariance = ac_led.fit_result[pixel, :, 0], ac_led.fit_result[pixel, :, 2:7:1]
        param_err = ac_led.fit_result[pixel, :, 1]
        xx = np.vstack([x1 ** (deg - i) for i in range(deg + 1)]).T
        yi = np.dot(xx, param)
        C_yi = np.dot(xx, np.dot(covariance, xx.T))
        sig_yi = np.sqrt(np.diag(C_yi))
        y_fit = function(param, x1)
        y_fit_max = function(param + param_err, x1)
        y_fit_min = function(param - param_err, x1)
        ax = plt.subplot(1, 2, 1)
        ax.cla()
        plt.errorbar(x, y, yerr=yerr, fmt='ok')
        # ax.set_yscale('log')
        ax.set_ylabel('$N_{\gamma}$')
        ax.set_xlabel('LED DAC %d'%options.pixel_list[pixel])
        ax.set_yscale('log')
        # plt.fill_between(x1, y_fit_max, y_fit_min, alpha=0.5, facecolor='blue', label='polyfit confidence level')
        plt.fill_between(x1, yi + sig_yi, yi - sig_yi, alpha=0.5, facecolor='red', label='polyfit confidence level')
        ax1 = plt.subplot(1, 2, 2)
        ax1.cla()
        ax1.set_ylabel('$N_{\gamma}$, $68\%$ C.L. (relative)')
        ax1.set_xlabel('LED DAC')
        # plt.plot(x1, (y_fit_max - y_fit_min) / yi, label='polyfit + 1 $\sigma$')
        # plt.plot(x1, y_fit_min, label='polyfit - 1 $\sigma$')
        # plt.fill_between(x1, yi + sig_yi, yi - sig_yi, alpha=0.5, facecolor='red', label='polyfit confidence level')
        plt.fill_between(x1, 2 * sig_yi / yi, alpha=0.5, facecolor='red', label='polyfit confidence level')
        print('param ', param, ' ± ', param_err)

        plt.show()
        input('bla')

    return
