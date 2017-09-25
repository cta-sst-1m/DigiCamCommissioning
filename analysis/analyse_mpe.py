#!/usr/bin/env python3

# external modules

# internal modules
from data_treatement import mpe_hist
from spectra_fit import fit_low_light,fit_high_light
from utils import display, histogram, geometry
import logging,sys
import numpy as np
import scipy
import logging
from tqdm import tqdm
from utils.logger import TqdmToLogger
import pickle
import utils.pdf
import matplotlib.pyplot as plt
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
    mpes_file = np.load(options.output_directory + options.histo_filename)
    mpes = histogram.Histogram(filename=options.output_directory + options.histo_filename)
    #setattr(mpes,'data',mpes_file['data'][:,0:1])
    #setattr(mpes,'errors',mpes_file['errors'][:,0:1])
    #setattr(mpes,'bin_centers',mpes_file['bin_centers'])
    print(mpes.data.shape)

    log = logging.getLogger(sys.modules['__main__'].__name__+__name__)
    pbar = tqdm(total=mpes.data.shape[0]*mpes.data.shape[1])
    tqdm_out = TqdmToLogger(log, level=logging.INFO)
    b = {}
    with open(options.output_directory + options.level_pickle, 'rb') as handle:
        b = pickle.load(handle)

    def std_dev(x, y):
        if np.sum(y)<=0: return 0.
        avg = np.average(x, weights=y)
        return np.sqrt(np.average((x - avg) ** 2, weights=y))

    g_logs = logs(2000)
    reduced_func = lambda p, x1, x2 : fit_function(p, x1, x2, g_logs)
    ## Now perform the mu and mu_XT fits

    plt.subplots(4, 4)
    param = None
    for pixel,real_pix in enumerate([options.pixel_list[0]]):#enumerate(options.pixel_list):#enumerate(options.pixel_list):
        patch = options.cts.camera.Pixels[real_pix].patch
        levels = np.array(b[patch])[0:40]
        slice_list = slice_function(mpes.data[0,0:40,pixel])
        bound_min,bound_max = bounds(None, mpes.data[0,0:40,pixel],levels,mpes.bin_centers)
        reduced_p0 = p0(None, mpes.data[0,0:40,pixel],levels,mpes.bin_centers)
        print('p0',reduced_p0)

        for i, level in enumerate(levels):
            if i > 15: continue
            plt.subplot(4, 4, 1 + i)

        residual = lambda p, x1,x2, y, y_err: residuals(reduced_func, p, x1,x2, y, y_err)
        out = scipy.optimize.least_squares(residual, reduced_p0, args=(levels,
            mpes.bin_centers[slice_list[0]:slice_list[1]:slice_list[2]],
            mpes.data[0,0:40,pixel,slice_list[0]:slice_list[1]:slice_list[2]],
            mpes.errors[0,0:40,pixel,slice_list[0]:slice_list[1]:slice_list[2]]),loss='soft_l1',bounds=(bound_min,bound_max))
                                          # bounds=(bound_min,bound_max),

        param = out.x
        #weight_matrix = np.diag(1. / mpes.errors[0,i,pixel,slice_list[0]:slice_list[1]:slice_list[2]])
        #weight_matrix = 1.  # TODO changed to one since pull study showed previous config is fine
        #cov = np.sqrt(np.diag(inv(np.dot(np.dot(out.jac.T, weight_matrix), out.jac))))
        #fit_result = np.append(out.x.reshape(out.x.shape + (1,)), cov.reshape(cov.shape + (1,)), axis=1)
    #param=

    ys = reduced_func(param, levels, mpes.bin_centers[slice_list[0]:slice_list[1]:slice_list[2]])
    for i, y in enumerate(ys):
        if i>15: continue
        plt.subplot(4, 4, 1 + i)
        plt.errorbar(mpes.bin_centers[slice_list[0]:slice_list[1]:slice_list[2]],
                     (y-mpes.data[0,i,pixel,slice_list[0]:slice_list[1]:slice_list[2]])/mpes.errors[0,i,pixel,slice_list[0]:slice_list[1]:slice_list[2]],fmt='ok')


    reduced_func = lambda p, x : led_polynomial(p, x)

    reduced_p0 = np.poly1d(np.polyfit(levels, param[0:40], 4))
    reduced_p0 = list(reduced_p0.coeffs)
    print(reduced_p0)
    residual = lambda p, x1, y, y_err: residuals_1D(reduced_func, p, x1, y, y_err)
    out = scipy.optimize.least_squares(residual, list(reduced_p0)+[1.2], args=(levels,param[0:40],np.sqrt(param[0:40])))


    plt.subplots(1, 1)
    plt.errorbar(levels[0:40], param[0:40],fmt = 'ok')
    plt.plot(np.arange(levels[0],levels[-1],0.1),led_polynomial(out.x,np.arange(levels[0],levels[-1],0.1)))
    plt.yscale('log')
    plt.show()



def display_results(options):
    """
    Display the analysis results

    :param options:

    :return:
    """
    return



def p0(prev_fit,y,x1,x2):
    mus = np.zeros(x1.shape,dtype=float)
    for i in range(x1.shape[0]):
        mus[i] = np.average(x2.astype(dtype=float)/23.,weights=y[i],axis=-1)*0.92
    p = list(mus)
    p +=[ 0.08 , 19., 0., 4.,1. ]
    p+=list(np.sum(y,axis=-1).reshape(-1)*3.)
    '''
    yy = np.zeros(x1.shape,dtype=float)
    for i in range(x1.shape[0]):
        yy[i] = np.average(x2.astype(dtype=float)/23.,weights=y[i],axis=-1)*0.92
    pol = np.poly1d(np.polyfit(x1.astype(dtype=float)[yy>1], yy[yy>1], 4))
    p = list(pol.coeffs)
    p +=[1.2, 0.08 , 19., 0., 4.,1. ]
    p+=list(np.sum(y,axis=-1).reshape(-1)*3.)
    '''
    return p

def bounds(prev_fit,y,x1,x2):
    '''
    bound_min = [-np.inf,
                 -np.inf,
                 -np.inf,
                 -np.inf,
                 -np.inf,
                 0.,
                 0.,
                 15.,
                 -10.,
                 0.,
                 0.,
                 ]+list(np.maximum(np.sum(y,axis=-1)-5* np.sqrt(np.sum(y,axis=-1)),0.).reshape(-1))
    #for a in np.maximum(np.sum(y,axis=-1)-5* np.sqrt(np.sum(y,axis=-1)),0.) :
    #    bound_min.append(float(a))

    bound_max = [np.inf,
                 np.inf,
                 np.inf,
                 np.inf,
                 np.inf,
                 2.,
                 0.2,
                 27.,
                 10.,
                 10.,
                 10.,
                 ]+list(3. * np.maximum(np.sum(y,axis=-1)+5* np.sqrt(np.sum(y,axis=-1)),0.).reshape(-1)*np.inf)
    return bound_min,bound_max
    '''

    mus = np.zeros(x1.shape,dtype=float)
    for i in range(x1.shape[0]):
        mus[i] = np.average(x2.astype(dtype=float)/23.,weights=y[i],axis=-1)*0.92
    bound_min = list(mus*0.1)+ [
                 0.,
                 15.,
                 -10.,
                 0.,
                 0.,
                 ] + list(np.maximum(np.sum(y, axis=-1) - 5 * np.sqrt(np.sum(y, axis=-1)), 0.).reshape(-1))
    # for a in np.maximum(np.sum(y,axis=-1)-5* np.sqrt(np.sum(y,axis=-1)),0.) :
    #    bound_min.append(float(a))
    bound_max = list(mus * 10.) + [
                 0.2,
                 27.,
                 10.,
                 10.,
                 10.,
    ] + list(np.maximum(3.*np.sum(y, axis=-1) + 5 * np.sqrt(3.*np.sum(y, axis=-1)), 0.).reshape(-1)*np.inf)
    return bound_min, bound_max

def led_polynomial(p , x):
    # p[0] is the level at which the LED do not produce light anymore
    # p[1],p[2],p[3] are negative
    # p[4] is the mu_dark
    poly = np.poly1d(p[0:5]) (x)
    poly[poly<0] = 0.
    poly+=p[5]
    return poly

def gaussian_array(p, x, normalised = False):
    sigma = p[0]
    mean = p[1]
    amplitude = p[2]
    if normalised :
        amplitude = 1.
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if len(sigma.shape)==1:
        sigma = sigma.reshape(-1,1)
    if len(mean.shape)==1:
        mean = mean.reshape(-1,1)
    if len(amplitude.shape)==1:
        amplitude = amplitude.reshape(-1,1)
    return amplitude.T / np.sqrt(2 * sigma.T ** 2 * np.pi) * np.exp(-(x - mean.T) ** 2 / (2 * sigma.T ** 2))

def logs(max_n):
    N = np.arange(1,max_n+1)
    _logs = np.zeros((max_n+1),dtype = float)
    _logs[1:]=np.log(N)
    return np.cumsum(_logs)

def generalized_poisson(k, mu, mu_xt , logs_k):
    klog = logs_k[list(k)]
    ### logs_k of N
    if len(k.shape) == 1:
        k=k.reshape(-1,1)
    if len(mu.shape) == 1:
        mu = mu.reshape(-1,1)
    return np.exp(np.log(1.) + np.log(mu) + np.log(mu + k.T * mu_xt)*(k.T - 1) + (-mu - k.T * mu_xt) - klog)

import time
def fit_function(p, x1, x2, g_logs):
    # get the mu as function of DAC
    #mu = led_polynomial(p[0:6],x1)
    #mu_xt, gain, baseline, sigma_e, sigma_1, amplitudes = p[6], p[7], p[8], p[9], p[10], np.array(p[11:])
    mu = np.array(p[0:x1.shape[0]])
    mu_xt, gain, baseline, sigma_e, sigma_1, amplitudes = p[x1.shape[0]], p[x1.shape[0]+1], p[x1.shape[0]+2], p[x1.shape[0]+3], p[x1.shape[0]+4],\
                                                          np.array(p[x1.shape[0]+5:])
    central_n = mu * (1 + mu_xt)
    n_peakmin = np.round(np.maximum(central_n - 4 * np.sqrt(central_n), 0),0).astype(dtype = int)
    n_peakmin[n_peakmin<0]=0
    n_peakmax = np.round(central_n + 4 * np.sqrt(central_n),0).astype(dtype = int)
    if n_peakmax[-1]>100000:
        return np.ones(x1.shape+x2.shape,dtype=float)*1.e8
    # Generate the gaussians that will be used
    N = np.arange(0, n_peakmax[-1] + 1)
    sigma_n = np.sqrt(sigma_e ** 2 + N * sigma_1 ** 2 + 1. / 12.)
    gaussian_pdfs = gaussian_array([sigma_n, N * gain, np.ones(N.shape, dtype=float)], x2)
    generalised_poisson_pdfs = generalized_poisson(N, mu.T, mu_xt,g_logs)
    y = np.dot(generalised_poisson_pdfs, gaussian_pdfs.T)* amplitudes.reshape(-1,1)
    return y

def slice_function(y):
    k = np.sum(y,axis=0)
    return [np.where(k != 0)[0][0], np.where(k != 0)[0][-1], 1]


def residuals(function, p, x1 , x2 , y, y_err):
        """
        Return the residuals of the data with respect to a function

        :param function: The function defined with arguments params and x (function)
        :param p: the parameters of the function                          (np.array)
        :param x: the x values                                            (np.array)
        :param y: the y values                                            (np.array)
        :param y_err: the y values errors                                 (np.array)
        :return: the residuals                                            (np.array)
        """
        return (y.reshape(-1) - function(p, x1, x2).reshape(-1)) / y_err.reshape(-1)

def residuals_1D(function, p, x1 , y, y_err):
        """
        Return the residuals of the data with respect to a function

        :param function: The function defined with arguments params and x (function)
        :param p: the parameters of the function                          (np.array)
        :param x: the x values                                            (np.array)
        :param y: the y values                                            (np.array)
        :param y_err: the y values errors                                 (np.array)
        :return: the residuals                                            (np.array)
        """
        return (y.reshape(-1) - function(p, x1).reshape(-1)) / y_err.reshape(-1)