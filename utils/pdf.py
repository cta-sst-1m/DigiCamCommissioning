import math
from scipy.special import factorial
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

def poisson(k, mu):
    return mu ** k * np.exp(-mu) / math.factorial(k)


def gaussian(p, x, amplitude=1.):

    sigma = p[0]
    mean = p[1]
    amplitude = p[2]

    if amplitude==1.:
        amplitude = 1.

    return amplitude / np.sqrt(2 * sigma ** 2 * math.pi) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

def generalized_poisson(k, mu, mu_xt, amplitude=1):
    if mu_xt < 0 or mu < 0 or k < 0:

        if isinstance(k, int):
            return 0
        else:
            return np.zeros(len(k))

    else:

        if k==0:
           log_k = np.log(1)
        else:
           log_k = np.sum([np.log(i) for i in range(1, k+1)])
        #return amplitude * mu * (mu + k * mu_xt) ** (k - 1) * np.exp(-mu - k * mu_xt) / factorial(k)
        return np.exp(np.log(amplitude) + np.log(mu) + np.log(mu + k * mu_xt)*(k - 1) + (-mu - k * mu_xt) - log_k)

def gaussian_sum(param, x):

    temp = np.zeros(x.shape)
    bin_width = x[1] - x[0]

    baseline = param[0]
    gain = param[1]
    sigma_e = param[2]
    sigma_1 = param[3]
    #offset = param[4]
    amplitudes = param[4:len(param)+1]

    n_peaks = len(amplitudes)

    for i in range(n_peaks):
        sigma = np.sqrt(sigma_e**2 + i*sigma_1**2 + bin_width**2/12.)
        #temp += gaussian(x, sigma, baseline + i*gain + (offset if i==0 else 0), amplitude=amplitudes[i])

        param = [sigma, baseline + i*gain, amplitudes[i]]
        temp += gaussian(param, x, amplitude=amplitudes[i])

    return temp

def gaussian_2(param, x):

    temp = np.zeros(x.shape)

    param = np.array([[param[0], param[1], param[2]],[param[3], param[4], param[5]]])


    for i in range(int(len(param)/3.)):

        temp += gaussian(param[i], x)

    return temp

def chi2(param, x):

    ndf = param[0]

    return scipy.stats.chi2.pdf(x, ndf)



def erlang_compound(x, mu, mu_xt):
    temp = 0
    mu_xt = mu_xt
    n = 15
    for k in range(n):

        if k == 0:
            temp += poisson(k, mu)
        else:
            temp += mu_xt ** k * x ** (k - 1) * np.exp(-mu_xt * x) / math.factorial(k - 1) * poisson(k, mu)

    return temp


def mpe_gaussian_distribution(p, x):
    p[0] = gain
    p[1] = sigma_e
    p[2] = sigma_1
    p[3] = offset
    amplitude = []
    for i in range(4, len(p) - 4):
        amplitude.append(p[i])
    n_peak = len(amplitude)
    temp = np.zeros(len(x))
    x = x - offset
    amplitude = np.array(amplitude)
    x = x - offset # TODO: Cyril check
    for n in range(int(n_peak)):
        sigma_n = np.sqrt(sigma_e ** 2 + n * sigma_1 ** 2) * gain
        temp += amplitude[n] * gaussian(x, sigma_n, n * gain)
    return temp


def mpe_distribution_general(p, x, config=None):
    mu, mu_xt, gain, offset, sigma_e, sigma_1, amplitude = p

    # print(p)

    temp = np.zeros(x.shape)
    x = x - offset
    n_peak = 40
    for n in range(0, n_peak, 1):
        sigma_n = np.sqrt(sigma_e ** 2 + n * sigma_1 ** 2)

        temp += generalized_poisson(n, mu, mu_xt) * gaussian(x, sigma_n, n * gain)

    return temp * amplitude


def mpe_distribution_general_sh(p, x, config=None):
    mu, mu_xt, gain, baseline, sigma_e, sigma_1, amplitude, offset = p

    # print(p)

    temp = np.zeros(x.shape)
    x = x - baseline
    n_peak = 15
    for n in range(0, n_peak, 1):
        sigma_n = np.sqrt(sigma_e ** 2 + n * sigma_1 ** 2)  # * gain

        temp += generalized_poisson(n, mu, mu_xt) * gaussian(x, sigma_n, n * gain + (offset if n != 0 else 0))

    return temp * amplitude


def multi_gaussian_with0( p, x):
    """
    Multiple gaussian fit for the SPE
    :param p:
    :param x:
    :return:
    """
    gaus0 = p[0] / (p[9]) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[7])) ** 2 / (2. * (p[9]**2)))
    gaus1 = p[4] / (np.sqrt(p[1]**2+1*(p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]+p[7]+p[8])) ** 2 / (2. * (p[1]**2+1*(p[2])**2)))
    gaus2 = p[5] / (np.sqrt(p[1]**2+2*(p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*2+p[7]+p[8])) ** 2 / (2. * (p[1]**2+2*(p[2])**2)))
    gaus3 = p[6] / (np.sqrt(p[1]**2+3*(p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*3+p[7]+[p[8]])) ** 2 / (2. * (p[1]**2+3*(p[2])**2)))
    gaus4 = p[10] / (np.sqrt(p[1]**2+4*(p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*4+p[7]+[p[8]])) ** 2 / (2. * (p[1]**2+4*(p[2])**2)))
    gaus5 = p[11] / (np.sqrt(p[1]**2+5*(p[2])**2)) / np.sqrt(2. * np.pi) * np.exp(-(np.asfarray(x)-(p[3]*5+p[7]+[p[8]])) ** 2 / (2. * (p[1]**2+5*(p[2])**2)))

    return gaus0+gaus1+gaus2+gaus3+gaus4+gaus5

def bspleval(x, knots, coeffs, order):
    '''
    Evaluate a B-spline at a set of points.

    Parameters
    ----------
    x : list or ndarray
        The set of points at which to evaluate the spline.
    knots : list or ndarray
        The set of knots used to define the spline.
    coeffs : list of ndarray
        The set of spline coefficients.
    order : int
        The order of the spline.

    Returns
    -------
    y : ndarray
        The value of the spline at each point in x.
    '''

    k = order
    t = knots
    m = np.alen(t)
    npts = np.alen(x)
    B = np.zeros((m-1,k+1,npts))

    ## Create the zero-order B-spline basis functions.
    for i in range(m-1):
        B[i,0,:] = np.float64(np.logical_and(x >= t[i], x < t[i+1]))

    if (k == 0):
        B[m-2,0,-1] = 1.0

    ## Next iteratively define the higher-order basis functions, working from lower order to higher.
    for j in range(1,k+1):
        for i in range(m-j-1):
            if (t[i+j] - t[i] == 0.0):
                first_term = 0.0
            else:
                first_term = ((x - t[i]) / (t[i+j] - t[i])) * B[i,j-1,:]

            if (t[i+j+1] - t[i+1] == 0.0):
                second_term = 0.0
            else:
                second_term = ((t[i+j+1] - x) / (t[i+j+1] - t[i+1])) * B[i+1,j-1,:]

            B[i,j,:] = first_term + second_term
        B[m-j-2,j,-1] = 1.0

    ## Evaluate the spline by multiplying the coefficients with the highest-order basis functions.
    y = np.zeros(npts)
    for i in range(m-k-1):
        y += coeffs[i] * B[i,k,:]


    return(y)

if __name__ == '__main__':
    x = np.arange(0, 200, 1)
    n_peak = 20
    gain = 5.6
    mu = 5.
    mu_xt = 0.08
    sigma_e = 0.09
    sigma_1 = 0.1
    type_list = ['generalized_poisson', 'poisson', 'erlang_compound']
    offset = 10.
    plt.figure()

    for type_pdf in type_list:
        y = mpe_distribution(x, n_peak, gain, mu, mu_xt, sigma_e, sigma_e, offset, type_pdf) #TODO: check

        plt.plot(x, y, label=type)

    plt.xlabel('ADC count [ADC]')
    plt.ylabel('P(ADC)')
    plt.legend(loc='best')
    plt.show()
