from cts_core import cameratestsetup
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils.histogram import Histogram

f = open('/data/datasets/CTA/DATA/20170322/scan_ac_level/coeff.txt')
coeffs = np.zeros((528,2),dtype = float)
lines = f.readlines()
for i,l in enumerate(lines):
    val=l.split('\n')[0].split(' ')
    for j in range(2):
        coeffs[i,j]=val[j]
f.close()

ac_led = Histogram(filename='/data/datasets/CTA/DATA/20170322/scan_ac_level/ac_led.npz', fit_only=True)

param, covariance = ac_led.fit_result[:, :, 0], ac_led.fit_result[:, :, 2:7:1]

param_err = ac_led.fit_result[:, :, 1]

#coeffs = np.load('/data/datasets/CTA/DATA/20170322/scan_ac_level/dc_led.npz')['fit_result']
#print(coeffs[323])
#coeffs = coeffs[([250, 272, 273, 274, 275, 296, 297, 298, 299, 300, 320, 321, 322, 323, 344, 345, 346, 347, 348, 369, 370])]
dac = np.arange(0,1000)
pixels=[518,552,553,554,555,588,589,590,591,592,624,625,626,627,660,661,662,663,664,697,698]
allnsb = np.zeros(1000)
nsbs = np.zeros((1000))
#nsb = coeffs[:,0,0]*np.exp(coeffs[:,1,0]*dac[:,None])
plt.ion()
for i,pix in enumerate([250, 272, 273, 274, 275, 296, 297, 298, 299, 300, 320, 321, 322, 323, 344, 345, 346, 347, 348, 369, 370]):
    #if pix not in [660,552,555]: continue
    nsb = coeffs[pix, 0] * np.exp(coeffs[pix, 1] * dac)
    allnsb+=nsb
    print(pix,pixels[i],nsb[250])
    plt.plot(dac,nsb,label='pixel %d'%pixels[i])

plt.plot(dac,allnsb/len(pixels),color='k',linewidth = 2 , label='Average per DAC')
plt.legend()
plt.ylim(40.9,1500.)
plt.xlim(250.,550.)
plt.show()


def get_DAC_DC(pixel,f_NSB):
    return 1./coeffs[pixel,1,0] * np.log(f_NSB/coeffs[pixel,0,0])

def get_DAC_AC(pixel,N_pe):
    inv_p = lambda y: np.max(np.real((np.poly1d(param[pixel]) - y).roots))
    inv_p_min = lambda y: np.max(np.real((np.poly1d(param[pixel] - param_err[pixel]) - y).roots))
    inv_p_max = lambda y: np.max(np.real((np.poly1d(param[pixel] + param_err[pixel]) - y).roots))
    return inv_p(N_pe),inv_p_min(N_pe),inv_p_max(N_pe)

plt.figure()
for i,pix in enumerate([250, 272, 273, 274, 275, 296, 297, 298, 299, 300, 320, 321, 322, 323, 344, 345, 346, 347, 348, 369, 370]):
    y = np.polyval(param[pix],dac)
    plt.plot(dac,y,label='pixel %d'%pixels[i])


plt.legend()
plt.ylim(1,3000.)
plt.xlim(20.,1000.)
plt.show()


#def get_f_NSB_DAC_pixels():

