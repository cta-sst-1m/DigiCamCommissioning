from ctapipe import visualization
from utils import  geometry
import numpy as np
from cts_core import cameratestsetup
import matplotlib.pyplot as plt

cts = cameratestsetup.CTS('/data/software/CTS/config/cts_config_120.cfg', '/data/software/CTS/config/camera_config.cfg', angle=120., connected=False)

geom,pixel_list = geometry.generate_geometry(cts)
geom0,pixel_list0 = geometry.generate_geometry(cts,all_camera=True)


def load_mc_event(pixel_values,cts,param):
    patch_dac,patch_pe = [],[]
    for patch in cts.LED_patches:
        pe_in_pixels = []
        for pix in patch.leds_camera_pixel_id:
            pe_in_pixels+=[pixel_values[pix]]
        dac=get_patch_DAC(patch.leds_camera_pixel_id, pe_in_pixels, param)
        pe = np.mean(pe_in_pixels)
        patch_dac.append(dac)
        patch_pe.append(pe)
    return patch_dac,patch_pe


def get_patch_DAC(pixels,pe_per_pixel,param):
    dac = 0
    total_pe = np.sum(pe_per_pixel)
    if total_pe==0.: return 0
    sum_param = np.zeros(param.shape[-1],dtype=param.dtype)
    pe_vs_dac = np.zeros((1000,))
    for p in pixels:
        poly =  np.polyval(param[pixel_list.index(p)],np.arange(1000))
        poly[poly<0.]=0.
        pe_vs_dac += poly
        #sum_param= sum_param+param[pixel_list.index(p)]
    dac = 0
    try:
        dac = np.argmin(np.abs(pe_vs_dac - total_pe))
        #for pix in pixels:
        #    print(pix,dac,np.polyval(param[pixel_list.index(pix)],dac))
        '''
        poles = np.real((np.poly1d(sum_param) - total_pe).roots)
        goodpole = 0
        for pole in poles:
            if pole<0: continue
            if np.count_nonzero(poles-pole)>2: goodpole = pole

        dac = goodpole#np.max(np.real((np.poly1d(sum_param) - total_pe).roots))
        '''
    except:
        dac = -1
    if dac < 1. or total_pe < 1.: dac = 0
    return int(np.round(dac))


ac_led_coefficient = np.load('/data/datasets/CTA/DATA/20170322/scan_ac_level/ac_led.npz')['fit_result']
param, covariance = ac_led_coefficient[:, :, 0], ac_led_coefficient[:, :, 2:7:1]

plt.subplots(1,2)
plt.subplot(1,2,1)
ac_array=np.zeros((len(pixel_list)*1000),dtype=float)
dac_array=np.zeros((len(pixel_list)*1000),dtype=float)
x_dac = np.arange(0,1000,1)
for i in range(len(pixel_list)):
    pes = np.polyval(param[i], x_dac)
    pes[pes<1e-1]=1.e-3
    ac_array[i:i+1000:1]=np.copy(np.log10(pes))
    dac_array[i:i+1000:1]=np.arange(0,1000,1)
    plt.plot(x_dac,pes,color='k')
plt.yscale('log')
plt.ylim(1.,1e5)



f = open('/data/datasets/CTA/DATA/20170322/scan_ac_level/coeff.txt')
coeffs = np.zeros((528,2),dtype = float)
lines = f.readlines()
for i,l in enumerate(lines):
    val=l.split('\n')[0].split(' ')
    for j in range(2):
        coeffs[i,j]=val[j]
f.close()
plt.ion()

plt.subplot(1,2,2)
dc_array=np.zeros((len(pixel_list)*1000),dtype=float)
for i in range(len(pixel_list)):
    pes = coeffs[i, 0] * np.exp(coeffs[i, 1] * x_dac)
    pes[pes<10.]=1.e-3
    pes[pes>1000.]=1.e4
    dc_array[i:i+1000:1]=pes
    plt.plot(x_dac,pes,color='k')

plt.yscale('log')
plt.ylim(10.,1000.)
plt.show()
dc_array[np.isnan(dc_array)]=10.
ac_array[np.isnan(ac_array)]=1.e-3
print(ac_array)

plt.subplots(1,2)
plt.subplot(1,2,1)
plt.hist2d(dac_array,ac_array)
plt.ylim(1.,5.)
plt.subplot(1,2,2)
plt.hist2d(dac_array,dc_array,bins=1000)
plt.ylim(10.,1000.)
plt.show()


