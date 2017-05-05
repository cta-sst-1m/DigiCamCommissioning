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
mc_events = np.load('/data/datasets/CTA/MC/mc_1.npz')['mc_pes']


data = np.load('/data/datasets/CTA/DATA/20170413/shower_plot.npz')

plt.ion()
plt.subplots(2, 1)
plt.subplot(2, 1, 1)
h = np.copy(data['pixel'].flatten())
p = np.copy(data['patch'].flatten())
h[np.isnan(h)]=0
p[np.isnan(p)]=0
h[h>2]=2
p[p>2]=2
plt.hist(h, bins=100)
plt.subplot(2, 1, 2)
plt.hist(p, bins=100)
plt.show()


fig, ax = plt.subplots(2,3,figsize=(16,12))

titles={
'true_pix':'True p.e',
           'injected_pix':'CTS injected p.e.',
           'measured_pix':'Measured p.e.',
           'true_patch':'True p.e. in patch',
           'injected_patch':'Injected p.e. in patch',
    'measured_patch':'Measured p.e. in patch',
    'diff_patch':'pixel relative difference measured,injected',
    'diff_pixel':'patch relative difference measured,injected'
}

camera_visu = {}
ax = {}
for i,val in enumerate(['true_pix','injected_pix','measured_pix','true_patch','injected_patch','measured_patch']):
    if 'true' in val:
        ax[val]=plt.subplot(2,3,i+1)
        camera_visu[val] = visualization.CameraDisplay(geom0, ax=ax[val], title=titles[val], norm='lin', cmap='viridis',allow_pick=False)
        camera_visu[val].image = np.ones(geom0.pix_x.shape[0])
        camera_visu[val].add_colorbar()
    else:
        ax[val]=plt.subplot(2,3,i+1)
        camera_visu[val+'base'] = visualization.CameraDisplay(geom0, ax=ax[val], title=titles[val], norm='lin', cmap='YlOrRd',allow_pick=False)
        camera_visu[val] = visualization.CameraDisplay(geom, ax=ax[val], title=titles[val], norm='lin' if not 'diff' in val else 'lin', cmap='viridis',allow_pick=False)
        camera_visu[val+'base'].image = np.zeros(geom0.pix_x.shape[0])
        camera_visu[val].image = np.ones(geom.pix_x.shape[0])
        camera_visu[val].add_colorbar()

kk = 0
f_out = open('evt_list.txt','w')
for i in range(mc_events.shape[0]):#enumerate([4,6,8,16,21,33,43,64,79,76,81]):
    ii = i
    print ('runnning evt,',i)
    pixel_true_pe = mc_events[i]
    patch_dac,patch_pe = load_mc_event(pixel_true_pe, cts, param)
    pixel_injected_pe = [0 for i in range(528)]
    patch_injected_pe = [0 for i in range(528)]
    pixel_injected_dac = [0 for i in range(528)]
    pixel_measured_pe = data['pixel'][ii] if ii<data['pixel'].shape[0] else np.zeros((528),dtype=int)
    patch_measured_pe = data['patch'][ii] if ii<data['pixel'].shape[0] else np.zeros((528),dtype=int)
    patch_true_pe = [0 for i in range(1296)]
    mask = np.array([True if i in pixel_list else False for i in range(1296)])
    for p in cts.camera.Patches:
        sum_pe = 0.
        for pix in p.pixelsID:
            sum_pe += pixel_true_pe[pix]
        for pix in p.pixelsID:
            patch_true_pe[pix] = sum_pe
    for p_id , p_dac in enumerate(patch_dac):
        tot_in_patch=0.
        for pix in cts.LED_patches[p_id].leds_camera_pixel_id:
            if p_dac < 0.5: continue
            pe = np.polyval(param[pixel_list.index(pix)],p_dac)
            if pe<0.: continue
            pixel_injected_pe[pixel_list.index(pix)]= pe
            tot_in_patch+=pe
        for pix in cts.LED_patches[p_id].leds_camera_pixel_id:
            pixel_injected_dac[pixel_list.index(pix)]=p_dac
            patch_injected_pe[pixel_list.index(pix)]=tot_in_patch

    pixel_injected_pe=np.array(pixel_injected_pe,dtype=float)
    patch_true_pe=np.array(patch_true_pe)
    pixel_true_pe=np.array(pixel_true_pe,dtype=float)
    patch_injected_pe=np.array(patch_injected_pe)
    goodevt=False

    if (np.sum(patch_injected_pe)/np.sum(patch_true_pe))>0.8:
        goodevt=True
    pixel_true_pe[pixel_true_pe<5.e-1]= 1.e-3
    patch_true_pe[patch_true_pe<5.e-1]= 1.e-3
    pixel_injected_pe[pixel_injected_pe<5.e-1]= 1.e-3
    patch_injected_pe[patch_injected_pe<5.e-1]= 1.e-3
    pixel_measured_pe[pixel_measured_pe<10.e-1]= 1.e-3
    patch_measured_pe[patch_measured_pe<30.e-1]= 1.e-3
    pixel_measured_pe[np.isnan(pixel_measured_pe)]=1e-3
    patch_measured_pe[np.isnan(patch_measured_pe)]=1e-3
    #diff_patch = (patch_measured_pe-patch_injected_pe)/patch_injected_pe
    #diff_pixel = (pixel_measured_pe-pixel_injected_pe)/pixel_injected_pe
    #diff_patch[patch_injected_pe<1]=0
    #diff_pixel[pixel_injected_pe<1]=0
    #patch_injected_pe[patch_injected_pe>100]=100
    #patch_true_pe[patch_true_pe>100]=100
    camera_visu['true_pix'].image = pixel_true_pe
    camera_visu['injected_pix'].image = pixel_injected_pe#pixel_injected_pe
    camera_visu['true_patch'].image = patch_true_pe#pixel_true_pe
    camera_visu['injected_patch'].image = patch_injected_pe  #pixel_injected_pe
    camera_visu['measured_pix'].image = pixel_measured_pe
    camera_visu['measured_patch'].image =  patch_measured_pe
    #camera_visu['diff_pixel'].image = diff_pixel
    #camera_visu['diff_patch'].image =  diff_patch

    if goodevt:
        print('Event %d',i)
        plt.show()
        f_out.write('# Event %d'%kk)
        for pixel in range(1296):
            if pixel in pixel_list:
                f_out.write('%d %f\n'%(pixel,pixel_injected_pe[pixel_list.index(pixel)]))
            else:
                f_out.write('%d 0.\n'%(pixel))
        kk+=1
        #h = input('press a key to go to next event')

f_out.close()
plt.subplots(1,2)
plt.subplot(1,2,1)
ac_array=np.array((len(pixel_list),1000))
x_dac = np.arange(0,1000,1)
for i in pixel_list:
    pes = np.polyval(param[i], x_dac)
    pes[pes<0.]=-1.
    ac_array[i]=pes
    plt.plot(x_dac,pes,color='k')


f = open('/data/datasets/CTA/DATA/20170322/scan_ac_level/coeff.txt')
coeffs = np.zeros((528,2),dtype = float)
lines = f.readlines()
for i,l in enumerate(lines):
    val=l.split('\n')[0].split(' ')
    for j in range(2):
        coeffs[i,j]=val[j]
f.close()

plt.subplot(1,2,2)
dc_array=np.array((len(pixel_list),1000))
x_dac = np.arange(0,1000,1)
for i in pixel_list:
    pes = np.polyval(param[i], x_dac)
    pes[pes<0.]=-1.
    ac_array[i]=pes
    plt.plot(x_dac,pes,color='k')


