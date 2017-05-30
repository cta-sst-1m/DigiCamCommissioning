import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from utils.histogram import Histogram

#data_0 = np.zeros((6,35))
data_0 = []
data_0 += [[100,90,80,70,60,50,45,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,23,21,16,11,6,1,0,4,8]]
data_0 += [[2,2,2,4,5,14,20,31,22,26,23,31,19,40,66,148,602,2072,6332,23588,77704,291857,1945509,2321467,
                      2013709,6096702,4582408,5374269,7011471,13538356,193393140,194399101,147856182,148037529,85815690]]
data_0+= [[120.477,120.173,120.7,120.66,185.63,120.0,165.966,196.9,163.11,133.19,130.25,163.98,103.62,
                      125.788,125.02,120.7,135.22,106.92,74.46,79.72,63.23,65.28,132.96,62.5898,30.586760,65.965293,
                      35.314039,33.266396,30.338799,30.714656,44.992524,39.657416736,30.162661,30.466780,36.673887]]

#Crate master Analog
data_1 = []
data_1 += [[50,45,40,35,70,30,25,20,15,10,5]]
data_1 += [[289,207,248,95,82,174,500,3327,179317,22090984,140665075]]
data_1 += [[705.811872968,467.455945760,444.195793216,106.391106976,401.586030728,136.514668576,66.501146888,26.714960224,31.106538896,39.321387064,32.522535528]]


data_2 = []
data_2 += [[5,10,15,20,25,30,35,40]]
data_2 += [[404008293,9373104,4837759,4207519,3246574,5519876,1559280,301]]
data_2 += [[93.121154128,30.274769496,20.012270576,20.692336952,20.196262232,48.895958104,47.560179536,139.314085048]]

data_3 = []
data_3 += [[40,35,30,25,20,15,15,10,5,0]]
data_3 += [[204,1578936,6911811,5567420,7664965,9833482,7795891,20363559,27910327,24152084]]
data_3 += [[131.819300096,49.919709520,62.109608488,34.970017952,37.977594856,38.433611448,30.76793376,70.508486504,11.021567320,10.413762928]]

data_4 = []
data_4 += [[]]
data_4 += [[]]
data_4 += [[]]
triggers = Histogram(filename='/home/alispach/data/digicam_commissioning/trigger/mc/trigger.npz')

dataset = [data_0,data_1,data_2,data_3]
colors=['k','b','g','r','y']
labels = ['Dark run, config ((1,1,0),(2,1,0))',
          'Dark run ((1,0,0),(2,0,2))',
          'Dark run ((0,1,0),(0,1,2))',
          'Dark run ((0,1,0),(1,1,1))']

def rate_calc(data):
    data = np.array(data)
    samples = data[2]/4e-9
    p = data[1]/data[2]*4e-9
    q = 1.-p
    data = np.append(data,(data[1]/data[2]).reshape(1,data.shape[1]),axis=0)
    data = np.append(data,(np.sqrt(samples*p*q)/data[2]).reshape(1,data.shape[1]),axis=0)
    data = np.append(data,(data[0]*4./5.6).reshape(1,data.shape[1]),axis=0)
    print(data.shape)
    sort0 = data[0,:].argsort()
    for i,d in enumerate(data):
        data[i]=data[i,sort0]
    return np.copy(data)

plt.ion()
def plot(datas,labels,colors=colors,xlim=[0,100]):
    fig = plt.figure(figsize=(14, 12))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    xlim_min = xlim[0]
    xlim_max = xlim[1]
    ax1.errorbar(triggers.bin_centers, triggers.data[0], yerr=triggers.errors[0], fmt='o', color='%s' % colors[1], label='MC')
    for i,data in enumerate(datas):
        data = np.array(data)
        print(data.shape)
        ax1.errorbar(data[0], data[3], yerr=data[4], fmt='%s'%colors[i], label=labels[i])
        data0str = '%s - x: ['%labels[i]
        opt = ''
        for jj in data[0]:
            data0str+=' %s %s'%(opt,jj)
            opt=','
        data0str+=']'
        print(data0str)
        data3str = '%s - y: [' % labels[i]
        opt = ''
        for jj in data[3]:
            data3str += ' %s %s' % (opt,jj)
            opt = ','
        data3str += ']'
    #ax1.plot(np.arange(xlim_min - 5, xlim_max + 5), np.ones(np.arange(xlim_min - 5, xlim_max + 5).shape[0]) * 500.,
    #         label='Max. physics rate (500Hz)', linestyle='--', color='k', linewidth=2.)
    #ax2.plot(data[5], np.ones(data[0].shape[0]) * 0.01, label='maximum readout rate')

    ax2.cla()
    ax2.set_xlabel('p.e. Threshold')
    ax1.set_yscale('log')
    ax1.set_xlabel('ADC Threshold (patch 7)')
    ax1.set_ylabel('Trigger Rate [Hz]')

    ax2.set_xlabel('p.e. Threshold (patch 7)', color='r')
    ax2.tick_params('x', colors='r')
    ax2.grid(True, color='r')
    ax1.set_xlim(xlim_min, xlim_max)
    ax2.set_xlim(xlim_min * 4. / 5.6, xlim_max * 4. / 5.6)
    ax1.legend()

    plt.show()



if __name__ == '__main__':
    dataset_np = []
    for d in dataset:
        dataset_np += [rate_calc(d)]
    plt.figure()
    plot(dataset_np, labels=labels)
