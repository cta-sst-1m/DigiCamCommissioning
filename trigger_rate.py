import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

data_0 = np.zeros((6,26))
data_0[0] = np.array([100,90,80,70,60,50,45,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25])
data_0[1] = np.array([0,5,1,8,9,12,12,50, 90 ,194,674,2584,10675,39765,153514,683494,1776577,4185052,9587268,8673146,3167977,3705663,4123883,6801906,4681228,6427427])
data_0[2] = np.array([200,121.459121,160.2258,180.156,230.58,169.5,103.19,121.18,121.73,60.72,60.75,60.30,60.71,60.84,60.24,71.75,65.41,76.79,126.37,94.58,30.29,31.14,31.14,47.69,30.47,39.])


data_1 = np.zeros((6,35))
data_1[0] = np.array([100,90,80,70,60,50,45,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,23,21,16,11,6,1,0,4,8])#31,30,29,28,27,26,25])
data_1[1] = np.array([2,2,2,4,5,14,20,31,22,26,23,31,19,40,66,148,602,2072,6332,23588,77704,291857,1945509,2321467,
                      2013709,6096702,4582408,5374269,7011471,13538356,193393140,194399101,147856182,148037529,85815690])
data_1[2] = np.array([120.477,120.173,120.7,120.66,185.63,120.0,165.966,196.9,163.11,133.19,130.25,163.98,103.62,
                      125.788,125.02,120.7,135.22,106.92,74.46,79.72,63.23,65.28,132.96,62.5898,30.586760,65.965293,
                      35.314039,33.266396,30.338799,30.714656,44.992524,39.657416736,30.162661,30.466780,36.673887])


data_2 = np.zeros((6,11))
data_2[0] = np.array([21,16,11,8,6,4,0,25,30,27,32])
data_2[1] = np.array([11383365,19607215,43971944,126098219,245779365,346878649,239881319,4631014,57520,1107002,6045])
data_2[2] = np.array([37.617683,44.264681,61.582067,39.785382,32.458567,36.705770,24.947657,27.955103952,44.360620,20.684224,61.862165])



data_3 = np.zeros((6,10))
data_3[0] = np.array([32,15,10,8,4,0,26, 20, 28, 23])
data_3[1] = np.array([3985,57736376,44996075,81764284,1104448160,594842163,10030355, 219921412, 679447, 51456224])
data_3[2] = np.array([47.184462,10.269594,5.806548824,6.75037588,10.493838712,4.758737608,23.179762472, 57.998529256, 30.274917048, 20.204114776])

data_4 = np.zeros((6,23))
data_4[0] = np.array([100,90,80,70,60,50,40,45,43,38,36,34,32,30,28,26,24,20,16,11,6,1,0])
data_4[1] = np.array([3,3,9,7,15,20,39470,187,1370,712279,293704,1789867,2771572,3627562,7705606,5893476,16636034,
                      8594071,38645444,181745023,305275985,163540854,116175317])
data_4[2] = np.array([180.4680,226.1087,180.6040,180.5078,180.3718,180.3721,180.60398,180.083927,138.418352112,
                      390.627916,30.538826320,52.743493232,37.849890168,32.074538896,52.231559312,33.202424344,
                      80.307328552,31.146668416,48.032202168,41.105295600,62.294018800,33.362335304,23.699764912])




def rate_calc(data):
    samples = data[2]/4e-9
    p = data[1]/data[2]*4e-9
    q = 1.-p
    data[3] = data[1]/data[2]
    data[4] = np.sqrt(samples*p*q)/data[2]
    data[5] = data[0]*4./5.6

    sort0 = data[0,:].argsort()
    for i,d in enumerate(data):
        data[i]=data[i,sort0]

rate_calc(data_0)
rate_calc(data_1)
rate_calc(data_2)
rate_calc(data_3)
rate_calc(data_4)

'''
def rate(trigger,time):
    samples = time/4e-9
    p = trigger/samples
    q = 1.-p
    errors=    np.sqrt(samples*p*q)
    return trigger/time , errors/time



rate,rate_err = rate(trigger=trigger,time=time)
'''
plt.ion()
def plot(datas,labels,colors=['k','b','g','r','y'],xlim=[0,100]):
    fig = plt.figure(figsize=(14, 12))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    xlim_min = xlim[0]
    xlim_max = xlim[1]
    for i,data in enumerate(datas):
        f = interp1d(data[0], data[3], kind='cubic')
        xnew = np.linspace(data[0,0],data[0,-1], num=200, endpoint=True)
        ax1.errorbar(data[0], data[3], yerr=data[4], fmt='o%s'%colors[i], label=labels[i])
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
        print(data3str)
        ax1.plot(xnew, f(xnew) , color='%s'%colors[i])


    ax1.plot(np.arange(xlim_min - 5, xlim_max + 5), np.ones(np.arange(xlim_min - 5, xlim_max + 5).shape[0]) * 6000.,
             label='Max. DAQ rate (6kHz)', linestyle='--',  color='k',linewidth=2.)

    #ax1.plot(np.arange(xlim_min - 5, xlim_max + 5), np.ones(np.arange(xlim_min - 5, xlim_max + 5).shape[0]) * 500.,
    #         label='Max. physics rate (500Hz)', linestyle='--', color='k', linewidth=2.)
    ax2.plot(data[5], np.ones(data[0].shape[0]) * 0.01, label='maximum readout rate')

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
plt.figure()
plot([data_0,data_1,data_2,data_3,data_4],labels=['v0','v1_PATCH7-50sample','v1_PATCH7-25sample','v1_PATCH7-1sample','v1_PATCH19-50sample'])

plt.figure()
plt.plot(data_1[5][0:-1],-np.diff(data_1[3]))
plt.yscale('log')