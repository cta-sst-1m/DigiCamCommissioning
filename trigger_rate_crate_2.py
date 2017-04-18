import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


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
data_2 += [[5,10,15,20,25,30,35,40,200,100,150]]
data_2 += [[113758836,5732173,47232,2166,146,57,56,34,1e-9,4,1]]
data_2 += [[30.074786769,31.170055432,30.954493936,60.430008728,70.508250416,130.075363824,200.144434464,210.590972840,634.365704768,882.296421152,227.9448120]]


data_3 = []
data_3 += [[5,10,15,20,25,30,35,40]]
data_3 += [[404008293,9373104,4837759,4207519,3246574,5519876,1559280,301]]
data_3 += [[93.121154128,30.274769496,20.012270576,20.692336952,20.196262232,48.895958104,47.560179536,139.314085048]]

data_4 = []
data_4 += [[150,20,15,10,5,25,30,35,7,8,6,1,27,9,50]]
data_4 += [[4,7064,326556,36072365,147205348,876,104,53,44341034,59700382,106337529,51243134,286,84254522,30]]
data_4 += [[1132.54651400,30.338796624,30.394805240,30.810795168,30.226575928,80.499492760,60.366143048,
            60.87811032,10.094250800,16.556873968,22.411948040,10.453599344,60.470089776,34.706044744,63.981699160]]



dataset = [data_0,data_3,data_1,data_2,data_4]
colors=['k','b','m','r','k']
style=['--','-','-','-','-']
'''
labels = ['Dark run, config ((1,1,0),(2,1,1)) (light off)',
          'Dark run ((1,0,0),(2,1,2)) analog (light on) ',
          'Dark run ((0,1,0),(1,2,1)) intersil (dark room)',
          'Dark run ((0,1,0),(1,1,2)) intersil (dark room)']
'''
labels = ['Dark run, analog PDP ON, intersil PDP OFF (light off)',
          'Dark run, intersil PDP OFF (dark room not tight)',
          'Dark run, analog PDP ON (dark room not tight) ',
          'Dark run, intersil PDP ON (dark room)',
          'Dark run, 3 crate PDP ON (dark room)']

def rate_calc(data):
    data = np.array(data)
    print(type(data[2]))
    samples = data[2]/4e-9
    p = data[1]/data[2]*4e-9
    q = 1.-p
    data = np.append(data,(data[1]/data[2]).reshape(1,data.shape[1]),axis=0)
    data = np.append(data,(np.sqrt(samples*p*q)/data[2]).reshape(1,data.shape[1]),axis=0)
    data[4][data[1]<1e-8]=1.
    data = np.append(data,(data[0]*4./5.6).reshape(1,data.shape[1]),axis=0)
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
    for i,data in enumerate(datas):
        data = np.array(data)
        ax1.plot(data[0], data[3],color='%s'%colors[i], linestyle='%s'%style[i] ,label=labels[i])
        #ax1.errorbar(data[0], data[3], yerr=data[4], fmt='%s'%colors[i], label=labels[i])
        ax1.fill_between(data[0], data[3]-data[4],data[3]+data[4], alpha= 0.5, edgecolor='%s'%colors[i], facecolor='%s'%colors[i])
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
    ax1.set_ylim(1e-3,250.e6)
    ax2.set_ylim(1e-3,250.e6)
    ax1.legend()

    plt.show()



if __name__ == '__main__':
    dataset_np = []
    for d in dataset:
        dataset_np += [rate_calc(d)]
    plot(dataset_np, labels=labels)
