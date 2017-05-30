import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d



data_0 = []
data_0 += [[150,20,15,10,5,25,30,35,7,8,6,1,27,9,50,70,40,60,100]]
data_0 += [[4,7064,326556,36072365,147205348,876,104,53,44341034,59700382,106337529,51243134,286,84254522,30,21,42,29,3]]
data_0 += [[1132.54651400,30.338796624,30.394805240,30.810795168,30.226575928,80.499492760,60.366143048,
            60.87811032,10.094250800,16.556873968,22.411948040,10.453599344,60.470089776,34.706044744,63.981699160,198.401153400,67.108859520,173.124773872,191.026240968]]



dataset = [data_0]
colors=['k','b','m','r','k']
style=['-','-','-','-','-']

labels = ['Dark run, Full Camera']

def rate_calc(data):
    data = np.array(data)
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
def plot(datas,labels,colors=colors,xlim=[0,200]):
    fig = plt.figure(figsize=(14, 12))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    xlim_min = xlim[0]
    xlim_max = xlim[1]
    for i,data in enumerate(datas):
        data = np.array(data)
        ax1.plot(data[0], data[3],color='%s'%colors[i], linestyle='%s'%style[i] ,label=labels[i])
        ax1.errorbar(data[0], data[3], yerr=data[4], fmt='o%s'%colors[i])
        ax1.fill_between(data[0], data[3]-1.*data[4],data[3]+1.*data[4], alpha= 0.5, edgecolor='%s'%colors[i], facecolor='%s'%colors[i])
        print(data[3]-3.*data[4])
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
