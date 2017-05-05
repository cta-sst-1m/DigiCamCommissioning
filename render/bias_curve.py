import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import logging,sys


def plot_gain_drop(datas,labels,xaxis='pe',colors=['k','b','g','r','c'],style=[],xlim=[0,200], ylim=[5e-3,1e8], title= 'Rates'):
    log = logging.getLogger(sys.modules['__main__'].__name__)
    fig = plt.figure(figsize=(14, 12))
    ax1 = fig.add_subplot(111)
    #ax2 = ax1.twiny()
    if xaxis=='pe':
        index=5
    else:
        index=0
    xlim_min = xlim[0]
    xlim_max = xlim[1]
    ylim_min = ylim[0]
    ylim_max = ylim[1]
    for i,data in enumerate(datas):
        log.info('---> Data for plot %s %s in %s -------------------------------'%(title,labels[i],xaxis))
        #f = UnivariateSpline(data[5], data[3], s=0, k=0)
        #xnew = np.linspace(data[5,0],data[5,-1], num=200, endpoint=True)
        ax1.errorbar(data[index], data[3], yerr=data[4], fmt='o%s'%(colors[i]), markersize = 7)

        ax1.plot(data[index], data[3],color='%s'%colors[i], linestyle='%s'%style[i] ,label=labels[i])
        ax1.fill_between(data[index], data[3]-data[4],data[3]+data[4], alpha= 0.5, edgecolor='%s'%colors[i], facecolor='%s'%colors[i])
        data0str = '%s - x: ['%labels[i]
        opt = ''
        for jj in data[0]:
            data0str+=' %s %s'%(opt,jj)
            opt=','
        data0str+=']'
        log.info(data0str)
        data3str = '%s - y: [' % labels[i]
        opt = ''
        for jj in data[3]:
            data3str += ' %s %s' % (opt,jj)
            opt = ','
        data3str += ']'
        log.info(data3str)
        #ax1.plot(xnew, f(xnew) , color='%s'%colors[i])

    #ax1.plot(np.arange(xlim_min - 5, xlim_max + 5), np.ones(np.arange(xlim_min - 5, xlim_max + 5).shape[0]) * 6000.,
    #         label='Max. DAQ rate (6kHz)', linestyle='--',  color='k',linewidth=2.)
    #ax1.plot(np.arange(xlim_min - 5, xlim_max + 5), np.ones(np.arange(xlim_min - 5, xlim_max + 5).shape[0]) * 500.,
    #         label='Max. physics rate (500Hz)', linestyle='--', color='k', linewidth=2.)

    ax1.set_yscale('log')
    ax1.set_xlabel('p.e. Threshold (Cluster 7)' if index==5 else 'ADC Threshold (Cluster 7)')
    ax1.set_ylabel('Rate [Hz]')
    ax1.set_title(title)

    ax1.set_xlim(xlim_min, xlim_max)
    ax1.set_ylim(ylim_min, ylim_max)
    ax1.legend()

    plt.show()

def load_data(file):
    f = open(file,'r')
    keys = []
    line = ''
    while '# HEADER' not in line:
        line = f.readline()
    keys = f.readline().split('# ')[1].split('\n')[0].split('\t')
    print(file)
    print(keys)
    while '# DATA' not in line:
        line = f.readline()
    readlines = f.readlines()
    print(len(readlines[0].split('\n')[0].split('\t')))
    #lines = []
    #for i in range(len(readlines[0].split('\n')[0].split('\t'))):
    #    lines+=[[]]
    #for l in readlines:
    #    for ii,v in enumerate(l.split('\n')[0].split('\t')):
    #        lines[ii].append(v)
    lines = list(
        map(list, zip(*[l.split('\n')[0].split('\t') for l in readlines])))

    _map_dict = dict(zip(keys, lines))
    print(keys)
    for k in _map_dict.keys():
        print(k)
        _map_dict[k] = [float(x) for x in _map_dict[k]]
    print('t',_map_dict.keys())
    # Sort by threshold
    for k in filter(lambda x: x != 'threshold', _map_dict.keys()):
        _map_dict[k] = np.array([v[1]for v in sorted(zip(_map_dict['threshold'], _map_dict[k]))])
        if k in ['trigger_cnt','readout_cnt'] : _map_dict[k]= _map_dict[k]-1
    # make it a list

    print('tt',_map_dict.keys())
    return _map_dict


def rate_calc(data,nsb=1.e9):
    # data[0,1,2]: threshold, cnt, time
    data = np.array(data)
    print('DATA2 ',type(data[2]),data[2])
    samples = data[2]/4e-9
    p = data[1]/data[2]*4e-9
    q = 1.-p
    # in data[3], put the rate in Hz
    data = np.append(data,(data[1]/data[2]).reshape(1,data.shape[1]),axis=0)
    # in data[4], put the binomial error on rate in Hz
    data = np.append(data,5*(np.sqrt(samples*p*q)/data[2]).reshape(1,data.shape[1]),axis=0)
    # in data[5], put the gain drop corrected NPE
    data = np.append(data,(data[0]*4./5.6 / (1. / (1 + 10000. * float(nsb) * 85e-15))).reshape(1,data.shape[1]),axis=0)
    #sort0 = data[0,:].argsort()
    #for i,d in enumerate(data):
    #    data[i]=data[i,sort0]
    return np.copy(data)


def get_datasets(options):
    datasets = []
    for i,legend in enumerate(options.legends):
        datasets.append([])
        for j, l in enumerate(legend):
            data = load_data(options.base_directory+options.dataset[i][j])
            print(data.keys())
            datasets[-1].append(rate_calc([data['threshold'],data[options.variable[i][j]],data['time']] , nsb=options.NSB[i][j]))

    return datasets

def plot(options):
    plt.ion()
    datasets = get_datasets(options)
    for i,d in enumerate(datasets):
        plot_gain_drop(d,labels=options.legends[i],xaxis=options.xaxis[i],colors=options.color[i],style=options.style[i],xlim=options.x_lim[i], ylim=[5e-3,1e8],title=options.title[i])

