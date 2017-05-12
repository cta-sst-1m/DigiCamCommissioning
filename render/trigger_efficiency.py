import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline,UnivariateSpline
import logging,sys
from utils import histogram

def plot_trigger_efficiency(datas,labels,colors=['k','b','g','r','c'],style=[],xlim=[0,200], ylim=[5e-3,1e8], title= 'Rates'):
    log = logging.getLogger(sys.modules['__main__'].__name__)
    fig = plt.figure(figsize=(14, 12))
    ax1 = fig.add_subplot(111)
    xlim_min = xlim[0]
    xlim_max = xlim[1]
    ylim_min = float(ylim[0])
    ylim_max = float(ylim[1])
    for i,data in enumerate(datas):
        log.info('---> Data for plot %s %s -------------------------------'%(title,labels[i]))

        f = UnivariateSpline(data[0], data[1], w = 1./data[2], s=3 , k=1)
        xnew = np.linspace(data[0][0],data[0][-1]+10, num=400, endpoint=True)
        ynew = f(xnew)
        pt50 = xnew[np.argmin(np.abs(ynew-50))]
        print('******************* Point at 50% efficiency:%d',pt50)

        ax1.errorbar(data[0], data[1], yerr=data[2], fmt='o',color='%s'%(colors[i]), ecolor='%s'%(colors[i]), markersize = 6, linewidth=2)
        #ax1.plot(data[0], data[1],color='%s'%colors[i], linestyle='%s'%style[i] ,label=labels[i], linewidth = 2)
        ax1.plot(xnew, ynew, color='%s' % colors[i], linestyle='%s' % style[i], label=labels[i])
        ax1.fill_between(data[0], data[1]-data[2],data[1]+data[2], alpha= 0.3, edgecolor='%s'%colors[i], facecolor='%s'%colors[i])
        data0str = '%s - x: ['%labels[i]
        opt = ''
        for jj in data[0]:
            data0str+=' %s %s'%(opt,jj)
            opt=','
        data0str+=']'
        log.info(data0str)
        data3str = '%s - y: [' % labels[i]
        opt = ''
        for jj in data[1]:
            data3str += ' %s %s' % (opt,jj)
            opt = ','
        data3str += ']'
        log.info(data3str)
        data4str = '%s - y_err: [' % labels[i]
        opt = ''
        for jj in data[2]:
            data4str += ' %s %s' % (opt,jj)
            opt = ','
        data4str += ']'
        log.info(data4str)
        #ax1.plot(xnew, f(xnew) , color='%s'%colors[i])

    #ax1.plot(np.arange(xlim_min - 5, xlim_max + 5), np.ones(np.arange(xlim_min - 5, xlim_max + 5).shape[0]) * 6000.,
    #         label='Max. DAQ rate (6kHz)', linestyle='--',  color='k',linewidth=2.)
    #ax1.plot(np.arange(xlim_min - 5, xlim_max + 5), np.ones(np.arange(xlim_min - 5, xlim_max + 5).shape[0]) * 500.,
    #         label='Max. physics rate (500Hz)', linestyle='--', color='k', linewidth=2.)

    ax1.set_xlabel('<N(p.e.)> in a single cluster (21 pixels)')
    ax1.set_ylabel('Trigger Efficiency [%]')
    ax1.xaxis.get_label().set_ha('right')
    ax1.xaxis.get_label().set_position((1, 0))
    ax1.yaxis.get_label().set_ha('right')
    ax1.yaxis.get_label().set_position((0, 1))
    ax1.set_title(title)
    ax1.set_xlim(xlim_min, xlim_max)
    ax1.set_ylim(ylim_min, ylim_max)
    ax1.legend()

    plt.show()

def load_data(file,evt_per_level):


    # Load the histogram
    triggers = histogram.Histogram(filename=file)
    test = np.argmax(np.sum(triggers.data,axis=0))
    trig = triggers.data[:,test]

    # Define Geometry
    p = trig /evt_per_level
    efficiencies = p
    q = 1. - p
    # in data[3], put the rate in Hz
    # in data[4], put the binomial error on rate in Hz
    errors = np.sqrt(evt_per_level * p * q)/evt_per_level*100.

    return p*100.,errors



def get_datasets(options):
    datasets = []
    for i,legend in enumerate(options.legends):
        datasets.append([])
        for j, l in enumerate(legend):
            eff,err= load_data(options.base_directory+options.dataset[i][j],options.evt_per_level)
            datasets[-1].append([np.array(options.levels),eff,err])
    return datasets

def plot(options):
    plt.ion()
    print(options.__dict__.keys())
    datasets = get_datasets(options)
    for i,d in enumerate(datasets):
        plot_trigger_efficiency(d,labels=options.legends[i],colors=options.color[i],style=options.style[i],xlim=options.x_lim[i],ylim=options.y_lim[i],title=options.title[i])

