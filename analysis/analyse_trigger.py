
#!/usr/bin/env python3

# external modules
import logging,sys
import numpy as np
from tqdm import tqdm
from utils.logger import TqdmToLogger


# internal modules
from data_treatement import trigger
from utils import display, histogram, geometry

__all__ = ["create_histo", "perform_analysis", "display_results", "save"]


def create_histo(options):
    """
    Create a list of ADC histograms and fill it with data
    :param options: a dictionary containing at least the following keys:
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram      (str)
        - 'synch_histo_filename'   : the name of the file containing the histogram      (str)
        - 'file_basename'    : the base name of the input files                   (str)
        - 'directory'        : the path of the directory containing input files   (str)
        - 'file_list'        : the list of base filename modifiers for the input files
                                                                                  (list(str))
        - 'evt_max'          : the maximal number of events to process            (int)
        - 'n_evt_per_batch'  : the number of event per fill batch. It can be
                               optimised to improve the speed vs. memory          (int)
        - 'n_pixels'         : the number of pixels to consider                   (int)
        - 'scan_level'       : number of unique poisson dataset                   (int)
        - 'adcs_min'         : the minimum adc value in histo                     (int)
        - 'adcs_max'         : the maximum adc value in histo                     (int)
        - 'adcs_binwidth'    : the bin width for the adcs histo                   (int)
    :return:
    """

    #temp = np.load(options.cts_directory + 'config/cluster.p')

    #camera.Clusters = temp['']
    #camera.Clusters.ID = np.arange(0, len(camera.Clusters), 1)
    #print(temp)
    #Clusters_patch_list = temp['patches_in_cluster'][0:options.n_clusters]

    triggers = histogram.Histogram(data=np.zeros((len(options.scan_level), options.n_clusters, len(options.threshold))), \
                                 bin_centers=np.array(options.threshold), label='Trigger', \
                                 xlabel='Threshold [ADC]', ylabel='trigger rate [Hz]')



    trigger_spectrum = trigger.run(triggers, options=options)
    triggers.save(options.output_directory + options.histo_filename)

    np.save(arr=trigger_spectrum.ravel(), file=options.output_directory + options.trigger_spectrum_filename)

    return


def perform_analysis(options):
    """
    Perform a simple gaussian fit of the ADC histograms
    :param options: a dictionary containing at least the following keys:
        - 'output_directory' : the directory in which the histogram will be saved (str)
        - 'histo_filename'   : the name of the file containing the histogram
                                                 whose fit contains the gain,sigmas etc...(str)
    :return:
    """

    print('Nothing implemented')

    return


def display_results(options):
    """
    Display the analysis results
    :param options:
    :return:
    """

    # Load the histogram
    triggers = histogram.Histogram(filename=options.output_directory + options.histo_filename)

    # Define Geometry
    #geom = geometry.generate_geometry_0(pixel_list=np.arange(0, 432, 1))

    #display.display_hist(dc_led, options, geom=geom, scale='linear')
    #display.display_hist(triggers, options, scale='log')

    trigger_spectrum = np.load(options.output_directory + options.trigger_spectrum_filename + '.npy')
    import matplotlib.pyplot as plt

    v0 = {'x': [0.0 , 1.0 , 4.0 , 6.0 , 8.0 , 11.0 , 16.0 , 21.0 , 23.0 , 25.0 , 26.0 , 27.0 , 28.0 , 29.0 , 30.0 , 31.0 , 32.0 , 33.0 , 34.0 , 35.0 , 36.0 , 37.0 , 38.0 , 39.0 , 40.0 , 41.0 , 42.0 , 43.0 , 45.0 , 50.0 , 60.0 , 70.0 , 80.0 , 90.0 , 100.0],
          'y': [4901960.80512 , 4901960.768 , 4858981.78278 , 4298339.43079 , 2339967.12702 , 440778.369779 , 231105.753395 , 161552.48678 , 129761.650883 , 92422.8745562 , 65835.9695502 , 37090.1808282 , 14632.2879061 , 4470.84865196 , 1228.91032738 , 295.885599599 , 85.0389470857 , 19.3789749345 , 4.4520041414 , 1.22618061309 , 0.527915533515 , 0.317995357268 , 0.183362285273 , 0.18904744481 , 0.176583493282 , 0.195209850589 , 0.134878302986 , 0.157440325038 , 0.120506609788 , 0.116666666667 , 0.026935301406 , 0.0331510028178 , 0.016570008285 , 0.0166426734791 , 0.0166006789678],
          'label': 'Dark run'
          }
    v1 = {'label':'<NSB rate> = 36 MHz (DAC=275) ',
          'x': [20.0 , 30.0 , 33.0 , 35.0 , 40.0 , 60.0 , 80.0 , 100.0],
          'y': [  202811.096965 , 19298.3269756 , 471.967324458 , 27.4354593718 , 0.246880256492 , 0.0327837143621 , 0.0219795194837 , 0.0133112603057]
            }
    v2 = {'label':'<NSB rate> = 129 MHz (DAC=295)',
          'x': [20.0 , 30.0 , 33.0 , 35.0 , 38.0 , 40.0],
          'y': [ 211616.989699 , 18941.5170433 , 438.623332188 , 28.4312285085 , 0.81161057626 , 0.30352186033]}

    v3 = {'label':'<NSB rate> = 385 MHz (DAC=314)',
          'x': [5.0 , 10.0 , 15.0 , 20.0 , 25.0 , 30.0 , 33.0 , 35.0 , 40.0 , 45.0 , 50.0],
          'y' : [4884752.37657 , 4026069.26625 , 1816363.25696 , 435999.809928 , 152648.233046 , 24366.0373277 , 845.362195339 , 155.659721909 , 9.27380909778 , 0.67516234774 , 0.0973843265094]
          }
    v4 = {'label':'<NSB rate> = 589 MHz (DAC=321)',
          'x': [5.0 , 10.0 , 15.0 , 20.0 , 25.0 , 30.0 , 33.0 , 35.0 , 40.0 , 45.0 , 50.0 , 60.0],
          'y': [4890965.05403 , 4347702.36203 , 2698611.60683 , 846260.054975 , 214739.526733 , 28409.0852957 , 2600.48913261 , 796.950807615 , 62.0006258289 , 4.56169579714 , 0.488678067917 , 0.123917038087]
          }
    """
    v0 = {'x': [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0,
             43.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], 'y': [164805.820513, 153634.000656, 142627.511009, 132430.411047, 119000.096339, 104588.213932, 91701.6916896,
             75866.6455646, 54499.9609324, 27160.6329307, 9526.04878049, 2548.37317397, 653.599605523, 175.835941361,
             42.8524046434, 11.0946502058, 3.19499341238, 0.739341164873, 0.412609341475, 0.116290338211,
             0.070796460177, 0.0390320062451, 0.0444059592797, 0.00624119211762, 0.0411661138236, 0.0], 'label':'v0'
          }
    v1 = {'label':'v1_PATCH7 - 50 sample', 'x': [0.0, 1.0, 4.0, 6.0, 8.0, 11.0, 16.0, 21.0, 23.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0,
                 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
          'y': [4901960.80512, 4901960.768, 4858981.78278, 4298339.43079, 2339967.12702, 440778.369779, 231105.753395,
                 161552.48678, 129761.650883, 92422.8745562, 65835.9695502, 37090.1808282, 14632.2879061, 4470.84865196,
                 1228.91032738, 295.885599599, 85.0389470857, 19.3789749345, 4.4520041414, 1.22618061309,
                 0.527915533515, 0.317995357268, 0.183362285273, 0.18904744481, 0.176583493282, 0.195209850589,
                 0.134878302986, 0.157440325038, 0.120506609788, 0.116666666667, 0.026935301406, 0.0331510028178,
                 0.016570008285, 0.0166426734791, 0.0166006789678]}
    v2 = {'label':'v1_PATCH7 - 25 sample', 'x': [0.0, 4.0, 6.0, 8.0, 11.0, 16.0, 21.0, 25.0, 27.0, 30.0, 32.0],
          'y': [9615384.68322, 9450248.53041, 7572095.37316, 3169461.05984, 714038.130614, 442953.943348,
                 302606.755445, 165658.979768, 53519.1457992, 1296.64553832, 97.7172396084]}
    v3 = {'label':'v1_PATCH7 - 1 sample', 'x': [0.0, 4.0, 8.0, 10.0, 15.0, 20.0, 23.0, 26.0, 28.0, 32.0],
          'y' : [124999992.015, 105247297.039, 12112552.7605, 7749194.29146, 5622069.96693, 3791844.63505,
                 2546819.03021, 432720.353028, 22442.5718136, 84.4557685113]}
    v4 = {'label':'v1_PATCH19 - 50 sample', 'x': [0.0, 1.0, 6.0, 11.0, 16.0, 20.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 43.0, 45.0,
                 50.0, 60.0, 70.0, 80.0, 90.0, 100.0], 'y': [4901960.73385, 4901960.62445, 4900566.55327, 4421450.33498, 804573.645506, 275922.640753,
                 207154.618389, 177501.375771, 147527.780168, 113097.869053, 73225.3643986, 33935.3139188,
                 9617.39645533, 1823.42062824, 218.54446397, 9.89753149851, 1.03840472115, 0.110881893597,
                 0.0831615585141, 0.0387794876454, 0.0498327833271, 0.0132679547492, 0.0166234457078]}
    """
    data_digicam = [v0, v1, v2, v3, v4]
    colors = ['k', 'b', 'g', 'r', 'c']

    fig_1 = plt.figure()
    axis_1 = fig_1.add_subplot(111)
    axis_1.set_title('window width : %d samples' %options.window_width)
    for i, level in enumerate((options.scan_level)):

        axis_1.errorbar(x=triggers.bin_centers, y=triggers.data[level, 0], yerr=triggers.errors[level, 0], label='$f_{nsb} = $ %0.1f [MHz]' %(options.nsb_rate[level]), color=colors[i], linestyle='-.')
        axis_1.plot(data_digicam[i]['x'], data_digicam[i]['y'], label=data_digicam[i]['label'], color=colors[i])

    axis_1.axhline(y=500, color = 'k', label='safe threshold', linestyle='-.')
    axis_1.set_xlabel(triggers.xlabel)
    plt.legend(loc='best', fontsize=12)
    axis_1.set_ylabel(triggers.ylabel)
    axis_1.set_yscale('log')


#    fig = plt.figure()
#    axis = fig.add_subplot(111)
#    axis.hist(trigger_spectrum, bins=np.arange(int(np.min(trigger_spectrum)), int(np.max(trigger_spectrum))+1, 1), normed=True, label='spectrum cluster, 3 [MHz]')
#    plt.xlabel('Threshold [ADC]')
#    plt.legend(loc='best')
#    plt.ylabel('P')
#    axis.set_yscale('log')


    return


def save(options):

    print('Nothing implemented')

    return




