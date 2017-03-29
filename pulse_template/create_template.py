import numpy as np
from utils.histogram import Histogram
import matplotlib.pyplot as plt
from scipy.interpolate import LSQUnivariateSpline,UnivariateSpline
plt.ion()
import pickle
from os import listdir
from os.path import isfile, join
import numpy as np
import utils.histogram as histogram

def get_data(filename='/data/datasets/CTA/pulses/templates_hist.npz'):
    onlytxt = [f for f in listdir('/data/datasets/CTA/pulses') if
               isfile(join('/data/datasets/CTA/pulses', f)) and f.endswith('.txt')]
    # read them all

    pes = []
    for f in onlytxt:
        file_txt = open(join('/data/datasets/CTA/pulses', f), 'r')
        if int(f.split('pe')[1].split('.txt')[0]) == 6: continue
        pes.append(int(f.split('pe')[1].split('.txt')[0]))

    pes.sort()
    templates = []
    axes = []
    for pe in pes:
        file_txt = open('/data/datasets/CTA/pulses/pe' + str(pe).zfill(5) + '.txt', 'r')
        templates.append([])
        axes.append([])
        firstline = True
        for l in file_txt.readlines():
            axes[-1].append(int(l.split('\t')[0]))
            templates[-1].append(float(l.split('\t')[1]))

    hist = histogram.Histogram(bin_center_min=-243, bin_center_max=len(axes[-1])-243, bin_width=1, data_shape=(len(pes),),
                               label='Template', xlabel='sample [4ns]', ylabel='ADC')

    # build npz
    sample_array = np.array(axes[-1], dtype=np.int)
    pe_array = np.array(pes, dtype=np.int)
    template = np.zeros((pe_array.shape + sample_array.shape))
    for i, pe in enumerate(pe_array):
        for j, sample in enumerate(sample_array):
            hist.data[i, j] = templates[i][j]
            template[i, j] = templates[i][j]
    hist.fit_axis = pe_array
    hist.data = (hist.data.T - np.mean(hist.data[:,0:200])).T
    hist.save(filename)
    np.savez_compressed('/data/datasets/CTA/pulses/templates_input.npz',
                        templates=template,
                        samples=sample_array,
                        pes=pe_array)


def bspleval(x, knots, coeffs, order, debug=False):
    '''
    Evaluate a B-spline at a set of points.

    Parameters
    ----------
    x : list or ndarray
        The set of points at which to evaluate the spline.
    knots : list or ndarray
        The set of knots used to define the spline.
    coeffs : list of ndarray
        The set of spline coefficients.
    order : int
        The order of the spline.

    Returns
    -------
    y : ndarray
        The value of the spline at each point in x.
    '''

    k = order
    t = knots
    m = np.alen(t)
    npts = np.alen(x)
    B = np.zeros((m-1,k+1,npts))

    if debug:
        print('k=%i, m=%i, npts=%i' % (k, m, npts))
        print('t=', t)
        print('coeffs=', coeffs)

    ## Create the zero-order B-spline basis functions.
    for i in range(m-1):
        B[i,0,:] = np.float64(np.logical_and(x >= t[i], x < t[i+1]))

    if (k == 0):
        B[m-2,0,-1] = 1.0

    ## Next iteratively define the higher-order basis functions, working from lower order to higher.
    for j in range(1,k+1):
        for i in range(m-j-1):
            if (t[i+j] - t[i] == 0.0):
                first_term = 0.0
            else:
                first_term = ((x - t[i]) / (t[i+j] - t[i])) * B[i,j-1,:]

            if (t[i+j+1] - t[i+1] == 0.0):
                second_term = 0.0
            else:
                second_term = ((t[i+j+1] - x) / (t[i+j+1] - t[i+1])) * B[i+1,j-1,:]

            B[i,j,:] = first_term + second_term
        B[m-j-2,j,-1] = 1.0

    if debug:
        plt.figure()
        for i in range(m-1):
            plt.plot(x, B[i,k,:])
        plt.title('B-spline basis functions')

    ## Evaluate the spline by multiplying the coefficients with the highest-order basis functions.
    y = np.zeros(npts)
    for i in range(m-k-1):
        y += coeffs[i] * B[i,k,:]

    if debug:
        plt.figure()
        plt.plot(x, y)
        plt.title('spline curve')
        plt.show()

    return(y)

def get_template(filename='/data/datasets/CTA/pulses/templates_hist.npz'):
    template = Histogram(filename=filename)
    return template

def show_template(idx):
    template = get_template()
    plt.figure()
    plt.step(template.bin_centers,template.data[idx])
    plt.show()


def create_template():
    pkl_file = open('templates_bspline.p', 'wb')
    dict_template = {}
    template = get_template()
    #template_per_sample = Histogram(data=np.swapaxes(np.copy(template.data),0,1), bin_centers=template.fit_axis)
    spl,list_coef = [],[]
    # get the spline for each pe
    for k in template.data:
        spl += [UnivariateSpline(template.bin_centers[200:-1], k[200:-1], s=0, k=5)]
        list_coef += [ list(spl[-1].get_coeffs())]
    dict_template['knots_sample'] = spl[-1].get_knots()-5
    dict_template['coeff_sample'] = np.swapaxes(list_coef, 0, 1)

    dict_template['spline_coeff_func_pe'] = []
    # get the spline of the coeff as function of pe
    for i in range(len(dict_template['coeff_sample'])):
        dict_template['spline_coeff_func_pe'].append(UnivariateSpline(template.fit_axis, dict_template['coeff_sample'][i], s=0, k=3))
    pickle.dump(dict_template, pkl_file)


def estimated_template(pe,start=0,stop=500,step=0.2):
    pkl_file = open('templates_bspline.p', 'rb')

    dict_template = pickle.load(pkl_file)
    xs = np.linspace(start,stop, (stop-start)*1./step)
    coeffs = []
    for coef in range(len(dict_template['coeff_sample'])):
        coeffs.append(dict_template['spline_coeff_func_pe'][coef](float(pe)))
    return xs,bspleval(xs, dict_template['knots_sample'], np.array(coeffs), 5, debug=False)


def plot_pes_template(list_pe):
    template = get_template()
    plt.figure()
    for pe in list_pe:
        x_template,y_template = estimated_template(pe)
        #y_template[0:-11] = y_template[10:-1]
        if pe in template.fit_axis:
            i = np.where(template.fit_axis == pe)[0][0]
            plt.errorbar(template.bin_centers[200:500:1], template.data[i][200:500:1], fmt='o',
                         label='Measured Shape ($N_{\gamma}=%d$)'%pe)
        plt.plot(x_template, y_template, '--', lw=2, label='$f(N_{\gamma}=%d)$'%pe)
    plt.legend()
    plt.show()

def amplitude():
    pes,gain,meas = [],[],[]
    for logpe in np.arange(1.,4.,0.1):
        pe = 10.**logpe
        x_template, y_template = estimated_template(pe, start=0, stop=500)
        pes+=[pe]
        gain+=[np.max(y_template) / pe]
        meas+=[np.max(y_template) *4.72 ]
    plt.figure()
    plt.plot(pes, gain)
    plt.show()
    plt.figure()
    plt.plot(pes, meas)
    plt.show()

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def integral():
    pes,gain,meas,integ,integ_2 = [],[],[],[],[]
    for logpe in np.arange(1.,4.,0.1):
        pe = 10.**logpe
        x_template, y_template = estimated_template(pe, start=0, stop=500)
        pes+=[pe]
        gain+=[np.sum(y_template[:80])/5.]
        meas+=[np.max(y_template)/4.72 ]
        integ+=[np.sum(y_template[:80])/5./22. ]
        if meas[-1]>570:

            good = consecutive(np.where(y_template>0)[0])
            h = np.zeros((len(good),))
            for i,g in enumerate(good):
                h[i] = np.sum(y_template[(g,)]) / 5. / 22.64
            integ_2 += [np.max(h)]
        else:
            integ_2+=[integ[-1]]
    plt.figure()
    plt.plot(pes, gain)
    plt.show()
    plt.figure()
    #plt.plot(pes, integ,label='Integral')
    plt.plot(pes, integ_2,label='Integral until 0')
    plt.plot(pes, meas,label='Peak amplitude')
    plt.ylim(10.,10000.)
    plt.xlim(10.,10000.)
    plt.xlabel('$\mathrm{N_{true}(p.e.)}$')
    plt.ylabel('$\mathrm{N_{evaluated}(p.e.)}$')
    plt.legend()
    plt.show()


def plot_pe(pe):
    plt.figure()
    plt.xlabel('ADC')
    plt.ylabel('A.U.')
    plt.ylim(-400.,4096.)
    plt.xlim(0.,300.)
    template = get_template()
    x_template,y_template = estimated_template(pe,start=0,stop=500)
    #y_template[0:-11] = y_template[10:-1]
    print(template.fit_axis)
    if pe in template.fit_axis:
        i = np.where(template.fit_axis == pe)[0][0]
        plt.errorbar(template.bin_centers[0:300:1], template.data[i][0:300:1], fmt='ok',
                     label='Measured Shape ($N_{\gamma}=%d$)'%pe)
    plt.plot(x_template, y_template, 'r', lw=2, label='$f(N_{\gamma}=%d),G=%0.3f$'%(pe,np.max(y_template)/pe))
    plt.legend()
    plt.show()


def dump_int_dat(pe):
    f = open('template_%s.dat'%str(pe),'w')
    x_template,y_template = estimated_template(pe,start=0,stop=291,step=1)
    y_template = y_template*0.4285714285714286
    f.write('-8.0 0.0\n')
    f.write('-7.0 0.0\n')
    f.write('-6.0 0.0\n')
    f.write('-5.0 0.0\n')
    f.write('-4.0 0.0\n')
    f.write('-3.0 0.0\n')
    f.write('-2.0 0.0\n')
    f.write('-1.0 0.0\n')
    f.write('0.0 0.0\n')
    for i in range(x_template.shape[0]):
        f.write('%0.1f %f\n'%(x_template[i],y_template[i]))
    f.close()



#def template_with_NSB(pe,gain,nsb_baseline_shift,baseline_offset):


'''
pes = np.copy(data_template['pes'])
samples = np.copy(data_template['samples'])
templates = np.copy(data_template['templates'])

templates = templates -np.mean(templates[:,0:219:1],axis=1)[:,None]

templates_by_sample = np.copy(templates)
templates_by_sample = np.swapaxes(templates_by_sample,0,1)
hist_per_pe = histogram(data=templates, bin_centers=samples)
hist_per_sample = histogram(data=templates_by_sample, bin_centers=pes)
plt.ion()

spl = []
# get the knots from the most difficult one
spl_0 = UnivariateSpline(hist_per_pe.bin_centers[220:511:1], hist_per_pe.data[10][220:511:1], s=0,k=3)
'''
'''
spl_0 = []
for i,pe in enumerate(pes):
    spl_0.append(UnivariateSpline(hist_per_pe.bin_centers[220:511:1], hist_per_pe.data[i][220:511:1], s=0,k=3))
    print(pe,len(spl_0[-1].get_knots()))
'''
'''
#plt.figure(0)
#plt.step(hist_per_pe.bin_centers[220:511:1],hist_per_pe.data[-1][220:511:1])
#plt.plot(xs, spl_0(xs), 'k-', lw=1)

knt0 = spl_0.get_knots()
wei0 = spl_0.get_coeffs()
print(knt0)

#knt0 = hist_per_pe.bin_centers[221:510:1]
#print(knt0)
#print(5./0)
list_coef = []
for i,pe in enumerate(pes):
    if i>99:continue
    #t = hist_per_pe.bin_centers[236:509:2]
    spl.append( LSQUnivariateSpline(hist_per_pe.bin_centers[220:511:1], hist_per_pe.data[i][220:511:1], knt0[1:-1]))
    list_coef.append(list(spl[-1].get_coeffs()))


# Get the knots

xs1 = np.linspace(0, 17000, 10000)
fig1,ax1 = plt.subplots(10,10)
list_coef = np.swapaxes(list_coef,0,1)
spl_1 = []
for i in range(len(list_coef)):
    spl_1.append( UnivariateSpline(pes[:-1],list_coef[i] , s=0, k=3))
    if i>99:continue
    plt.subplot(10, 10,i+1)
    plt.step(pes[:-1],list_coef[i])
    plt.plot(xs1, spl_1[-1](xs1), 'k-', lw=3)


xs = np.linspace(hist_per_pe.bin_centers[220:400:1][0], hist_per_pe.bin_centers[220:400:1][-1], 1000)
fig,ax = plt.subplots(1,1,figsize=(8,6))
for i,pe in enumerate(pes):
    if i!=99:continue
    plt.subplot(1, 1,1)
    plt.xlabel('ADC')
    plt.ylabel('A.U.')
    plt.ylim(-400.,4000.)
    plt.xlim(900.,1600.)
    plt.errorbar(hist_per_pe.bin_centers[220:400:1],hist_per_pe.data[i][220:400:1],fmt='ok',label='Measured Shape ($N_{\gamma}=16366$)')
    #plt.plot(xs, spl[i](xs), 'g-', lw=2,label='1D Spline')
    #generate list of coefficient
    coeffs = []
    for coef in range(len(list_coef)):
        coeffs.append(spl_1[coef](float(pe)))
    y_eval = bspleval(xs, knt0, np.array(coeffs), 3, debug=False)
    y_eval[0:-17]=y_eval[16:-1]
    plt.plot(xs, y_eval, 'r--', lw=2,label='$f(N_{\gamma})$')
    plt.legend()


plt.show()


def myspline(pe):
    xs = np.linspace(hist_per_pe.bin_centers[220:400:1][0], hist_per_pe.bin_centers[220:400:1][-1], 1000)
    coeffs = []
    for coef in range(len(list_coef)):
        coeffs.append(spl_1[coef](float(pe)))
    return bspleval(xs, knt0, np.array(coeffs), 3, debug=False)

'''
