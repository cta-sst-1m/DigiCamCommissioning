import iminuit
import probfit
from spectra_fit import fit_hv_off
import numpy as np


def gauss_pdf_name(x , sigma):

    """Normalized Gaussian"""
    return 1 / np.sqrt(2 * np.pi) / p[1] * np.exp(-(x - p[0]) ** 2 / 2. / args[1] ** 2)

def gauss_pdf(x, p , labels):
    subdet = dict(zip(labels, p))
    gauss_pdf_name(x,**subdet)

gauss_pdf(np.array([1,2,3]),*(0.,1.))


def fit_binnedLH( x , y , func , p0 , bounds=None, fixed_param=None):
    # func should be a pdf
    print(p0,bounds)
    #pdf = lambda *args , **kwargs : func( x, *args , **kwargs )
    binned_likelihood = probfit.BinnedLH(func, y)
    param = iminuit.describe(func)
    print(param)
    5/0
    p0_dict = {}
    for i,p in enumerate(param):
        if p == 'x': continue
        p0_dict[p]=p0[i+1]
    print(p0_dict)
    minuit = iminuit.Minuit(binned_likelihood, **p0_dict)
    minuit.migrad()

    print('Value: {}'.format(minuit.values))
    print('Error: {}'.format(minuit.errors))
    return minuit,binned_likelihood

import numpy as np
h = np.load('/data/datasets/CTA/20161214/FullChain/adc_hv_off_w7.npz')
y = h['data']
x = h['bin_centers']
p0 = fit_hv_off.p0_func(y[700],x)
bounds = fit_hv_off.bounds_func(y[700],x)
labels = fit_hv_off.labels_func()
print(labels)

m,l = fit_binnedLH(x,y[700],gauss_pdf,p0,bounds=bounds,fixed_param=None)
'''
# Create the minuit
# and give an initial value for the sigma parameter
minuit = iminuit.Minuit(binned_likelihood, sigma=3)
# Remember: minuit.errordef is automatically set to 0.5
# as required for likelihood fits (this was explained above)
binned_likelihood.draw(minuit);

# <codecell>

minuit.migrad()
# Like in all binned fit with long zero tail. It will have to do something about the zero bin
# probfit.BinnedLH does handle them gracefully but will give you a warning;

# <codecell>

# Visually check if the fit succeeded by plotting the model over the data
binned_likelihood.draw(minuit) # uncertainty is given by symmetric Poisson;

# <codecell>

# Let's see the result
print('Value: {}'.format(minuit.values))
print('Error: {}'.format(minuit.errors))

# <codecell>

# That printout can get out of hand quickly
minuit.print_fmin()
# Also print the correlation matrix
minuit.print_matrix()

# <codecell>

# Looking at a likelihood profile is a good method
# to check that the reported errors make sense
minuit.draw_mnprofile('mu');

# <codecell>

# Plot a 2d contour error
# You can notice that it takes some time to draw
# We will this is because our PDF is defined in Python
# We will show how to speed this up later
minuit.draw_mncontour('mu', 'sigma');

# <markdowncell>

# ## Chi^2 fit of a Gaussian distribution
#
# Let's explore another popular cost function chi^2.
# Chi^2 is bad when you have bin with 0.
# ROOT just ignore.
# ROOFIT does something I don't remember.
# But it's best to avoid using chi^2 when you have bin with 0 count.

# <codecell>
'''