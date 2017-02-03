import iminuit
import probfit

def fit_binnedLH( x , y , func , p0 , bounds=None, fixed_param=None):
    # func should be a pdf
    pdf = lambda *args , **kwargs : func( x, *args , **kwargs )
    binned_likelihood = probfit.BinnedLH(func, y)
    param = iminuit.describe(pdf)
    p0_dict = {}
    for i,p in enumerate(param):
        if p == 'x': continue
        p0_dict[p]=p0[i+1]
    minuit = iminuit.Minuit(binned_likelihood, **p0_dict)
    minuit.migrad()

    print('Value: {}'.format(minuit.values))
    print('Error: {}'.format(minuit.errors))
# <codecell>

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
