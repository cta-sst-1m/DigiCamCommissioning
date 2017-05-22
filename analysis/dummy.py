import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def dc_led_fit_function(dc_level, a, b):


    return a * np.exp(b * dc_level)
x = np.array([295, 323, 353])
y = np.array([40, 125, 660])

fit_parameters = curve_fit(dc_led_fit_function, xdata=x, ydata=y, p0=[3, 0.001], bounds=[[2.5, 0.000001], [3.5, 0.001]])

print(fit_parameters[0])

fig = plt.figure()
axis = fig.add_subplot(111)

pixel = 0

axis.semilogy(x, y, linestyle='None', marker='x', color='k', label='data')

axis.semilogy(x, dc_led_fit_function(y, fit_parameters[0][0], fit_parameters[0][1] ), linestyle='--', color='k',
              label='fit')
axis.set_xlabel('DC DAC')
axis.set_ylabel('$f_{nsb}$ [MHz]')
axis.legend(loc='best', prop={'size': 10})

plt.show(   )