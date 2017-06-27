import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splrep, splev

if __name__ == '__main__':

    def wave(x, y):

        return np.sin(x - y)

    x = np.linspace(-np.pi, np.pi, num=100)
    y = np.linspace(-np.pi, np.pi, num=100)
    z = wave(x, y)

    X, Y = np.meshgrid(x, y)
    Z = wave(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z)
    plt.show()

    plt.figure()
    plt.plot(x, wave(x, 0), label='x')
    plt.plot(y, wave(0, y), label='y')
    plt.legend()
    plt.show()