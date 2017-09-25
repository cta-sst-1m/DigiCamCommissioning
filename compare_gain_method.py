import numpy as np
import matplotlib.pyplot as plt
from utils import histogram
from cts_core.cameratestsetup import CTS
from utils.geometry import generate_geometry


def compare_gain_method(dark_spe, full_mpe, angle=0):

    cts_path = '/data/software/CTS/'
    cts = CTS(cts_path + 'config/cts_config_' + str(angle) + '.cfg',
                      cts_path + 'config/camera_config.cfg', angle=angle, connected=True)

    pixel_list = generate_geometry(cts, available_board=None)[1]

    gain_dark = dark_spe.fit_result[pixel_list, 1, 0]
    gain_full_mpe = full_mpe.fit_result[:, 1, 0]

    chi2_dark = dark_spe.fit_chi2_ndof[pixel_list, 0] / dark_spe.fit_chi2_ndof[pixel_list, 1]
    chi2_full_mpe = full_mpe.fit_chi2_ndof[:, 0] / full_mpe.fit_chi2_ndof[:, 1]

    mask = ~np.isnan(gain_dark) * ~np.isnan(gain_full_mpe) * ~np.isnan(chi2_dark) * ~np.isnan(chi2_full_mpe) *\
           (chi2_full_mpe > 2) * (chi2_full_mpe < 10) * (chi2_dark > 3) * (chi2_dark < 35)

    plt.figure()
    plt.hist(gain_dark[mask]/gain_full_mpe[mask], bins='auto')
    plt.xlabel('$G_{dark}/G_{full-mpe}$')

    plt.figure()
    plt.hist(gain_dark[mask], bins=np.arange(0, 50), label='dark')
    plt.hist(gain_full_mpe[mask], bins=np.arange(0, 50), label='full mpe')
    plt.xlabel('$G$')
    plt.legend()

    plt.figure()
    plt.hist((gain_dark[mask] - gain_full_mpe[mask])/gain_full_mpe[mask], bins='auto')
    plt.xlabel('$G_{dark} - G_{full-mpe}$')

    plt.figure()
    plt.scatter(chi2_dark[mask], chi2_full_mpe[mask], s=(gain_dark/gain_full_mpe)[mask]**2)
    plt.xlabel('$\chi^2_{dark}$ /ndf ')
    plt.ylabel('$\chi^2_{full-mpe}$ /ndf')

    plt.figure()
    plt.scatter(chi2_dark[mask], chi2_full_mpe[mask], s=(gain_dark - gain_full_mpe)[mask]**2)
    plt.xlabel('$\chi^2_{dark}$ /ndf ')
    plt.ylabel('$\chi^2_{full-mpe}$ /ndf')


    """
    best_gain = np.zeros(gain_dark.shape)
    chi2_diff = chi2_dark - chi2_full_mpe

    for i in range(best_gain.shape[0]):

        if chi2_diff[i] > 0:

            best_gain[i] = gain_full_mpe[i]

        else:

            best_gain[i] = gain_dark[i]
    """

    return np.mean(gain_full_mpe[mask]/gain_dark[mask]), np.std(gain_full_mpe[mask]/gain_dark[mask])/np.sqrt(gain_dark[mask].shape[0])


if __name__ == '__main__':

    directory = '/data/datasets/CTA/DATA/FULLSEQ/'
    full_mpe_filename = 'full_mpe_0.npz'
    dark_spe_filename = 'dark_spe.npz'

    full_mpe = histogram.Histogram(filename=directory + full_mpe_filename)
    dark_spe = histogram.Histogram(filename=directory + dark_spe_filename)

    average_ratio = compare_gain_method(dark_spe=dark_spe, full_mpe=full_mpe)

    print(average_ratio[0], average_ratio[1])

    plt.show()
