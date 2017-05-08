import numpy as np
from ctapipe.io.camera import CameraGeometry
from ctapipe.io.camera import find_neighbor_pixels
from astropy import units as u


def generate_geometry(cts, available_board=None, all_camera= False):
    """
    Generate the SST-1M geometry from the CTS configuration
    :param cts: a CTS instance
    :param available_board:  which board per sector are available (dict)
    :return: the geometry for visualisation and the list of "good" pixels
    """
    pix_x = []
    pix_y = []
    pix_id = []
    pix_good_id = []

    for pix in cts.camera.Pixels:
        if pix.ID in cts.pixel_to_led['AC'].keys():
            pix_x.append(pix.center[0])
            pix_y.append(pix.center[1])
            pix_id.append(pix.ID)
        elif all_camera:
            pix_x.append(pix.center[0])
            pix_y.append(pix.center[1])
            pix_id.append(pix.ID)


    neighbors_pix = find_neighbor_pixels(pix_x, pix_y, 30.)
    geom = CameraGeometry(0, pix_id, pix_x * u.mm, pix_y * u.mm, np.ones(1296) * 400., neighbors_pix, 'hexagonal')
    return geom, pix_id


    '''

        if not available_board or pix.fadc in available_board[pix.sector]:
            pix_x.append(pix.center[0])
            pix_y.append(pix.center[1])
            pix_id.append(pix.ID)
            pix_good_id.append(True)
        else:
            pix_x.append(pix.center[0])
            pix_y.append(pix.center[1])
            pix_id.append(pix.ID)
            pix_good_id.append(False)

    pix_good_id = np.array(pix_good_id)
    neighbors_pix = find_neighbor_pixels(pix_x, pix_y, 30.)
    geom = CameraGeometry(0, pix_id, pix_x * u.mm, pix_y * u.mm, np.ones(1296) * 400., neighbors_pix, 'hexagonal')
    return geom, pix_good_id
    '''

def update_pixels_quality(bad_id, pix_good_id):
    """
    Update the pixel quality array
    :param bad_id: list of bad pixels
    :param pix_good_id: pixel quality array
    :return:
    """
    for pix in bad_id:
        pix_good_id[pix] = False
