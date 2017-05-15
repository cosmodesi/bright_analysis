from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import healpy as hp
import numpy as np

############################################################
def list_healpix_areas():
    """
    """
    print('{nside:6s} | {area:16s} | {npix:12s}'.format(nside ='NSIDE',
                                                        area  ='AREA (sq. deg.)',
                                                        npix  ='NPIX'))
    for i in range(0,10):
        nside = 2**i
        print('{nside:6d} | {area:16.4f} | {npix:12d}'.format(nside=nside,
                                                              area=hp.nside2pixarea(nside,degrees=True),
                                                              npix=hp.nside2npix(nside)))
    return

############################################################
def ra_dec_ang2pix(nside,ra,dec,nest=True):
    """
    Converts (ra, dec) in degrees to Healpix pixels, using ang2pix, assuming
    nest and taking into account differences in coordinate conventions.
    """
    theta = np.deg2rad(90.0-dec)
    phi   = np.deg2rad(ra)
    return hp.ang2pix(nside,theta,phi,nest=nest)

############################################################
def ra_dec_pix2ang(nside,ipix,nest=True):
    """
    Converts Healpix ipix to (ra, dec) in degrees, using pix2ang, assuming
    nest and taking into account differences in coordinate conventions.
    """
    theta, phi = hp.pix2ang(nside,ipix,nest=nest)
    dec = -np.rad2deg(theta) + 90.0
    ra  = np.rad2deg(phi)
    return ra, dec

############################################################
def pix_counts(nside,ra,dec):
    """Count in each pixel"""
    all_ipix           = ra_dec_ang2pix(nside,ra,dec)
    ipix,ipix_count    = np.unique(all_ipix,return_counts=True)
    allpix_count       = np.zeros(hp.nside2npix(nside),dtype=np.float32)
    allpix_count[ipix] = ipix_count
    return allpix_count

############################################################
def l_b_ang2pix(nside,l,b,nest=True):
    """
    Functionally identical to ra_dec_ang2pix().
    """
    return ra_dec_ang2pix(nside,l,b,nest=nest)


