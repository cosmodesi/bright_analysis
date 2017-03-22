from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import time
import sys
import healpy as hp
import desiutil.plots as desiplot
import desitarget.io
import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as pl
from   astropy.table import Table
from   matplotlib import rcParams

rcParams['font.family'] = 'monospace'

from bright_analysis.sweeps.io import prepare_sweep_data

############################################################
def ra_dec_ang2pix(nside,ra,dec,nest=False):
    """
    """
    theta = (dec+90.0)*np.pi/180.0
    phi   = ra*np.pi/180.0
    return hp.ang2pix(nside,theta,phi,nest=nest)
    
############################################################
def plot_epoch_summary(sweep_root_dir,data=None,epoch=0,
                       lgrid_deg=1,
                       filetype='observed',
                       savepath=None):
    """
    """
    if savepath is not None:
        os.path.splitext(savepath)[-1] in ['.png','.pdf']

    # Load the data if not passed directly
    data = prepare_sweep_data(sweep_root_dir,data,epoch,filetype=filetype)
    
    ra_edges  = np.linspace(0,360,360.0/float(lgrid_deg))
    dec_edges = np.linspace(-90,90,180.0/float(lgrid_deg))
    grid, xbin, ybin = np.histogram2d(data['RA'],data['DEC'],bins=(ra_edges,dec_edges))

    # Units of density are stars per square degree
    grid = grid/(lgrid_deg**2)

    figure = pl.figure(figsize=(9,7))
    desiplot.plot_grid_map(grid.T,ra_edges,dec_edges,label='Epoch %d, N stars per sq. deg.'%(epoch))

    pl.draw()
    
    if savepath is not None:
        pl.savefig(savepath,bbox_inches='tight',pad_inches=0.1)
        print('Saved figure to {}'.format(savepath))
    return

############################################################
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sweep_root_dir')
    parser.add_argument('-e','--epoch',default=0,  type=int)
    parser.add_argument('-l','--lgrid',default=1.0,type=float)
    parser.add_argument('-s','--savepath',default=None)
    args = parser.parse_args()
    plot_epoch_summary(args.sweep_root_dir,epoch=args.epoch,
                       lgrid_deg=args.lgrid,
                       savepath=args.savepath)
  
