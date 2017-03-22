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
def hide_end_ticklabels(ax):
    pl.setp(ax.get_xticklabels()[0],visible=False)
    pl.setp(ax.get_yticklabels()[0],visible=False)
    pl.setp(ax.get_xticklabels()[-1],visible=False)
    pl.setp(ax.get_yticklabels()[-1],visible=False)
    pl.draw()
    return

############################################################
def plot_epoch_distance_ratio(sweep_root_dir=None,
                              data_obs=None,data_uno=None,
                              epoch=0,
                              ymax=0.0,
                              group_disc=False,
                              split_pop=True,
                              savepath=None,**kwargs):
    """
    """
    if savepath is not None:
        assert(os.path.splitext(savepath)[-1] in ['.png','.pdf'])

    # Load the data if not passed directly
    data_obs = prepare_sweep_data(sweep_root_dir,data_obs,epoch,filetype='observed')
    data_uno = prepare_sweep_data(sweep_root_dir,data_uno,epoch,filetype='unobserved')

    bin_size = 0.1 # dex
    dhelio_log_bins = np.arange(-1,3,0.1)
    bin_r           = 10**(dhelio_log_bins)
    bin_volume      = (4.0*np.pi/3.0)*(bin_r**3)
    bin_shell_vol   = bin_volume[1:]-bin_volume[:-1]

    dhelio_obs  = data_obs['d_helio'] 
    dhelio_uno  = data_uno['d_helio'] 
    hist_obs, _  = np.histogram(np.log10(dhelio_obs),bins=dhelio_log_bins)
    hist_uno, _  = np.histogram(np.log10(dhelio_uno),bins=dhelio_log_bins)

    ratio = np.array(hist_obs,dtype=np.float64)/(hist_obs+hist_uno)

    figure = pl.figure(figsize=(5,5))

    axmain  = pl.gca()
    axtop   = axmain.twiny()
    pl.sca(axmain)

    plot_kwargs = dict(c='k',
                       drawstyle='steps-post',
                       lw=1.5,
                       zorder=10,
                       label='All obs.')
    plot_kwargs.update(**kwargs)


    pl.plot(dhelio_log_bins[:-1],ratio,**plot_kwargs)

    if split_pop and 'popid' in data_obs.dtype.names:
        popids = np.unique(data_obs['popid'])
        c      = [pl.cm.viridis(i) for i in np.linspace(0,0.9,len(popids))]
        for i,jpop in enumerate(popids):
            if group_disc and jpop < 7:
                continue


            mask_obs    = data_obs['popid'] == jpop
            mask_uno    = data_uno['popid'] == jpop
            bin_midpoints = dhelio_log_bins[:-1] + 0.5*(dhelio_log_bins[1]-dhelio_log_bins[0])
            hist_obs, _ = np.histogram(np.log10(dhelio_obs[mask_obs]),bins=dhelio_log_bins)
            hist_uno, _ = np.histogram(np.log10(dhelio_uno[mask_uno]),bins=dhelio_log_bins)
    
            ratio = np.array(hist_obs,dtype=np.float64)/(hist_obs+hist_uno)

            plot_kwargs = dict(c=c[i],drawstyle='solid',label='Pop %d'%(jpop))
            plot_kwargs.update(**kwargs)
            pl.plot(bin_midpoints,ratio,**plot_kwargs)

        if group_disc:
            # All disk components
            mask_obs    = (data_obs['popid'] != 8) & (data_obs['popid'] != 7)
            mask_uno    = (data_uno['popid'] != 8) & (data_uno['popid'] != 7)

            bin_midpoints = dhelio_log_bins[:-1] + 0.5*(dhelio_log_bins[1]-dhelio_log_bins[0])
            hist_obs, _ = np.histogram(np.log10(dhelio_obs[mask_obs]),bins=dhelio_log_bins)
            hist_uno, _ = np.histogram(np.log10(dhelio_uno[mask_uno]),bins=dhelio_log_bins)
    
            ratio = np.array(hist_obs,dtype=np.float64)/(hist_obs+hist_uno)
            
            plot_kwargs = dict(c='b',linestyle='--',label='Pop 0-6',lw=1.5)
            pl.plot(bin_midpoints,ratio,**plot_kwargs)


    pl.sca(axtop)
    axtop.set_xlim(5*np.log10(0.1*1000.0)-5.0,5*np.log10(400*1000.0)-5.0)
    axtop.set_ylim(0,max(0.5,ymax))
    #pl.axvline(19.0,ls='--',c='grey',zorder=-20)
    #pl.axvline(20.0,ls='--',c='grey',zorder=-20)
    pl.xlabel('$\mathtt{Distance\ Modulus}$',fontsize=12)
    hide_end_ticklabels(pl.gca())
    
    pl.sca(axmain)
    leg   = pl.legend(loc='upper right',fontsize=8,frameon=True,ncol=2)
    leg.set_zorder(5)
    frame = leg.get_frame()
    frame.set_facecolor('white')

    pl.xlabel('$\mathtt{\log_{10} \, D_{helio}/kpc}$',fontsize=12)
    pl.ylabel(r'$\mathtt{Fraction\ of\ targets\ observed}$',fontsize=12)
   
    pl.xlim(np.log10(0.1),np.log10(400.0))
    pl.ylim(0,max(0.5,ymax))
    pl.grid(color='grey',linestyle=':')
    hide_end_ticklabels(pl.gca())
    
    pl.draw()

    if savepath is not None:
        pl.savefig(savepath,bbox_inches='tight',pad_inches=0.1)
        print('Saved figure to {}'.format(savepath))

    return

############################################################
def plot_epoch_distance(sweep_root_dir=None,data=None,epoch=0,
                        split_pop=True,
                        filetype='observed',
                        group_disc=False,
                        savepath=None,**kwargs):
    """
    """
    if savepath is not None:
        os.path.splitext(savepath)[-1] in ['.png','.pdf']

    # Load the data if not passed directly
    data = prepare_sweep_data(sweep_root_dir,data,epoch,filetype=filetype)

    # Heliocentric distance is in kpc
    bin_size = 0.1 # dex
    dhelio_log_bins = np.arange(-1,3,0.1)

    dhelio  = data['d_helio'] 
    hist, _ = np.histogram(np.log10(dhelio),bins=dhelio_log_bins)

    figure = pl.figure(figsize=(5,5))
    plot_kwargs = dict(c='k',
                       drawstyle='steps-post',
                       lw=1.5,
                       zorder=10,
                       label='All')
    plot_kwargs.update(**kwargs)

    pl.plot(dhelio_log_bins[:-1],np.log10(hist),**plot_kwargs)

    if split_pop and 'popid' in data.dtype.names:
        popids = np.unique(data['popid'])
        c      = [pl.cm.viridis(i) for i in np.linspace(0,0.9,len(popids))]
        for i,jpop in enumerate(popids):
            if group_disc and jpop < 7:
                continue

            mask    = data['popid'] == jpop
            bin_midpoints = dhelio_log_bins[:-1] + 0.5*(dhelio_log_bins[1]-dhelio_log_bins[0])
            hist, _ = np.histogram(np.log10(dhelio[mask]),bins=dhelio_log_bins)

            plot_kwargs = dict(c=c[i],drawstyle='solid',label='Pop %d'%(jpop))
            plot_kwargs.update(**kwargs)
            pl.plot(bin_midpoints,np.log10(hist),**plot_kwargs)

        if group_disc:
            mask    = (data['popid'] != 7) & (data['popid'] != 8)
            bin_midpoints = dhelio_log_bins[:-1] + 0.5*(dhelio_log_bins[1]-dhelio_log_bins[0])
            hist, _ = np.histogram(np.log10(dhelio[mask]),bins=dhelio_log_bins)

            plot_kwargs = dict(c='b',linestyle='--',label='Pop 0-6',lw=1.5)
            plot_kwargs.update(**kwargs)
            pl.plot(bin_midpoints,np.log10(hist),**plot_kwargs)

    pl.legend(loc='upper right',fontsize=8,frameon=False,ncol=2)

    pl.xlabel('$\mathtt{\log_{10} \, D_{helio}/kpc}$',fontsize=12)
    pl.ylabel(r'$\mathtt{d\,\log_{10}\,N \,\, {[bins\, of\, %2.1f\, dex]}}$'%(bin_size),fontsize=12)
   
    pl.xlim(-1,2.5)
    pl.ylim(1,7)
    pl.grid(color='grey',linestyle=':')
    hide_end_ticklabels(pl.gca())
    
    pl.draw()

    if savepath is not None:
        pl.savefig(savepath,bbox_inches='tight',pad_inches=0.1)
        print('Saved figure to {}'.format(savepath))
    return

############################################################
def plot_epoch_distance_cumulative(sweep_root_dir=None,data=None,epoch=0,
                                   split_pop=True,filetype='observed',
                                   group_disc=False,
                                   savepath=None,**kwargs):
    """
    """
    if savepath is not None:
        os.path.splitext(savepath)[-1] in ['.png','.pdf']

    # Load the data if not passed directly
    data = prepare_sweep_data(sweep_root_dir,data,epoch,filetype=filetype)

    # Heliocentric distance is in kpc
    dhelio  = data['d_helio'] 
    rsort   = np.argsort(dhelio)

    figure = pl.figure(figsize=(5,5))

    axmain  = pl.gca()
    axtop   = axmain.twiny()

    plot_kwargs = dict(c='k',
                       drawstyle='solid',
                       lw=2,
                       zorder=10,
                       label='All')
    plot_kwargs.update(**kwargs)

    axmain.plot(np.log10(dhelio[rsort]),np.log10(len(dhelio)-np.arange(0,len(dhelio))),**plot_kwargs)
    axtop.plot(5*np.log10(1000.0*dhelio[rsort])-5.0,np.log10(len(dhelio)-np.arange(0,len(dhelio))),**plot_kwargs)
    
    pl.sca(axmain)
    if split_pop and 'popid' in data.dtype.names:
        popids = np.unique(data['popid'])
        c      = [pl.cm.viridis(i) for i in np.linspace(0,0.8,len(popids))]
        for i,jpop in enumerate(popids):
            if group_disc and jpop < 7:
                continue

            mask    = data['popid'] == jpop
            nmask   = np.sum(mask)
            dhelio  = data['d_helio'][mask]
            rsort   = np.argsort(dhelio)
            
            plot_kwargs = dict(c=c[i],linestyle='solid',label='Pop %d'%(jpop))
            plot_kwargs.update(**kwargs)
            pl.plot(np.log10(dhelio[rsort]),np.log10(nmask-np.arange(0,nmask)),**plot_kwargs)
        
        if group_disc:
            # All disk components
            mask    = (data['popid'] != 8) & (data['popid'] != 7)
            nmask   = np.sum(mask)
            dhelio  = data['d_helio'][mask]
            rsort   = np.argsort(dhelio)

            plot_kwargs = dict(c='b',linestyle='--',label='Pop 0-6',lw=1.5)
            plot_kwargs.update(**kwargs)
            axmain.plot(np.log10(dhelio[rsort]),np.log10(nmask-np.arange(0,nmask)),**plot_kwargs)

    pl.sca(axtop)
    axtop.set_xlim(5*np.log10(0.1*1000.0)-5.0,5*np.log10(400*1000.0)-5.0)
    axtop.set_ylim(2,7.5)
    axtop.set_xlabel('$\mathtt{Distance\ Modulus}$',fontsize=12)
    axtop.set_yticklabels(axtop.get_yticks(),family='monospace')
    hide_end_ticklabels(pl.gca())

    pl.sca(axmain)
    pl.legend(loc='upper right',fontsize=8,frameon=False,ncol=2,columnspacing=0.6)

    pl.xlabel('$\mathtt{\log_{10} \ D_{helio}/kpc}$',fontsize=12)
    pl.ylabel(r'$\mathtt{\log_{10}\,N(>D)}$',fontsize=12)
    axmain.set_yticklabels(axtop.get_yticks(),family='monospace')

    pl.title('Observed Targets',y=1.12,fontsize=12)

    axmain.set_xlim(np.log10(0.1),np.log10(400))
    axmain.set_ylim(2,7.5)
    pl.grid(color='grey',linestyle=':')
    hide_end_ticklabels(pl.gca())
 
    pl.draw()

    if savepath is not None:
        pl.savefig(savepath,bbox_inches='tight',pad_inches=0.1)
        print('Saved figure to {}'.format(savepath))
    return

############################################################
def plot_epoch_distance_ratio_cumulative(sweep_root_dir=None,
                                         data_obs=None,data_uno=None,
                                         epoch=0,
                                         split_pop=True,group_disc=False,
                                         savepath=None,**kwargs):
    """
    """
    if savepath is not None:
        os.path.splitext(savepath)[-1] in ['.png','.pdf']

    # Load the data if not passed directly
    data_obs = prepare_sweep_data(sweep_root_dir,data_obs,epoch,filetype='observed')
    data_uno = prepare_sweep_data(sweep_root_dir,data_uno,epoch,filetype='unobserved')

    dhelio_obs  = data_obs['d_helio'] 
    rsort_obs   = np.argsort(dhelio_obs)
    n_obs       = np.arange(1,len(rsort_obs)+1)

    dhelio_uno    = data_uno['d_helio'] 
    rsort_uno     = np.argsort(dhelio_uno)
    n_uno         = np.arange(1,len(rsort_uno)+1)
    n_uno_at_robs = np.interp(dhelio_obs[rsort_obs],dhelio_uno[rsort_uno],n_uno)

    # Fraction of stars observed to a given distance/
    ratio = n_obs/n_uno_at_robs

    figure = pl.figure(figsize=(5,5))

    axmain  = pl.gca()
    axtop   = axmain.twiny()

    plot_kwargs = dict(c='k',
                       drawstyle='solid',
                       lw=2,
                       zorder=10,
                       label='All')
    plot_kwargs.update(**kwargs)

    axmain.plot(np.log10(dhelio_obs[rsort_obs]), ratio,**plot_kwargs)
    axtop.plot(5*np.log10(dhelio_obs[rsort_obs]*1000.0) - 5.0, ratio,**plot_kwargs)

    pl.sca(axmain)
    if split_pop and 'popid' in data_obs.dtype.names:
        popids = np.unique(data_obs['popid'])
        c      = [pl.cm.viridis(i) for i in np.linspace(0,0.8,len(popids))]
        for i,jpop in enumerate(popids):
            if group_disc and jpop < 7:
                continue

            mask_obs    = data_obs['popid'] == jpop
            mask_uno    = data_uno['popid'] == jpop

            dhelio_obs  = data_obs['d_helio'][mask_obs]
            rsort_obs   = np.argsort(dhelio_obs)
            n_obs       = np.arange(1,len(rsort_obs)+1)

            dhelio_uno    = data_uno['d_helio'][mask_uno]
            rsort_uno     = np.argsort(dhelio_uno)
            n_uno         = np.arange(1,len(rsort_uno)+1)
            n_uno_at_robs = np.interp(dhelio_obs[rsort_obs],dhelio_uno[rsort_uno],n_uno)
    
            ratio = n_obs/n_uno_at_robs

            plot_kwargs = dict(c=c[i],linestyle='solid',label='Pop %d'%(jpop))
            plot_kwargs.update(**kwargs)
            axmain.plot(np.log10(dhelio_obs[rsort_obs]),ratio,**plot_kwargs)
        
        if group_disc:
            # All disk components
            mask_obs    = (data_obs['popid'] != 8) & (data_obs['popid'] != 7)
            mask_uno    = (data_uno['popid'] != 8) & (data_uno['popid'] != 7)

            dhelio_obs  = data_obs['d_helio'][mask_obs]
            rsort_obs   = np.argsort(dhelio_obs)
            n_obs       = np.arange(1,len(rsort_obs)+1)

            dhelio_uno    = data_uno['d_helio'][mask_uno]
            rsort_uno     = np.argsort(dhelio_uno)
            n_uno         = np.arange(1,len(rsort_uno)+1)
            n_uno_at_robs = np.interp(dhelio_obs[rsort_obs],dhelio_uno[rsort_uno],n_uno)
            
            ratio = n_obs/n_uno_at_robs

            plot_kwargs = dict(c='b',linestyle='--',label='Pop 0-6',lw=1.5)
            plot_kwargs.update(**kwargs)
            axmain.plot(np.log10(dhelio_obs[rsort_obs]),ratio,**plot_kwargs)

    pl.sca(axtop)
    axtop.set_xlim(5*np.log10(0.1*1000.0)-5.0,5*np.log10(400*1000.0)-5.0)
    axtop.set_ylim(0,1)
    axtop.set_xlabel('$\mathtt{Distance\ Modulus}$',fontsize=12)
    axtop.set_yticklabels(axtop.get_yticks(),family='monospace')
    hide_end_ticklabels(pl.gca())


    pl.sca(axmain)
    pl.legend(loc='upper right',fontsize=8,frameon=False,ncol=2,columnspacing=0.6)

    pl.xlabel('$\mathtt{\log_{10} \ D_{helio}/kpc}$',fontsize=12)
    pl.ylabel(r'$\mathtt{N_{obs}(<D)/N_{tot}(<D)}$',fontsize=12)
    axmain.set_yticklabels(axtop.get_yticks(),family='monospace')

    pl.title('Observed/Total',y=1.12,fontsize=12)

    axmain.set_xlim(np.log10(0.1),np.log10(400))
    axmain.set_ylim(0,1)
    pl.grid(color='grey',linestyle=':')
    hide_end_ticklabels(pl.gca())
 
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
    parser.add_argument('-s','--savepath',default=None)
    args = parser.parse_args()
    plot_epoch_summary(args.sweep_root_dir,epoch=args.epoch,
                       savepath=args.savepath)
  
