# Adds table of Gaia photometry/astrometry to a DESI MWS mock file

import sys
import os
import numpy      as np
import datetime
from   astropy.io    import fits
from   astropy.table import Table, Column

import astropy.units as u

############################################################
def make_gaia_observables(desi_table,dust='galaxia'):
    """
    """
    import pygaia
    import pygaia.errors.astrometric   as gaia_astro
    import pygaia.errors.spectroscopic as gaia_spectro
    import pygaia.errors.photometric   as gaia_photo

    np.random.seed(2019)
    gaia_table = Table()

    # Intrinsic heliocentric distance in parsecs
    d_parsecs   = desi_table['d_helio']*1000.0

    # Heliocentric RV in km/s
    v_helio_kms = desi_table['v_helio']

    # Intrinsic photometry (using values without extinction)
    g = desi_table['SDSSg_true_nodust']
    r = desi_table['SDSSr_true_nodust']
    i = desi_table['SDSSi_true_nodust']

    # Lupton conversions
    magV_nodust = g - 0.5784*(g-r)-0.0038
    magI_nodust = r - 1.2444*(r-i)-0.3820

    gaia_table.add_column(Column(magV_nodust,name='V_Lupton_nodust',unit=1.0,
                                 description='J-C V-band magnitude converted from SDSS g, gmr following Lupton 2005'))
    gaia_table.add_column(Column(magI_nodust,name='I_Lupton_nodust',unit=1.0,
                                 description='J-C I-band magnitude converted from SDSS r, rmi following Lupton 2005'))

    # Add extinction back (currently in a poor way!). If we care in detail we
    # should know exactly which V and I pygaia wants!
    if dust == 'galaxia':
        # Galaxia dust equations use E(B-V) and coefficients Am/E(B-V).
        #
        # E_schelgel is seletive absorption to star in fiducial bands (V and B in this case).
        #
        #       Am = E_schlegel * A_over_EBV[band]
        # mag_true = mag_nodust + Am
        #
        # Hence:
        # mag_nodust = mag_true - E_schelgel*A_over_EBV[band]

        # Extinction coefficients for SDSS from galaxia documentation
        # These are Am/E(B-V) for Rv = 3.1 according to the Finkbeiner 99 law.
        # These give the coefficients to use with E(B-V)_Schelgel get reddenings
        # consistent with S&F 2011 and Schlafly 2010.
        ext_coeffs_ctio = {'V':3.240,
                           'I':1.962}

        magV = magV_nodust + desi_table['ABV']*ext_coeffs_ctio['V']
        magI = magI_nodust + desi_table['ABV']*ext_coeffs_ctio['I']

    elif dust == 'galfast':
        # GalFast dust equations use Ar and coefficients Am/Ar.
        # Am0 is total absorption to star in fiducial band (r, in this case).
        #
        #       Am = Am0 * reddening[band]
        # mag_true = mag_nodust + Am
        #
        # Hence:
        # mag_nodust = mag_true - Am0*reddening[band]
        
        Ar_coeffs_ctio = {'V':1.1800,
                          'I':0.5066}

        magV = magV_nodust + desi_table['Ar']*Ar_coeffs_ctio['V']
        magI = magI_nodust + desi_table['Ar']*Ar_coeffs_ctio['I']

    gaia_table.add_column(Column(magV,name='V_Lupton',unit=1.0,
                                 description='J-C V-band magnitude converted from SDSS g, gmr following Lupton 2005, with extinction'))
    gaia_table.add_column(Column(magI,name='I_Lupton',unit=1.0,
                                 description='J-C I-band magnitude converted from SDSS r, rmi following Lupton 2005, with extinction'))

    # Gaia routines need this colour as observed (something to do with PSF size)
    VminI = magV - magI
    from pygaia.photometry.transformations import gminvFromVmini, vminGrvsFromVmini  

    # Calculate the value of (G-V) from (V-I)
    # Should this be the true or extincted V-I?
    GmV     = gminvFromVmini(VminI)
    GmVrvs  = vminGrvsFromVmini(VminI)
    magG    = GmV    + magV
    magGrvs = GmVrvs + magV

    gaia_table.add_column(Column(magV,name='G_gaia',unit=1.0,
                                 description='Gaia G apparent magnitude (pygaia)'))
    gaia_table.add_column(Column(magI,name='G_gaia_rvs',unit=1.0,
                                 description='Gaia G apparent magnitude for RVS (pygaia)'))

    # Sky coordinates and intrinsic PMs
    ra     = desi_table['RA']     # Degrees
    dec    = desi_table['DEC']    # Degrees
    pm_ra  = desi_table['pm_RA']  # mas/yr
    pm_dec = desi_table['pm_DEC'] # mas/yr

    import pygaia.astrometry.coordinates as gaia_coordinates
    matrix_equ_to_ecl = gaia_coordinates.Transformations.ICRS2ECL
    equ_to_ecl        = gaia_coordinates.CoordinateTransformation(matrix_equ_to_ecl)
  
    # Note input in radians and output in radians. This is only used for input
    # into the proper motion error routine.
    ecl_lambda, ecl_beta = equ_to_ecl.transformSkyCoordinates(ra*np.pi/180.0,dec*np.pi/180.0)

    # The error in mu_alpha* and the error in mu_delta, in that order, in
    # micro-arcsecond/year. The * on mu_alpha* indicates
    # mu_alpha*=mu_alpha*cos(delta).  These are in MICRO arcsec/yr so convert
    # to MILLI arcsec/yr.
    mu_alpha_star_err, mu_delta_err = gaia_astro.properMotionError(magG,VminI,ecl_beta)
    
    mu_alpha_star_err = mu_alpha_star_err/1000.0
    mu_delta_err      = mu_delta_err/1000.0

    gaia_table.add_column(Column(mu_alpha_star_err,name='pm_RA_gaia_error',unit=1.0,
                                 description='Gaia error on proper motion in RA (mu_alpha_star; pygaia) [mas/yr]'))
    gaia_table.add_column(Column(mu_delta_err,name='pm_DEC_gaia_error',unit=1.0,
                                 description='Gaia error on proper motion in DEC (mu_delta; pygaia) [mas/yr]'))

    # Error-convolved proper motions. Question here whether pm_ra from Galfast
    # is mu_alpha or mu_alpha_star. Give Mario the benefit of the doubt and
    # assume it is mu_alpha_star.
    GALFAST_PMRA_IS_MU_ALPHA_STAR = True
    if GALFAST_PMRA_IS_MU_ALPHA_STAR:
        RA_FIX_FACTOR      = 1.0
    else:
        RA_FIX_FACTOR      = np.cos(dec*np.pi/180.0)

    gaia_mu_alpha_star = np.random.normal(pm_ra*RA_FIX_FACTOR, mu_alpha_star_err)
    gaia_mu_delta      = np.random.normal(pm_dec,mu_delta_err)

    gaia_table.add_column(Column(gaia_mu_alpha_star,name='pm_RA_star_gaia',unit=1.0,
                                 description='Proper motion in RA convolved with Gaia error (mu_alpha_star; pygaia) [mas/yr]'))
    gaia_table.add_column(Column(gaia_mu_delta,name='pm_DEC_gaia',unit=1.0,
                                 description='Proper motion in DEC convolved with Gaia error (mu_delta; pygaia) [mas/yr]'))

    # True parallax in **milli-arcsec**
    true_parallax_arcsec      = 1.0/d_parsecs
    true_parallax_milliarcsec = true_parallax_arcsec*1e3

    # Pygaia error on parallax is returned in micro-arcsec
    # Convert to **milli-arcsec**
    MICRO_TO_MILLI                = 1.0/1e3
    gaia_parallax_err_milliarcsec = gaia_astro.parallaxError(magG,VminI,ecl_beta)*MICRO_TO_MILLI

    # Error convolved parallax
    gaia_parallax_milliarcsec     = np.random.normal(true_parallax_milliarcsec,gaia_parallax_err_milliarcsec)

    gaia_table.add_column(Column(gaia_parallax_err_milliarcsec,name='parallax_gaia_error',unit=1.0,
                                 description='Gaia error on parallax (pygaia) [mas]'))

    gaia_table.add_column(Column(gaia_parallax_milliarcsec,name='parallax_gaia',unit=1.0,
                                 description='Parallax convolved with Gaia error (pygaia) [mas]'))

    # Error convolved RV
    for spectype in ['G0V','F0V','K1III', 'K1IIIMP']:
        gaia_rv_error_kms = gaia_spectro.vradErrorSkyAvg(magV,spectype)
        gaia_rv_kms       = np.random.normal(v_helio_kms,gaia_rv_error_kms)

        gaia_table.add_column(Column(gaia_rv_error_kms,name='v_helio_gaia_error_%s'%(spectype),unit=1.0,
                                     description='Gaia error on heliocentric radial velocity assuming star is type %s (pygaia) [km/s]'%(spectype)))

        gaia_table.add_column(Column(gaia_rv_kms,name='v_helio_gaia_%s'%(spectype),unit=1.0,
                                     description='Heliocentric radial velocity convolved with Gaia error assuming star is type %s (pygaia) [km/s]'%(spectype)))

    return gaia_table

