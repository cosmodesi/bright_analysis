import numpy as np
import sys
import os
import time

from astropy.table import Table

from bright_analysis.util.match import match

############################################################
def read_config_file(config_file):
    """
    Read parameters for quicksurvey.
    """
    import yaml
    with open(config_file,'r') as pfile:
        params = yaml.load(pfile)
    return config_file

############################################################
def read_all_tiles(epoch_dir):
    """
    Read all fiber maps (tiles) in a given epoch dir.
    """
    import glob

    tilefiles = glob.glob('{}/fiberassign/tile_*.fits'.format(epoch_dir))
    ntiles    = len(tilefiles)
    print("Have {} tile files".format(ntiles))

    # Read all the tiles
    t0 = time.time()
    tiledata = list()
    for tilefile in tilefiles:
        f = fits.open(tilefile,'readonly',memmap=False)
        tiledata.append(f[1].data)
        f.close()
    t1 = time.time()

    print('Read tile data in {}s'.format(t1-t0))

    return tilefiles,tiledata

############################################################
def load_zcat(epoch_dir):
    """
    Load redshift catalogue for this epoch
    """
    t0 = time.time()
    zcat = Table.read(os.path.join(epoch_dir, 'zcat.fits'))
    t1 = time.time()
    print('Read {} rows from zcat in {}s'.format(len(zcat), t1-t0))
    return zcat


############################################################
TRUTH_CACHE  = None
TARGET_CACHE = None
def match_zcat_truth(input_dir,epoch_dir):
    """
    """
    global TRUTH_CACHE, TARGET_CACHE

    # Time the whole routine
    print('Loading truth and zcat...'.format())
    ttot0 = time.time()

    # Load truth table 
    if TRUTH_CACHE is None:
        t0 = time.time()
        truth_table = Table.read(os.path.join(input_dir, 'truth.fits'))
        t1 = time.time()
        print('Read {} rows from truth in {}s'.format(len(truth_table), t1-t0))
        TRUTH_CACHE = truth_table
    else:
        truth_table = TRUTH_CACHE
        print('Have {} rows from truth in memory'.format(len(truth_table)))

    # Load Target Table
    if TARGET_CACHE is None:
        t0 = time.time()
        target_table = Table.read(os.path.join(input_dir, 'targets.fits'))
        t1 = time.time()
        print('Read {} rows from target in {}s'.format(len(target_table), t1-t0))
        TARGET_CACHE = target_table
    else:
        target_table = TARGET_CACHE
        print('Have {} rows from target in memory'.format(len(target_table)))

    # Load redshift catalogue for this epoch
    zcat = load_zcat(epoch_dir)

    t0 = time.time()
    itruth_for_izcat = match(zcat['TARGETID'],truth_table['TARGETID'])
    t1 = time.time()
    matched = np.where(itruth_for_izcat >= 0)[0]
    print('Matched {} rows from zcat in {}s'.format(len(matched), t1-t0))

    ttot1 = time.time()
    print('Total time for match_zcat_truth: {}s'.format(ttot1-ttot0))

    return zcat, truth_table, target_table, itruth_for_izcat[matched]


