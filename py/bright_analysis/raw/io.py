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
TRUTH_CACHE  = None
TARGET_CACHE = None
def match_zcat_truth(input_dir,epoch_dir):
    """
    """
    global TRUTH_CACHE, TARGET_CACHE
    assert(os.path.exists(config_file))

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
    t0 = time.time()
    zcat = Table.read(os.path.join(epoch_dir, 'zcat.fits'))
    t1 = time.time()
    print('Read {} rows from zcat in {}s'.format(len(zcat), t1-t0))

    t0 = time.time()
    itruth_for_izcat = match(zcat['TARGETID'],truth_table['TARGETID'])
    t1 = time.time()
    matched = np.where(itruth_for_izcat >= 0)[0]
    print('Matched {} rows from zcat in {}s'.format(len(matched), t1-t0))

    ttot1 = time.time()
    print('Total time for match_zcat_truth: {}s'.format(ttot1-ttot0))

    return zcat, truth_table, target_table, itruth_for_izcat[matched]


