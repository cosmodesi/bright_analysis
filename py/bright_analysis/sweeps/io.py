from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import time
import sys
import desitarget.io
import astropy.io.fits as fits
import numpy as np
from   astropy.table import Table

############################################################
def load_all_epoch(sweep_root_dir,epoch=0,filetype='observed'):
    """
    """
    print('Loading data for epoch {:d} under {}'.format(epoch,sweep_root_dir))

    # Walk directories
    iter_sweep_files = desitarget.io.iter_files(sweep_root_dir, '',
                                                ext="{}.fits".format(filetype))

    t0 = time.time()
    data = list()
    for fpath in iter_sweep_files:
        fpath_epoch = int(os.path.split(os.path.split(fpath)[0])[-1])
        if fpath_epoch == epoch:
            data.append(fits.getdata(fpath))
    nfiles = len(data)
    if nfiles == 0:
        __fname__ = sys._getframe().f_code.co_name
        raise Exception('{}({},{},{}) read zero files!'.format(__fname__,
                                                               sweep_root_dir,
                                                               epoch,
                                                               filetype))

    data   = np.concatenate(data)
    t1 = time.time()

    print('Read {:d} rows from {:d} files in {:f}s'.format(len(data),nfiles,t1-t0))

    return data

############################################################
def prepare_sweep_data(sweep_root_dir,data=None,epoch=0,filetype='observed'):
    """
    """
    if data is None:
        # Load the data if not passed directly
        data = load_all_epoch(sweep_root_dir,epoch=epoch,filetype=filetype)
    elif isinstance(data,str):
        # Data is a path to read the data from
        print('Got data path: {}'.format(data))

        # Only accept zipped fits as likely correct path
        if not os.path.splitext(data)[-1] == '.fits.gz':
            raise Exception('Data path does not have .fits.gz extension!')

        # If the data exists, read it; if it doesn't exist, read all epochs as
        # if data=None, but then also write the result out to the specified
        # file.
        if os.path.exists(data):
            print('Reading cached data!')
            data = Table.read(data)
        else:
            raise Exception('Cannot read data from {}, no such path'.format(data))
    elif isinstance(data,Table):
        pass
    else:
        data = np.array(data)
        
    return data
