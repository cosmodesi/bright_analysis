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
def sweep_mock_roots(input_yaml,sweep_root_dir='./output/sweeps'):
    """
    Returns a dict, keys are the input.yaml names of each source class and
    values are the path to the sweeps for the corresponding mocks. Use to get
    the input paths for the other sweep io routines.

    Args:
        input_yaml     :   path to the survey configuration yaml
        sweep_root_dir :   top of the sweep directory tree

    """
    import yaml
    with open(input_yaml,'r') as f:
        param = yaml.load(f)

    roots = list()
    for source,v in param['sources'].items():
        # Don't use os.path.join because of behaviour on leading / for
        # v['root_mock_dir'].
        sweep_subroot = os.path.normpath(sweep_root_dir +
                                         os.path.sep +
                                         v['root_mock_dir'])
        roots.append((source,sweep_subroot))
    return dict(roots)

############################################################
def load_all_epoch(sweep_mock_root_dir,epoch=0,filetype='observed'):
    """
    As written this will only work if passed a sweep *sub*-root path (i.e. the
    node above on particular type of mock) rather than the base sweep root
    (output/sweep).
    """
    print('Loading data for epoch {:d} under {}'.format(epoch,sweep_mock_root_dir))

    # Walk directories
    iter_sweep_files = desitarget.io.iter_files(sweep_mock_root_dir, '',
                                                ext="{}.fits".format(filetype))

    t0 = time.time()
    data = list()
    for fpath in list(iter_sweep_files):
        fpath_epoch = int(os.path.split(os.path.split(fpath)[0])[-1])
        if fpath_epoch == epoch:
            data.append(fits.getdata(fpath))
    nfiles = len(data)
    if nfiles == 0:
        __fname__ = sys._getframe().f_code.co_name
        raise Exception('{}({},{},{}) read zero files!'.format(__fname__,
                                                               sweep_mock_root_dir,
                                                               epoch,
                                                               filetype))

    data   = np.concatenate(data)
    t1 = time.time()

    print('Read {:d} rows from {:d} files in {:f}s'.format(len(data),nfiles,t1-t0))

    return data

############################################################
def prepare_sweep_data(sweep_mock_root_dir,data=None,epoch=0,filetype='observed'):
    """
    """
    if data is None:
        # Load the data if not passed directly
        data = load_all_epoch(sweep_mock_root_dir,epoch=epoch,filetype=filetype)
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
        # The data was passed in and was an astropy table, pass it back out
        # again with no change.
        print('Using existing table with {:d} rows'.format(len(data)))
        pass
    else:
        # The data was passed in but was not an astropy table. Pass it back out
        # again.
        data = np.array(data)
        print('Using existing table with {:d} rows'.format(len(data)))
        
    return data

############################################################
def combine_sweep_files(source_name,input_yaml,sweep_root_dir,
                        output_dir=None,
                        data=None,epoch=0,filetype='observed'):
    """
    Concatenates all the sweep files for a given target class.
    """
    roots = sweep_mock_roots(input_yaml,sweep_root_dir=sweep_root_dir)

    if not source_name in roots.keys():
        raise Exception("No source class {} in config {}".format(source_name,input_yaml))

    sweep_mock_root_dir = roots[source_name]
    t = prepare_sweep_data(sweep_mock_root_dir,data=data,epoch=epoch,filetype=filetype)

    if output_dir is None:
        output_dir = os.path.join(sweep_root_dir,'combined',source_name)
    else:
        output_dir = os.path.join(output_dir,source_name)
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    output_name = '{}_{}_epoch{:d}.fits.gz'.format(source_name,filetype,epoch)
    output_path = os.path.join(output_dir,output_name)
    Table(t).write(output_path,overwrite=True)
    print('Wrote combined sweep file to {}'.format(output_path))
    return

