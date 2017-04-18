"""

Creates a directory structure parallel to that of the mock. Within this
structure, generates the following files:

zcat_sweep* : these have the same format as the original mock files, but only
contain rows for targets that were observed on at least one epoch.

row_obseved_{epoch} : These have the same number of rows as the input mock
file. They contain one boolean column, which is True if the target was observed
at {epoch}.

In some cases, brick files in the mock will have no observed targets. These
will get empty zcat_sweep files, which can cause problems.


Let's say the mocks are found under a path like this:

/project/projectdirs/desi/mocks/mws/galaxia/alpha/v0.0.3/bricks/005/0050p000/allsky_galaxia_desi_0050p000.fits

This part is specified as mock root in the yaml:
/project/projectdirs/desi/mocks/mws/galaxia/alpha/v0.0.3/bricks


"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import sys
import os
import yaml
import glob
import time
from astropy.table import Table, Column
from desitarget.mock.io import decode_rownum_filenum
from astropy.io import fits

import desimodel
import desimodel.footprint as footprint
import desimodel.io

from bright_analysis.util.match import match

class SweepDirExistsError(Exception): pass

############################################################
def mock_map_for_source_from_file(map_id_file,source):
    """
    Read mock ids and mock paths from the automatically generated list.
    """
    all_mock_sources = np.loadtxt(map_id_file,usecols=[0],dtype=np.str)
    all_mock_ids     = np.loadtxt(map_id_file,usecols=[1],dtype=np.str)
    all_mock_paths   = np.loadtxt(map_id_file,usecols=[2],dtype=np.str)

    w_this_source = np.where(np.array([x.startswith(source) for x in all_mock_sources]))[0]
    mock_ids      = all_mock_ids[w_this_source]
    mock_paths    = all_mock_paths[w_this_source]

    assert(False)
    return mock_ids, mock_paths

############################################################
def make_directory_structure(config_file,source_name,map_id_filename,sweep_mock_root,
                             override_root=None,dry_run=True):
    """
    map_id_file:
        :
        MWS_MAIN 0 /project/projectdirs/desi/mocks/mws/galaxia/alpha/v0.0.3/bricks/005/...
        :
    """
    assert(os.path.exists(config_file))
    assert(os.path.exists(map_id_filename))

    # Read parameters for quicksurvey
    with open(config_file,'r') as pfile:
        params = yaml.load(pfile)

    # mock_ids, mock_paths = mock_map_for_source_from_file(map_id_file,source)

    map_id_file = np.loadtxt(map_id_filename,
                             dtype={'names': ('SOURCENAME', 'FILEID', 'FILENAME'),
                                    'formats': ('S10', 'i4', 'S256')})

    w = np.where(map_id_file['SOURCENAME']==source_name.encode())
    mock_paths = map_id_file['FILENAME'][w]

    # Original mock path.
    root_mock_dir = params['sources'][source_name]['root_mock_dir']

    if override_root is not None:
        print('User override for mock root directory mocks from survey config yaml.')
        print('    YAML: %s'%(root_mock_dir))
        print('    USER: %s'%(override_root))
        original_root_mock_dir = root_mock_dir
        root_mock_dir          = override_root

    # Recreate the variable part of the path for this mock under output_dir. To
    # avoid trouble, insist that the output root cannot exist.
    print('Sweep mock output root: {}'.format(sweep_mock_root))
    if os.path.exists(sweep_mock_root):
        raise SweepDirExistsError('Output sweep root dir already exists, you need to manually delete it')

    # No need to wrap this in try, since we guarentee the directory doesn't
    # exist.
    if not dry_run: os.makedirs(sweep_mock_root)

    print('%d mock paths'%len(mock_paths))
    for mock_path in mock_paths:
        mock_path = mock_path.decode()

        if override_root:
            mock_path = mock_path.replace(original_root_mock_dir,root_mock_dir)

        if mock_path.startswith(os.path.sep):
            mock_path = os.path.curdir + mock_path

        new_path = os.path.normpath(os.path.join(sweep_mock_root,mock_path))
        new_dir  = os.path.split(new_path)[0]
        print('Creating path: %s'%(new_dir))

        if not dry_run: os.makedirs(new_dir)
    return

############################################################
def concatenate_tilefiles(epoch_dir,sweep_mock_root):
    """
    """
    epoch = int(os.path.split(epoch_dir.strip(os.path.sep))[-1])
    print('Epoch: {}'.format(epoch))

    tilefiles = glob.glob('{}/fiberassign/tile_*.fits'.format(epoch_dir))
    ntiles    = len(tilefiles)
    print("Have {} tile files".format(ntiles))

    # Numbers of each tile
    tilenum = np.array([int(os.path.splitext(os.path.basename(s))[0].split('_')[-1]) for s in tilefiles])

    # Read all the tiles
    t0 = time.time()
    tiledata = list()
    for tilefile in tilefiles:
        f = fits.open(tilefile,'readonly',memmap=False)
        tiledata.append(f[1].data)
        f.close()
    t1 = time.time()
    
    print('Read tile data in {}s'.format(t1-t0))

    # Stores original tile index after concatenation
    itile = np.concatenate([np.repeat(int(i),len(x)) for i,x in enumerate(tiledata)])
    # Concatenate the tiles
    tiledata = Table(np.concatenate(tiledata))
    # Add the tile number (not the index)
    tiledata.add_column(Column(tilenum[itile],'TILE'),index=0)
    
    # Save the output
    tiledata_dir = os.path.normpath(os.path.join(sweep_mock_root,'tiles'))
    if not os.path.exists(tiledata_dir): os.makedirs(tiledata_dir)
    tiledata_path = os.path.join(tiledata_dir,'tiles_{}.fits'.format(epoch))
    tiledata.write(tiledata_path)
    print('Wrote {}'.format(tiledata_path))
    return

############################################################
TRUTH_CACHE  = None
TARGET_CACHE = None
def match_zcat_truth(config_file,input_dir,epoch_dir):
    """
    """
    global TRUTH_CACHE, TARGET_CACHE
    assert(os.path.exists(config_file))

    # Read parameters for quicksurvey
    with open(config_file,'r') as pfile:
        params = yaml.load(pfile)

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

############################################################
def make_mock_sweeps(config_file,source_name,input_dir,epoch_dir,map_id_file_path,
                     sweep_mock_root,override_root=None,dry_run=True,match_on_objid=False):
    """
    """
    if match_on_objid:
        print('NOTE: Matching on mock column "objid", only makes sense for 2016 data challenge outputs!')

    # Load the tile definitions
    TILES  = desimodel.io.load_tiles()

    epoch = int(os.path.split(epoch_dir.strip(os.path.sep))[-1])
    print('Epoch: {}'.format(epoch))

    # Read parameters for quicksurvey
    with open(config_file,'r') as pfile:
        params = yaml.load(pfile)

    # Original mock path.
    root_mock_dir = params['sources'][source_name]['root_mock_dir']

    if override_root is not None:
        original_root_mock_dir = root_mock_dir
        root_mock_dir          = override_root

    # Read zcat and truth, and construct the mapping between them
    zcat, truth_table, target_table, itruth_for_izcat = match_zcat_truth(config_file,input_dir,epoch_dir)

    # Read the table of mock paths for this source
    map_id_file = np.loadtxt(map_id_file_path,
                             dtype={'names': ('SOURCENAME', 'FILEID', 'FILENAME'),
                                    'formats': ('S10', 'i4', 'S256')})
    
    # Filter truth by source
    truth_this_source = truth_table['SOURCETYPE'] == source_name
    # Decode rowid and fileid for all targets associated with this source
    rowid, fileid = decode_rownum_filenum(truth_table['MOCKID'][truth_this_source])

    # Store the row indices for this source, will save them later
    truth_rows_for_source = np.where(truth_this_source)[0]

    # Filter truth by source for sources in zcat
    obs_this_source       = truth_table['SOURCETYPE'][itruth_for_izcat] == source_name
    # Decode rowid and fileid for observed targets associated with this source
    obs_rowid, obs_fileid = decode_rownum_filenum(truth_table['MOCKID'][itruth_for_izcat][obs_this_source])

    # Determine if target/truth rows for this source are in footprint
    print('Testing truth coordinates against DESIMODEL v %s'%(desimodel.__version__))
    footprint_flag_targets_truth   = footprint.is_point_in_desi(TILES,
                                                                truth_table['RA'][truth_rows_for_source],
                                                                truth_table['DEC'][truth_rows_for_source])

    # Read original files
    fileid_to_read = np.array(list(set(fileid)))


    # Different mocks have different names for sky coordinate columns
    # Auto-detect this based on a list of possibilities.
    ra_column, dec_column = None, None
    # Names to try in order of priority
    ra_names  = ['RA','ra','_RA','_ra']
    dec_names = ['DEC','dec','DE','_DEC','_dec','_DE']

    for ifile in fileid_to_read:

        # Filemap entry for this file ID
        row_in_map = (map_id_file['SOURCENAME']==source_name.encode()) & (map_id_file['FILEID']==ifile)
        filename   = map_id_file['FILENAME'][row_in_map]
        filename   = filename[0].decode()

        if override_root:
            filename = filename.replace(original_root_mock_dir,root_mock_dir)

        print('Reading mock file: {}'.format(filename))
        sys.stdout.flush()

        if source_name == 'MWS_MAIN':
            n_this_mock_file = fits.getheader(filename,1)['NAXIS2']
        else:
            raise Exception('Unrecognized source: {}'.format(source))

        print('Mock file nrows: {}'.format(n_this_mock_file))

        # This will be of the length of the total number of targets in this
        # mock file. Not all of these will be selected, and not all of these
        # will be observed.
        selected_as_target    = np.repeat(False,n_this_mock_file) 
        observed_this_epoch   = np.repeat(False,n_this_mock_file)

        # Read the mock data for this file ID
        data  = fits.getdata(filename,1)

        # Set column names if not already set
        if ra_column is None:
            for ra_name in ra_names:
                if ra_name in data.dtype.names:
                    ra_column = ra_name
                    break
            if ra_column is None:
                raise Exception('No RA column found in mock data columns')
 
        if dec_column is None:
            for dec_name in dec_names:
                if dec_name in data.dtype.names:
                    dec_column = dec_name
                    break
            if dec_column is None:
                raise Exception('No DEC column found in mock data columns')
        
        # Flag mock rows in footprint
        print('Testing mock coordinates against DESIMODEL v %s'%(desimodel.__version__))
        footprint_flag = footprint.is_point_in_desi(TILES,
                                                    data[ra_column],
                                                    data[dec_column])

        # Select truth rows for this file ID
        fileid_mask     = fileid == ifile
        obs_fileid_mask = obs_fileid == ifile

        if match_on_objid:
            # In the Dec 2016 data challenge, the 'row number' is actually the
            # Galaxia mock objid for MWS_MAIN.
            objid = data['objid']

            objid_this_file_all = rowid[fileid_mask]
            objid_this_file_obs = obs_rowid[obs_fileid_mask]

            m = match(objid_this_file_all,objid)
            rows_this_file_all = m[np.where(m>=0)[0]]

            m = match(objid_this_file_obs,objid)
            rows_this_file_obs = m[np.where(m>=0)[0]]
        else:
            rows_this_file_all = rowid[fileid_mask]
            rows_this_file_obs = obs_rowid[obs_fileid_mask]

        print('  -- N in truth =  {}'.format(len(rows_this_file_all)))
        print('  -- N observed =  {}'.format(len(rows_this_file_obs)))

        selected_as_target[rows_this_file_all]   = True
        observed_this_epoch[rows_this_file_obs]  = True
        
        assert(not np.any((observed_this_epoch) & (~selected_as_target)))
        
        # Targets that could have been observed this epoch, but were not.
        selected_not_observed = np.where((selected_as_target) & (~observed_this_epoch))[0]

        # Sort rows in this file by row number. Can use this to reorder subsets
        # of targets and truth, if we want to output them.
        rows_this_file_all_sort = np.argsort(rows_this_file_all)

        # Write various outputs
        filename_path, filename_file = os.path.split(filename)
        base, ext                   = filename_file.split(os.path.extsep)
        new_filename_status         = base + os.path.extsep + 'status'     + os.path.extsep + ext
        new_filename_observed       = base + os.path.extsep + 'observed'   + os.path.extsep + ext
        new_filename_unobserved     = base + os.path.extsep + 'unobserved' + os.path.extsep + ext
        new_filename_targets_subset = base + os.path.extsep + 'targets'    + os.path.extsep + ext
        new_filename_truth_subset   = base + os.path.extsep + 'truth'      + os.path.extsep + ext

        if override_root:
            filename_path = filename_path.replace(original_root_mock_dir,root_mock_dir)

        if filename_path.startswith(os.path.sep):
            filename_path = os.path.curdir + filename_path

        os.makedirs(os.path.normpath(os.path.join(sweep_mock_root,filename_path,str(epoch))))

        new_path_status         = os.path.normpath(os.path.join(sweep_mock_root,filename_path,str(epoch),new_filename_status))
        new_path_observed       = os.path.normpath(os.path.join(sweep_mock_root,filename_path,str(epoch),new_filename_observed))
        new_path_unobserved     = os.path.normpath(os.path.join(sweep_mock_root,filename_path,str(epoch),new_filename_unobserved))
        new_path_targets_subset = os.path.normpath(os.path.join(sweep_mock_root,filename_path,str(epoch),new_filename_targets_subset))
        new_path_truth_subset   = os.path.normpath(os.path.join(sweep_mock_root,filename_path,str(epoch),new_filename_truth_subset))

        # Write the status table, which has the same number of rows as the
        # origianl mock file (hence generally more than the sweep).
        t = Table((selected_as_target,observed_this_epoch), names=('SELECTED','OBSERVED'))
        t.write(new_path_status)
        print('Wrote {}'.format(new_path_status))

        # Provided that at least one target in this mock file has been
        # selected, write subsets of the mock file.

        # Previous had rows_this_file_obs here
        if len(rows_this_file_all) > 0:
            
            # For each row in mock that appears in targets/truth, get the
            # corresponding row number in the target/truth.
            input_target_row = np.zeros(n_this_mock_file,dtype=np.int32) - 1
            # ... These are the t/t rows that are in this mock file.
            w_fileid_mask = np.where(fileid_mask)[0] 
            # Store row
            input_target_row[rows_this_file_all] = truth_rows_for_source[w_fileid_mask]
            assert(np.all(input_target_row[observed_this_epoch]>=0))
            
            # 1. Make the mock sweep for observed targets
            t = Table(data[observed_this_epoch])
            t.add_column(Column(input_target_row[observed_this_epoch],name='TARGETROW'))
            t.add_column(Column(target_table[input_target_row[observed_this_epoch]]['TARGETID'],name='TARGETID'))
            # Include flag for targets in actual footprint
            t.add_column(Column(footprint_flag[observed_this_epoch],name='IN_FOOTPRINT'))

            # Strict check
            assert(np.allclose(t['RA'],target_table[t['TARGETROW']]['RA'],atol=1e-5))
            t.write(new_path_observed)
            print('Wrote {}'.format(new_path_observed))
        
            # 2. Make the mock sweep for unobserved targets
            t = Table(data[selected_not_observed])
            t.add_column(Column(input_target_row[selected_not_observed],name='TARGETROW'))
            t.add_column(Column(target_table[input_target_row[selected_not_observed]]['TARGETID'],name='TARGETID'))
            # Include flag for targets in actual footprint
            t.add_column(Column(footprint_flag[selected_not_observed],name='IN_FOOTPRINT'))

            # Strict check
            assert(np.allclose(t['RA'],target_table[t['TARGETROW']]['RA'],atol=1e-5))
            t.write(new_path_unobserved)
            print('Wrote {}'.format(new_path_unobserved)) 

            # 3. Make a file extracted from targets for this mock brick
            # Save in rowid order
            subset_this_file = truth_rows_for_source[fileid_mask]
            t = Table(target_table[subset_this_file])[rows_this_file_all_sort]
            t.add_column(Column((footprint_flag_targets_truth[fileid_mask])[rows_this_file_all_sort],name='IN_FOOTPRINT'))
            t.write(new_path_targets_subset)
            print('Wrote {}'.format(new_path_targets_subset))

            # 4. Make a file extracted from truth for this mock brick
            subset_this_file = truth_rows_for_source[fileid_mask]
            t = Table(truth_table[subset_this_file])[rows_this_file_all_sort]
            t.add_column(Column((footprint_flag_targets_truth[fileid_mask])[rows_this_file_all_sort],name='IN_FOOTPRINT'))
            t.write(new_path_truth_subset)
            print('Wrote {}'.format(new_path_truth_subset))
    return
