from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

############################################################
def match(arr1,arr2,arr2_sorted=False):
    """
    For each element in arr1, return the index of the element with the same
    value in arr2, or -1 if there is no element with the same value.

    Neither arr1 nor arr2 have to be sorted first. Only arr2 is sorted in
    operation. If it's already sorted, save time by setting arr2_sorted=True.

    Code by John Helly, Andrew Cooper
    """
    from numpy import arange, argsort, array, isscalar, searchsorted
    from numpy import where

    if arr2_sorted:
        idx  = slice(0,len(arr2))
        tmp2 = arr2
    else:
        idx  = argsort(arr2)
        tmp2 = arr2[idx]

    # Find where the elements of arr1 can be inserted in arr2
    ptr_l = searchsorted(tmp2,arr1,side='left')
    ptr_r = searchsorted(tmp2,arr1,side='right')

    if isscalar(ptr_l):
        ptr_l = array([ptr_l])
        ptr_r = array([ptr_r])

    # Return -1 where no match is found. Note that searchsorted returns
    # len(tmp2) for values beyond the maximum of tmp2. This cannot be used as
    # an index.
    filter        = ptr_r-ptr_l == 0
    ptr_l[filter] = -1
    del(ptr_r,filter)

    # Put ptr back in original order
    ind   = arange(len(arr2))[idx]
    ptr_l = where(ptr_l >= 0, ind[ptr_l], -1)

    return ptr_l

