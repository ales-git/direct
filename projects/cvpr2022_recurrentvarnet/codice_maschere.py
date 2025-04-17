# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 09:04:16 2025

@author: Sonia.Colombo
"""

r"""Random vertical line mask function.

    The mask selects a subset of columns from the input k-space data. If the k-space data has :math:`N` columns,
    the mask picks out:

        #.  :math:`N_{\text{low freqs}} = (N \times \text{center_fraction})` columns in the center corresponding
            to low-frequencies if center_fraction < 1.0, or :math:`N_{\text{low freqs}} = \text{center_fraction}`
            if center_fraction >= 1 and is integer.
        #.  The other columns are selected uniformly at random with a probability equal to:
            :math:`\text{prob} = (N / \text{acceleration} - N_{\text{low freqs}}) / (N - N_{\text{low freqs}})`.
            This ensures that the expected number of columns selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which case one possible
    (center_fraction, acceleration) is chosen uniformly at random each time the MaskFunc object is
    called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04], then there
    is a 50% probability that 4-fold acceleration with 8% center fraction is selected and a 50%
    probability that 8-fold acceleration with 4% center fraction is selected.

    Parameters
    ----------
    accelerations : Union[list[Number], tuple[Number, ...]]
        Amount of under-sampling.
    center_fractions : Union[list[Number], tuple[Number, ...]]
        If < 1.0 this corresponds to the fraction of low-frequency columns to be retained.
        If >= 1 (integer) this corresponds to the exact number of low-frequency columns to be retained.
    uniform_range : bool, optional
        If True then an acceleration will be uniformly sampled between the two values, by default False.
    mode : MaskFuncMode
        Mode of the mask function. Can be MaskFuncMode.STATIC, MaskFuncMode.DYNAMIC, or MaskFuncMode.MULTISLICE.
        If MaskFuncMode.STATIC, then a single mask is created independent of the requested shape, and will be
        broadcasted to the shape by expanding other dimensions with 1, if applicable. If MaskFuncMode.DYNAMIC,
        this expects the shape to have more then 3 dimensions, and the mask will be created for each time frame
        along the fourth last dimension. Similarly for MaskFuncMode.MULTISLICE, the mask will be created for each
        slice along the fourth last dimension. Default: MaskFuncMode.STATIC.
    """


import random
import numpy as np
import math
import h5py

def ACS_list(matrix_size, acceleration):
    r_ACS = 1/(acceleration*3.125)
    n_ACS = int(matrix_size * r_ACS)
    ACS = [item for item in range(int(matrix_size/2)-int(n_ACS/2),int(matrix_size/2)+int(n_ACS/2))]    
    return ACS


def random_cartesian_list(matrix_size, acceleration):
    phase_list = [item for item in range(matrix_size)]
    ACS = ACS_list(matrix_size, acceleration) 
    prob = (matrix_size/acceleration - len(ACS))/(matrix_size)# - n_ACS)
    n_ext_rows = int(matrix_size*prob)
    EXT = random.sample(phase_list, n_ext_rows)
    return ACS + EXT

def equispaced_list(matrix_size, acceleration):
    phase_list = [item for item in range(matrix_size)]    
    ACS = ACS_list(matrix_size, acceleration) 
    prob = (matrix_size/acceleration - len(ACS))/(matrix_size)
    n_ext_rows = int(matrix_size*prob)    
    n = int((matrix_size - len(ACS))/n_ext_rows)
    EXT = phase_list[0:min(ACS):n] + phase_list[max(ACS):matrix_size:n]
    return ACS + EXT

def gaussian_list(matrix_size, acceleration):
    ACS = ACS_list(matrix_size, acceleration) 
    prob = (matrix_size/acceleration - len(ACS))/(matrix_size)
    n_ext_rows = int(matrix_size*prob)    
    mean = matrix_size / 2
    std = 4 * math.sqrt(mean)  
    #genereo il doppi dei numeri per essere sicura che non ci siano sovrapposizioni nella regione ACS, poi ne prendo il numero che mi servono
    EXT = [int(item) for item in np.random.normal(mean, std, n_ext_rows*4) if (0 <= item < min(ACS)) or (max(ACS) < item < matrix_size)]
    EXT = np.unique(EXT).tolist()[0:n_ext_rows] ## CORREGGI!
    return ACS + EXT

def array(matrix_size_y, matrix_size_z, rows_list):
    array = np.zeros((matrix_size_z, matrix_size_y), dtype=int)
    for row in rows_list:
        array[:,row] = 1
    return array

# =============================================================================
# TEST CODICE
# =============================================================================
import matplotlib.pyplot as plt
matrix_size_y = 128
matrix_size_z = 128
acceleration = 4

#rows_list = gaussian_list(matrix_size_y, acceleration)
#mask = array(matrix_size_y, matrix_size_z, rows_list)
#plt.imshow(mask, cmap = "gray")
#plt.show()

#rows_list = equispaced_list(matrix_size_y, acceleration)
#mask = array(matrix_size_y, matrix_size_z, rows_list)
#plt.imshow(mask, cmap = "gray")
#plt.show()

rows_list = random_cartesian_list(matrix_size_y, acceleration)
mask = array(matrix_size_y, matrix_size_z, rows_list)
plt.imshow(mask, cmap = "gray")
plt.show()






