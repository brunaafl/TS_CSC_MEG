r"""
===========================================================
Extracting :math:`\mu`-wave from the somato-sensory dataset
===========================================================

This example illustrates how to learn rank-1 atoms [1]_ on the multivariate
somato-sensorymotor dataset from :code:`mne`. The displayed results highlight
the presence of :math:`\mu`-waves located in the SI cortex.

.. [1] Dupré La Tour, T., Moreau, T., Jas, M., & Gramfort, A. (2018).
    `Multivariate Convolutional Sparse Coding for Electromagnetic Brain Signals
    <https://arxiv.org/abs/1805.09654v2>`_. Advances in Neural Information
    Processing Systems (NIPS).
"""
from functions import display_atom, display_atoms, display_topomap, display_ffts

sfreq = 150.

# Define the shape of the dictionary
n_atoms = 25
n_times_atom = int(round(sfreq * 1.0))  # 1000. ms

###############################################################################
# Next, we define the parameters for multivariate CSC

# This fix is not working. I will import my forked version of alphacsc instead
# Reload alphacsc to ensure the patch is applied
import importlib
from alphacsc import BatchCDL

cdl = BatchCDL(
    # Shape of the dictionary
    n_atoms=n_atoms,
    n_times_atom=n_times_atom,
    # Request a rank1 dictionary with unit norm temporal and spatial maps
    rank1=True, uv_constraint='separate',
    # Initialize the dictionary with random chunk from the data
    D_init='chunk',
    # rescale the regularization parameter to be 20% of lambda_max
    lmbd_max="scaled", reg=.2,
    # Number of iteration for the alternate minimization and cvg threshold
    n_iter=100, eps=1e-4,
    # solver for the z-step
    solver_z="lgcd", solver_z_kwargs={'tol': 1e-2, 'max_iter': 1000},
    # solver for the d-step
    solver_d='alternate_adaptive', solver_d_kwargs={'max_iter': 300},
    # Technical parameters
    verbose=1, random_state=0, n_jobs=6)


###############################################################################
# Here, we load the data from the somato-sensory dataset and preprocess them
# in epochs. The epochs are selected around the stim, starting 2 seconds
# before and finishing 4 seconds after.

from alphacsc.datasets.mne_data import load_data
t_lim = (-2, 4)
X, info = load_data(dataset='somato', epoch=t_lim, sfreq=sfreq)


###############################################################################
# Fit the model and learn rank1 atoms
cdl.fit(X)

###############################################################################
# Display the 4-th atom, which displays a :math:`\mu`-waveform in its temporal
# pattern.

import mne
import numpy as np
import matplotlib.pyplot as plt

# plot atom from mu wave
display_atom(cdl,3, info)

# display all
display_atoms(cdl, n_atoms, 5, 5, sfreq)
display_ffts(cdl, n_atoms, 5, 5, sfreq)
display_topomap(cdl, n_atoms, 5, 5, info)