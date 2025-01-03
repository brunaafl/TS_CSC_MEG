r"""
===========================================================
Comparing atoms found when separating the regions
===========================================================

Separating the channels into two regions of biggest correlation,
defining a minimization problem for each, and comparing found atoms.

"""
import mne
import copy

import numpy as np
import pandas as pd

from functions import display_atoms, display_ffts, display_topomap, display_atom
from dtw import dtw
###############################################################################
# Let us first define the parameters of our model.

sfreq = 150.

# Define the shape of the dictionary
n_atoms = 25
n_times_atom = int(round(sfreq * 1.0))  # 1000. ms


###############################################################################
# Here, we load the data from the somato-sensory dataset and preprocess them
# in epochs. The epochs are selected around the stim, starting 2 seconds
# before and finishing 4 seconds after.

from mne_data import load_data
t_lim = (-2, 4)

X, info= load_data(dataset='somato', epoch=t_lim, sfreq=sfreq)


###############################################################################
# Next, we define the parameters for multivariate CSC

import scipy.signal.windows
from alphacsc.init_dict import init_dictionary


# Monkey-patch scipy.signal.tukey to point to the correct function
scipy.signal.tukey = scipy.signal.windows.tukey

from alphacsc import BatchCDL


cdl = BatchCDL(
    # Shape of the dictionary
    n_atoms,
    n_times_atom,
    rank1=False,
    uv_constraint='auto',
    # Number of iteration for the alternate minimization and cvg threshold
    n_iter=100, eps=1e-4,
    # number of workers to be used for dicodile
    n_jobs=4,
    # solver for the z-step
    solver_z='dicodile',
    solver_z_kwargs={'max_iter': 10000},
    window=True,
    D_init='chunk',
    random_state=60)



###############################################################################
# Fit the model and learn atoms

cdl.fit(X)

###############################################################################
# Display the 4-th atom, which displays a :math:`\mu`-waveform in its temporal
# pattern.

# plot atom from mu wave
display_atom(cdl,3, info)

# display all
display_atoms(cdl, n_atoms, 5, 5, sfreq, savefig="atoms_no_rank1")