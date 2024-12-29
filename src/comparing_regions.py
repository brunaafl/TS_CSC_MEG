r"""
===========================================================
Comparing atoms found when separating the regions
===========================================================

Separating the channels into two regions of biggest correlation,
defining a minimization problem for each, and comparing found atoms.

"""


###############################################################################
# Let us first define the parameters of our model.

sfreq = 150.

# Define the shape of the dictionary
n_atoms = 25
n_times_atom = int(round(sfreq * 1.0))  # 1000. ms

###############################################################################
# Next, we define the parameters for multivariate CSC

import scipy.signal.windows

# Monkey-patch scipy.signal.tukey to point to the correct function
scipy.signal.tukey = scipy.signal.windows.tukey

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

motor = []
X_motor, info_motor = load_data(dataset='somato', epoch=t_lim, sfreq=sfreq)

meg_channels = [
    "MEG 1521", "MEG 1721", "MEG 1921", "MEG 1941",
    "MEG 2031", "MEG 2321", "MEG 2521", "MEG 2631",
    "MEG 1711", "MEG 1731", "MEG 1931", "MEG 2111",
    "MEG 2341", "MEG 2511", "MEG 2531",
    "MEG 1741", "MEG 2131", "MEG 2141", "MEG 2541"
]

###############################################################################
# Separate the data in two parts depending on the region
# Select the channels from visual cortex (mu rhythms) and
###############################################################################
# Fit the model and learn rank1 atoms
cdl.fit(X)

###############################################################################
# Display the 4-th atom, which displays a :math:`\mu`-waveform in its temporal
# pattern.
