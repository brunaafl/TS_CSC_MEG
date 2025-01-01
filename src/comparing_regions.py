r"""
===========================================================
Comparing atoms found when separating the regions
===========================================================

Separating the channels into two regions of biggest correlation,
defining a minimization problem for each, and comparing found atoms.

"""
import copy

import numpy as np
import pandas as pd

from functions import display_atoms, display_ffts, display_topomap
from dtw import dtw
###############################################################################
# Let us first define the parameters of our model.

sfreq = 150.

# Define the shape of the dictionary
n_atoms = 10
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

from mne_data import load_data
t_lim = (-2, 4)

X, info= load_data(dataset='somato', epoch=t_lim, sfreq=sfreq)

n_split = 110
X1, ch_name_1 = X[:, :n_split], info.ch_names[:n_split]
X2, ch_name_2 = X[:, n_split:], info.ch_names[n_split:]
###############################################################################
# Separate the data in two parts depending on the region
# Select the channels from visual cortex (mu rhythms) and

"""
t_lim = (-2, 4)

motor = []
X_motor, info_motor = load_data(dataset='somato', epoch=t_lim, sfreq=sfreq)
X_visual, info_visual = load_data(dataset='somato', epoch=t_lim, sfreq=sfreq, channels=visual)


visual = [
    "MEG 1521", "MEG 1721", "MEG 1921", "MEG 1941",
    "MEG 2031", "MEG 2321", "MEG 2521", "MEG 2631",
    "MEG 1711", "MEG 1731", "MEG 1931", "MEG 2111",
    "MEG 2341", "MEG 2511", "MEG 2531",
    "MEG 1741", "MEG 2131", "MEG 2141", "MEG 2541"
]
X_visual, info_visual = load_data(dataset='somato', epoch=t_lim, sfreq=sfreq, channels=visual)
"""

###############################################################################
# Fit the model and learn rank1 atoms

cdl_1 = copy.deepcopy(cdl)
cdl_2 = copy.deepcopy(cdl)

cdl_1.fit(X1)
cdl_2.fit(X2)

###############################################################################
# Display the 4-th atom, which displays a :math:`\mu`-waveform in its temporal
# pattern.

display_atoms(cdl_1, n_atoms, 5, 5, sfreq, savefig="atoms_somato_1")
display_ffts(cdl_1, n_atoms, 5, 5, sfreq, savefig = "topomap_ffts_1")
display_topomap(cdl_1, n_atoms, 5, 5, info, savefig = "topomap_somato_1")

display_atoms(cdl_2, n_atoms, 5, 5, sfreq, savefig="atoms_somato_2")
display_ffts(cdl_2, n_atoms, 5, 5, sfreq, savefig = "topomap_ffts_2")
display_topomap(cdl_2, n_atoms, 5, 5, info, savefig = "topomap_somato_2")

##########################################

import matplotlib.pyplot as plt

# Compare the atoms found in the two regions
v_hat_1 = cdl_1.v_hat_
v_hat_2 = cdl_2.v_hat_

table = np.zeros(shape=(n_atoms, n_atoms))

for i in range(n_atoms):
    for j in range(i+1,n_atoms):

        distance = dtw(v_hat_1[i], v_hat_2[j])
        table[i,j]=distance
        table[j,i]=distance

columns = [f"Atom {i}" for i in range(1,1+n_atoms)]

min_index = np.argmin(table)
row, col = np.unravel_index(min_index, table.shape)

# plot the most similar atoms
min_distance = table[row,col]
atom_row = v_hat_1[row]
atom_col = v_hat_2[col]

figsize = (5, 11)
fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)

t = np.arange(v_hat_1.size) / sfreq

ax1 = axes[0]
ax1.plot(t, atom_row)
ax1.set(xlabel='Time (sec)', title=f'Atom {row + 1}')
ax1.grid(True)

ax2 = axes[1]
ax2.plot(t, atom_col)
ax2.set(xlabel='Time (sec)', title=f'Atom {col + 1}')
ax2.grid(True)

plt.tight_layout()
plt.savefig("../figures/most_similar_atoms.pdf", dpi=300)
plt.show()

table_df = pd.DataFrame(table, columns=columns)

print(table)
