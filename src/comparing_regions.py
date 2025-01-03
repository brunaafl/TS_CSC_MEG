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
import seaborn as sns

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
n_channels = len(info['ch_names'])

all_channels = info['ch_names']
channels_1 = all_channels[:n_split] + ['STI 014']
channels_2 = all_channels[n_split:] + ['STI 014']

X1, info1= load_data(dataset='somato', epoch=t_lim, sfreq=sfreq,channels=channels_1)
X2, info2= load_data(dataset='somato', epoch=t_lim, sfreq=sfreq,channels=channels_2)

def create_info(ch_names, sfreq=150):
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, )
    return info

###############################################################################
# Learn rank-1 atoms for each separate part of the brain

# Separate the problem into 2 different
cdl_1 = copy.deepcopy(cdl)
cdl_2 = copy.deepcopy(cdl)

cdl_1.fit(X1)
cdl_2.fit(X2)

###############################################################################
# Display the 4-th atom, which displays a :math:`\mu`-waveform in its temporal
# pattern.

display_atoms(cdl_1, n_atoms, 2, 5, sfreq, savefig="atoms_somato_1")
display_ffts(cdl_1, n_atoms, 2, 5, sfreq, savefig = "topomap_ffts_1")
display_topomap(cdl_1, n_atoms, 2, 5, info1, savefig = "topomap_somato_1")

display_atoms(cdl_2, n_atoms, 2, 5, sfreq, savefig="atoms_somato_2")
display_ffts(cdl_2, n_atoms, 2, 5, sfreq, savefig = "topomap_ffts_2")
display_topomap(cdl_2, n_atoms, 2, 5, info2, savefig = "topomap_somato_2")

##########################################

import matplotlib.pyplot as plt

# Compare the atoms found in the two regions
v_hat_1 = cdl_1.v_hat_
v_hat_2 = cdl_2.v_hat_

table = np.zeros(shape=(n_atoms, n_atoms))

for i in range(n_atoms):
    align_row = []
    for j in range(i,n_atoms):

        alignment = dtw(v_hat_1[i], v_hat_2[j],keep_internals=True)
        align_row.append(alignment)
        table[i,j]=alignment.distance
        table[j,i]=alignment.distance

    #min_row = table[i,:].argmin(axis=0)
    #closest = align_row[min_row]
    #print(f"atom {i+1} and atom {min_row+1}")
    #closest.plot(type="twoway", offset=10)
    #plt.clf()

columns = [f"Atom {i}" for i in range(1,1+n_atoms)]

min_index = np.argmin(table)
row, col = np.unravel_index(min_index, table.shape)

# plot the most similar atoms
min_distance = table[row,col]
atom_row = v_hat_1[row]
atom_col = v_hat_2[col]

figsize = (11,5)
fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)

t = np.arange(atom_row.size)/sfreq

ax1 = axes[0,0]
ax1.plot(t, atom_row)
ax1.set(xlabel='Time (sec)', title=f'Atom {row + 1}')
ax1.grid(True)

ax2 = axes[0,1]
ax2.plot(t, atom_col)
ax2.set(xlabel='Time (sec)', title=f'Atom {col + 1}')
ax2.grid(True)

plt.tight_layout()
plt.savefig("../figures/most_similar_atoms.pdf", dpi=300)
plt.show()

table_df = pd.DataFrame(table, columns=columns)
table_df.index = columns

fig, ax = plt.subplots()
sns.heatmap(table_df, annot=True, cmap="YlGnBu", linewidths=0.5, ax=ax)
plt.xticks(rotation=45)
plt.savefig("../figures/distance_atoms.pdf", dpi=300)
# Show the plot
plt.show()
