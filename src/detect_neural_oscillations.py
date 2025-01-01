# noinspection PyPackageRequirements
r"""
===========================================================
Finding other neural rhythms
===========================================================

We can do that by computing the spectrum/fourier transform of
each basic brain wave and compare with the spectrum of each atom.

Another way to do that would be use some distance, such as dtw, to
compare the raw signal of the atoms and the brain waves

"""
import mne
import numpy as np
import scipy
from matplotlib import pyplot as plt

###############################################################################
# Let us first define the parameters of our model.

sfreq = 150.

# Define the shape of the dictionary
n_atoms = 25
n_times_atom = int(round(sfreq * 1.0))  # 1000. ms

###############################################################################
# Next, we define the parameters for multivariate CSC

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

rhythms = {4:'Delta',
           8:'Theta',
           12:'Alpha-Mu',
           30:'Beta'}

def find_peaks(model, n_atoms, n=5, figure=False, rows=1, columns=1):

    if figure:
        figsize = (columns * 5, rows * 5.5)
        fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)

    main_rhythm = {}

    for i_atom in range(n_atoms):

        print(f"Atom {i_atom+1}")

        v_hat = model.v_hat_[i_atom]
        u_hat = model.u_hat_[i_atom]
        psd = np.abs(np.fft.rfft(v_hat)) ** 2
        frequencies = np.linspace(0, sfreq / 2.0, len(psd))

        mask = frequencies<=30
        frequencies = frequencies[mask]
        psd = psd[mask]
        print(frequencies)

        peaks_idx = np.argsort(psd)[-n:][::-1]
        peaks_freq = frequencies[peaks_idx]
        #peaks_idx, _ = scipy.signal.find_peaks(psd)
        #peaks_freq = np.sort(frequencies[peaks_idx])[::-1]

        print(peaks_freq)

        for v in rhythms.keys():
            if peaks_freq[0]<v:

                print(f"    {rhythms[v]} wave")

                # n most relevant channels
                idx_sorted = np.argpartition(u_hat, -n)[-n:]
                #idx_sorted = idx_sorted[np.argsort(u_hat[idx_sorted])[::-1]]

                # most relevant channels
                channels = np.array(info.ch_names)[idx_sorted]

                print(f"    {n} most relevant channels:")
                print(f'    {channels}')

                main_rhythm[i_atom] = {"rhythm": rhythms[v],
                                       "channels": channels}

                if figure:
                    row = i_atom // columns
                    col = i_atom % columns
                    ax = axes[row, col]

                    u_hat = model.u_hat_[i_atom]
                    mne.viz.plot_topomap(u_hat, info, axes=ax, show=False)
                    ax.set(title=f'Atom {i_atom + 1} - Rhythm {rhythms[v]}',)

                break

    if figure:
        for i in range(n_atoms, rows * columns):
            row = i // columns
            col = i % columns
            axes[row, col].axis('off')
        plt.tight_layout()
        plt.savefig("../figures/waves_per_region.pdf", dpi=300)
        plt.show()

    return main_rhythm

main_rhythm = find_peaks(cdl,n_atoms, figure=True, rows=5, columns=5)
