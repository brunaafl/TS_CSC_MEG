import mne
import numpy as np
import scipy
from matplotlib import pyplot as plt


rhythms = {4:'Delta',
           8:'Theta',
           12:'Alpha-Mu',
           30:'Beta'}


def display_atom(model, i_atom, info, sfreq=150):

    n_plots = 3
    figsize = (n_plots * 5, 5.5)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, squeeze=False)

    # Plot the spatial map of the learn atom using mne topomap
    ax = axes[0, 0]
    u_hat = model.u_hat_[i_atom]
    mne.viz.plot_topomap(u_hat, info, axes=ax, show=False)
    ax.set(title='Learned spatial pattern')

    # Plot the temporal pattern of the learn atom
    ax = axes[0, 1]
    v_hat = model.v_hat_[i_atom]
    t = np.arange(v_hat.size) / sfreq
    ax.plot(t, v_hat)
    ax.set(xlabel='Time (sec)', title='Learned temporal waveform')
    ax.grid(True)

    # Plot the psd of the time atom
    ax = axes[0, 2]
    psd = np.abs(np.fft.rfft(v_hat)) ** 2
    frequencies = np.linspace(0, sfreq / 2.0, len(psd))
    ax.semilogy(frequencies, psd)
    ax.set(xlabel='Frequencies (Hz)', title='Power Spectral Density')
    ax.grid(True)
    ax.set_xlim(0, 30)

    plt.tight_layout()
    plt.show()


def display_atoms(model, n_atoms, rows, columns, sfreq, savefig="atoms_somato"):
    if rows * columns < n_atoms:
        raise ValueError("The grid size (rows x columns) must be at least equal to n_atoms")

    figsize = (columns * 5, rows * 5.5)
    fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)

    for i_atom in range(n_atoms):
        row = i_atom // columns
        col = i_atom % columns
        ax = axes[row, col]

        v_hat = model.v_hat_[i_atom]
        t = np.arange(v_hat.size) / sfreq

        ax.plot(t, v_hat)
        ax.set(xlabel='Time (sec)', title=f'Atom {i_atom + 1}')
        ax.grid(True)

    for i in range(n_atoms, rows * columns):
        row = i // columns
        col = i % columns
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(f"../figures/{savefig}.pdf", dpi=300)
    plt.show()


def display_ffts(model, n_atoms, rows, columns, sfreq, savefig="topomap_ffts"):
    if rows * columns < n_atoms:
        raise ValueError("The grid size (rows x columns) must be at least equal to n_atoms")

    figsize = (columns * 5, rows * 5.5)
    fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)

    for i_atom in range(n_atoms):
        row = i_atom // columns
        col = i_atom % columns
        ax = axes[row, col]

        v_hat = model.v_hat_[i_atom]
        psd = np.abs(np.fft.rfft(v_hat)) ** 2
        frequencies = np.linspace(0, sfreq / 2.0, len(psd))
        ax.semilogy(frequencies, psd)
        ax.set(xlabel='Frequencies (Hz)', title=f'Atom {i_atom + 1}')
        ax.grid(True)
        ax.set_xlim(0, 30)

    for i in range(n_atoms, rows * columns):
        row = i // columns
        col = i % columns
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(f"../figures/{savefig}.pdf", dpi=300)
    plt.show()

def display_topomap(model, n_atoms, rows, columns, info, savefig="topomap_somato"):
    if rows * columns < n_atoms:
        raise ValueError("The grid size (rows x columns) must be at least equal to n_atoms")

    figsize = (columns * 5, rows * 5.5)
    fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)

    for i_atom in range(n_atoms):
        row = i_atom // columns
        col = i_atom % columns
        ax = axes[row, col]

        u_hat = model.u_hat_[i_atom]
        mne.viz.plot_topomap(u_hat, info, axes=ax, show=False)
        ax.set(title=f'Atom {i_atom + 1}')

    for i in range(n_atoms, rows * columns):
        row = i // columns
        col = i % columns
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(f"../figures/{savefig}.pdf", dpi=300)
    plt.show()



def find_peaks(model, n_atoms, info, n=5, figure=False, rows=1, columns=1, sfreq=150):

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
        plt.show()

    return main_rhythm

