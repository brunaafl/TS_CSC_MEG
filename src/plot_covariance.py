from mne_data import load_data
from pyriemann.utils.covariance import covariances
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# We can use the correlation between regions to test the hypotesis
# of one source!!
# We can divide the data into the two main correlated regions and see
# what is detected

def plot_covariance(savefig='channel_correlation'):
    t_lim = (-2, 4)
    sfreq = 150
    X, info = load_data(dataset='somato', epoch=t_lim, sfreq=sfreq)
    corr = covariances(X, estimator='corr')
    mean_corr = corr.mean(axis=0)

    # Now, for the figure with the histograms
    row_sums = mean_corr.sum(axis=1)
    col_sums = mean_corr.sum(axis=0)

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0.05, hspace=0.05)

    ax_heatmap = fig.add_subplot(gs[1, 0])
    sns.heatmap(mean_corr, cmap="hot", square=True, cbar=True, ax=ax_heatmap)
    ax_heatmap.set_title("Mean Covariance Matrix")

    # vertical
    ax_row_hist = fig.add_subplot(gs[1, 1], sharey=ax_heatmap)
    ax_row_hist.barh(np.arange(len(row_sums)), row_sums, color="orange")
    ax_row_hist.axis("off")

    # horizontal
    ax_col_hist = fig.add_subplot(gs[0, 0], sharex=ax_heatmap)
    ax_col_hist.bar(np.arange(len(col_sums)), col_sums, color="orange")
    ax_col_hist.axis("off")
    plt.savefig(f"../figures/{savefig}.pdf", dpi=300)
    plt.show()

