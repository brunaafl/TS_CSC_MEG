import mne
import matplotlib.pyplot as plt
from mne.datasets import somato

data_path = somato.data_path()
subject = "01"
task = "somato"
raw_fname = data_path / f"sub-{subject}" / "meg" / f"sub-{subject}_task-{task}_meg.fif"

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname)
# Plot the sensor layout
raw.plot_sensors(show_names=True, kind='topomap', sphere=0.07)

