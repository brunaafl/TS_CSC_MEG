from os.path import join
from copy import deepcopy

import mne
import numpy as np
from joblib import Memory
from scipy.signal.windows import tukey

from utils.config import ALPHACSC_CACHE_DIR

mem = Memory(location=ALPHACSC_CACHE_DIR, verbose=0)

## Adaptation of the code at alphacsc
@mem.cache(ignore=['n_jobs'])
def load_data(dataset="somato", n_splits=10, sfreq=None, epoch=None, channels= None,
              filter_params=[2., None], return_array=True, n_jobs=1):
    """Load and prepare the somato dataset for multiCSC

    Parameters
    ----------
    dataset : str in {'somato', 'sample'}
        Dataset to load.
    n_splits : int
        Split the signal in n_split signals of same length before returning it.
        If epoch is provided, the signal is instead splitted according to the
        epochs and this option is not followed.
    sfreq : float
        Sampling frequency of the signal. The data are resampled to match it.
    epoch : tuple or None
        If set to a tuple, extract epochs from the raw data, using
        t_min=epoch[0] and t_max=epoch[1]. Else, use the raw signal, divided
        in n_splits chunks.
    filter_params : tuple of length 2
        Boundaries of filtering, e.g. (2, None), (30, 40), (None, 40).
    return_array : boolean
        If True, return an NumPy array, instead of mne objects.
    n_jobs : int
        Number of jobs that can be used for preparing (filtering) the data.

    Returns
    -------
    X : array, shape (n_splits, n_channels, n_times)
        The loaded dataset.
    info : dict
        MNE dictionary of information about recording settings.
    """

    if dataset == 'somato':
        pick_types_epoch = dict(meg='grad', eeg=False, eog=True, stim=False)
        pick_types_final = dict(meg='grad', eeg=False, eog=False, stim=False)

        data_path = mne.datasets.somato.data_path()
        subjects_dir = None
        file_name = join(data_path, 'sub-01', 'meg',
                         'sub-01_task-somato_meg.fif')
        raw = mne.io.read_raw_fif(file_name, preload=True)

        raw_copy = raw.copy()

        # Keep a copy for event extraction
        raw_stim = raw.copy()
        raw_stim.pick_types(meg=False, stim=True)  # Only keep stim channels for event detection

        # Extract events from stim channels
        event_id = {'somato': 1}
        events = mne.find_events(raw_stim, stim_channel='STI 014')
        events = mne.pick_events(events, include=list(event_id.values()))

        raw_copy.notch_filter(np.arange(50, 101, 50), n_jobs=n_jobs)

        baseline = (None, 0)

    elif dataset == 'sleep':
        pick_types_epoch = dict(meg=False, eeg=True, eog=True, stim=False)
        pick_types_final = dict(meg=False, eeg=True, eog=False, stim=False)

        # Load the sleep PhysioNet dataset
        subject = 1
        subjects_dir = None
        [data_fetch] = mne.datasets.sleep_physionet.age.fetch_data(subjects=[subject], recording=[1])

        raw = mne.io.read_raw_edf(data_fetch[0],stim_channel="Event marker",infer_types=True,preload=True,)

        annot_train = mne.read_annotations(data_fetch[1])
        print(annot_train)
        raw.set_annotations(annot_train, emit_warning=False)

        raw_copy = raw.copy()

        # remove 6 and 7 labels

        annotation_event_id = {
            "Sleep stage W": 1,
            "Sleep stage 1": 2,
            "Sleep stage 2": 3,
            "Sleep stage 3": 4,
            "Sleep stage 4": 4,
            "Sleep stage R": 5,
        }

        # Set reference for EEG channels
        annot_train.crop(annot_train[1]["onset"] - 30 * 60, annot_train[-2]["onset"] + 30 * 60)
        raw_copy.set_annotations(annot_train, emit_warning=False)

        # Extract events based on annotations
        events, _ = mne.events_from_annotations(raw_copy, event_id=annotation_event_id, chunk_duration=30.0)

        # create a new event_id that unifies stages 3 and 4
        event_id = {
            "Sleep stage W": 1,
            "Sleep stage 1": 2,
            "Sleep stage 2": 3,
            "Sleep stage 3/4": 4,
            "Sleep stage R": 5,
        }

        baseline = None

    else:
        ValueError("Dataset must be somato or auditory")

    if channels is not None:
        raw_copy = raw_copy.pick_channels(channels, ordered=True)

    # Dipole fit information
    cov = None  # see below
    file_trans = None
    file_bem = None

    raw_copy.filter(*filter_params, n_jobs=n_jobs)

    # Now pick final channel types for the main raw object
    raw_copy.pick_types(**pick_types_final)

    if dataset == 'somato':
        # compute the covariance matrix for somato
        picks_cov = mne.pick_types(raw_copy.info, **pick_types_epoch)
        epochs_cov = mne.Epochs(raw_copy, events, event_id, tmin=-4, tmax=0,
                                picks=picks_cov, baseline=baseline,
                                reject=dict(grad=4000e-13),
                                preload=True)
        epochs_cov.pick_types(**pick_types_final)
        cov = mne.compute_covariance(epochs_cov)

    if epoch:
        t_min, t_max = epoch
        print(events)
        picks = mne.pick_types(raw_copy.info, **pick_types_epoch)
        epochs = mne.Epochs(raw_copy, events, event_id, t_min, t_max, picks=picks,
                            baseline=baseline,preload=True)
        epochs.pick_types(**pick_types_final)
        info = epochs.info

        print(epochs)
        if sfreq is not None:
            epochs = epochs.resample(sfreq, npad='auto', n_jobs=n_jobs)

        if return_array:
            X = epochs.get_data()

    else:
        events[:, 0] -= raw_copy.first_samp
        if channels is not None:
            raw_copy = raw_copy.pick_channels(channels, ordered=True)
        raw_copy.pick_types(**pick_types_final)
        info = raw_copy.info

        if sfreq is not None:
            raw_copy, events = raw_copy.resample(sfreq, events=events, npad='auto',
                                       n_jobs=n_jobs)

        if return_array:
            X = raw_copy.get_data()
            n_channels, n_times = X.shape
            n_times = n_times // n_splits
            X = X[:, :n_splits * n_times]
            X = X.reshape(n_channels, n_splits, n_times).swapaxes(0, 1)

    # Deep copy before modifying info to avoid issues when saving EvokedArray
    info = deepcopy(info)
    event_info = dict(event_id=event_id,
                      events=events,
                      subject=dataset,
                      subjects_dir=subjects_dir,
                      cov=cov,
                      file_bem=file_bem,
                      file_trans=file_trans)

    info['temp'] = event_info

    if return_array:
        n_splits, n_channels, n_times = X.shape
        X *= tukey(n_times, alpha=0.1)[None, None, :]
        X /= np.std(X)
        return X, info
    elif epoch:
        return epoch, info
    else:
        return raw_copy, info


def separate_sleep_stages(X, info):

    event_id = info['temp']['event_id']  # Mapping of sleep stages
    events = info['temp']['events']  # Event data

    data_by_stage = {stage: [] for stage in event_id.keys()}

    # Sampling frequency (sfreq) is required to slice the data
    sfreq = info['sfreq']  # Replace with the correct sampling frequency of your data

    # Iterate over events to separate data
    j=0
    for _, _, stage_id in events:
        # Find corresponding stage name
        stage_name = next((name for name, sid in event_id.items() if sid == stage_id), None)
        if stage_name is None:
            continue  # Skip if the event ID does not match any stage
        # Store data from each stage in the dict
        data_by_stage[stage_name].append(X[j])
        j += 1

    X_stage = {}
    for stage in data_by_stage.keys():
        value = data_by_stage[stage]
        if len(value)>0:
            concat_values = np.concatenate(value)
            X_stage[stage] = concat_values

    return X_stage