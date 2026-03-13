# Copyright (C) 2024 ETH Zurich. All rights reserved.
# Author: Carlos Santos, ETH Zurich

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions
# and limitations under the License.
# SPDX-License-Identifier: Apache-2.0

"""Main preprocessing pipeline for BioWolf in-ear ECG recordings.

This script reproduces the MATLAB ``data_window_processing.m`` pipeline in
Python.  Filtering is performed with MNE's IIR filter routines (zero-phase
Butterworth), R peaks are detected with :func:`scipy.signal.find_peaks`, and
processed windows are saved as ``.mat`` files via :func:`data_save`.

Usage
-----
Edit the ``--- Configuration ---`` section below, then run::

    python data_window_processing.py
"""

import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.io as sio
from scipy.signal import find_peaks

from convert_data import convert_data
from data_save import data_save
from get_max import get_max

# ---------------------------------------------------------------------------
# --- Configuration ---------------------------------------------------------
# ---------------------------------------------------------------------------

DATA_FOLDER = "Data"

# TODO: adapt subject_id, bin_file_name and recording_id to your recording
SUBJECT_ID = "S1"
BIN_FILE_NAME = "Data_20241206_174157.bin"
RECORDING_ID = "R1"

# Sampling frequencies
FS = 500    # Hz – original
DOWN_FS = 250  # Hz – after downsampling

# Electrode mapping (1-based indices converted to 0-based below)
LARM_ELECTRODE = 7  # 0-based (MATLAB col 8 → Python index 7)
RARM_ELECTRODE = 6  # 0-based (MATLAB col 7 → Python index 6)
LEAR_ELECTRODE = 1  # 0-based (MATLAB col 2 → Python index 1)
REAR_ELECTRODE = 0  # 0-based (MATLAB col 1 → Python index 0)

# ---------------------------------------------------------------------------
# --- Helper: MNE IIR filter wrappers ---------------------------------------
# ---------------------------------------------------------------------------

_IIR_NOTCH = dict(order=6, ftype='butter')
_IIR_BP = dict(order=2, ftype='butter')


def _apply_notch(data_2d, sfreq, freq=50.0):
    """Zero-phase notch filter (filtfilt) via MNE – operates on (n_samples, n_ch)."""
    # MNE expects (n_channels, n_times)
    filtered = mne.filter.notch_filter(
        data_2d.T.copy(),
        Fs=sfreq,
        freqs=freq,
        method='iir',
        iir_params=_IIR_NOTCH,
        verbose=False,
    )
    return filtered.T


def _apply_bandpass(data_2d, sfreq, l_freq, h_freq):
    """Zero-phase bandpass filter (filtfilt) via MNE – operates on (n_samples, n_ch)."""
    filtered = mne.filter.filter_data(
        data_2d.T.copy(),
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method='iir',
        iir_params=_IIR_BP,
        verbose=False,
    )
    return filtered.T


def _apply_bandpass_causal(data_2d, sfreq, l_freq, h_freq):
    """Causal (one-pass) bandpass filter via MNE – operates on (n_samples, n_ch).

    MNE's ``filter_data`` with ``phase='minimum'`` applies the filter causally,
    analogous to MATLAB's ``filter()`` function.
    """
    filtered = mne.filter.filter_data(
        data_2d.T.copy(),
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        method='iir',
        iir_params=_IIR_BP,
        phase='minimum',
        verbose=False,
    )
    return filtered.T


def _apply_notch_causal(data_2d, sfreq, freq=50.0):
    """Causal notch filter via MNE – analogous to MATLAB's ``filter()``."""
    filtered = mne.filter.notch_filter(
        data_2d.T.copy(),
        Fs=sfreq,
        freqs=freq,
        method='iir',
        iir_params=_IIR_NOTCH,
        phase='minimum',
        verbose=False,
    )
    return filtered.T


# ---------------------------------------------------------------------------
# --- Pipeline --------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    # --- Paths ---
    save_dir = os.path.join("..", DATA_FOLDER, "Processed", SUBJECT_ID, RECORDING_ID)
    os.makedirs(save_dir, exist_ok=True)

    path_to_bin = os.path.join("..", DATA_FOLDER, "Raw", SUBJECT_ID, BIN_FILE_NAME)

    # --- Load binary recording ---
    exg_data = convert_data(path_to_bin, 'uV', 's')
    data = exg_data['Data']                  # (n_samples, 8)
    recording_state = exg_data['Trigger']    # (n_samples,)

    # --- Trim noisy edges ---
    avoid_seconds = 5
    recording = data[avoid_seconds * FS: -avoid_seconds * FS, :]

    # =====================================================================
    # Arm ECG – chest Lead I
    # =====================================================================

    arm_ch_lo = min(RARM_ELECTRODE, LARM_ELECTRODE)
    arm_ch_hi = max(RARM_ELECTRODE, LARM_ELECTRODE)
    arm_recording = recording[:, arm_ch_lo: arm_ch_hi + 1]

    # Step I – notch 50 Hz (zero-phase)
    arm_notch = _apply_notch(arm_recording, FS)

    # Step II – bandpass 0.5–30 Hz (zero-phase)
    arm_bandpass = _apply_bandpass(arm_notch, FS, l_freq=0.5, h_freq=30.0)

    # Chest Lead I = LA – RA
    if LARM_ELECTRODE > RARM_ELECTRODE:
        chest_lead_i = arm_bandpass[:, 1] - arm_bandpass[:, 0]
    else:
        chest_lead_i = arm_bandpass[:, 0] - arm_bandpass[:, 1]

    transient_response = 2000
    chest_lead_i = chest_lead_i[transient_response:]

    # Normalise (z-score)
    chest_lead_i = (chest_lead_i - chest_lead_i.mean()) / chest_lead_i.std()

    # Downsample to DOWN_FS
    chest_lead_i = chest_lead_i[::int(FS / DOWN_FS)]

    # --- R peak detection on –differential ---
    chest_lead_i_der = -np.diff(chest_lead_i)
    chest_peaks, _ = find_peaks(
        chest_lead_i_der,
        height=0.7,
        distance=80,
    )

    # Refine each peak to the true local maximum (±5 samples)
    chest_lead_i_ones = np.zeros(len(chest_lead_i))
    for j, pk in enumerate(chest_peaks):
        lo = max(0, pk - 5)
        hi = min(len(chest_lead_i) - 1, pk + 5)
        window = chest_lead_i[lo: hi + 1]
        refined = get_max(window, pk)
        chest_peaks[j] = int(np.clip(refined, 0, len(chest_lead_i) - 1))
        p = chest_peaks[j]
        chest_lead_i_ones[max(0, p - 1): p + 2] = 1.0

    # --- Optional visualisation ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(chest_lead_i, 'b', label='Lead I')
    for pk in chest_peaks:
        ax.axvline(pk, color='r', linewidth=1.5)
    ax.set_title("Chest Lead I with detected R peaks (close window to continue)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # TODO: manually correct any mismatching peaks before continuing.

    # =====================================================================
    # Full biopotential filtering
    # =====================================================================

    # Causal notch (analogous to MATLAB ``filter``)
    recording_notched = _apply_notch_causal(recording, FS)

    # Causal bandpass 0.5–30 Hz
    recording_f1 = _apply_bandpass_causal(recording_notched, FS, l_freq=0.5, h_freq=30.0)
    recording_f1 = recording_f1[transient_response:, :]

    # Causal bandpass 0.5–45 Hz
    recording_f6 = _apply_bandpass_causal(recording_notched, FS, l_freq=0.5, h_freq=45.0)
    recording_f6 = recording_f6[transient_response:, :]

    # =====================================================================
    # Sliding-window extraction and saving
    # =====================================================================

    window_size = 1000   # samples  (2 s @ 500 Hz)
    overlap = 900        # samples  (1.8 s)

    slice_count = 0
    start_ear = 0
    start_ecg = 0

    while start_ear + window_size <= len(recording_f1):
        slice_count += 1
        slice_name = f"{RECORDING_ID}_{slice_count}.mat"

        ecg_window_size = window_size // 2
        lead_i_window = chest_lead_i[start_ecg: start_ecg + ecg_window_size]
        lead_i_ones_window = chest_lead_i_ones[start_ecg: start_ecg + ecg_window_size]

        f1_window = recording_f1[start_ear: start_ear + window_size, :]
        f6_window = recording_f6[start_ear: start_ear + window_size, :]

        data_save(
            save_dir, slice_name,
            f1_window, f6_window,
            lead_i_window, lead_i_ones_window,
            LEAR_ELECTRODE, REAR_ELECTRODE,
            FS, DOWN_FS,
        )

        step_ecg = (window_size - overlap) // 2
        step_ear = window_size - overlap
        start_ecg += step_ecg
        start_ear += step_ear

    # --- Save ground-truth ECG trace and peaks ---
    sio.savemat(
        os.path.join(save_dir, 'gt.mat'),
        {'chest_LeadI': chest_lead_i, 'chest_peaks': chest_peaks},
    )
    print(f"Done. Saved {slice_count} windows to {save_dir}")


if __name__ == '__main__':
    main()
