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

import os

import numpy as np
import scipy.io as sio

from downsample import downsample


def data_save(
    save_dir,
    slice_name,
    f1_window,
    f6_window,
    lead_i_chest,
    lead_i_chest_ones,
    lear_electrode,
    rear_electrode,
    fs,
    down_fs,
):
    """Normalise, downsample and save a processed biopotential window.

    Saves a ``.mat`` file containing the in-ear electrode signals (both filter
    bands) together with the reference chest Lead-I ECG trace and its R-peak
    indicator vector.

    Parameters
    ----------
    save_dir : str
        Directory where the ``.mat`` file will be written.
    slice_name : str
        Filename for the output ``.mat`` file (e.g. ``'R1_1.mat'``).
    f1_window : numpy.ndarray, shape (n_samples, n_channels)
        Signal filtered with the narrow bandpass (0.5–30 Hz).
    f6_window : numpy.ndarray, shape (n_samples, n_channels)
        Signal filtered with the wide bandpass (0.5–45 Hz).
    lead_i_chest : numpy.ndarray, shape (n_ecg_samples,)
        Chest Lead-I ECG trace (already downsampled to *down_fs*).
    lead_i_chest_ones : numpy.ndarray, shape (n_ecg_samples,)
        Binary R-peak indicator vector aligned with *lead_i_chest*.
    lear_electrode : int
        0-based column index of the left in-ear electrode.
    rear_electrode : int
        0-based column index of the right in-ear electrode.
    fs : int or float
        Original sampling frequency in Hz.
    down_fs : int or float
        Target downsampled frequency in Hz.
    """
    f1_window = np.asarray(f1_window, dtype=np.float64)
    f6_window = np.asarray(f6_window, dtype=np.float64)

    # Normalise (z-score, column-wise)
    f1_window = _zscore(f1_window)
    f6_window = _zscore(f6_window)

    # Downsample
    f1_window_down = downsample(f1_window, fs, down_fs)
    f6_window_down = downsample(f6_window, fs, down_fs)

    # Extract in-ear channels (1-D row vectors to match MATLAB layout)
    L_inear_f1 = f1_window_down[:, lear_electrode]
    L_inear_f6 = f6_window_down[:, lear_electrode]
    R_inear_f1 = f1_window_down[:, rear_electrode]
    R_inear_f6 = f6_window_down[:, rear_electrode]

    os.makedirs(save_dir, exist_ok=True)
    sio.savemat(
        os.path.join(save_dir, slice_name),
        {
            'LeadI_chest': lead_i_chest,
            'LeadI_chest_ones': lead_i_chest_ones,
            'L_inear_f1': L_inear_f1,
            'L_inear_f6': L_inear_f6,
            'R_inear_f1': R_inear_f1,
            'R_inear_f6': R_inear_f6,
        },
    )


def _zscore(arr):
    """Z-score normalise *arr* column-wise, avoiding division by zero."""
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    return (arr - mean) / std
