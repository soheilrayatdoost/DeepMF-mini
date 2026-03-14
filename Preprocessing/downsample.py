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

import numpy as np


def downsample(original_array, fs, new_fs):
    """Downsample a signal by keeping every *jump*-th sample.

    Parameters
    ----------
    original_array : numpy.ndarray
        Input array of shape (n_samples,) or (n_samples, n_channels).
    fs : int or float
        Original sampling frequency in Hz.
    new_fs : int or float
        Target sampling frequency in Hz.  *fs* must be an integer multiple of
        *new_fs*.

    Returns
    -------
    numpy.ndarray
        Downsampled array with the same number of dimensions as *original_array*.
    """
    original_array = np.asarray(original_array)
    ratio = fs / new_fs
    rounded_ratio = int(round(ratio))
    if ratio < 1 or not np.isclose(ratio, rounded_ratio):
        raise ValueError(
            "fs must be an integer multiple of new_fs and new_fs must not exceed fs; "
            f"got fs={fs}, new_fs={new_fs} (fs/new_fs={ratio})."
        )
    jump = rounded_ratio
    return original_array[::jump]
