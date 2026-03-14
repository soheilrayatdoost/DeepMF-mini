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


def get_max(signal, aprox_max):
    """Refine an approximate R-peak location to the true maximum.

    Searches within a fixed window around *aprox_max* in the provided ``signal``
    and returns the index of the actual maximum within that window.

    Parameters
    ----------
    signal : array-like
        1-D signal in which the peak is to be refined.
    aprox_max : int
        0-based index of the approximate peak location within ``signal``.

    Returns
    -------
    int
        0-based index of the refined maximum within ``signal``.
    """
    signal = np.asarray(signal)
    aprox_max = int(aprox_max)

    if signal.size == 0:
        # Degenerate case: nothing to refine; return the approximate index.
        return aprox_max

    # Define a fixed 11-sample window (±5 samples) around the approximate peak,
    # with clipping at the signal boundaries.
    half_window = 5
    start_idx = max(0, aprox_max - half_window)
    end_idx = min(signal.size, aprox_max + half_window + 1)  # end index is exclusive

    window = signal[start_idx:end_idx]
    # Index of the local maximum within the window
    local_argmax = int(np.argmax(window))

    # Position of aprox_max within the window
    center_offset = aprox_max - start_idx

    # Refined global index, matching aprox_max + (local_argmax - center_offset)
    refined_index = aprox_max + (local_argmax - center_offset)
    return refined_index
