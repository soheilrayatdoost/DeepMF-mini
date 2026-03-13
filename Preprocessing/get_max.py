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

    Searches in a ±5 sample neighbourhood around *aprox_max* and returns the
    index of the actual maximum within that window.

    Parameters
    ----------
    signal : array-like
        1-D signal segment (typically a short window around the approximate peak).
    aprox_max : int
        0-based index of the approximate peak location in the *full* signal.

    Returns
    -------
    int
        0-based index of the true maximum in the full signal.
    """
    signal = np.asarray(signal)
    max_location = int(np.argmax(signal))  # 0-based index within the window
    diff = max_location - 5               # window was centred at position 5
    return aprox_max + diff
