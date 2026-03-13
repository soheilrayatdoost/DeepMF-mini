# Copyright (C) 2024 ETH Zurich. All rights reserved.
# Author: ETH Zurich

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions
# and limitations under the License.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for reading various EEG/ECG file formats.

Supported formats
-----------------
* **EDF / EDF+** – via :func:`mne.io.read_raw_edf`
* **BrainVision (VHDR)** – via :func:`mne.io.read_raw_brainvision`
* **Micromed TRC** – via :mod:`neo.io.MicromedIO`

All functions return a dictionary with at least the keys:

``data``
    :class:`numpy.ndarray` of shape ``(n_samples, n_channels)``.
``sfreq``
    Sampling frequency in Hz.
``ch_names``
    List of channel names.

For EDF and VHDR files the raw :class:`mne.io.Raw` object is also returned
under the ``raw`` key so that the full MNE preprocessing API remains
accessible.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# EDF
# ---------------------------------------------------------------------------

def read_edf(file_path: str, preload: bool = True) -> dict:
    """Read an EDF or EDF+ file using MNE.

    Parameters
    ----------
    file_path : str
        Path to the ``.edf`` file.
    preload : bool
        If ``True`` (default) the data are loaded into memory immediately.

    Returns
    -------
    dict
        Keys: ``raw``, ``data`` (n_samples × n_channels), ``sfreq``,
        ``ch_names``.
    """
    import mne

    raw = mne.io.read_raw_edf(file_path, preload=preload, verbose=False)
    data, _ = raw[:]               # shape: (n_channels, n_times)
    return {
        'raw': raw,
        'data': data.T,            # → (n_samples, n_channels)
        'sfreq': raw.info['sfreq'],
        'ch_names': raw.ch_names,
    }


# ---------------------------------------------------------------------------
# BrainVision VHDR
# ---------------------------------------------------------------------------

def read_vhdr(file_path: str, preload: bool = True) -> dict:
    """Read a BrainVision header (``.vhdr``) file using MNE.

    Parameters
    ----------
    file_path : str
        Path to the ``.vhdr`` file.
    preload : bool
        If ``True`` (default) the data are loaded into memory immediately.

    Returns
    -------
    dict
        Keys: ``raw``, ``data`` (n_samples × n_channels), ``sfreq``,
        ``ch_names``.
    """
    import mne

    raw = mne.io.read_raw_brainvision(file_path, preload=preload, verbose=False)
    data, _ = raw[:]
    return {
        'raw': raw,
        'data': data.T,
        'sfreq': raw.info['sfreq'],
        'ch_names': raw.ch_names,
    }


# ---------------------------------------------------------------------------
# Micromed TRC
# ---------------------------------------------------------------------------

def read_trc(file_path: str) -> dict:
    """Read a Micromed TRC file using the Neo library.

    Parameters
    ----------
    file_path : str
        Path to the ``.trc`` file.

    Returns
    -------
    dict
        Keys: ``data`` (n_samples × n_channels), ``sfreq``, ``ch_names``.

    Raises
    ------
    ImportError
        If the ``neo`` package is not installed.
    """
    try:
        import neo.io as nio
    except ImportError as exc:
        raise ImportError(
            "The 'neo' package is required to read TRC files.  "
            "Install it with: pip install neo"
        ) from exc

    reader = nio.MicromedIO(filename=file_path)
    block = reader.read_block(signal_group_mode='split-all', lazy=False)

    # Collect all analog signals from the first segment
    segment = block.segments[0]
    signals = []
    ch_names = []
    sfreq = None

    for sig in segment.analogsignals:
        arr = np.asarray(sig).squeeze()  # (n_times,) or (n_times, n_ch)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        signals.append(arr)
        ch_names.extend([str(sig.name)] * arr.shape[1])
        if sfreq is None:
            sfreq = float(sig.sampling_rate.magnitude)

    if not signals:
        raise ValueError(f"No analog signals found in TRC file: {file_path}")

    data = np.concatenate(signals, axis=1)  # (n_samples, n_channels)
    return {
        'data': data,
        'sfreq': sfreq,
        'ch_names': ch_names,
    }


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

def read_data(file_path: str, **kwargs) -> dict:
    """Auto-detect the file format from the extension and read accordingly.

    Parameters
    ----------
    file_path : str
        Path to the data file.  Supported extensions: ``.edf``, ``.vhdr``,
        ``.trc``.
    **kwargs
        Additional keyword arguments forwarded to the format-specific reader.

    Returns
    -------
    dict
        Format-specific dictionary (see individual reader functions).

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    """
    ext = file_path.rsplit('.', 1)[-1].lower()
    readers = {
        'edf': read_edf,
        'vhdr': read_vhdr,
        'trc': read_trc,
    }
    if ext not in readers:
        raise ValueError(
            f"Unsupported file extension '.{ext}'.  "
            f"Supported formats: {list(readers.keys())}"
        )
    return readers[ext](file_path, **kwargs)
