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
    :class:`numpy.ndarray` of shape ``(n_samples, n_channels)`` when the
    underlying reader is called with ``preload=True``; otherwise ``None``.
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
        If ``False``, the returned ``raw`` object is set up for on-demand
        reading and the ``data`` entry will be ``None`` to avoid loading the
        full dataset into memory.

    Returns
    -------
    dict
        Keys:

        ``raw``
            The :class:`mne.io.Raw` object.
        ``data``
            Array of shape ``(n_samples, n_channels)`` when ``preload=True``,
            otherwise ``None``.
        ``sfreq``
            Sampling frequency in Hz.
        ``ch_names``
            List of channel names.
    """
    import mne

    raw = mne.io.read_raw_edf(file_path, preload=preload, verbose=False)
    if preload:
        # Only slice when preloading, as raw[:] loads the entire dataset
        data, _ = raw[:]               # shape: (n_channels, n_times)
        data_out = data.T              # → (n_samples, n_channels)
    else:
        data_out = None
    return {
        'raw': raw,
        'data': data_out,
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
        If ``False``, the returned ``raw`` object is set up for on-demand
        reading and the ``data`` entry will be ``None`` to avoid loading the
        full dataset into memory.

    Returns
    -------
    dict
        Keys:

        ``raw``
            The :class:`mne.io.Raw` object.
        ``data``
            Array of shape ``(n_samples, n_channels)`` when ``preload=True``,
            otherwise ``None``.
        ``sfreq``
            Sampling frequency in Hz.
        ``ch_names``
            List of channel names.
    """
    import mne

    raw = mne.io.read_raw_brainvision(file_path, preload=preload, verbose=False)
    if preload:
        # Only slice when preloading, as raw[:] loads the entire dataset
        data, _ = raw[:]
        data_out = data.T
    else:
        data_out = None
    return {
        'raw': raw,
        'data': data_out,
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
    if not getattr(block, "segments", None):
        raise ValueError(f"No segments found in TRC file: {file_path}")

    segment = block.segments[0]
    if getattr(segment, "analogsignals", None) is None:
        raise ValueError(
            f"No analog signals container present in first segment of TRC file: {file_path}"
        )

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
        Additional keyword arguments forwarded to the format-specific reader
        where supported (e.g., EDF/VHDR readers).

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
    reader = readers[ext]
    # Only forward keyword arguments to readers that are known to accept them.
    if ext in ('edf', 'vhdr'):
        return reader(file_path, **kwargs)
    return reader(file_path)
