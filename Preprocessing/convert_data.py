# Copyright (C) 2024 ETH Zurich. All rights reserved.
# Author: Victor Kartsch, ETH Zurich

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions
# and limitations under the License.
# SPDX-License-Identifier: Apache-2.0

import struct
import warnings

import numpy as np


def convert_data(file_including_path, voltage_scale, timestamp_scale):
    """Read a BioWolf binary recording and convert it to physical units.

    Parameters
    ----------
    file_including_path : str
        Path to the .bin file.
    voltage_scale : str
        Desired voltage unit: 'V', 'mV', 'uV', or 'none' (skip scaling).
    timestamp_scale : str
        Desired time unit: 's', 'ms', or 'us'.

    Returns
    -------
    dict
        ExGData with keys: Data, Trigger, timestamp, ImuData,
        SampleRate, SignalGain, and optional metadata fields.
    """
    # --- Constants ---
    lsb_map = {
        0: 1,           # legacy
        1: 1 / 7000700,
        2: 1 / 14000800,
        3: 1 / 20991300,
        4: 1 / 27990100,
        6: 1 / 41994600,
        8: 1 / 55994200,
        12: 1 / 83970500,
    }
    HEADER_SIZE = 7
    BT_PCK_SIZE = 32

    # --- Validate scales ---
    tscale_map = {'s': 1, 'ms': 1e3, 'us': 1e6}
    if timestamp_scale not in tscale_map:
        raise ValueError(
            "Enter a valid timestamp scale. Available options: ['s', 'ms', 'us']"
        )
    tscale_factor = tscale_map[timestamp_scale]

    vscale_map = {'V': 1, 'mV': 1e3, 'uV': 1e6, 'none': 1}
    if voltage_scale not in vscale_map:
        raise ValueError(
            "Enter a valid voltage scale. Available options: ['V', 'mV', 'uV', 'none']"
        )
    vscale_factor = vscale_map[voltage_scale]

    # --- Read file ---
    with open(file_including_path, 'rb') as fid:
        raw = fid.read()
    A = np.frombuffer(raw, dtype=np.uint8)

    # --- Parse experimental notes header (<<IEP,) ---
    # Byte pattern: 60 60 62 62 73 69 80 44
    MARKER = bytes([60, 60, 62, 62, 73, 69, 80, 44])
    exg_data = {}
    end_of_data = 0

    for i in range(len(A) - HEADER_SIZE):
        if bytes(A[i: i + 8]) == MARKER:
            data_recovered = ''.join(chr(b) for b in A[i:])
            end_of_data = i
            break

    if end_of_data != 0:
        params = data_recovered.split(',')
        for t_value in params[1:]:
            if not t_value:
                continue
            key = t_value[0]
            val = t_value[1:]
            if key == 'T':
                exg_data['TestName'] = val
            elif key == 'S':
                exg_data['SubjectName'] = val
            elif key == 'A':
                exg_data['SubjectAge'] = float(val)
            elif key == 'R':
                exg_data['Remarks'] = val
            elif key == 'F':
                exg_data['SampleRate'] = float(val)
            elif key == 'G':
                exg_data['SignalGain'] = float(val)
    else:
        warnings.warn(
            'The file does not contain information about the experimental '
            'parameters. Hence, conversion of the data to the specified '
            'voltage scale is skipped.'
        )
        end_of_data = len(A)
        exg_data['SampleRate'] = 500
        exg_data['SignalGain'] = 12
        vscale_factor = 1e6

    # --- Parse binary packets ---
    payload = A[:end_of_data]
    n_packets = len(payload) // BT_PCK_SIZE
    ADS = payload[: n_packets * BT_PCK_SIZE].reshape(n_packets, BT_PCK_SIZE)

    def _decode_channel_24bit(b0, b1, b2):
        """Reconstruct a signed 24-bit ADC value stored as 3 bytes (MSB first),
        left-shifted by 8 to fill a 32-bit integer (matching MATLAB typecast)."""
        raw_u32 = (int(b0) << 24) | (int(b1) << 16) | (int(b2) << 8)
        return struct.unpack('>i', struct.pack('>I', raw_u32))[0]

    def _decode_acc_16bit(b0, b1):
        """Reconstruct a signed 16-bit accelerometer value stored MSB-first,
        left-shifted into the top 16 bits of a 32-bit integer."""
        raw_u32 = (int(b0) << 24) | (int(b1) << 16)
        return struct.unpack('>i', struct.pack('>I', raw_u32))[0]

    n = ADS.shape[0]
    channels = np.zeros((n, 8), dtype=np.int32)
    acc = np.zeros((n, 3), dtype=np.int32)

    # Channel byte offsets (0-based): ch1=0,1,2  ch2=3,4,5  ... ch8=21,22,23
    for ch_idx in range(8):
        base = ch_idx * 3
        for i in range(n):
            channels[i, ch_idx] = _decode_channel_24bit(
                ADS[i, base], ADS[i, base + 1], ADS[i, base + 2]
            )

    for i in range(n):
        acc[i, 0] = _decode_acc_16bit(ADS[i, 24], ADS[i, 25])
        acc[i, 1] = _decode_acc_16bit(ADS[i, 26], ADS[i, 27])
        acc[i, 2] = _decode_acc_16bit(ADS[i, 28], ADS[i, 29])
        if i % 5000 == 0:
            print(f'Processing {(i / n) * 100:.2f}%')

    # --- Gain scaling ---
    signal_gain = int(exg_data.get('SignalGain', 12))
    if voltage_scale != 'none':
        gain_scaling = lsb_map.get(signal_gain, 1)
    else:
        gain_scaling = 1

    t_data = (channels.astype(np.float64) / 256.0) * gain_scaling * vscale_factor
    t_trigger = ADS[:, 31]

    skipped_samples = 1
    exg_data['Data'] = t_data[skipped_samples + 1:, :]
    exg_data['Trigger'] = t_trigger[skipped_samples + 1:]
    n_samples = exg_data['Data'].shape[0]
    fs = exg_data['SampleRate']
    exg_data['timestamp'] = np.arange(n_samples) / fs * tscale_factor
    exg_data['ImuData'] = acc.astype(np.float64) / (255.0 * 255.0)

    return exg_data
