from igor2 import binarywave as bw
import numpy as np
import pandas as pd
import re
import os

from . import curves


def _get_notes(wave: dict) -> dict:
    '''Utility function for processing the 'notes' section of a .ibw file,
    the end user should not call this function'''

    note_raw = wave['wave']['note']
    # Asylum seems to store the degree sign in a broken way that
    # python can't parse, replace all occurances of this invalid
    # byte sequence with 'deg'
    note_raw = note_raw.replace(b'\xb0', b'deg')
    all_notes = note_raw.split(b'\r')
    note_dict = dict()
    for line in all_notes:
        split_line = line.split(b':')
        key = split_line[0]
        value = b':'.join(split_line[1:]).strip()
        note_dict[key.decode()] = value.decode()
    return note_dict


def _get_data(wave: dict) -> pd.DataFrame:
    '''Utility function for processing the 'data' section of a .ibw file,
    the end user should not call this function'''

    wave_labels = wave['wave']['labels']
    wave_data = wave['wave']['wData']
    labels = [label.decode() for label in wave_labels[1] if label]
    col_indices = {
        'rawz': labels.index('Raw'),
        'defl': labels.index('Defl'),
        'z': labels.index('ZSnsr')
    }

    df = dict(
        rawz=wave_data[:, col_indices['rawz']],
        z=wave_data[:, col_indices['z']],
        defl=wave_data[:, col_indices['defl']]
    )

    wave_frame = pd.DataFrame(df)
    wave_frame.loc[:, 'ind'] = wave_frame['z'] - wave_frame['defl']
    return wave_frame


def load_ibw(filename: str) -> curves.Curve:
    wave = bw.load(filename)
    data = _get_data(wave)
    notes = _get_notes(wave)
    trigger_index = int(np.argmax(data.loc[:, 'defl']))

    sample_time = wave['wave']['wave_header']['sfA'][0]
    t = np.arange(data.shape[0]) * sample_time
    data.loc[:, 't'] = t

    if notes.get('DwellTime'):
        dwell_time = float(notes['DwellTime'])
    else:
        print("Missing DwellTime")
        dwell_time = 0.0

    dwell_start_time = data['t'].loc[trigger_index]
    dwell_end_time = dwell_start_time + dwell_time

    dwell_end_index = int(np.argmin(np.abs(data.loc[:, 't'] - dwell_end_time)))
    dwell_range = [trigger_index, dwell_end_index-1]

    if notes.get('SpringConstant'):
        k = float(notes['SpringConstant'])
    else:
        print("Missing SpringConstant")
        k = 0.0

    data.loc[:, 'f'] = data.loc[:, 'defl'] * k

    if notes.get('InvOLS'):
        invOLS = float(notes['InvOLS'])
    else:
        print("Missing InvOLS")
        invOLS = 0.0

    this_curve = curves.Curve(
        filename=filename.split(os.path.sep)[-1],
        data=data,
        notes=notes,
        invOLS=invOLS,
        k=k,
        dwell_range=dwell_range
    )

    return this_curve


def load_curveset_ibw(folder: str, ident_labels: list[str]) -> curves.CurveSet:
    """
    Loads a set of .ibw files from a specified directory into a curves.CurveSet object.

    This function scans a directory for .ibw files associated with Asylum Atomic Force Microscopy (AFM) measurements. 
    It filters and loads these files into a CurveSet object based on specified identification labels found in the filenames. 
    These labels help distinguish between different measurements within the set.

    Parameters:
    - folder (str): The path to the directory containing .ibw files to be loaded. 
      The directory should exclusively contain relevant .ibw files created by an Asylum AFM.

    - ident_labels (list[str]): A list of substrings that are consistently present in the filenames of interest, 
      in the order they appear. These labels are used to filter and differentiate between measurements. 
      Intervening strings not included in this list will be considered as part of the unique identifiers for each measurement.

    Example:
    Consider a directory at '~/experiment' with the following files:
    - Sample1Measurement0.ibw
    - Sample1Measurement1.ibw
    - Sample2Measurement0.ibw
    - Sample2Measurement1.ibw

    To load these files into a CurveSet, use:
    >>> load_curveset_ibw('~/experiment', ['Sample', 'Measurement'])

    Returns:
    - curves.CurveSet: An object containing all loaded force curves that match the identification labels,
      organized as specified by the 'folder' and 'ident_labels' parameters.
    """

    regex_str = ""
    for label in ident_labels:
        regex_str = regex_str + label + f'(?P<{label}>.*)'
    regex_str = regex_str + '\\.ibw'
    regex = re.compile(regex_str)

    all_filenames = os.listdir(folder)
    all_matches = [regex.match(a) for a in all_filenames]

    curve_dict = dict()
    for m in all_matches:
        if not m:
            continue
        idents = tuple([m.group(label) for label in ident_labels])
        filename = m.group(0)
        filepath = os.path.join(folder, filename)

        print(f"   > Reading {filename}")
        curve_dict[idents] = load_ibw(filepath)

    curveset = curves.CurveSet(
        ident_labels=ident_labels, curve_dict=curve_dict)
    print("   > Experiment Loaded.")
    return curveset
