from ..igor import binarywave as bw
import numpy as np
import pandas as pd
from . import curves
import re
import os


def _get_notes(wave: pd.DataFrame) -> pd.DataFrame:
    '''Utility function for processing the 'notes' section of a .ibw file, the end user should not call this function'''

    note_raw = wave['wave']['note']
    # Asylum seems to store the degree sign in a broken way that python can't parse, replace all occurances of this invalid byte sequence with 'deg'
    note_raw = note_raw.replace(b'\xb0', b'deg')
    all_notes = note_raw.split(b'\r')
    note_dict = dict()
    for line in all_notes:
        split_line = line.split(b':')
        key = split_line[0]
        value = b':'.join(split_line[1:]).strip()
        note_dict[key.decode()] = value.decode()
    return note_dict


def _get_data(wave: pd.DataFrame):
    '''Utility function for processing the 'data' section of a .ibw file, the end user should not call this function'''

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
    '''Load a .ibw file as a Curve object

    --------------------Arguments--------------------
    filename: File path to load. Should be a .ibw 
    file created by an Asylum AFM

    ---------------------Returns---------------------
    A pyrtz.curves.Curve object which contains the 
    force curve stored in the .ibw file located at 
    filename'''

    wave = bw.load(filename)
    data = _get_data(wave)
    notes = _get_notes(wave)
    trigger_index = np.argmax(data.loc[:, 'defl'])

    sample_time = wave['wave']['wave_header']['sfA'][0]
    t = np.arange(data.shape[0]) * sample_time
    data.loc[:, 't'] = t

    dwell_time = float(notes['DwellTime'])
    dwell_start_time = data.loc[trigger_index, 't']
    dwell_end_time = dwell_start_time + dwell_time

    dwell_end_index = np.argmin(np.abs(data.loc[:, 't'] - dwell_end_time))
    dwell_range = [trigger_index, dwell_end_index]
    k = float(notes['SpringConstant'])

    data.loc[:, 'f'] = data.loc[:, 'defl'] * k

    invOLS = float(notes['InvOLS'])

    this_curve = curves.Curve(
        filename=filename.split(os.path.sep)[-1],
        data=data,
        parameters=notes,
        z_col='z',
        t_col='t',
        f_col='f',
        ind_col='ind',
        invOLS=invOLS,
        k=k,
        dwell_range=dwell_range
    )

    return this_curve


def load_curveset_ibw(folder: str, ident_labels: list[str]) -> curves.CurveSet:
    '''Load a folder of .ibw files as a curves.CurveSet

    --------------------Arguments--------------------
    folder: Path to a directory containing .ibw files
    created by an Asylum AFM to be loaded

    ident_labels: A list of character sequences always
    found (in the order given) in the files of
    interest in folder. Intervening strings will be
    used to distinguish between different measurements

    For example, assume a directory located at
    ~/experiment contains files with names such as:
    Sample1Measurement0.ibw
    Sample1Measurement1.ibw
    Sample2Measurement0.ibw
    Sample2Measurement1.ibw

    This dataset could then be loaded by calling
    pyrtz.asylum.load_curveset_ibw('~/experiment',
    ['Sample','Measurement'])

    ---------------------Returns---------------------
    A curves.CurveSet object containing all
    force curves with matching filenames contained
    in the directory located at folder'''

    ident_labels = tuple(ident_labels)
    regex_str = ""
    for l in ident_labels:
        regex_str = regex_str + l + f'(?P<{l}>.*)'
    regex_str = regex_str + '\.ibw'
    regex = re.compile(regex_str)

    all_filenames = os.listdir(folder)
    all_matches = [regex.match(a) for a in all_filenames]

    curve_dict = dict()
    for m in all_matches:
        if not m:
            continue
        idents = tuple([m.group(a) for a in ident_labels])
        filename = m.group(0)
        filepath = os.path.join(folder, filename)
        curve_dict[idents] = load_ibw(filepath)

    curveset = curves.CurveSet(
        ident_labels=ident_labels, curve_dict=curve_dict)
    return curveset
