from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy import signal

MAX_MANUAL_R_PEAK_ERROR = 1  # in sample count; how much can the manually labeled R-peak differ from MIT-BIH dataset
# R-peak


def slice_signal(ECG,
                 freq,
                 slice_start_sec,
                 seconds_to_slice):
    """

    :param ECG: ECG signal to be sliced
    :param freq: sampling rate of the signal
    :param slice_start_sec: from where to slice the signal in seconds
    :param seconds_to_slice: how many seconds of the signal to slice
    :return:
    """
    dt = 1 / freq
    time_vector = np.arange(slice_start_sec,
                            slice_start_sec + seconds_to_slice,
                            dt)
    sliced_ECG = ECG[slice_start_sec * freq: (slice_start_sec + seconds_to_slice) * freq]

    return time_vector, sliced_ECG


def signal_filtering(ecg_signal,
                     freq):
    Ts = 1 / freq
    Fs = 1 / Ts
    Fn = Fs / 2

    # Stopband Frequency Vector (Normalised)
    Ws = np.asarray([0.2, 160])
    # Passband Frequency Vector (Normalised)
    Wp = np.asarray([0.5, 150])
    # Passband Ripple (dB)
    Rp = 1
    # Stopband Attenuation (dB)
    Rs = 50

    n, Wp = signal.ellipord((Wp / Fn),
                            (Ws / Fn),
                            Rp,
                            Rs)
    z, p, k = signal.ellip(n,
                           Rp,
                           Rs,
                           Wp,
                           btype = 'bandpass',
                           output = 'zpk')

    # https://stackoverflow.com/questions/51328872/how-to-find-gain-g-from-z-p-k-in-python
    sos1 = signal.zpk2sos(z,
                          p,
                          k)

    sos1 = signal.sosfiltfilt(sos1,
                              ecg_signal)
    EKGII_filtered = sos1

    ecg_signal_mean = np.mean(EKGII_filtered)
    isoeletric_line = np.ones((len(EKGII_filtered), 1)) * ecg_signal_mean

    return EKGII_filtered, isoeletric_line


def annotation_recorder(index,
                        path, time):
    manual_annotation = []
    for i in index:
        y = time - i
        z = np.sign(y)
        u = np.diff(z,
                    axis = 0)
        v = np.abs(u)
        m = np.nonzero(v)[0]

        manual_annotation = np.append(manual_annotation,
                                      m)

    if len(index) % 11 == 0:
        manual_annotation = manual_annotation.reshape(len(index) // 11,
                                                      11)
        manual_annotation_df = pd.DataFrame(manual_annotation,
                                            columns = ['P_s', 'P', 'P_e', 'Q_s', 'Q', 'R', 'S', 'S_e', 'T_s', 'T',
                                                       'T_e'],
                                            dtype = int)
        print(manual_annotation_df)
        manual_annotation_df.to_csv((path + '.csv'),
                                    sep = ';')


def load_record(record_path: str,
                column_names: str = None) -> pd.DataFrame:
    if not column_names:
        column_names = ['MLII', 'V1']

    # Loading the record
    record_wfdb = wfdb.rdrecord(record_path)

    # Converting the record into Data Frame
    return pd.DataFrame(record_wfdb.p_signal,
                        columns = column_names)


def load_record_annotation(record_path: str) -> pd.DataFrame:
    record_annotation_wfdb = wfdb.rdann(record_path,
                                        'atr')

    # Converting the annotation into Data Frame
    record_annotation_data = {
        'sample': record_annotation_wfdb.sample,
        'symbol': record_annotation_wfdb.symbol
    }
    return pd.DataFrame(record_annotation_data,
                        columns = ['sample', 'symbol'])


def get_beats_by_symbols(record_annotation: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """
    Get beats of specific type by their symbol from annotated data

    :param record_annotation: DataFrame with 2 columns: ['sample', 'symbol']. Each row denotes a separate heartbeat.
    Sample is the sample ID of the R-peak of the beat.

    :param symbols: Will return only beats annotated with these characters
    :return: DataFrame filtered by the symbol
    """
    beats = record_annotation[record_annotation['symbol'].isin(symbols)]  # Change the letter and run all below to
    # explore
    # this
    # beat's data in the selected record
    return beats


def get_beat_slices(beats: pd.DataFrame, signal: np.array, window_size: int, manual_label_dict: Dict[int,
                                                                                                     np.ndarray] =
None, moving_average_range: Optional[int] = None) -> \
    (np.ndarray, list):
    """
    For each row in the beats DataFrame this function will return an array of size 2 * window_size.
    Each array are signal values for the beat.
    Middle of the beat is in the beats['sample'] Series, then we also consider window_size samples before and after
    middle sample.

    :param beats:
    :param signal:
    :param window_size:
    :return: beat slice array, indices of selected slices
    """
    beat_slices = []
    beat_slice_shape = (list(manual_label_dict.values())[0].shape[0] + 1, 2 * window_size) if manual_label_dict else 2\
                                                                                                                    * \
                                                                                                           window_size
    beat_slice_indices = []
    if manual_label_dict:
        manual_r_peaks = np.array(list(manual_label_dict.keys()))

    for row in beats.itertuples():
        manual_label_dict_sample_key = None
        if manual_label_dict:
            # Try to match manually labeled R-peaks with dataset R-peaks, within given window of error
            manual_label_dict_sample_key = manual_r_peaks[np.where((row.sample - MAX_MANUAL_R_PEAK_ERROR <=
                                                              manual_r_peaks) & (manual_r_peaks
                                                          <= row.sample + MAX_MANUAL_R_PEAK_ERROR))]
            if not manual_label_dict_sample_key.size == 1:
                continue
            manual_label_dict_sample_key = manual_label_dict_sample_key[0]
            print(f'{row.sample=} {manual_label_dict_sample_key=}')
            beat_slice_indices.append(row.Index)

        beat_slice = np.zeros(beat_slice_shape)
        if isinstance(beat_slice_shape, int):
            if row.sample - window_size < 0:
                beat_slice[abs(row.sample - window_size):] = signal[0: row.sample + window_size]
            elif row.sample + window_size > signal.size:
                beat_slice[:-abs(row.sample + window_size - signal.size)] = signal[row.sample - window_size:]
            else:
                beat_slice[:] = signal[row.sample - window_size: row.sample + window_size]
            if moving_average_range:
                beat_slice[:] = np.convolve(beat_slice[:], np.ones(moving_average_range), 'same') / moving_average_range
        else:
            if row.sample - window_size < 0:
                beat_slice[0, abs(row.sample - window_size):] = signal[0: row.sample + window_size]
            elif row.sample + window_size > signal.size:
                beat_slice[0, :-abs(row.sample + window_size - signal.size)] = signal[row.sample - window_size:]
            else:
                beat_slice[0, :] = signal[row.sample - window_size: row.sample + window_size]
            beat_slice[1:, :] = manual_label_dict[manual_label_dict_sample_key]
            if moving_average_range:
                beat_slice[0, :] = np.convolve(beat_slice[0, :], np.ones(moving_average_range),
                                               'same') / moving_average_range

        beat_slices.append(beat_slice)
    beat_slices = np.array(beat_slices)
    return beat_slices, beat_slice_indices