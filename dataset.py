import os
from typing import (
    Dict,
    List,
    Optional,
)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.nn.functional as Fun
import torch

from sklearn.preprocessing import OneHotEncoder

from common import (
    get_beat_slices,
    get_beats_by_symbols,
    load_record,
    load_record_annotation,
)


class ArrhythmiaDataset(Dataset):
    """MIT-BIH Arrhythmia dataset."""

    def __init__(self, root_dir: str, window_size: int, only_include_labels: Optional[List[str]] = None, load_data =
    True,
                 encode_labels =
    True, include_manual_labels = False, moving_average_range: Optional[int] = None, include_raw_signal: bool = True,
                 subset_from_manual_labels = True):
        """

        :param root_dir: Directory path containing all .atr etc. record files
        :param window_size: Slice the signal into 2 * window_size arrays; these are the samples
        :param only_include_labels: Only load samples for these labels
        """
        self.window_size = window_size
        self.root_dir = root_dir
        self.only_include_labels = only_include_labels

        self.data = None
        self.labels = []

        # This allows to initialize training and test dataset as an object of this class without reloading the data
        # files, by simply setting
        # appropriate fields after the call to the constructor
        if load_data:
            self._load_data(include_manual_labels, moving_average_range, include_raw_signal, subset_from_manual_labels)
        if encode_labels:
            self.encode_labels()

    def _df_to_channels(self, df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """

        :param df: Columns are integers that indicate starting, middle, and final sample for each structural feature of
        PQRST ECG wave as well as patient record number
        :return: Maps R-peaks of ECG wave to array of 5 rows, one for each of P, Q, R, S, T segments. Parts of the
        feature in a given row are masked with 1's, other parts are 0's
        """
        r_to_data = dict()
        for row in df.itertuples(index = False):
            row_arr = np.zeros((5, 2 * self.window_size))  # P, Q, R, S, T - 5 input channels for manual features
            idx_normalization_term = - row.R + self.window_size
            if not np.isnan(row.P_s) and not np.isnan(row.P_e):
                row_arr[0, row.P_s + idx_normalization_term: row.P_e + idx_normalization_term] = 1
            if not np.isnan(row.Q_s):
                row_arr[1, row.Q_s + idx_normalization_term: row.R + idx_normalization_term] = 1
            row_arr[2, row.R + idx_normalization_term] = 1
            if not np.isnan(row.S_e):
                row_arr[3, row.R + idx_normalization_term + 1: row.S_e + idx_normalization_term] = 1
            if not np.isnan(row.T_s) and not np.isnan(row.T_e):
                row_arr[4, row.T_s + idx_normalization_term: row.T_e + idx_normalization_term] = 1
            r_to_data[(row.R, row.Record_number)] = row_arr
        return r_to_data

    def _load_data(self, include_manual_labels: bool, moving_average_range: Optional[int] = None, include_raw_signal:
    bool = True, subset_from_manual_labels = True):
        manual_label_dict, manual_r_peaks, manual_record_numbers = None, None, None
        if include_manual_labels or subset_from_manual_labels:
            manual_label_dict = dict()
            dirpath = os.path.join(os.path.dirname(self.root_dir), 'manual_labels')
            for filename in os.listdir(dirpath):
                df = pd.read_csv(os.path.join(dirpath, filename))
                manual_label_dict.update(self._df_to_channels(df))
                print(f'{filename=} Unique_keys={len(manual_label_dict.keys())}')
            manual_r_peaks, manual_record_numbers = np.array(list(zip(*manual_label_dict.keys())))

        # Only include .atr patient files whose record numbers are present in manual label files
        data_files = filter(lambda name: name.endswith('.atr') and name.replace('.atr',
                                                                                   '').isdigit() and ((int(name.replace(
            '.atr',
                                                                                   '')) in manual_record_numbers) if
        manual_record_numbers is not None else True),
                               os.listdir(self.root_dir))

        for filename in data_files:
            curr_patient_manual_label_dict = None
            patient_record_number = int(filename.replace('.atr',
                                                        ''))
            print(f'{filename=} {patient_record_number=}')

            if manual_label_dict:
                # Only include manual label dict entries for the current patient
                curr_patient_manual_label_dict = {r_peak: v for (r_peak, record_number),
                                                                 v in manual_label_dict.items() if record_number ==
                                                  patient_record_number}


            record_path = os.path.join(self.root_dir, str(patient_record_number))
            record = load_record(record_path)
            signal = np.array(record['MLII'])

            record_annotation = load_record_annotation(record_path)
            if self.only_include_labels:
                record_annotation = get_beats_by_symbols(record_annotation, self.only_include_labels)

            # Get data
            beat_slice_array, beat_slice_indices = get_beat_slices(record_annotation,
                                          signal,
                                          self.window_size, curr_patient_manual_label_dict, moving_average_range,
                                                                   include_raw_signal, include_manual_labels)
            beat_slices = torch.tensor(beat_slice_array)
            print(f'{beat_slice_array.shape=} {beat_slices.shape=}')

            # Get labels
            if curr_patient_manual_label_dict:
                label_list = record_annotation.loc[beat_slice_indices].symbol.tolist()
            else:
                label_list = record_annotation.symbol.tolist()
            self.labels += label_list

            # Save data
            if self.data is None:
                self.data = beat_slices
            else:
                print(f'{self.data.shape=} {beat_slices.shape=}')
                self.data = torch.cat((self.data, beat_slices), dim = 0)
            assert self.data.shape[
                       0] == len(self.labels), f'Data contains {self.data.shape[0]} samples, but there a' \
                                            f're {len(self.labels)} ' \
                                          f'labels'

    def encode_labels(self):
        # Translation list + dict to allow Pytorch one_hot() function
        self._label_list = list(set(self.labels))
        self._label_dict = {label: idx for idx, label in enumerate(self._label_list)}
        self.labels_encoded = Fun.one_hot(torch.tensor([self._label_dict[label] for label in self.labels]))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx], self.labels_encoded[idx]

    def get_label_from_tensor(self, t: torch.Tensor) -> str:
        unique_vals = t.unique()
        assert len(unique_vals) == 2
        assert torch.sum(t) == 1
        assert 0 in unique_vals and 1 in unique_vals
        return self._label_list[torch.argmax(t)]

    def train_test_split(self, test_size: float, shuffle = True, random_state = 0):
        # TODO: This is BAD; many fields are not copied here, labels could be handled more elegantly, new datasets
        #  should have identical properties to the parent dataset, except for the actual data and targets
        # Split data
        train_idx, valid_idx = train_test_split(
            np.arange(len(self.labels)),
            test_size = test_size,
            random_state = random_state,
            shuffle = shuffle,
            stratify = self.labels_encoded)
        (train_data, train_labels), (test_data, test_labels) = self[train_idx], self[valid_idx]

        # Prepare training dataset
        train_dataset = ArrhythmiaDataset(self.root_dir, self.window_size, self.only_include_labels, load_data =
        False, encode_labels = False)
        train_dataset.data, train_dataset.labels_encoded = train_data, train_labels
        train_dataset._label_list = self._label_list
        train_dataset._label_dict = self._label_dict

        # Prepare test dataset
        test_dataset = ArrhythmiaDataset(self.root_dir, self.window_size, self.only_include_labels, load_data =
        False, encode_labels = False)
        test_dataset.data, test_dataset.labels_encoded = test_data, test_labels
        test_dataset._label_list = self._label_list
        test_dataset._label_dict = self._label_dict

        return train_dataset, test_dataset

