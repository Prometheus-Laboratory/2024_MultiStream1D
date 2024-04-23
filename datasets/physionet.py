import re
from copy import deepcopy
from os import listdir
from os.path import abspath, join, isdir
from typing import Optional, Union, List, Any

import einops
import mne
import torch

from datasets.motor_imagery_base import MotorImageryBaseDataset


class PhysioNetMotorImageryDataset(MotorImageryBaseDataset):

    def __init__(self, data_path: str,
                 subjects_to_include: Optional[Union[Union[str, int], List[Union[str, int]]]] = None):
        super().__init__(data_path, subjects_to_include)

        # retrieves path to the trials
        edf_raws = mne.concatenate_raws([
            mne.io.read_raw_edf(
                join(self.data_path, folder, filename
                     ), stim_channel='auto', verbose=False)
            for folder in listdir(self.data_path)
            if isdir(join(self.data_path, folder))
               and int(re.search("[0-9]+", folder)[0]) in [int(s_id) for s_id in self.subjects_to_include]
            for filename in listdir(join(self.data_path, folder))
            if filename.endswith(".edf")
               and re.search(f"{folder}R(04|08|12)", filename)
        ])

        # bundles data into epochs
        epochs = mne.Epochs(
            edf_raws,
            mne.events_from_annotations(edf_raws, verbose=False)[0],
            event_id=dict(hands_or_left=2, feet_or_right=3),
            tmin=-0.5,
            tmax=4.1,
            proj=False,
            picks=mne.pick_types(edf_raws.info, meg=False, eeg=True, exclude="bads"),
            baseline=None,
            preload=False,
            verbose=False
        )

        self._x = torch.from_numpy(epochs.get_data() * 1e3).float()
        self._y = torch.as_tensor(epochs.events[:, 2] - 2).long()

    def __len__(self):
        return len(self._x)

    def __getitem__(self, i):
        x, y = self._x[i], self._y[i]
        return x, y

    @staticmethod
    def get_subject_ids():
        return [str(s_id) for s_id in range(1, 109 + 1)
                if s_id not in (43, 88, 89, 92, 100, 104)]

    @staticmethod
    def get_in_channels():
        return 64
