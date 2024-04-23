import re
from os import listdir
from os.path import abspath, join, isdir, splitext
from pprint import pprint
from typing import Optional, Union, List

import einops
import mne
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from datasets.motor_imagery_base import MotorImageryBaseDataset


class BCICompetition3Dataset3a(MotorImageryBaseDataset):

    def __init__(self, data_path: str,
                 subjects_to_include: Optional[Union[str, List[str]]] = None):
        super().__init__(data_path, subjects_to_include)

        edf_raws = mne.concatenate_raws([
            mne.io.read_raw_gdf(
                join(self.data_path, filename
                     ), stim_channel='auto', verbose=False)
            for filename in listdir(self.data_path)
            if re.fullmatch(pattern=r".*\.gdf", string=filename)
        ])
        # event codes can be found at https://github.com/donnchadh/biosig/blob/master/biosig/doc/eventcodes.txt

        # bundles data into epochs
        self._epochs = mne.Epochs(
            edf_raws,
            mne.events_from_annotations(edf_raws, verbose=False)[0],
            event_id=dict(left_hand=3, right_hand=4),
            tmin=-0.5,
            tmax=4.1,
            proj=False,
            picks=mne.pick_types(edf_raws.info, meg=False, eeg=True, exclude="bads"),
            baseline=None,
            preload=True,
            verbose=False
        )

    def __len__(self):
        return len(self._epochs)

    def __getitem__(self, i):
        x = einops.rearrange(torch.from_numpy(self._epochs.get_data(item=i) * 1e5), "b c s -> (b c) s").float()
        y = torch.as_tensor(self._epochs.events[i, 2] - 3).long()
        return x, y

    @staticmethod
    def get_subject_ids():
        return ["k3b", "k6b", "l1b"]

    @staticmethod
    def get_in_channels():
        return 60