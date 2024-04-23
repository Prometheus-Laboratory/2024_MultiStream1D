import re
from copy import deepcopy
from os import listdir
from os.path import abspath, join, isdir
from typing import Optional, Union, List, Any

import einops
import mne
import torch
from torch.utils.data import Dataset
import abc
from abc import ABC, abstractmethod

class MotorImageryBaseDataset(ABC, Dataset):

    def __init__(self, data_path: str,
                 subjects_to_include: Optional[Union[Union[str, int], List[Union[str, int]]]] = None):
        assert isdir(data_path)
        self.data_path = data_path

        # retrieves the list of subject ids and exclude subjects with bad trials
        subject_ids = self.get_subject_ids()

        # eventually exclude patients
        if subjects_to_include is None:
            self.subjects_to_include = deepcopy(subject_ids)
        elif isinstance(subjects_to_include, str) or isinstance(subjects_to_include, int):
            self.subjects_to_include = [str(subjects_to_include)]
        self.subjects_to_include = [str(s_id) for s_id in self.subjects_to_include]
        for s_id in self.subjects_to_include:
            assert s_id in subject_ids, f"{s_id} is not a valid id"

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, i):
        pass

    @staticmethod
    def get_subject_ids():
        raise NotImplementedError

    @staticmethod
    def get_in_channels():
        return NotImplementedError
