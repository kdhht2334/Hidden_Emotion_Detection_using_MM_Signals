#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kdh
@email: kdhht5022@gmail.com
"""


import os
from PIL import Image

import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class FaceDataset(Dataset):
    """Face dataset."""

    def __init__(self, csv_file, root_dir, transform=None, inFolder=None, landmarks=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.training_sheet = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        if inFolder.any() == None:
            self.inFolder = np.full((len(self.training_sheet),), True)
        
        self.loc_list = np.where(inFolder)[0]
        self.infold = inFolder
        
    def __len__(self):
        return  np.sum(self.infold*1)     

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.training_sheet.iloc[idx, 0])
        arousal = self.training_sheet.iloc[idx,1]
        valence = self.training_sheet.iloc[idx,2]
        
        eeg = os.path.join(self.root_dir,
                                self.training_sheet.iloc[idx, 3])
        eeg_signal = np.load(eeg)
        
        image = Image.open(img_name)
        sample = image
        
        if self.transform:
            sample = self.transform(sample)
        return {'image': sample, 'va': [valence, arousal], 'eeg': eeg_signal}