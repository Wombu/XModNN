import copy

from torch.utils import data
import numpy as np
import torch
import random
import pandas as pd

class Dataset(data.Dataset):
    def __init__(self, x=None, y=None):
        self.x = x
        self.length_ = len(x.iloc[0].tolist())

        self.label_list = sorted(list(set(y.iloc[0].tolist())))
        prefix_tmp = "prefix_that_needs_to_be_removed_afterwards_and_should_not_match_the_label"
        y = pd.get_dummies(y.T, prefix=prefix_tmp, prefix_sep="").T
        y.index = [i.replace(prefix_tmp, "") for i in list(y.index)]
        self.y = y.loc[self.label_list]

    def __len__(self):
        # Denotes the total number of samples
        return self.length_

    def __getitem__(self, index):
        # Generates one sample of data
        # Select sample
        if isinstance(index, int):
            index = [index]

        if isinstance(index, slice):
            index = list(range(index.stop)[index])  #https://stackoverflow.com/questions/13855288/turn-slice-into-range

        x = self.x.iloc[:, index].T.to_dict(orient="list")
        y = self.y.iloc[:, index].T.to_dict(orient="list")

        x = {key: torch.tensor(item) for key, item in x.items()}
        y = {key: torch.tensor(item) for key, item in y.items()}

        return x, y
