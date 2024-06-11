#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Marwa Mahmood intership (UCD student, May-July 2024)

Pytorch datasets
"""
import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset


class TemperatureLandcover1D(Dataset):
    """This is a custom dataset following the tutorial:

    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files


    Examples
    --------
    >>> ds = TemperatureLandcover1D("data/mera-sample-v1.2010-2011-123h.nc")
    >>> x, c, y = ds[0]
    >>> x
    283.5603
    >>> c
    1
    >>> y
    283.8921
    """

    def __init__(self, path, subset="train", seed=87, normalize=False, reduce_to=None):
        self.path = path
        self.subset = subset
        self.seed = seed

        # We open the netCDF file and load the data we need
        ds = xr.open_dataset(self.path)
        t2m = ds.air_temperature_at_2_metres.values  # (143, 489, 529)
        t30m = ds.air_temperature_at_30_metres.values  # (143, 489, 529)
        landcover = ds.landcover.values[0, ::]  # (489, 529)
        ds.close()

        # The landcover is static, so we repeat it
        landcover = np.broadcast_to(landcover, t2m.shape)  # (143, 489, 529)

        # Then, as we are in 1D, we flatten the data
        t2m = t2m.flatten()
        t30m = t30m.flatten()
        landcover = landcover.flatten()

        # Normalization (if needed)
        if normalize:
            t2m = (t2m - t2m.mean()) / t2m.std()
            t30m = (t30m - t30m.mean()) / t30m.std()

        # We split the data in train/test/val randomly, but with a fixed seed to ensure repeatable results
        np.random.seed(self.seed)
        if reduce_to is None:
            reduce_to = t2m.size

        train_idxs = np.random.randint(0, t2m.size, reduce_to // 2)  # 50% in training
        test_idxs = np.random.randint(0, t2m.size, reduce_to // 4)  # 25% in test
        val_idxs = np.random.randint(0, t2m.size, reduce_to // 4)  # 25% in validation

        if self.subset == "train":
            idxs = train_idxs
        elif self.subset == "test":
            idxs = test_idxs
        elif self.subset == "val":
            idxs = val_idxs
        else:
            raise ValueError("Unknown subset")

        # We keep only the selected indices as tensors
        self.t2m = torch.tensor(t2m[idxs])
        self.t30m = torch.tensor(t30m[idxs])
        self.landcover = torch.tensor(landcover[idxs])

    def __len__(self):
        return len(self.t2m)

    def __getitem__(self, idx):
        """Return an item of the dataset.


        Parameters
        ----------
        idx: int
            Index of the item to return


        Returns
        -------
        x: float tensor of shape (1,)
            Temperature at 2m
        c: int64
            Landcover
        y: float tensor of shape (1,)
            Temperature at 30m
        """
        x = self.t30m[idx].unsqueeze(0)
        c = self.landcover[idx].long()
        y = self.t2m[idx].unsqueeze(0)
        return x, c, y
