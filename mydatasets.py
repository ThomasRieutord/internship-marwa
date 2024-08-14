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








""" Custom datatset including all features """ 

class FeaturesLandcover1D(Dataset):
    

    def __init__(self, path, subset="train", seed=87, normalize=False, reduce_to=None):


        self.path = path
        self.subset = subset
        self.seed = seed

        # We open the netCDF file and load the data we need
        ds = xr.open_dataset(self.path)
        
        # Temperatures:
        t2m = ds.air_temperature_at_2_metres.values  # (143, 489, 529)
        t30m = ds.air_temperature_at_30_metres.values  # (143, 489, 529)
        
        t500hPa = ds.air_temperature_at_500_hPa.values  # (143, 489, 529)
        t850hPa = ds.air_temperature_at_850_hPa.values  # (143, 489, 529)
        
        # Pressures: 
        psurface = ds.air_pressure_at_surface_level.values  # (143, 489, 529)
        psea = ds.air_pressure_at_sea_level.values          # (143, 489, 529)
        
        # Net flux: 
        nlong = ds.net_upward_longwave_flux_in_air.values   # (143, 489, 529)
        nshort = ds.net_upward_shortwave_flux_in_air.values  # (143, 489, 529)
        
        # Humidity:
        h2m = ds.relative_humidity_at_2_metres.values   # (143, 489, 529)
        h30m = ds.relative_humidity_at_30_metres.values  # (143, 489, 529)
        
        # Wind: 
        e30m = ds.eastward_wind_at_30_metres.values    # (143, 489, 529)
        n30m = ds.northward_wind_at_30_metres.values   # (143, 489, 529)
        
        e850hPa = ds.eastward_wind_at_850_hPa.values   # (143, 489, 529)
        n850hPa = ds.northward_wind_at_850_hPa.values  # (143, 489, 529)
        
        # Landcover (categorical):
        landcover = ds.landcover.values[0, :, :]  # (489, 529)
        
        # 2D variables 
        orography = ds.orography.values     # (489, 529)
        eastings = ds.eastings.values       # (489, 529)
        northings = ds.northings.values     # (489, 529)
        
        ds.close()

        # The landcover is static, so we repeat it
        landcover = np.broadcast_to(landcover, t2m.shape)  # (143, 489, 529)
        
        # Similarly with the rest of the 2D variables
        orography = np.broadcast_to(orography, t2m.shape)  # (143, 489, 529)
        eastings = np.broadcast_to(eastings, t2m.shape)    # (143, 489, 529)
        northings = np.broadcast_to(northings, t2m.shape)  # (143, 489, 529)


        
        # Define a list with the numerical input 
        self.numeric_input = [t30m, t500hPa, t850hPa,  psurface, psea, nlong, nshort, h2m,
                          h30m, e850hPa, n850hPa, orography, eastings,  northings]

        
        # Flatten the data
        t2m = t2m.flatten()
        
        landcover = landcover.flatten()
        
        # Flatten numeric input using a loop
        for i in range(len(self.numeric_input)):
            self.numeric_input[i] = self.numeric_input[i].flatten()
            
            
      # Normalization (if needed)
        if normalize:
            t2m = (t2m - t2m.mean()) / t2m.std()
            t30m = (t30m - t30m.mean()) / t30m.std()
            
            # normalize numeric input using a loop
            for i in range(len(self.numeric_input)):
                self.numeric_input[i] = (self.numeric_input[i] - self.numeric_input[i].mean()) / self.numeric_input[i].std()

                
        # We split the data in train/test/val randomly, but with a fixed seed to ensure repeatable results
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        
        # Minimize size of dataset (if needed)
        if reduce_to is None:
            reduce_to = t2m.size

        
        train_idxs = np.random.randint(0, t2m.size, reduce_to // 2)  # 50% in training
        test_idxs = np.random.randint(0, t2m.size, reduce_to // 4)   # 25% in test
        val_idxs = np.random.randint(0, t2m.size, reduce_to // 4)    # 25% in validation

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
        self.landcover = torch.tensor(landcover[idxs])

        # similarly with the input variable using a loop
        for i in range(len(self.numeric_input)):
            self.numeric_input[i] = torch.tensor(self.numeric_input[i][idxs]) 
            
    
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
            Number of features in the Data
        c: int64
            Landcover
        y: float tensor of shape (1,)
            Temperature at 30m
        """
        
        # concatenate the numeric input tensors along the dimension 0 so we have a tensor of shape (1, ...)
        x = torch.cat([self.numeric_input[i][idx].unsqueeze(0) for i in range(len(self.numeric_input))], 0)

        c = self.landcover[idx].long()
        y = self.t2m[idx].unsqueeze(0)
        return x, c, y









