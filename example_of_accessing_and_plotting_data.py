#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Marwa Mahmood intership (UCD student, May-July 2024)

Example code to show how to access the data and plot it
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

from metplotlib import plots
import myfunctions


# Open netcdf file
#-------------
nc_filename = "data/mera-sample-v1.2010-2011-123h.nc"
ds = xr.open_dataset(nc_filename)
print(ds)


# Plot orography
#----------------
lcc = myfunctions.cartopy_crs_from_proj4(ds.crs)
x = ds.eastings.values
y = ds.northings.values
z = ds.orography.values

fig = plt.figure()
ax = plt.subplot(projection=ccrs.PlateCarree())
ax.set_extent([-22, 15, 40, 65])
ax.coastlines(resolution="50m", color="black", linewidth=0.5)
c = ax.pcolormesh(x, y, z, transform = lcc, cmap="terrain", shading="nearest")
plt.colorbar(c, shrink=0.5, label = "m asl")
ax.set_title("Orography in MERA")
fig.show()


# Plot spatial fields at a random date
#--------------------------------------
t_idx = 67
t = ds.valtimes.values[t_idx]
mslp = ds.air_pressure_at_sea_level.values[t_idx, ::] / 100
t2m = ds.air_temperature_at_2_metres.values[t_idx, ::] - 273.15

fig, ax = plots.twovar_plot(mslp, t2m, lons=x, lats=y, cl_varfamily = "temp", title = str(t).split(":")[0], figcrs=lcc, datcrs=lcc)
fig.show()


# Plot the time series of average temperatures
#----------------------------------------------
avg_t2m = ds.air_temperature_at_2_metres.values.mean(axis = (1,2)) - 273.15
avg_t30m = ds.air_temperature_at_30_metres.values.mean(axis = (1,2)) - 273.15
avg_t850 = ds.air_temperature_at_850_hPa.values.mean(axis = (1,2)) - 273.15
avg_t500 = ds.air_temperature_at_500_hPa.values.mean(axis = (1,2)) - 273.15
t = ds.valtimes.values

fig, ax = plt.subplots()
ax.plot(t, avg_t2m, label = "T 2m")
ax.plot(t, avg_t30m, label = "T 30m")
ax.plot(t, avg_t850, label = "T 850")
ax.plot(t, avg_t500, label = "T 500")
ax.grid()
ax.legend()
ax.set_title("Time series of average temperatures")
fig.show()
