#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Marwa Mahmood intership (UCD student, May-July 2024)

Module with useful functions


Notes
-----
This file must contain only functions. No main code.

To use it in a main code (located in the same directory), simply include
`import myfunctions` at the begining of your program or notebook.
"""

import cartopy.crs as ccrs


def lineparser(line, startword, stopword=None):
    """Extract content from a line of text.

    Parameters
    ----------
    line: str
        Line of text from which we extract content

    startword: str
        Text pattern marking the begining of the content to extract

    stopword: str, optional
        Text pattern marking the end of the content to extract. If not
        provided, the function extracts up to the end of the line.


    Example
    -------
    >>> line = "This is a line from a log file. Recorded value=0.25645"
    >>> lineparser(line, "Recorded value=")
    "0.25645"
    >>> lineparser(line, "This is a line from ", ".")
    "a log file"
    """
    matchidx = line.index(startword)
    startidx = matchidx + len(startword)
    if stopword is not None:
        endidx = startidx + line[startidx:].index(stopword)
    else:
        endidx = len(line) + 1

    return line[startidx:endidx]


def cartopy_crs_from_proj4(proj4):
    """Return the Cartopy Lambert conformal conic projection corresponding
    to the given string in Proj4 convention.


    Example
    -------
    >>> proj4 = '+proj=lcc +lat_0=53.5 +lon_0=5.0 +lat_1=53.5 +lat_2=53.5 +x_0=0.0 +y_0=0.0 +datum=WGS84 +units=m +no_defs'
    >>> crs = cartopy_crs_from_proj4(proj4)
    """
    lat_0 = float(lineparser(proj4, "+lat_0=", " "))
    lon_0 = float(lineparser(proj4, "+lon_0=", " "))
    lat_1 = float(lineparser(proj4, "+lat_1=", " "))
    lat_2 = float(lineparser(proj4, "+lat_2=", " "))

    return ccrs.LambertConformal(
        central_longitude=lon_0,
        central_latitude=lat_0,
        standard_parallels=(lat_1, lat_2),
    )
